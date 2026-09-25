"""Answer forward KL and same-prompt capped soft InfoNCE with live donor gradients."""

from __future__ import annotations

import torch
import torch.distributed as dist


def c_components(
    model,
    *,
    prompt_ids,
    prompt_mask,
    direct_prompt_ids,
    direct_prompt_mask,
    native_cot_ids,
    native_cot_mask,
    view_to_sample,
    true_z,
    global_z,
    wrong_candidate_mask,
    global_match_sample_count,
    global_specific_sample_count,
    distill_row_weights,
    physical_chunk_size,
    wrong_control_chunk_size,
    generated_prefix,
    specificity_tau=0.1,
    timer=None,
    staged=True,
    collect_diagnostics=False,
    distill_cot_ids=None,
    distill_cot_mask=None,
    specificity_donor_prompt_ids=None,
    specificity_donor_prompt_mask=None,
    true_z_views=None,
    specificity_positive_mask=None,
):
    """Teacher||student KL on a shared stopped student-answer prefix for the configured native population.

    Specificity uses capped soft InfoNCE over all dropout true-z views.
    Students use pure z. The teacher is frozen; true and donor z stay live.
    Match uses B+C row weights. Specificity uses its own eligible-C denominator;
    its owner mask was sealed by the global-window sampler.

    Production consumes each donor F graph immediately, retaining its scalar
    distance and per-pair latent VJP. Once all candidate distances are known,
    scalar-loss derivatives weight those VJPs. This is valid because F has no
    cross-example operations; it avoids retaining or replaying donor F graphs.
    """
    from think_bridge.model.parallel_model import (
        Route1SpecificityStats,
        _route1_rng_state,
        _route1_replay_with_rng,
        _route1_vector_vjp,
        _route1_accumulate_indexed_gradient,
    )
    from think_bridge.model.objectives import (
        _streamed_forward_kl_rows_from_hidden_validated,
    )
    from think_bridge.model.trajectory import (
        StoppedAnswerPrefix,
        generate_true_z_prefix,
    )
    from think_bridge.stage1.methods.bridge.specificity import (
        masked_capped_soft_infonce,
        similarity_negative_weights,
        mean_one_donor_weights,
    )

    if true_z_views is None:
        true_z_views = (true_z,)
    else:
        true_z_views = tuple(true_z_views)
        if not true_z_views:
            raise ValueError("true_z_views cannot be empty")
        if any(z.ndim != 3 or z.shape != true_z.shape for z in true_z_views):
            raise ValueError("all true z views must share [B,K,D] geometry")
        if true_z_views[0] is not true_z:
            raise ValueError("the first true_z_view must be the owner z tensor")
    view_count = len(true_z_views)

    if (
        distill_row_weights.shape != view_to_sample.shape
        or distill_row_weights.dtype != torch.float32
        or distill_row_weights.requires_grad
        or not bool(torch.isfinite(distill_row_weights).all())
        or bool((distill_row_weights <= 0).any())
    ):
        raise ValueError("invalid native distillation row weights")
    if (distill_cot_ids is None) != (distill_cot_mask is None):
        raise ValueError("distillation CoT prefix ids/mask must be jointly supplied")
    if distill_cot_ids is None:
        distill_cot_ids = native_cot_ids[:, :0]
        distill_cot_mask = native_cot_mask[:, :0]
    if (
        distill_cot_ids.ndim != 2
        or distill_cot_ids.shape != distill_cot_mask.shape
        or distill_cot_ids.size(0) != view_to_sample.numel()
    ):
        raise ValueError("distillation context must align with C views")
    if bool(distill_cot_mask.any()):
        raise ValueError("native distillation must not contain curriculum CoT")
    if specificity_donor_prompt_ids is None:
        specificity_donor_prompt_ids = true_z.new_empty((0, 1), dtype=torch.long)
        specificity_donor_prompt_mask = true_z.new_empty((0, 1), dtype=torch.bool)
    if specificity_donor_prompt_mask is None:
        raise ValueError("specificity donor prompt ids/mask must be supplied together")
    if (
        specificity_donor_prompt_ids.ndim != 2
        or specificity_donor_prompt_ids.shape != specificity_donor_prompt_mask.shape
    ):
        raise ValueError("specificity donor prompt bank geometry is invalid")
    wrong_live = True
    same_prompt_loss = True
    normalized_capped_loss = True
    capped_loss = True
    window_donor_bank = bool(specificity_donor_prompt_ids.numel() > 0)
    compute_match = global_match_sample_count > 0
    compute_specific = global_specific_sample_count > 0
    positive_candidates = torch.zeros_like(wrong_candidate_mask, dtype=torch.bool)
    if same_prompt_loss and specificity_positive_mask is not None:
        if (
            specificity_positive_mask.shape != wrong_candidate_mask.shape
            or specificity_positive_mask.dtype != torch.bool
            or bool((specificity_positive_mask & wrong_candidate_mask).any())
        ):
            raise ValueError(
                "same-prompt positive mask must align and exclude negative candidates"
            )
        positive_candidates = (
            specificity_positive_mask & wrong_candidate_mask.any(1)[:, None]
        )
    if bool(positive_candidates.any()) and not window_donor_bank:
        raise ValueError(
            "same-prompt positives require an optimizer-window prompt bank"
        )
    world = (
        dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    )
    if view_count == 1:
        gm = torch.zeros_like(true_z, dtype=torch.float32)
        gs = torch.zeros_like(true_z, dtype=torch.float32)
    else:
        gm = torch.zeros(
            (view_count, *true_z.shape), dtype=torch.float32, device=true_z.device
        )
        gs = torch.zeros_like(gm)
    gd = torch.zeros_like(global_z, dtype=torch.float32)
    mv = true_z.new_zeros((), dtype=torch.float32)
    sv = mv.clone()
    counts = dict(
        c_view_count=int(view_to_sample.numel()),
        legal_wrong_pair_count=0,
        wrong_physical_chunk_count=0,
    )
    covered = physical = 0
    distances = (
        true_z.new_zeros(3, dtype=torch.float32) if collect_diagnostics else None
    )
    margin_diagnostics = (
        true_z.new_zeros(2, dtype=torch.float32)
        if collect_diagnostics and capped_loss
        else None
    )
    cap_diagnostics = (
        true_z.new_zeros(2, dtype=torch.float32)
        if collect_diagnostics and capped_loss
        else None
    )
    donor_z_bank = None
    donor_z_indices = None
    donor_grad_bank = None
    donor_surrogate = None
    positive_banks = []
    positive_grad_banks = []
    positive_selected = None
    if compute_specific and window_donor_bank:
        selected_mask = (
            (wrong_candidate_mask | positive_candidates).any(0).to(dtype=torch.int32)
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(selected_mask, op=dist.ReduceOp.MAX)
        selected = torch.nonzero(selected_mask.bool(), as_tuple=False).flatten()
        if selected.numel():
            donor_z_indices = selected
            donor_prompt_ids = specificity_donor_prompt_ids.index_select(0, selected)
            donor_prompt_mask = specificity_donor_prompt_mask.index_select(0, selected)
            donor_z_bank = model.reason(donor_prompt_ids, donor_prompt_mask)
            donor_grad_bank = torch.zeros_like(donor_z_bank, dtype=torch.float32)
            pos_used = positive_candidates.any(0).to(torch.int32)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(pos_used, op=dist.ReduceOp.MAX)
            positive_selected = torch.nonzero(pos_used.bool(), as_tuple=False).flatten()
            if positive_selected.numel():
                first = donor_z_bank.index_select(
                    0, torch.searchsorted(selected, positive_selected)
                )
                positive_banks = [first]
                for _ in range(1, view_count):
                    positive_banks.append(
                        model.reason(
                            specificity_donor_prompt_ids.index_select(
                                0, positive_selected
                            ),
                            specificity_donor_prompt_mask.index_select(
                                0, positive_selected
                            ),
                        )
                    )
                positive_grad_banks = [
                    torch.zeros_like(v, dtype=torch.float32) for v in positive_banks
                ]
            physical += 1

    def prompt_repr(ids, mask):
        """Detached frozen-F prompt representation used only as a soft prior."""
        embedding = model.executor.get_input_embeddings().weight
        hidden = (
            embedding.index_select(0, ids.reshape(-1)).reshape(*ids.shape, -1).float()
        )
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    donor_prompt_repr = (
        prompt_repr(
            specificity_donor_prompt_ids, specificity_donor_prompt_mask
        ).detach()
        if window_donor_bank and specificity_donor_prompt_ids.numel() > 0
        else None
    )
    match_parts, specific_parts, owner_views, donor_views = [], [], [], []
    head = model.executor.lm_head

    def timed(stage, fn):
        return (
            fn()
            if timer is None
            else timer.call(stage.replace("-", "_") + "_seconds", fn, stage=stage)
        )

    for start in range(0, view_to_sample.numel(), physical_chunk_size):
        if not (compute_match or compute_specific):
            break
        stop = min(start + physical_chunk_size, view_to_sample.numel())
        indices = view_to_sample[start:stop].long()
        z = true_z.index_select(0, indices)
        pids, pmask = prompt_ids[indices], prompt_mask[indices]
        prefix = (
            StoppedAnswerPrefix(
                generated_prefix.token_ids[start:stop].detach(),
                generated_prefix.token_mask[start:stop].detach(),
            )
            if generated_prefix is not None
            else generate_true_z_prefix(
                model.executor,
                embedding=model.executor.get_input_embeddings(),
                prompt_ids=pids,
                prompt_mask=pmask,
                z=z.detach(),
                boundary_ids=model.boundary_ids,
                eos_token_id=model.eos_token_id,
                max_steps=model.trajectory_max_steps,
                temperature=model.route1_generation_temperature,
                seed=model.generation_seed,
                cot_ids=None,
                cot_mask=None,
            )
        )
        width = int(prefix.token_mask.sum(1).max())
        aids, amask = prefix.token_ids[:, :width], prefix.token_mask[:, :width]
        weights = distill_row_weights[start:stop] * world

        def branch(
            selected_z=None,
            *,
            own_rows=None,
            cot=None,
            cotmask=None,
            no_z=False,
            live=False,
        ):
            rows = (
                torch.arange(stop - start, device=z.device)
                if own_rows is None
                else own_rows
            )
            teacher_branch = cot is not None
            geometry = (
                "natural_cot"
                if teacher_branch
                else ("direct" if no_z else "deployed_z")
            )
            h, b = model._branch_hidden(
                prompt_ids=(direct_prompt_ids[indices][rows] if no_z else pids[rows]),
                prompt_mask=(
                    direct_prompt_mask[indices][rows] if no_z else pmask[rows]
                ),
                answer_ids=aids[rows],
                answer_mask=amask[rows],
                geometry=geometry,
                live=live,
                checkpoint_backbone=bool(model.route1_gradient_checkpointing and live),
                z=selected_z,
                cot_ids=cot,
                cot_mask=cotmask,
            )
            return h, b.target_mask

        teacher, valid = timed(
            "specificity-teacher",
            lambda: branch(
                cot=native_cot_ids[start:stop], cotmask=native_cot_mask[start:stop]
            ),
        )
        teacher = teacher.detach()
        physical += 1

        def kl(h, mask, rows=None):
            t = teacher if rows is None else teacher[rows]
            return _streamed_forward_kl_rows_from_hidden_validated(
                t[:, : h.size(1)],
                h,
                mask,
                mask.sum(1),
                lm_head_weight=head.weight,
                lm_head_bias=getattr(head, "bias", None),
            )

        own_rng = _route1_rng_state(z.device)
        owns, leaves, dtrues, dhs = [], [], [], []
        for view_z in true_z_views:
            view_z_chunk = view_z.index_select(0, indices)
            own, _ = timed(
                "specificity-student",
                lambda view_z_chunk=view_z_chunk: branch(view_z_chunk, live=not staged),
            )
            physical += 1
            leaf = own.detach().requires_grad_(True) if staged else own
            dtrue = timed("specificity-head", lambda leaf=leaf: kl(leaf, valid))
            owns.append(own)
            leaves.append(leaf)
            dtrues.append(dtrue)
            if staged:
                dhs.append(torch.autograd.grad(dtrue.sum(), leaf)[0])
        dtrue_matrix = torch.stack(dtrues, dim=1)
        # Reuse the same true KL and hidden derivatives for Match and
        # Specificity.  Match is the mean over stochastic views; Specificity
        # keeps all views as simultaneous positives below.
        if compute_match:
            if staged:
                mv = mv + (dtrue_matrix.detach().mean(dim=1) * weights).sum()
            else:
                match_parts.append(dtrue_matrix.mean(dim=1))

        true_coeff = torch.zeros_like(dtrue_matrix)
        local_specific = False
        if compute_specific:
            dcontrol = None
            legal = wrong_candidate_mask[indices].bool()
            soft_weights = None
            if donor_prompt_repr is not None:
                owner_repr = prompt_repr(pids, pmask).detach()
                soft_weights = similarity_negative_weights(
                    owner_repr, donor_prompt_repr
                )

            def specific_rows(positive, negative, positive_mask=None):
                return masked_capped_soft_infonce(
                    positive,
                    negative,
                    legal,
                    donor_weights=soft_weights,
                    negative_kl_cap=model.route1_specificity_negative_kl_cap,
                    temperature=model.route1_specificity_temperature,
                    normalize_weights=normalized_capped_loss,
                    positive_mask=positive_mask,
                    normalize_candidate_counts=same_prompt_loss,
                )

            # Match is an empirical B+C mean. Specificity averages only
            # eligible C (or label-free reference) owners with a wrong donor.
            specific_weights = legal.any(dim=1).to(weights.dtype) * (
                world / max(int(global_specific_sample_count), 1)
            )
            covered += int(legal.any(1).sum())
            pairs = torch.nonzero(legal, as_tuple=False)
            local_specific = pairs.size(0) > 0
            counts["legal_wrong_pair_count"] += pairs.size(0)
            wrong = dtrue_matrix.new_zeros(legal.shape)
            donor_vjps = []
            for pos in range(0, pairs.size(0), wrong_control_chunk_size):
                pair = pairs[pos : pos + wrong_control_chunk_size]
                local, donor = pair[:, 0].long(), pair[:, 1].long()
                if window_donor_bank:
                    donor_local = torch.searchsorted(donor_z_indices, donor)
                    donor_z = donor_z_bank.index_select(0, donor_local)
                else:
                    donor_local = donor
                    donor_z = global_z.index_select(0, donor)
                sw, mask = timed(
                    "specificity-donor-student",
                    lambda: branch((donor_z), own_rows=local, live=wrong_live),
                )
                dw = timed("specificity-head", lambda: kl(sw, mask, local))
                if staged and wrong_live:
                    dz = timed(
                        "specificity-donor-backward",
                        lambda: _route1_vector_vjp(
                            dw,
                            donor_z,
                            torch.ones_like(dw),
                            retain_graph=window_donor_bank,
                        ),
                    )
                    donor_vjps.append((local, donor, donor_local, dz.detach().float()))
                    wrong[local, donor] = dw.detach()
                wrong = wrong.index_put((local, donor), dw)
                donor_views.append(donor_z)
                physical += 1
                counts["wrong_physical_chunk_count"] += 1
                del sw, dw, donor_z
            # Score each extra positive against THIS owner's teacher and prefix.
            positive_legal = positive_candidates[indices]
            positive_pairs = torch.nonzero(positive_legal, as_tuple=False)
            extra = dtrue_matrix.new_zeros(
                (stop - start, positive_legal.size(1) * view_count)
            )
            extra_mask = positive_legal.repeat_interleave(view_count, dim=1)
            positive_vjps = []
            for pos in range(0, positive_pairs.size(0), wrong_control_chunk_size):
                pair = positive_pairs[pos : pos + wrong_control_chunk_size]
                local, donor = pair[:, 0].long(), pair[:, 1].long()
                bank_index = torch.searchsorted(positive_selected, donor)
                for view_index, bank in enumerate(positive_banks):
                    selected_z = bank.index_select(0, bank_index)
                    h, mask = timed(
                        "specificity-positive-student",
                        lambda: branch(selected_z, own_rows=local, live=True),
                    )
                    dp = timed("specificity-head", lambda: kl(h, mask, local))
                    columns = donor * view_count + view_index
                    if staged:
                        dz = timed(
                            "specificity-positive-backward",
                            lambda: _route1_vector_vjp(
                                dp, selected_z, torch.ones_like(dp), retain_graph=True
                            ),
                        )
                        positive_vjps.append(
                            (
                                local,
                                columns,
                                bank_index,
                                view_index,
                                dz.detach().float(),
                            )
                        )
                        extra[local, columns] = dp.detach()
                    else:
                        extra = extra.index_put((local, columns), dp)
                    physical += 1
                    del h, dp, selected_z
            all_positive = torch.cat((dtrue_matrix, extra), dim=1)
            all_positive_mask = torch.cat(
                (torch.ones_like(dtrue_matrix, dtype=torch.bool), extra_mask), dim=1
            )
            if staged:
                true_leaf = all_positive.detach().requires_grad_(True)
                wrong_leaf = wrong.detach().requires_grad_(wrong_live)
                values = specific_rows(true_leaf, wrong_leaf, all_positive_mask)
                contribution = (values * specific_weights).sum()
                true_coeff, wrong_coeff = torch.autograd.grad(
                    contribution, (true_leaf, wrong_leaf)
                )
                extra_coeff = true_coeff[:, view_count:]
                true_coeff = true_coeff[:, :view_count]
                for local, columns, bank_index, view_index, dz in positive_vjps:
                    _route1_accumulate_indexed_gradient(
                        positive_grad_banks[view_index],
                        bank_index,
                        dz * extra_coeff[local, columns, None, None].float(),
                    )
                del positive_vjps, extra_coeff
                sv = sv + contribution.detach()
                for local, donor, donor_local, dz in donor_vjps:
                    weighted = dz * wrong_coeff[local, donor, None, None].float()
                    if window_donor_bank:
                        _route1_accumulate_indexed_gradient(
                            donor_grad_bank, donor_local, weighted
                        )
                    else:
                        _route1_accumulate_indexed_gradient(gd, donor, weighted)
                if window_donor_bank and wrong_live and donor_z_bank is not None:
                    raw = (donor_z_bank.float() * donor_grad_bank.detach()).sum()
                    donor_surrogate = raw - raw.detach()
                del donor_vjps, wrong_coeff
            else:
                values = specific_rows(all_positive, wrong, all_positive_mask)
                # compose_route1_population_loss applies the Match row weights;
                # compensate here so the non-staged path has the same C mean.
                specific_parts.append(
                    values * specific_weights / weights + global_z.sum() * 0.0
                )
            if distances is not None:
                w = specific_weights / world
                wrong_mean = (wrong.detach() * legal).sum(1) / legal.sum(1).clamp_min(1)
                direct_mean = (
                    (dcontrol * w).sum()
                    if dcontrol is not None
                    else dtrue_matrix.new_zeros(())
                )
                distances += torch.stack(
                    (
                        (dtrue_matrix.detach().mean(dim=1) * w).sum(),
                        direct_mean,
                        (wrong_mean * w).sum(),
                    )
                )
                if margin_diagnostics is not None:
                    prior = (
                        torch.ones_like(wrong) if soft_weights is None else soft_weights
                    )
                    prior = mean_one_donor_weights(prior, legal)
                    active_mean = torch.zeros_like(w)
                    weight_mean = (prior * legal).sum(1) / legal.sum(1).clamp_min(1)
                    margin_diagnostics += torch.stack(
                        ((active_mean * w).sum(), (weight_mean * w).sum())
                    )
                if cap_diagnostics is not None:
                    raw_wrong = wrong.detach().masked_fill(~legal, 0.0)
                    cap = model.route1_specificity_negative_kl_cap
                    denominator = legal.sum(1).clamp_min(1)
                    saturated = ((raw_wrong >= cap) & legal).float().sum(
                        1
                    ) / denominator
                    effective = raw_wrong.clamp_max(cap).sum(1) / denominator
                    cap_diagnostics += torch.stack(
                        ((saturated * w).sum(), (effective * w).sum())
                    )

        if staged:
            # A globally active C objective can have no eligible owner in this
            # physical chunk (e.g. B-only). Its owner VJP is exactly zero.
            # These are local frozen-F autograd calls, not collectives; leave
            # the window donor bank and downstream R synchronization intact.
            for view_index, view_z in enumerate(true_z_views):
                if not (compute_match or local_specific):
                    continue
                view_z_chunk = view_z.index_select(0, indices)
                outer = _route1_rng_state(view_z.device)
                replay, _ = timed(
                    "specificity-owner-replay",
                    lambda view_z=view_z: _route1_replay_with_rng(
                        lambda: branch(view_z_chunk, live=True),
                        replay_state=own_rng,
                        outer_state=outer,
                        device=view_z.device,
                    ),
                )
                if compute_match:
                    dz = _route1_vector_vjp(
                        replay.reshape(-1),
                        view_z_chunk,
                        (
                            dhs[view_index].float()
                            * (weights / float(view_count))[:, None, None]
                        )
                        .to(replay.dtype)
                        .reshape(-1),
                        retain_graph=local_specific,
                    )
                    target_gradient = gm if view_count == 1 else gm[view_index]
                    _route1_accumulate_indexed_gradient(target_gradient, indices, dz)
                if local_specific:
                    dz = _route1_vector_vjp(
                        replay.reshape(-1),
                        view_z_chunk,
                        (
                            dhs[view_index].float()
                            * true_coeff[:, view_index, None, None]
                        )
                        .to(replay.dtype)
                        .reshape(-1),
                    )
                    target_gradient = gs if view_count == 1 else gs[view_index]
                    _route1_accumulate_indexed_gradient(target_gradient, indices, dz)
                physical += 1
                del replay, dz
            del dhs
        else:
            owner_views.append(z)
    if staged and same_prompt_loss:
        positive_surrogate = true_z.sum() * 0.0
        for bank, gradient in zip(positive_banks, positive_grad_banks):
            raw = (bank.float() * gradient.detach()).sum()
            positive_surrogate = positive_surrogate + raw - raw.detach()
        negative_surrogate = (
            donor_surrogate if donor_surrogate is not None else true_z.sum() * 0.0
        )
        donor_surrogate = (positive_surrogate, negative_surrogate)
    stats = Route1SpecificityStats(
        covered if compute_specific else 0,
        covered,
        counts["legal_wrong_pair_count"],
        *([None] * 3 if distances is None else [x.detach() for x in distances]),
        margin_active_fraction=(None),
        soft_weight_mean=(
            margin_diagnostics[1].detach() if margin_diagnostics is not None else None
        ),
        negative_cap_fraction=(
            cap_diagnostics[0].detach() if cap_diagnostics is not None else None
        ),
        effective_wrong_kl=(
            cap_diagnostics[1].detach() if cap_diagnostics is not None else None
        ),
    )
    if staged:
        return gm, gs, gd, mv, sv, physical, counts, stats, True, donor_surrogate
    empty = true_z.sum().mul(0.0).expand(0)
    return (
        torch.cat(match_parts) if match_parts else empty,
        torch.cat(specific_parts) if specific_parts else empty,
        physical,
        (
            tuple(owner_views) if compute_match else (),
            tuple(owner_views) if compute_specific else (),
            tuple(donor_views),
            True,
        ),
        counts,
        stats,
        donor_surrogate,
    )
