"""Independent HF replicas; partition whole groups before loading each GPU model."""
from __future__ import annotations
import copy
import json
import multiprocessing
from pathlib import Path
import time


def shard_groups(rows, *, rank, world_size, size=8):
    from think_bridge.model.evaluation_groups import evaluation_group_indices
    return [rows[i] for group in evaluation_group_indices(len(rows), rank=rank,
            world_size=world_size, group_size=size) for i in group]


def _worker(arguments, rank, world_size, output):
    args = copy.copy(arguments)
    args.hf_workers = 1
    args._hf_rank, args._hf_world_size = rank, world_size
    args.local_device = f'cuda:{rank}'
    args.output_dir = Path(output) / 'workers' / f'rank-{rank}'
    args.no_progress = bool(args.no_progress) or rank != 0
    if not args.dataset:
        # Keep the requested worker topology even when a rank has no batches.
        # No model needs loading for an idle worker.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / 'benchmark_summary.json').write_text(json.dumps({
            'mode': args.mode, 'datasets': [], 'complete': True, 'idle': True,
        }))
        return
    import torch
    torch.cuda.set_device(rank)
    from think_bridge.eval.multiturn import run_multiturn_benchmark
    with torch.inference_mode():
        run_multiturn_benchmark(args)


def merge_workers(output, world_size, *, expected_ids=None, group_size=8,
                  expected_dataset_ranks=None):
    """Merge per-question evidence; never average rank accuracies or TTFT means."""
    from think_bridge.eval.benchmark import _write_json_new

    paths = [output / 'workers' / f'rank-{rank}' for rank in range(world_size)]
    if expected_ids is not None:
        expected_ids = {str(Path(path).expanduser().resolve()): ids for path, ids in expected_ids.items()}
    if expected_dataset_ranks is not None:
        expected_dataset_ranks = {str(Path(path).expanduser().resolve()): set(ranks)
                                  for path, ranks in expected_dataset_ranks.items()}
    runs = [json.loads((p / 'benchmark_summary.json').read_text()) for p in paths]
    active_runs = [r for r in runs if not r.get('idle')]
    run = copy.deepcopy(active_runs[0])
    for other in active_runs[1:]:
        if (other.get('history_generation') != run.get('history_generation') or
                any(other.get('identity', {}).get(k) != run.get('identity', {}).get(k)
                    for k in ('runtime', 'generation', 'timing'))):
            raise ValueError('HF workers mixed runtime, timing or batch configurations')
    run['datasets'] = []
    multi = run['mode'] == 'multi-turn'
    if not multi:
        raise ValueError('HF replica merge requires multi-turn results')
    run['device_layout'] = {'backend': 'hf', 'replicas': world_size, 'group_size': group_size,
                            'assignment': 'whole-groups-round-robin'}
    run['worker_summaries'] = [str(p / 'benchmark_summary.json') for p in paths]
    if multi:
        # Worker identity hashes refer to the exact worker input slices, not the merge.
        for key in ('identity', 'identity_sha256', 'prediction_files', 'execution'):
            run.pop(key, None)
    rank_names = [{p.name for p in path.glob('*.predictions.json')} for path in paths]
    names = set().union(*rank_names)
    if not names or (expected_dataset_ranks is None and any(local != names for local in rank_names)):
        raise ValueError('HF workers produced different dataset/history outputs')
    seen_datasets = set()
    merged_predictions = {}
    for name in sorted(names):
        ranks = [rank for rank, local in enumerate(rank_names) if name in local]
        source_paths = [paths[rank] for rank in ranks]
        bundles = [json.loads((p / name).read_text()) for p in source_paths]
        rows = [row for bundle in bundles for row in bundle['rows']]
        metadata = copy.deepcopy(bundles[0]['metadata'])
        if any(b['metadata']['dataset'] != metadata['dataset'] for b in bundles):
            raise ValueError('HF worker dataset identities differ')
        dataset_key = str(Path(metadata['dataset']).expanduser().resolve())
        seen_datasets.add(dataset_key)
        if expected_dataset_ranks is not None and set(ranks) != expected_dataset_ranks.get(dataset_key):
            raise ValueError('HF worker outputs do not match the dataset assignment')
        if multi:
            if any(b['metadata'].get('reader_context') != metadata.get('reader_context') for b in bundles):
                raise ValueError('HF workers mixed Reader context interventions')
            if any((b['metadata'].get('protocol'), b['metadata'].get('history_contract')) !=
                   (metadata.get('protocol'), metadata.get('history_contract')) for b in bundles):
                raise ValueError('HF workers mixed multi-turn history protocols')
            from think_bridge.eval.multiturn import _summarize, _timed_accuracy_summary
            from think_bridge.eval.conversation_history import load_conversations
            # Preserve original conversation and turn ordering in the merged artifact.
            conversations = load_conversations(Path(metadata['dataset']))
            # Conversation IDs can be strings or integers; sort from source order.
            order = {str(row['id']): i for i, row in enumerate(row for c in conversations for row in c)}
            rows.sort(key=lambda r: order[str(r['id'])])
            if len({str(r['id']) for r in rows}) != len(rows):
                raise ValueError('Duplicate multi-turn shard rows')
            actual = [str(r['id']) for r in rows]
            measurement = metadata.get('measurement', 'accuracy')
            if any(b['metadata'].get('measurement', 'accuracy') != measurement for b in bundles):
                raise ValueError('HF workers mixed measurement protocols')
            metric = _summarize(rows, measurement)
            merged_predictions.setdefault(dataset_key, {})[metadata['history_source']] = rows
            if measurement == 'ttft':
                metadata['accuracy'] = _timed_accuracy_summary(rows)
        if expected_ids is not None and actual != expected_ids[str(Path(metadata['dataset']).expanduser().resolve())]:
            raise ValueError('HF shards do not cover the entire requested dataset')
        metadata[metadata.get('measurement', 'accuracy')] = metric
        metadata['device_layout'] = {**run['device_layout'],
                                     'active_replicas': len(ranks), 'active_ranks': ranks}
        _write_json_new(output / name, {'metadata': metadata, 'rows': rows,
                         'worker_sources': [str(p / name) for p in source_paths]})
        _write_json_new(output / name.replace('.predictions.json', '.summary.json'), metadata)
        run['datasets'].append(metadata)
    if expected_dataset_ranks is not None and seen_datasets != set(expected_dataset_ranks):
        raise ValueError('HF workers omitted a requested dataset')
    if multi:
        from think_bridge.eval.multiturn import _unique_execution_costs
        costs = {}
        for predictions in merged_predictions.values():
            for key, value in _unique_execution_costs(predictions, run['measurement']).items():
                costs[key] = costs.get(key, 0) + value
        run['execution'] = {'unique_costs': costs}
    run['complete'] = True
    _write_json_new(output / 'benchmark_summary.json', run)


def run_parallel(arguments):
    import torch
    if arguments.mode != 'multi-turn' or arguments.backend != 'hf':
        raise ValueError('HF replicas support multi-turn evaluation')
    counts, expected = [], {}
    from think_bridge.eval.conversation_history import load_conversations
    for path in arguments.dataset:
        conversations = load_conversations(path, maximum=arguments.max_conversations)
        counts.append(len(conversations))
        expected[str(path)] = [str(row['id']) for c in conversations for row in c]
    group_size = int(arguments.batch_size)
    if group_size <= 0:
        raise ValueError('batch_size must be positive')
    if not counts or any(n <= 0 for n in counts):
        raise ValueError('HF evaluation datasets must be nonempty')
    world = int(arguments.hf_workers)
    if world < 1:
        raise ValueError('hf_workers must be positive')
    if torch.cuda.device_count() < world:
        raise ValueError('hf_workers exceeds visible CUDA GPUs')
    dataset_ranks = {str(path): list(range(min(world, (n + group_size - 1) // group_size)))
                     for path, n in zip(arguments.dataset, counts)}
    for path, ranks in dataset_ranks.items():
        print(f'[benchmark HF] dataset={Path(path).name} active_workers={len(ranks)}/{world}', flush=True)
    output = arguments.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=False)
    processes = []
    try:
        for rank in range(world):
            worker_args = copy.copy(arguments)
            worker_args.dataset = [path for path in arguments.dataset if rank in dataset_ranks[str(path)]]
            process = multiprocessing.get_context('spawn').Process(
                target=_worker, args=(worker_args, rank, world, output))
            process.start()
            processes.append(process)
        print(f'[benchmark HF] requested_workers={world} launched_workers={len(processes)} '
              f'answer_batch={group_size}', flush=True)
        while any(p.is_alive() for p in processes):
            if any(p.exitcode not in (None, 0) for p in processes):
                raise RuntimeError('HF evaluation worker failed; see its traceback, partial results retained')
            time.sleep(.2)
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError('HF evaluation worker failed; partial results retained')
        merge_workers(output, world, expected_ids=expected, group_size=group_size,
                      expected_dataset_ranks=dataset_ranks)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    return 0
