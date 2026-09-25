"""Small, testable Stage1 loss and record-reduction primitives.

Every branch produces a per-record mean.  The functions below apply
prompt-balanced record weights and return an *unnormalized* numerator plus its
active weight.  Each phase sampler first projects onto its declared domain.
Answer supervision and reachable-function objectives cover the selected
prompt-balanced reasoner views.
Every written coefficient therefore applies to its declared population once.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WeightedLoss:
    numerator: torch.Tensor
    detached_mean: torch.Tensor
    active_weight: torch.Tensor
    total_weight: torch.Tensor


@dataclass(frozen=True)
class NativeDecisionSpecificity:
    loss: torch.Tensor
    absolute: torch.Tensor
    relative: torch.Tensor
    target_margin: torch.Tensor
    donor_margin: torch.Tensor
    target_wins: torch.Tensor


class _ForwardKLFromTeacherLogProb(torch.autograd.Function):
    """Full-vocabulary forward KL with an analytic student-only backward.

    The teacher distribution is stopped and may be shared by several stopped
    controls before this function is called.  Saving only ``p_student -
    q_teacher`` avoids retaining the ordinary softmax/log-softmax graph and
    avoids checkpoint-time re-projection through the frozen LM head.  The
    formula is exactly the unit-temperature forward KL used by
    :func:`forward_kl_per_probe`; only the physical backward implementation is
    different.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        teacher_logprob: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_logprob.shape != student_logits.shape or teacher_logprob.ndim < 2:
            raise ValueError(
                "teacher_logprob and student_logits must share shape [...,V]"
            )
        teacher_logprob_fp32 = teacher_logprob.detach().float()
        student_logprob = F.log_softmax(student_logits.float(), dim=-1)
        teacher_prob = teacher_logprob_fp32.exp()
        student_prob = student_logprob.exp()
        loss = (teacher_prob * (teacher_logprob_fp32 - student_logprob)).sum(dim=-1)
        # Reuse the student-probability buffer after the value is complete.
        # The analytic backward needs only p_student-q_teacher; avoiding a
        # fourth full-vocabulary allocation keeps the fast path's peak bounded.
        student_prob.sub_(teacher_prob)
        ctx.save_for_backward(student_prob)
        ctx.student_dtype = student_logits.dtype
        return loss

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[None, torch.Tensor]:
        (student_minus_teacher,) = ctx.saved_tensors
        grad_student = grad_output.float().unsqueeze(-1) * student_minus_teacher
        return None, grad_student.to(dtype=ctx.student_dtype)


def forward_kl_from_teacher_logprob(
    teacher_logprob: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    """Return full-vocabulary ``KL(q_teacher || p_student)`` per row.

    ``teacher_logprob`` must be the stopped, already-normalized FP32 teacher
    log distribution.  Only ``student_logits`` receives gradients.
    """

    return _ForwardKLFromTeacherLogProb.apply(teacher_logprob, student_logits)
