"""AdamW optimizer, cosine learning-rate schedule and global gradient clipping."""

import math
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


class AdamW(Optimizer):
    """Adam with decoupled weight decay.

    Weight decay is applied directly to the parameters rather than through the
    gradient, which is what separates AdamW from Adam with L2 regularization.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """Validate hyperparameters and register the parameter groups.

        Args:
            params: Parameters to optimize.
            lr: Learning rate.
            betas: Exponential decay rates for the first and second moments.
            eps: Term added to the denominator for numerical stability.
            weight_decay: Decoupled weight decay coefficient.

        Raises:
            ValueError: If any hyperparameter is outside its valid range.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Apply one update to every parameter that has a gradient.

        Args:
            closure: Optional callable that recomputes the loss and its
                gradients. Used only by optimizers that need it; if given, its
                return value is passed back to the caller.

        Returns:
            The value returned by ``closure``, or ``None``.

        Raises:
            RuntimeError: If a parameter carries a sparse gradient.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Learning rate at a given step: linear warmup then cosine decay.

    Args:
        it: Current iteration.
        max_learning_rate: Peak learning rate reached at the end of warmup.
        min_learning_rate: Floor the cosine decay approaches.
        warmup_iters: Number of linear warmup steps.
        cosine_cycle_iters: Step at which the cosine decay finishes.

    Returns:
        The learning rate to use at step ``it``.
    """
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_decay = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)

    return min_learning_rate


def clip_gradient(
    parameters: Iterable[nn.Parameter] | Iterable[Tensor],
    max_norm: float,
    eps: float = 1e-6,
) -> None:
    """Rescale gradients in place so their global L2 norm stays within ``max_norm``.

    Args:
        parameters: Parameters or tensors whose ``.grad`` fields are clipped.
            Entries without a gradient are skipped.
        max_norm: Maximum allowed global gradient norm.
        eps: Small constant added to the norm for numerical stability.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    total_norm_sq = torch.zeros(1, dtype=torch.float32, device=grads[0].device)
    for g in grads:
        total_norm_sq += torch.sum(g.detach().to(torch.float32) ** 2)

    total_norm = torch.sqrt(total_norm_sq)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for g in grads:
            g.detach().mul_(scale.to(g.dtype))
