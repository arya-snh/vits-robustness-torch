"""Functions and classes for adversarial training and for generating adversarial examples.

The way the attacks are instantiated is inspired by DeepMind's repository for adversarial robustness,
which is implemented in JAX, and can be found here:
https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness/.

The original license can be found here:
https://github.com/deepmind/deepmind-research/blob/master/LICENSE
"""

import functools
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from autoattack import AutoAttack
from timm.bits import DeviceEnv
from torch import nn

AttackFn = Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]
TrainAttackFn = Callable[[nn.Module, torch.Tensor, torch.Tensor, int], torch.Tensor]
Boundaries = Tuple[float, float]
ProjectFn = Callable[[torch.Tensor, torch.Tensor, float, Boundaries], torch.Tensor]
InitFn = Callable[[torch.Tensor, float, ProjectFn, Boundaries], torch.Tensor]
EpsSchedule = Callable[[int], float]
ScheduleMaker = Callable[[float, int, int], EpsSchedule]
Norm = str


def project_linf(x: torch.Tensor, x_adv: torch.Tensor, eps: float, boundaries: Boundaries) -> torch.Tensor:
    clip_min, clip_max = boundaries
    d_x = torch.clamp(x_adv - x.detach(), -eps, eps)
    x_adv = torch.clamp(x + d_x, clip_min, clip_max)
    return x_adv


def init_linf(x: torch.Tensor, eps: float, project_fn: ProjectFn, boundaries: Boundaries) -> torch.Tensor:
    x_adv = x.detach() + torch.empty_like(x.detach(), device=x.device).uniform_(-eps, eps) + 1e-5
    return project_fn(x, x_adv, eps, boundaries)


def init_l2(x: torch.Tensor, eps: float, project_fn: ProjectFn, boundaries: Boundaries) -> torch.Tensor:
    x_adv = x.detach() + torch.empty_like(x.detach(), device=x.device).normal_(-eps, eps) + 1e-5
    return project_fn(x, x_adv, eps, boundaries)


def project_l2(x: torch.Tensor, x_adv: torch.Tensor, eps: float, boundaries: Boundaries) -> torch.Tensor:
    clip_min, clip_max = boundaries
    d_x = x_adv - x.detach()
    d_x_norm = d_x.renorm(p=2, dim=0, maxnorm=eps)
    x_adv = torch.clamp(x + d_x_norm, clip_min, clip_max)
    return x_adv


def pgd(model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        eps: float,
        step_size: float,
        steps: int,
        boundaries: Tuple[float, float],
        init_fn: InitFn,
        project_fn: ProjectFn,
        criterion: nn.Module,
        targeted: bool = False,
        num_classes: Optional[int] = None,
        random_targets: bool = False,
        logits_y: bool = False,
        take_sign=True,
        normalize=False,
        dev_env: Optional[DeviceEnv] = None,
        return_losses: bool = False) -> Union[torch.Tensor, Tuple[List[float], torch.Tensor]]:
    losses = []
    local_project_fn = functools.partial(project_fn, eps=eps, boundaries=boundaries)
    x_adv = init_fn(x, eps, project_fn, boundaries)
    if random_targets:
        assert num_classes is not None
        y = torch.randint_like(y, 0, num_classes, device=y.device)
    
    if len(y.size()) > 1 and not logits_y:
        y = y.argmax(dim=-1)
    
    for _ in range(steps):
        x_adv.requires_grad_()
        loss = criterion(
            F.log_softmax(model(x_adv), dim=-1),
            y,
        )
        if return_losses:
            losses.append(loss)
        grad = torch.autograd.grad(loss, x_adv)[0]
        # Differentiate between L2 and Linf, though this can be probably abstracted better
        # Take sign for Linf
        if take_sign:
            d_x = torch.sign(grad)
        # Or normalize for L2
        elif normalize:
            # from the robustness library
            l = len(x.shape) - 1
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
            d_x = grad / (grad_norm + 1e-10)
        else:
            d_x = grad
        if targeted:
            # Minimize the loss if the attack is targeted
            x_adv = x_adv.detach() - step_size * d_x
        else:
            # Otherwise maximize
            x_adv = x_adv.detach() + step_size * d_x

        # Project into the allowed domain
        x_adv = local_project_fn(x, x_adv)

        if dev_env is not None:
            # Mark step here to keep XLA program size small and speed-up compilation time
            # It also seems to improve overall speed when `steps` > 1.
            dev_env.mark_step()
    
    if return_losses:
        return x_adv, torch.as_tensor(losses)
    return x_adv


_ATTACKS = {
    "pgd": pgd,
    "targeted_pgd": functools.partial(pgd, targeted=True, random_targets=True),
}
_INIT_PROJECT_FN: Dict[str, Tuple[InitFn, ProjectFn, bool, bool]] = {
    "linf": (init_linf, project_linf, True, False),
    "l2": (init_l2, project_l2, False, True)
}


def make_sine_schedule(final: float, warmup: int, zero_eps_epochs: int) -> Callable[[int], float]:

    def sine_schedule(step: int) -> float:
        if step < zero_eps_epochs:
            return 0.0
        if step < warmup:
            return 0.5 * final * (1 + math.sin(math.pi * ((step - zero_eps_epochs) / warmup - 0.5)))
        return final

    return sine_schedule


def make_linear_schedule(final: float, warmup: int, zero_eps_epochs: int) -> Callable[[int], float]:

    def linear_schedule(step: int) -> float:
        if step < zero_eps_epochs:
            return 0.0
        if step < warmup:
            return (step - zero_eps_epochs) / warmup * final
        return final

    return linear_schedule


_SCHEDULES: Dict[str, ScheduleMaker] = {
    "linear": make_linear_schedule,
    "sine": make_sine_schedule,
    "constant": (lambda eps, _1, _2: (lambda _: eps))
}


def make_train_attack(attack_name: str, schedule: str, final_eps: float, period: int, zero_eps_epochs: int,
                      step_size: float, steps: int, norm: Norm, boundaries: Tuple[float, float],
                      criterion: nn.Module, num_classes: int, logits_y: bool, **kwargs) -> TrainAttackFn:
    attack_fn = _ATTACKS[attack_name]
    init_fn, project_fn, take_sign, normalize = _INIT_PROJECT_FN[norm]
    schedule_fn = _SCHEDULES[schedule](final_eps, period, zero_eps_epochs)

    def attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, step: int) -> torch.Tensor:
        eps = schedule_fn(step)
        return attack_fn(model,
                         x,
                         y,
                         eps,
                         step_size=step_size,
                         steps=steps,
                         boundaries=boundaries,
                         init_fn=init_fn,
                         project_fn=project_fn,
                         criterion=criterion,
                         num_classes=num_classes,
                         logits_y=logits_y,
                         take_sign=take_sign,
                         normalize=normalize,
                         **kwargs)

    return attack


def make_attack(attack: str,
                eps: float,
                step_size: float,
                steps: int,
                norm: Norm,
                boundaries: Tuple[float, float],
                criterion: nn.Module,
                device: Optional[torch.device] = None,
                **attack_kwargs) -> AttackFn:
    if attack not in {"autoattack", "apgd-ce"}:
        attack_fn = _ATTACKS[attack]
        init_fn, project_fn, take_sign, normalize = _INIT_PROJECT_FN[norm]
        return functools.partial(attack_fn,
                                 eps=eps,
                                 step_size=step_size,
                                 steps=steps,
                                 boundaries=boundaries,
                                 init_fn=init_fn,
                                 project_fn=project_fn,
                                 criterion=criterion,
                                 take_sign=take_sign,
                                 normalize=normalize,
                                 **attack_kwargs)
    if attack in {"apgd-ce"}:
        attack_kwargs["version"] = "custom"
        attack_kwargs["attacks_to_run"] = [attack]
        if "dev_env" in attack_kwargs:
            del attack_kwargs["dev_env"]

    def autoattack_fn(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert isinstance(eps, float)
        adversary = AutoAttack(model, norm.capitalize(), eps=eps, device=device, **attack_kwargs)
        x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
        return x_adv  # type: ignore

    return autoattack_fn


@dataclass
class AttackCfg:
    name: str
    eps: float
    eps_schedule: str
    eps_schedule_period: int
    zero_eps_epochs: int
    step_size: float
    steps: int
    norm: str
    boundaries: Tuple[float, float]


class AdvTrainingLoss(nn.Module):

    def __init__(self,
                 attack_cfg: AttackCfg,
                 natural_criterion: nn.Module,
                 dev_env: DeviceEnv,
                 num_classes: int,
                 eval_mode: bool = False):
        super().__init__()
        self.criterion = natural_criterion
        self.attack = make_train_attack(attack_cfg.name,
                                        attack_cfg.eps_schedule,
                                        attack_cfg.eps,
                                        attack_cfg.eps_schedule_period,
                                        attack_cfg.zero_eps_epochs,
                                        attack_cfg.step_size,
                                        attack_cfg.steps,
                                        attack_cfg.norm,
                                        attack_cfg.boundaries,
                                        criterion=nn.NLLLoss(reduction="sum"),
                                        num_classes=num_classes,
                                        logits_y=False,
                                        dev_env=dev_env)
        self.eval_mode = eval_mode

    def forward(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                epoch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.eval_mode:
            model.eval()
        x_adv = self.attack(model, x, y, epoch)
        model.train()
        logits, logits_adv = model(x), model(x_adv)
        loss = self.criterion(logits_adv, y)
        return loss, logits, logits_adv


class TRADESLoss(nn.Module):
    """Adapted from https://github.com/yaodongyu/TRADES/blob/master/trades.py#L17"""

    def __init__(self,
                 attack_cfg: AttackCfg,
                 natural_criterion: nn.Module,
                 beta: float,
                 gamma: float,
                 dev_env: DeviceEnv,
                 num_classes: int,
                 eval_mode: bool = False):
        super().__init__()
        self.attack = make_train_attack(attack_cfg.name,
                                        attack_cfg.eps_schedule,
                                        attack_cfg.eps,
                                        attack_cfg.eps_schedule_period,
                                        attack_cfg.zero_eps_epochs,
                                        attack_cfg.step_size,
                                        attack_cfg.steps,
                                        attack_cfg.norm,
                                        attack_cfg.boundaries,
                                        criterion=nn.KLDivLoss(reduction="sum"),
                                        num_classes=num_classes,
                                        logits_y=True,
                                        dev_env=dev_env)
        self.natural_criterion = natural_criterion
        self.kl_criterion = nn.KLDivLoss(reduction="sum")
        self.gamma = gamma
        self.beta = beta
        self.eval_mode = eval_mode

    def forward(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                epoch: int, anchor_inputs_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        # Avoid setting the model in eval mode if on XLA (it crashes)
        if self.eval_mode:
            model.eval()  # FIXME: understand why with eval the gradient
        # of BatchNorm crashes
        output_softmax = F.softmax(model(x.detach()), dim=-1)
        x_adv = self.attack(model, x, output_softmax, epoch)
        model.train()
        logits, logits_adv = model(x), model(x_adv)
        loss_natural = self.natural_criterion(logits, y)

        softmax = F.softmax(logits, dim=1)  # Shape: (batch_size, num_classes)
        softmax_adv = F.softmax(logits_adv, dim=1)  # Shape: (batch_size, num_classes)

        # ## EDITED
        # num_classes = softmax.size(1)
        # loss_robust_mean_kl_old = 0
        # loss_robust_mean_kl_new = 0

        # softmax_mean = torch.zeros(size=(num_classes, num_classes))
        # adv_softmax_mean = torch.zeros(size=(num_classes, num_classes))

        # for class_idx in range(num_classes):
        #     class_softmax_vectors = softmax[(softmax.argmax(dim=1) == class_idx)]
        #     softmax_per_class = torch.mean(class_softmax_vectors, dim=0)
        #     softmax_mean[class_idx] = softmax_per_class
        # print(softmax_mean)

        # for class_idx in range(num_classes):
        #     adv_class_softmax_vectors = softmax_adv[(softmax_adv.argmax(dim=1) == class_idx)]
        #     adv_softmax_per_class = torch.mean(adv_class_softmax_vectors, dim=0)
        #     adv_softmax_mean[class_idx] = adv_softmax_per_class
        # # print(adv_softmax_mean)

        # for i in range(num_classes):
        #     if torch.isnan(adv_softmax_mean[i]).any() or torch.isnan(softmax_mean[i]).any():
        #         continue
        #         # print(f"Mean softmax vector for class {i} contains NaN values.")
        #     else:
        #         loss_robust_mean_kl_old += self.kl_criterion(torch.log(adv_softmax_mean[i]), softmax_mean[i])
        #         loss_robust_mean_kl_new += self.kl_criterion(torch.log(adv_softmax_mean[i]), softmax_mean[i])
        # # print(loss_robust_mean_kl_old)

        # for i in range(num_classes):
        #     if torch.isnan(softmax_mean[i]).any():
        #         continue
        #         # print(f"Mean softmax vector for class {i} contains NaN values.")
        #     else:
        #         for j in range(i+1, num_classes):
        #             if torch.isnan(softmax_mean[j]).any():
        #                 continue
        #                 # print(f"Mean softmax vector for class {j} contains NaN values.")
        #             else:
        #                 loss_robust_mean_kl_old -= self.kl_criterion(torch.log(softmax_mean[j]), softmax_mean[i])

        # for i in range(num_classes):
        #     if torch.isnan(softmax_mean[i]).any():
        #         continue
        #         # print(f"Mean softmax vector for class {i} contains NaN values.")
        #     else:
        #         indices_except_i = [j for j in range(num_classes) if j != i and not torch.isnan(softmax_mean[j]).any()]
        #         average_except_i = torch.mean(softmax_mean[indices_except_i].float())
        #         loss_robust_mean_kl_new -= self.kl_criterion(torch.log(average_except_i), softmax_mean[i])

        # loss_robust_mean_kl_old = self.relu(loss_robust_mean_kl_old)
        # loss_robust_mean_kl_new = self.relu(loss_robust_mean_kl_new)
                    
        #   CHECK EVAL MODE / TRAIN MODE
        with torch.no_grad():
            model.eval() 
            anchor_logits = model(anchor_inputs_tensor)

        model.train()

        anchor_vectors_softmax = F.softmax(anchor_logits/3.0, dim=1)

        anchor_vectors_transpose = anchor_logits.t()

        M1 = torch.matmul(softmax, anchor_vectors_transpose)
        M1_softmax = F.softmax(M1, dim=1)

        M2 = torch.matmul(softmax_adv, anchor_vectors_transpose)
        M2_softmax = F.softmax(M2, dim=1)

        cluster_loss = (1.0 / batch_size) * self.kl_criterion(torch.log(M2_softmax), M1_softmax)
        # # Create S matrix
        # S = (y.unsqueeze(1) == y.unsqueeze(0)).float()  # Shape: (batch_size, batch_size)
        # num_classes = softmax.size(1)

        # # Expand dimensions for broadcasting
        # softmax_expanded = softmax.unsqueeze(1)
        # softmax_adv_expanded = softmax_adv.unsqueeze(0)

        # # Repeat A and B to match each other's size
        # softmax_repeated = softmax_expanded.repeat(1, softmax_adv.size(0), 1) 
        # softmax_adv_repeated = softmax_adv_expanded.repeat(softmax.size(0), 1, 1)  

        # # Concatenate A and B along the second dimension to form the combination matrix
        # combination_matrix = torch.cat((softmax_repeated, softmax_adv_repeated), dim=2)
        
        # expanded_matrix = S.unsqueeze(2).repeat(1, 1, num_classes*2)
        
        # # Reshape to 1D matrix
        # S = expanded_matrix.view(S.size(0), S.size(1), num_classes*2)
        # all_combinations = S * combination_matrix

        # # Remove zero rows
        # non_zero_combinations = all_combinations[~torch.all(all_combinations == 0, dim=2)]

        # # Split into two matrices column-wise
        # matrix1 = non_zero_combinations[:, :num_classes]
        # matrix2 = non_zero_combinations[:, num_classes:]

        # loss_robust = (1.0 / matrix1.size(0)) * self.kl_criterion(matrix2, matrix1)

        loss_robust = (1.0 / batch_size) * self.kl_criterion(F.log_softmax(logits_adv, dim=1),
                                                             F.softmax(logits, dim=1))
        # if loss_robust_mean_kl > 0.0:
        #     ratio = loss_natural / loss_robust_mean_kl
        #     print(ratio)
        wandb.log({"natural loss": loss_natural, "KL loss": loss_robust, "cluster loss": cluster_loss})
        
        loss = loss_natural + self.gamma * cluster_loss
                    
        return loss, logits, logits_adv
