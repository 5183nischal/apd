"""Run SPD on a model."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

import einops
import matplotlib.pyplot as plt
import torch
from jaxtyping import Float
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.hooks import HookedRootModule
from spd.log import logger
from spd.models.base import SPDModel
from spd.module_utils import collect_nested_module_attrs, get_nested_module_attr
from spd.types import ModelPath, Probability
from spd.utils import (
    calc_recon_mse,
    calc_topk_mask,
    calculate_attributions,
    get_lr_schedule_fn,
    get_lr_with_warmup,
)

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb = None
    wandb_available = False


class TMSTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["tms"] = "tms"
    feature_probability: Probability
    train_bias: bool
    bias_val: float
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = (
        "at_least_zero_active"
    )
    pretrained_model_path: ModelPath  # e.g. wandb:spd-tms/runs/si0zbfxf


class ResidualMLPTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["residual_mlp"] = "residual_mlp"
    feature_probability: Probability
    init_scale: float = 1.0
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = "at_least_zero_active"
    pretrained_model_path: ModelPath  # e.g. wandb:spd-resid-mlp/runs/j9kmavzi


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_run_name_prefix: str = ""
    seed: int = 0
    topk: PositiveFloat | None = None
    batch_topk: bool = True
    exact_topk: bool = False
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    image_freq: PositiveInt | None = None
    image_on_first_step: bool = True
    slow_images: bool = False
    save_freq: PositiveInt | None = None
    lr: PositiveFloat
    out_recon_coeff: NonNegativeFloat | None = None
    act_recon_coeff: NonNegativeFloat | None = None
    param_match_coeff: NonNegativeFloat | None = 1.0
    topk_recon_coeff: NonNegativeFloat | None = None
    schatten_coeff: NonNegativeFloat | None = None
    schatten_pnorm: NonNegativeFloat | None = None
    lp_sparsity_coeff: NonNegativeFloat | None = None
    distil_from_target: bool = False
    pnorm: PositiveFloat | None = None
    C: PositiveInt
    m: PositiveInt | None = None
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = "constant"
    lr_exponential_halflife: PositiveFloat | None = None
    lr_warmup_pct: Probability = 0.0
    sparsity_loss_type: Literal["jacobian"] = "jacobian"
    unit_norm_matrices: bool = False
    attribution_type: Literal["gradient", "ablation", "activation"] = "gradient"
    task_config: TMSTaskConfig | ResidualMLPTaskConfig = Field(..., discriminator="task_name")

    DEPRECATED_CONFIG_KEYS: ClassVar[list[str]] = [
        "topk_param_attrib_coeff",
        "orthog_coeff",
        "hardcode_topk_mask_step",
        "pnorm_end",
        "topk_l2_coeff",
        "spd_type",
        "sparsity_warmup_pct",
    ]
    RENAMED_CONFIG_KEYS: ClassVar[dict[str, str]] = {"topk_act_recon_coeff": "act_recon_coeff"}

    @model_validator(mode="before")
    def handle_deprecated_config_keys(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove deprecated config keys and change names of any keys that have been renamed."""
        # Move k from task_config to Config and rename it to C
        if "task_config" in config_dict and "k" in config_dict["task_config"]:
            logger.warning("task_config.k is deprecated, please use C in the main Config instead")
            config_dict["C"] = config_dict["task_config"]["k"]
            del config_dict["task_config"]["k"]

        for key in list(config_dict.keys()):
            val = config_dict[key]
            if key in cls.DEPRECATED_CONFIG_KEYS:
                logger.warning(f"{key} is deprecated, but has value: {val}. Removing from config.")
                del config_dict[key]
            elif key in cls.RENAMED_CONFIG_KEYS:
                logger.info(f"Renaming {key} to {cls.RENAMED_CONFIG_KEYS[key]}")
                config_dict[cls.RENAMED_CONFIG_KEYS[key]] = val
                del config_dict[key]
        return config_dict

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        # Check valid combinations of topk and batch_size
        if self.topk is not None:
            if self.batch_topk:
                if not (self.batch_size * self.topk).is_integer():
                    logger.warning(
                        f"batch_size * topk={self.batch_size * self.topk} is not an integer, will "
                        f"round down from {self.batch_size * self.topk} to "
                        f"{int(self.batch_size * self.topk)} when calculating topk_mask"
                    )
            else:
                if not self.topk.is_integer():
                    raise ValueError("topk must be an integer when not using batch_topk")

        # Warn if neither topk_recon_coeff nor lp_sparsity_coeff is set
        if not self.topk_recon_coeff and not self.lp_sparsity_coeff:
            logger.warning("Neither topk_recon_coeff nor lp_sparsity_coeff is set")

        # If topk_recon_coeff is set, topk must be set
        if self.topk_recon_coeff is not None:
            assert self.topk is not None, "topk must be set if topk_recon_coeff is set"

        # If lp_sparsity_coeff is set, pnorm must be set
        if self.lp_sparsity_coeff is not None:
            assert self.pnorm is not None, "pnorm must be set if lp_sparsity_coeff is set"

        # Check that topk_recon_coeff is None if topk is None
        if self.topk is None:
            assert self.topk_recon_coeff is None, "topk_recon_coeff is not None but topk is"

        # Give a warning if both out_recon_coeff and param_match_coeff are > 0
        if (
            self.param_match_coeff is not None
            and self.param_match_coeff > 0
            and self.out_recon_coeff is not None
            and self.out_recon_coeff > 0
        ):
            logger.warning(
                "Both param_match_coeff and out_recon_coeff are > 0. It's typical to only set one."
            )

        # If any of the coeffs are 0, raise a warning
        msg = "is 0, you may wish to instead set it to null to avoid calculating the loss"
        if self.topk_recon_coeff == 0:
            logger.warning(f"topk_recon_coeff {msg}")
        if self.lp_sparsity_coeff == 0:
            logger.warning(f"lp_sparsity_coeff {msg}")
        if self.param_match_coeff == 0:
            logger.warning(f"param_match_coeff {msg}")

        # Check that lr_exponential_halflife is not None if lr_schedule is "exponential"
        if self.lr_schedule == "exponential":
            assert (
                self.lr_exponential_halflife is not None
            ), "lr_exponential_halflife must be set if lr_schedule is exponential"

        if self.schatten_coeff is not None:
            assert (
                self.schatten_pnorm is not None
            ), "schatten_pnorm must be set if schatten_coeff is set"

        return self


def get_common_run_name_suffix(config: Config) -> str:
    """Generate a run suffix based on Config that is common to all experiments."""
    run_suffix = ""
    if config.pnorm is not None:
        run_suffix += f"p{config.pnorm:.2e}_"
    if config.lp_sparsity_coeff is not None:
        run_suffix += f"lpsp{config.lp_sparsity_coeff:.2e}_"
    if config.topk is not None:
        run_suffix += f"topk{config.topk:.2e}_"
    if config.topk_recon_coeff is not None:
        run_suffix += f"topkrecon{config.topk_recon_coeff:.2e}_"
    if config.schatten_pnorm is not None:
        run_suffix += f"schatp{config.schatten_pnorm:.2e}_"
    if config.schatten_coeff is not None:
        run_suffix += f"schatten{config.schatten_coeff:.2e}_"
    if config.act_recon_coeff is not None:
        run_suffix += f"actrecon_{config.act_recon_coeff:.2e}_"
    run_suffix += f"C{config.C}_"
    run_suffix += f"sd{config.seed}_"
    run_suffix += f"attr-{config.attribution_type[:3]}_"
    run_suffix += f"lr{config.lr:.2e}_"
    run_suffix += f"bs{config.batch_size}_"
    return run_suffix


def calc_schatten_loss(
    As: dict[str, Float[Tensor, "C d_layer_in m"] | Float[Tensor, "n_instances C d_layer_in m"]],
    Bs: dict[str, Float[Tensor, "C m d_layer_out"] | Float[Tensor, "n_instances C m d_layer_out"]],
    mask: Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"],
    p: float,
    n_params: int,
    device: str,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the Schatten p-norms of the topk subnetworks and sum them.

    Args:
        As: Dictionary of A matrices for each layer
        Bs: Dictionary of B matrices for each layer
        mask: The mask to use for the Schatten p-norm penalty. May be a binary mask (if topk) or
            a float mask (if lp sparsity).
        p: The Schatten p-norm to use (from config.schatten_pnorm)
        n_params: The number of parameters in the model
        device: The device to use for calculations
    Returns:
        The Schatten p-norm penalty for the topk subnetworks
    """
    assert As.keys() == Bs.keys(), "As and Bs must have the same keys"
    n_instances = mask.shape[1] if mask.ndim == 3 else None
    accumulate_shape = (n_instances,) if n_instances is not None else ()

    schatten_penalty = torch.zeros(accumulate_shape, device=device)
    batch_size = mask.shape[0]

    for name in As:
        A = As[name]  # [C, d_in, m] or [n_instances, C, d_in, m]
        B = Bs[name]  # [C, m, d_out] or [n_instances, C, m, d_out]
        # mask: [batch, C] or [batch, n_instances, C]

        # Compute S_A = A^T A and S_B = B B^T
        S_A = einops.einsum(A, A, "... C d_in m, ... C d_in m -> ... C m")
        S_B = einops.einsum(B, B, "... C m d_out, ... C m d_out -> ... C m")

        S_AB = S_A * S_B

        # Apply topk mask
        S_AB_topk = einops.einsum(S_AB, mask, "... C m, batch ... C -> batch ... C m")

        # Sum the Schatten p-norm
        schatten_penalty = schatten_penalty + ((S_AB_topk + 1e-16) ** (0.5 * p)).sum(
            dim=(0, -2, -1)
        )

    return schatten_penalty / n_params / batch_size


def _calc_param_mse(
    params1: dict[str, Float[Tensor, "d_in d_out"] | Float[Tensor, "n_instances d_in d_out"]],
    params2: dict[str, Float[Tensor, "d_in d_out"] | Float[Tensor, "n_instances d_in d_out"]],
    n_params: int,
    device: str,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the MSE between params1 and params2, summing over the d_in and d_out dimensions.

    Normalizes by the number of parameters in the model.

    Args:
        params1: The first set of parameters
        params2: The second set of parameters
        n_params: The number of parameters in the model
        device: The device to use for calculations
    """
    param_match_loss = torch.tensor(0.0, device=device)
    for name in params1:
        param_match_loss = param_match_loss + ((params2[name] - params1[name]) ** 2).sum(
            dim=(-2, -1)
        )
    return param_match_loss / n_params


def calc_param_match_loss(
    param_names: list[str],
    target_model: HookedRootModule,
    spd_model: SPDModel,
    n_params: int,
    device: str,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the MSE between the target model weights and the SPD model weights.

    Args:
        param_names: The names of the parameters to be matched.
        target_model: The target model to match.
        spd_model: The SPD model to match.
        n_params: The number of parameters in the model. Used for normalization.
        device: The device to use for calculations.
    """
    target_params = {}
    spd_params = {}
    for param_name in param_names:
        target_params[param_name] = get_nested_module_attr(target_model, param_name + ".weight")
        spd_params[param_name] = get_nested_module_attr(spd_model, param_name + ".weight")
    return _calc_param_mse(
        params1=target_params,
        params2=spd_params,
        n_params=n_params,
        device=device,
    )


def calc_lp_sparsity_loss(
    out: Float[Tensor, "batch d_model_out"] | Float[Tensor, "batch n_instances d_model_out"],
    attributions: Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"],
    step_pnorm: float,
) -> Float[Tensor, "batch C"] | Float[Tensor, "batch n_instances C"]:
    """Calculate the Lp sparsity loss on the attributions.

    Args:
        out: The output of the model.
        attributions: The attributions to use for the sparsity loss.
        step_pnorm: The pnorm to use for the sparsity loss.
    Returns:
        The Lp sparsity loss. Will have an n_instances dimension if the model has an n_instances
            dimension. Note that we keep the batch and C dimensions as we need them if calculating
            the schatten loss.
    """
    # Average the attributions over the output dimensions
    d_model_out = out.shape[-1]
    attributions = attributions / d_model_out

    # step_pnorm * 0.5 is because we have the squares of sparsity_inner terms above
    lp_sparsity_loss_per_k = (attributions.abs() + 1e-16) ** (step_pnorm * 0.5)
    return lp_sparsity_loss_per_k


def calc_act_recon(
    target_post_weight_acts: dict[
        str, Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"]
    ],
    layer_acts: dict[str, Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"]],
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """MSE between all target model activations and the output of each subnetwork in the SPD model.

    Args:
        target_post_weight_acts: The activations after each layer in the target model.
        layer_acts: The activations after each subnetwork in the SPD model.

    Returns:
        The activation reconstruction loss. Will have an n_instances dimension if the model has an
            n_instances dimension, otherwise a scalar.
    """
    assert (
        target_post_weight_acts.keys() == layer_acts.keys()
    ), f"Layer keys must match: {target_post_weight_acts.keys()} != {layer_acts.keys()}"

    device = next(iter(layer_acts.values())).device

    total_act_dim = 0  # Accumulate the d_out over all layers for normalization
    loss = torch.zeros(1, device=device)
    for layer_name in target_post_weight_acts:
        total_act_dim += target_post_weight_acts[layer_name].shape[-1]

        error = ((target_post_weight_acts[layer_name] - layer_acts[layer_name]) ** 2).sum(dim=-1)
        loss = loss + error

    # Normalize by the total number of output dimensions and mean over the batch dim
    return (loss / total_act_dim).mean(dim=0)


def optimize(
    model: SPDModel,
    config: Config,
    device: str,
    dataloader: DataLoader[tuple[Float[Tensor, "... n_features"], Float[Tensor, "... n_features"]]],
    target_model: HookedRootModule,
    param_names: list[str],
    plot_results_fn: Callable[..., dict[str, plt.Figure]] | None = None,
    out_dir: Path | None = None,
) -> None:
    model.to(device=device)
    target_model.to(device=device)

    has_instance_dim = hasattr(model, "n_instances")

    # Note that we expect weight decay to be problematic for spd
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)

    n_params = 0
    for param_name in param_names:
        n_params += get_nested_module_attr(target_model, param_name + ".weight").numel()

    if has_instance_dim:
        # All subnetwork param have an n_instances dimension
        n_params = n_params / model.n_instances

    epoch = 0
    total_samples = 0
    data_iter = iter(dataloader)
    for step in tqdm(range(config.steps + 1), ncols=0):
        if config.unit_norm_matrices:
            assert isinstance(model, SPDModel), "Can only norm matrices in SPDModel instances"
            model.set_As_to_unit_norm()

        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_schedule_fn=lr_schedule_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )
        for group in opt.param_groups:
            group["lr"] = step_lr

        opt.zero_grad(set_to_none=True)
        try:
            batch = next(data_iter)[0]  # Ignore labels here, we use the output of target_model
        except StopIteration:
            tqdm.write(f"Epoch {epoch} finished, starting new epoch")
            epoch += 1
            data_iter = iter(dataloader)
            batch = next(data_iter)[0]

        batch = batch.to(device=device)
        total_samples += batch.shape[0]

        target_cache_filter = lambda k: k.endswith((".hook_pre", ".hook_post"))
        target_out, target_cache = target_model.run_with_cache(
            batch, names_filter=target_cache_filter
        )

        # Do a forward pass with all subnetworks
        spd_cache_filter = lambda k: k.endswith((".hook_post", ".hook_component_acts"))
        out, spd_cache = model.run_with_cache(batch, names_filter=spd_cache_filter)

        # Calculate losses
        out_recon_loss = calc_recon_mse(out, target_out, has_instance_dim)

        param_match_loss = None
        if config.param_match_coeff is not None:
            param_match_loss = calc_param_match_loss(
                param_names=param_names,
                target_model=target_model,
                spd_model=model,
                n_params=n_params,
                device=device,
            )

        post_weight_acts = {k: v for k, v in target_cache.items() if k.endswith("hook_post")}
        attributions = calculate_attributions(
            model=model,
            batch=batch,
            out=out,
            target_out=target_out,
            pre_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_pre")},
            post_weight_acts=post_weight_acts,
            component_acts={
                k: v for k, v in spd_cache.items() if k.endswith("hook_component_acts")
            },
            attribution_type=config.attribution_type,
        )

        lp_sparsity_loss_per_k = None
        if config.lp_sparsity_coeff is not None:
            assert config.pnorm is not None, "pnorm must be set if lp_sparsity_coeff is set"
            lp_sparsity_loss_per_k = calc_lp_sparsity_loss(
                out=out, attributions=attributions, step_pnorm=config.pnorm
            )

        (
            out_topk,
            schatten_loss,
            topk_recon_loss,
            topk_mask,
            layer_acts_topk,
        ) = None, None, None, None, None
        if config.topk is not None:
            # We always assume the final subnetwork is the one we want to distil
            topk_attrs: Float[Tensor, "batch ... C"] = (
                attributions[..., :-1] if config.distil_from_target else attributions
            )
            if config.exact_topk:
                # Currently only valid for batch_topk and n_instances = 1. Would need to change the
                # topk argument in calc_topk_mask to allow for tensors if relaxing these constraints
                assert config.batch_topk, "exact_topk only works if batch_topk is True"
                assert (
                    hasattr(model, "n_instances") and model.n_instances == 1
                ), "exact_topk only works if n_instances = 1"
                # Get the exact number of active features over the batch
                exact_topk = ((batch != 0).sum() / batch.shape[0]).item()
                topk_mask = calc_topk_mask(topk_attrs, exact_topk, batch_topk=True)
            else:
                topk_mask = calc_topk_mask(topk_attrs, config.topk, batch_topk=config.batch_topk)
            if config.distil_from_target:
                # Add back the final subnetwork index to the topk mask and set it to True
                last_subnet_mask = torch.ones(
                    (*topk_mask.shape[:-1], 1), dtype=torch.bool, device=device
                )
                topk_mask = torch.cat((topk_mask, last_subnet_mask), dim=-1)

            # Do a forward pass with only the topk subnetworks
            out_topk, topk_spd_cache = model.run_with_cache(
                batch, names_filter=spd_cache_filter, topk_mask=topk_mask
            )
            layer_acts_topk = {k: v for k, v in topk_spd_cache.items() if k.endswith("hook_post")}

            if config.topk_recon_coeff is not None:
                assert out_topk is not None
                topk_recon_loss = calc_recon_mse(out_topk, target_out, has_instance_dim)

        act_recon_loss = None
        if config.act_recon_coeff is not None:
            if isinstance(config.task_config, ResidualMLPTaskConfig):
                # For now, we treat resid-mlp special in that we take the post-relu activations
                # We ignore the mlp_out layers
                assert layer_acts_topk is not None
                post_relu_acts = {}
                layer_acts_topk_after_relu = {}
                for i in range(len(target_model.layers)):
                    post_relu_acts[f"layers.{i}.mlp_in.hook_post"] = torch.nn.functional.relu(
                        post_weight_acts[f"layers.{i}.mlp_in.hook_post"]
                    )
                    layer_acts_topk_after_relu[f"layers.{i}.mlp_in.hook_post"] = (
                        torch.nn.functional.relu(layer_acts_topk[f"layers.{i}.mlp_in.hook_post"])
                    )

                act_recon_loss = calc_act_recon(
                    target_post_weight_acts=post_relu_acts, layer_acts=layer_acts_topk_after_relu
                )
            else:
                act_recon_layer_acts = (
                    layer_acts_topk
                    if layer_acts_topk is not None
                    else {k: v for k, v in spd_cache.items() if k.endswith("hook_post")}
                )
                act_recon_loss = calc_act_recon(
                    target_post_weight_acts=post_weight_acts,
                    layer_acts=act_recon_layer_acts,
                )

        if config.schatten_coeff is not None:
            mask = topk_mask if topk_mask is not None else lp_sparsity_loss_per_k
            assert mask is not None
            schatten_pnorm = config.schatten_pnorm if config.schatten_pnorm is not None else 1.0
            # Use the attributions as the mask in the lp case, and topk_mask otherwise
            schatten_loss = calc_schatten_loss(
                As=collect_nested_module_attrs(model, attr_name="A", include_attr_name=False),
                Bs=collect_nested_module_attrs(model, attr_name="B", include_attr_name=False),
                mask=mask,
                p=schatten_pnorm,
                n_params=n_params,
                device=device,
            )

        lp_sparsity_loss = None
        if lp_sparsity_loss_per_k is not None:
            # Sum over the C dimension (-1) and mean over the batch dimension (0)
            lp_sparsity_loss = lp_sparsity_loss_per_k.sum(dim=-1).mean(dim=0)

        loss_terms = {
            "param_match_loss": (param_match_loss, config.param_match_coeff),
            "out_recon_loss": (out_recon_loss, config.out_recon_coeff),
            "lp_sparsity_loss": (lp_sparsity_loss, config.lp_sparsity_coeff),
            "topk_recon_loss": (topk_recon_loss, config.topk_recon_coeff),
            "act_recon_loss": (act_recon_loss, config.act_recon_coeff),
            "schatten_loss": (schatten_loss, config.schatten_coeff),
        }
        # Add up the loss terms
        loss = torch.tensor(0.0, device=device)
        for loss_name, (loss_term, coeff) in loss_terms.items():
            if coeff is not None:
                assert loss_term is not None, f"{loss_name} is None but coeff is not"
                loss = loss + coeff * loss_term.mean()  # Mean over n_instances dimension

        # Logging
        if step % config.print_freq == 0:
            tqdm.write(f"Step {step}")
            tqdm.write(f"Total loss: {loss.item()}")
            tqdm.write(f"lr: {step_lr}")
            for loss_name, (val, _) in loss_terms.items():
                if val is not None:
                    val_repr = f"\n{val.tolist()}" if val.numel() > 1 else f" {val.item()}"
                    tqdm.write(f"{loss_name}:{val_repr}")

            if wandb_available and wandb is not None and config.wandb_project:
                metrics = {
                    "pnorm": config.pnorm,
                    "lr": step_lr,
                    "total_loss": loss.item(),
                    **{
                        name: val.mean().item() if val is not None else None
                        for name, (val, _) in loss_terms.items()
                    },
                }
                wandb.log(metrics, step=step)

        # Make plots
        if (
            plot_results_fn is not None
            and config.image_freq is not None
            and step % config.image_freq == 0
            and (step > 0 or config.image_on_first_step)
        ):
            fig_dict = plot_results_fn(
                model=model,
                target_model=target_model,
                step=step,
                out_dir=out_dir,
                device=device,
                config=config,
                topk_mask=topk_mask,
                batch=batch,
            )
            if wandb_available and wandb is not None and config.wandb_project:
                wandb.log(
                    {k: wandb.Image(v) for k, v in fig_dict.items()},
                    step=step,
                )

        # Save model
        if (
            (config.save_freq is not None and step % config.save_freq == 0 and step > 0)
            or step == config.steps
        ) and out_dir is not None:
            torch.save(model.state_dict(), out_dir / f"spd_model_{step}.pth")
            tqdm.write(f"Saved model to {out_dir / f'spd_model_{step}.pth'}")
            if wandb_available and wandb is not None and config.wandb_project:
                wandb.save(str(out_dir / f"spd_model_{step}.pth"), base_path=out_dir, policy="now")

        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            loss.backward()

            if step % config.print_freq == 0 and wandb_available and wandb is not None and config.wandb_project:
                # Calculate gradient norm
                grad_norm: float = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm()  # type: ignore
                wandb.log({"grad_norm": grad_norm}, step=step)

            if config.unit_norm_matrices:
                model.fix_normalized_adam_gradients()

            opt.step()
