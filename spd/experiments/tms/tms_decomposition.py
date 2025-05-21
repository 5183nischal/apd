"""Run spd on a TMS model.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from spd.experiments.tms.models import (
    TMSModel,
    TMSModelConfig,
    TMSSPDModel,
    TMSSPDModelConfig,
)
from spd.log import logger
from spd.run_spd import Config, TMSTaskConfig, get_common_run_name_suffix, optimize
from spd.utils import (
    DatasetGeneratedDataLoader,
    SparseFeatureDataset,
    collect_subnetwork_attributions,
    load_config,
    set_seed,
)

try:
    import wandb
    from spd.wandb_utils import init_wandb
    wandb_available = True
except ImportError:
    wandb = None
    init_wandb = None
    wandb_available = False

if wandb_available and wandb:
    wandb.require("core")


def get_run_name(config: Config, tms_model_config: TMSModelConfig) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        run_suffix += f"ft{tms_model_config.n_features}_"
        run_suffix += f"hid{tms_model_config.n_hidden}"
        run_suffix += f"hid-layers{tms_model_config.n_hidden_layers}"
    return config.wandb_run_name_prefix + run_suffix


def plot_A_matrix(x: torch.Tensor, pos_only: bool = False) -> plt.Figure:
    n_instances = x.shape[0]

    fig, axs = plt.subplots(
        1, n_instances, figsize=(2.5 * n_instances, 2), squeeze=False, sharey=True
    )

    cmap = "Blues" if pos_only else "RdBu"
    ims = []
    for i in range(n_instances):
        ax = axs[0, i]
        instance_data = x[i, :, :].detach().cpu().float().numpy()
        max_abs_val = np.abs(instance_data).max()
        vmin = 0 if pos_only else -max_abs_val
        vmax = max_abs_val
        im = ax.matshow(instance_data, vmin=vmin, vmax=vmax, cmap=cmap)
        ims.append(im)
        ax.xaxis.set_ticks_position("bottom")
        if i == 0:
            ax.set_ylabel("k", rotation=0, labelpad=10, va="center")
        else:
            ax.set_yticks([])  # Remove y-axis ticks for all but the first plot
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("n_features")

    plt.subplots_adjust(wspace=0.1, bottom=0.15, top=0.9)
    fig.subplots_adjust(bottom=0.2)

    return fig


def plot_subnetwork_attributions_multiple_instances(
    attribution_scores: Float[Tensor, "batch n_instances C"],
    out_dir: Path,
    step: int | None,
) -> plt.Figure:
    """Plot subnetwork attributions for multiple instances in a row."""
    n_instances = attribution_scores.shape[1]

    # Create a wide figure with subplots in a row
    fig, axes = plt.subplots(1, n_instances, figsize=(5 * n_instances, 5), constrained_layout=True)

    axes = np.array([axes]) if isinstance(axes, plt.Axes) else axes

    images = []
    for idx, ax in enumerate(axes):
        instance_scores = attribution_scores[:, idx, :]
        im = ax.matshow(instance_scores.detach().cpu().numpy(), aspect="auto", cmap="Reds")
        images.append(im)

        # Annotate each cell with the numeric value
        for i in range(instance_scores.shape[0]):
            for j in range(instance_scores.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{instance_scores[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

        ax.set_xlabel("Subnetwork Index")
        if idx == 0:  # Only set ylabel for leftmost plot
            ax.set_ylabel("Batch Index")
        ax.set_title(f"Instance {idx}")

    # Add a single colorbar that references all plots
    norm = plt.Normalize(vmin=attribution_scores.min().item(), vmax=attribution_scores.max().item())
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axes)

    fig.suptitle(f"Subnetwork Attributions (Step {step})")
    filename = (
        f"subnetwork_attributions_s{step}.png"
        if step is not None
        else "subnetwork_attributions.png"
    )
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"Saved subnetwork attributions to {out_dir / filename}")
    return fig


def plot_subnetwork_attributions_statistics_multiple_instances(
    topk_mask: Float[Tensor, "batch_size n_instances C"], out_dir: Path, step: int | None
) -> plt.Figure:
    """Plot a row of vertical bar charts showing active subnetworks for each instance."""
    n_instances = topk_mask.shape[1]
    fig, axes = plt.subplots(1, n_instances, figsize=(5 * n_instances, 5), constrained_layout=True)

    axes = np.array([axes]) if isinstance(axes, plt.Axes) else axes

    for instance_idx in range(n_instances):
        ax = axes[instance_idx]
        instance_mask = topk_mask[:, instance_idx]

        values = instance_mask.sum(dim=1).cpu().detach().numpy()
        bins = list(range(int(values.min().item()), int(values.max().item()) + 2))
        counts, _ = np.histogram(values, bins=bins)

        bars = ax.bar(bins[:-1], counts, align="center", width=0.8)
        ax.set_xticks(bins[:-1])
        ax.set_xticklabels([str(b) for b in bins[:-1]])
        ax.set_title(f"Instance {instance_idx}")

        if instance_idx == 0:  # Only set y-label for leftmost plot
            ax.set_ylabel("Count")
        ax.set_xlabel("Number of active subnetworks")

        # Add value annotations on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    fig.suptitle(f"Active subnetworks per instance (batch_size={topk_mask.shape[0]})")
    filename = (
        f"subnetwork_attributions_statistics_s{step}.png"
        if step is not None
        else "subnetwork_attributions_statistics.png"
    )
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"Saved subnetwork attributions statistics to {out_dir / filename}")
    return fig


def plot_component_weights(model: TMSSPDModel, step: int, out_dir: Path, **_) -> plt.Figure:
    """Plot the component weight matrices."""
    component_weights = model.linear1.component_weights

    # component_weights: [n_instances, k, n_features, n_hidden]
    n_instances, C, dim1, dim2 = component_weights.shape

    fig, axs = plt.subplots(
        C,
        n_instances,
        figsize=(2 * n_instances, 2 * C),
        constrained_layout=True,
    )

    for i in range(n_instances):
        instance_max = np.abs(component_weights[i].detach().cpu().numpy()).max()
        for j in range(C):
            ax = axs[j, i]  # type: ignore
            param = component_weights[i, j].detach().cpu().numpy()
            ax.matshow(param, cmap="RdBu", vmin=-instance_max, vmax=instance_max)
            ax.set_xticks([])

            if i == 0:
                ax.set_ylabel(f"k={j}", rotation=0, ha="right", va="center")
            if j == C - 1:
                ax.set_xlabel(f"Inst {i}", rotation=45, ha="right")

    fig.suptitle(f"Component Weights (Step {step})")
    fig.savefig(out_dir / f"component_weights_{step}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"Saved component weights to {out_dir / f'component_weights_{step}.png'}")
    return fig


def plot_batch_frequencies(
    frequencies: Float[Tensor, "n_instances C"],
    xlabel: str,
    ax: plt.Axes,
    batch_size: int,
    title: str | None = None,
) -> None:
    """Plot frequency of C activations for each instance on a given axis.

    Args:
        frequencies: Tensor counting frequencies for each instance
        xlabel: Label for x-axis
        ax: Matplotlib axis to plot on
        batch_size: Size of the batch
        title: Optional title for the subplot
    """
    n_instances = frequencies.shape[0]
    C = frequencies.shape[1]

    for instance_idx in range(n_instances):
        bars = ax.bar(
            np.arange(C) + instance_idx * (C + 1),  # Add spacing between instances
            frequencies[instance_idx].detach().cpu().numpy(),
            align="center",
            width=0.8,
            label=f"Instance {instance_idx}",
        )

        # Add value annotations on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Activation Count (batch_size={batch_size})")
    if title:
        ax.set_title(title)

    # Set x-ticks for each instance group
    all_ticks = []
    all_labels = []
    for i in range(n_instances):
        ticks = np.arange(C) + i * (C + 1)
        all_ticks.extend(ticks)
        all_labels.extend([str(j) for j in range(C)])
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels)


def plot_batch_statistics(
    batch: Float[Tensor, "batch n_instances n_features"],
    topk_mask: Float[Tensor, "batch n_instances C"],
    out_dir: Path,
    step: int | None,
) -> dict[str, plt.Figure]:
    # Count the number of active features over the batch
    active_input_feats = (batch != 0).sum(dim=0)
    topk_activations = topk_mask.sum(dim=0)

    # Create figure with two vertically stacked subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

    # Plot input features
    ax1 = fig.add_subplot(gs[0])
    plot_batch_frequencies(
        frequencies=active_input_feats,
        xlabel="Input feature index",
        ax=ax1,
        batch_size=batch.shape[0],
        title="Input feature frequencies across batch",
    )

    # Plot subnetwork frequencies
    ax2 = fig.add_subplot(gs[1])
    plot_batch_frequencies(
        frequencies=topk_activations,
        xlabel="Component index",
        ax=ax2,
        batch_size=batch.shape[0],
        title="Component frequencies across batch",
    )

    # Ensure that each ax has the same y-axis maximum
    y_lims = [ax.get_ylim() for ax in [ax1, ax2]]
    y_max = max(y_lims[0][1], y_lims[1][1])
    for ax in [ax1, ax2]:
        ax.set_ylim(0, y_max)

    # fig.suptitle(f"Batch Statistics (Step {step})")

    # Save the combined figure
    filename = f"batch_statistics_s{step}.png" if step is not None else "batch_statistics.png"
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"Saved batch statistics to {out_dir / filename}")

    return {"batch_statistics": fig}


def make_plots(
    model: TMSSPDModel,
    target_model: TMSModel,
    step: int,
    out_dir: Path,
    device: str,
    config: Config,
    topk_mask: Float[Tensor, "batch n_instances C"] | None,
    batch: Float[Tensor, "batch n_instances n_features"],
    **_,
) -> dict[str, plt.Figure]:
    plots = {}
    if model.hidden_layers is not None:
        logger.warning("Only plotting the W matrix params and not the hidden layers.")
    plots["component_weights"] = plot_component_weights(model, step, out_dir)

    if config.topk is not None:
        assert topk_mask is not None
        assert isinstance(config.task_config, TMSTaskConfig)
        n_instances = model.config.n_instances if hasattr(model, "config") else model.n_instances
        attribution_scores = collect_subnetwork_attributions(
            spd_model=model,
            target_model=target_model,
            device=device,
            n_instances=n_instances,
        )
        plots["subnetwork_attributions"] = plot_subnetwork_attributions_multiple_instances(
            attribution_scores=attribution_scores, out_dir=out_dir, step=step
        )
        plots["subnetwork_attributions_statistics"] = (
            plot_subnetwork_attributions_statistics_multiple_instances(
                topk_mask=topk_mask, out_dir=out_dir, step=step
            )
        )

        batch_stat_plots = plot_batch_statistics(batch, topk_mask, out_dir, step)
        plots.update(batch_stat_plots)

    return plots


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    tms_model: TMSModel,
    tms_model_train_config_dict: dict[str, Any],
) -> None:
    torch.save(tms_model.state_dict(), out_dir / "tms.pth")

    with open(out_dir / "tms_train_config.yaml", "w") as f:
        yaml.dump(tms_model_train_config_dict, f, indent=2)

    if wandb_available and wandb is not None and save_to_wandb:
        wandb.save(str(out_dir / "tms.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "tms_train_config.yaml"), base_path=out_dir, policy="now")


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(config_path_or_obj, config_model=Config)

    if wandb_available and wandb is not None and init_wandb is not None and config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)

    task_config = config.task_config
    assert isinstance(task_config, TMSTaskConfig)

    set_seed(config.seed)
    logger.info(config)

    target_model, target_model_train_config_dict = TMSModel.from_pretrained(
        task_config.pretrained_model_path
    )
    target_model = target_model.to(device)
    target_model.eval()

    run_name = get_run_name(config=config, tms_model_config=target_model.config)
    if wandb_available and wandb is not None and init_wandb is not None and config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "final_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if wandb_available and wandb is not None and init_wandb is not None and config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")

    save_target_model_info(
        save_to_wandb=wandb_available and wandb is not None and init_wandb is not None and config.wandb_project is not None,
        out_dir=out_dir,
        tms_model=target_model,
        tms_model_train_config_dict=target_model_train_config_dict,
    )

    tms_spd_model_config = TMSSPDModelConfig(
        **target_model.config.model_dump(mode="json"),
        C=config.C,
        m=config.m,
        bias_val=task_config.bias_val,
    )
    model = TMSSPDModel(config=tms_spd_model_config)

    # Manually set the bias for the SPD model from the bias in the pretrained model
    model.b_final.data[:] = target_model.b_final.data.clone()

    if not task_config.train_bias:
        model.b_final.requires_grad = False

    param_names = ["linear1", "linear2"]
    if model.hidden_layers is not None:
        for i in range(len(model.hidden_layers)):
            param_names.append(f"hidden_layers.{i}")

    synced_inputs = target_model_train_config_dict.get("synced_inputs", None)
    dataset = SparseFeatureDataset(
        n_instances=target_model.config.n_instances,
        n_features=target_model.config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
        data_generation_type=task_config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=synced_inputs,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)

    optimize(
        model=model,
        config=config,
        device=device,
        dataloader=dataloader,
        target_model=target_model,
        param_names=param_names,
        out_dir=out_dir,
        plot_results_fn=make_plots,
    )

    if wandb_available and wandb is not None and init_wandb is not None and config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
