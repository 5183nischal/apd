# Standalone script for training a Toy Model of Superposition (TMS).
#
# Usage:
#   python train_tms_standalone.py
#
# Description:
#   This script trains a TMS model as defined in the original APD repository.
#   It is self-contained and does not require the rest of the `spd` library.
#   Model checkpoints (.pth) and training configurations (.yaml) are saved locally.
#
# Hyperparameter Modification:
#   - To change training parameters (e.g., model architecture, learning rate, steps),
#     modify the `default_config` object within the `if __name__ == "__main__":`
#     block at the end of this script.
#   - Key parameters in `TMSTrainConfig` and `TMSModelConfig` include:
#     - `tms_model_config.n_features`, `tms_model_config.n_hidden`
#     - `tms_model_config.n_hidden_layers`, `tms_model_config.n_instances`
#     - `lr`, `steps`, `batch_size`, `seed`, `feature_probability`
#
# Outputs:
#   - Saved model state_dict (e.g., `tms.pth`)
#   - Saved training configuration (e.g., `tms_train_config.yaml`)
#   - These are saved in a timestamped subdirectory under `./out_standalone_tms/`.
#   - Optionally, plots like feature vector cosine similarities.
import datetime
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, List, Literal, Self, Tuple # Self for Pydantic model_validator

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from matplotlib import collections as mc
# Pydantic imports
from pydantic import (BaseModel, ConfigDict, NonNegativeInt, PositiveInt,
                      model_validator)
from torch import Tensor
from tqdm import tqdm, trange


# Copied from spd/module_utils.py
def init_param_(
    param: torch.Tensor,
    scale: float = 1.0,
    init_type: Literal["kaiming_uniform", "xavier_normal"] = "kaiming_uniform",
) -> None:
    if init_type == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(param)
        with torch.no_grad():
            param.mul_(scale)
    elif init_type == "xavier_normal":
        torch.nn.init.xavier_normal_(param, gain=scale)


# Adapted from spd/models/components.py
class Linear(nn.Module):
    """A linear transformation with an optional n_instances dimension."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_instances: int | None = None,
        init_type: Literal["kaiming_uniform", "xavier_normal"] = "kaiming_uniform",
        init_scale: float = 1.0,
    ):
        super().__init__()
        shape = (n_instances, d_in, d_out) if n_instances is not None else (d_in, d_out)
        self.weight = nn.Parameter(torch.empty(shape))
        init_param_(self.weight, scale=init_scale, init_type=init_type)

    def forward(
        self, x: Float[Tensor, "batch ... d_in"], *args: Any, **kwargs: Any
    ) -> Float[Tensor, "batch ... d_out"]:
        out = einops.einsum(x, self.weight, "batch ... d_in, ... d_in d_out -> batch ... d_out")
        return out


class TransposedLinear(nn.Module):
    """Linear layer that uses a transposed weight from another Linear layer."""

    def __init__(self, original_weight: nn.Parameter):
        super().__init__()
        self.register_buffer("original_weight", original_weight, persistent=False)

    @property
    def weight(self) -> Float[Tensor, "... d_out d_in"]:
        return einops.rearrange(self.original_weight, "... d_in d_out -> ... d_out d_in")

    def forward(
        self, x: Float[Tensor, "batch ... d_out"], *args: Any, **kwargs: Any
    ) -> Float[Tensor, "batch ... d_in"]:
        # einsum for TransposedLinear: x is (batch ... d_out), weight is (... d_out d_in)
        # Output should be (batch ... d_in)
        out = einops.einsum(x, self.weight, "batch ... d_out, ... d_out d_in -> batch ... d_in")
        return out


# Adapted from spd/experiments/tms/models.py
class TMSModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_instances: PositiveInt
    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt 
    device: str


def _tms_forward(
    x: Float[Tensor, "batch n_instances n_features"],
    linear1: Linear, 
    linear2: TransposedLinear, 
    b_final: Float[Tensor, "n_instances n_features"],
    topk_mask: Any | None = None, 
    hidden_layers: nn.ModuleList | None = None,
) -> Float[Tensor, "batch n_instances n_features"]:
    """Forward pass used for TMSModel.
    Note that topk_mask is only used for TMSSPDModel, so it's ignored here.
    """
    hidden = linear1(x) 
    if hidden_layers is not None:
        for layer in hidden_layers:
            hidden = layer(hidden) 
    out_pre_relu = linear2(hidden) + b_final 
    out = F.relu(out_pre_relu)
    return out


class TMSModel(nn.Module): 
    def __init__(self, config: TMSModelConfig):
        super().__init__()
        self.config = config

        self.linear1 = Linear( 
            d_in=config.n_features,
            d_out=config.n_hidden,
            n_instances=config.n_instances,
            init_type="xavier_normal",
        )
        self.linear2 = TransposedLinear(self.linear1.weight) 

        self.b_final = nn.Parameter(torch.zeros((config.n_instances, config.n_features)))

        self.hidden_layers = None
        if config.n_hidden_layers > 0:
            self.hidden_layers = nn.ModuleList()
            for _ in range(config.n_hidden_layers):
                layer = Linear( 
                    d_in=config.n_hidden,
                    d_out=config.n_hidden,
                    n_instances=config.n_instances,
                    init_type="xavier_normal",
                )
                self.hidden_layers.append(layer)
        

    def forward(
        self, x: Float[Tensor, "... n_instances n_features"], **_: Any
    ) -> Float[Tensor, "... n_instances n_features"]:
        return _tms_forward(
            x=x,
            linear1=self.linear1,
            linear2=self.linear2,
            b_final=self.b_final,
            hidden_layers=self.hidden_layers,
        )


# Copied from spd/utils.py
DataGenerationType = Literal[
    "exactly_one_active",
    "exactly_two_active",
    "exactly_three_active",
    "exactly_four_active",
    "exactly_five_active",
    "at_least_zero_active",
]

class SparseFeatureDataset(
    torch.utils.data.Dataset[ 
        Tuple[
            Float[Tensor, "batch n_instances n_features"],
            Float[Tensor, "batch n_instances n_features"],
        ]
    ]
):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        feature_probability: float,
        device: str,
        data_generation_type: DataGenerationType = "at_least_zero_active",
        value_range: tuple[float, float] = (0.0, 1.0),
        synced_inputs: list[list[int]] | None = None,
    ):
        self.n_instances = n_instances
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device
        self.data_generation_type = data_generation_type
        self.value_range = value_range
        self.synced_inputs = synced_inputs

    def __len__(self) -> int:
        # Effective length for dataloader iterations.
        # Could be very large if truly streaming.
        # For practical purposes in training, this defines batches per "epoch".
        return 2**20 

    def sync_inputs(
        self, batch: Float[Tensor, "batch n_instances n_features"]
    ) -> Float[Tensor, "batch n_instances n_features"]:
        assert self.synced_inputs is not None
        all_indices = [item for sublist in self.synced_inputs for item in sublist]
        assert len(all_indices) == len(set(all_indices)), "Synced inputs must be non-overlapping"
        for indices in self.synced_inputs:
            mask = torch.zeros_like(batch, dtype=torch.bool)
            non_zero_samples = (batch[..., indices] != 0.0).any(dim=-1)
            for idx in indices:
                mask[..., idx] = non_zero_samples
            max_val, min_val = self.value_range
            random_values = torch.rand(
                batch.shape[0], self.n_instances, self.n_features, device=self.device
            )
            random_values = random_values * (max_val - min_val) + min_val
            batch = torch.where(mask, random_values, batch)
        return batch

    def generate_batch(
        self, batch_size: int
    ) -> Tuple[ 
        Float[Tensor, "batch n_instances n_features"], Float[Tensor, "batch n_instances n_features"]
    ]:
        number_map = {
            "exactly_one_active": 1,
            "exactly_two_active": 2,
            "exactly_three_active": 3,
            "exactly_four_active": 4,
            "exactly_five_active": 5,
        }
        if self.data_generation_type in number_map:
            n = number_map[self.data_generation_type]
            batch = self._generate_n_feature_active_batch(batch_size, n=n)
        elif self.data_generation_type == "at_least_zero_active":
            batch = self._generate_multi_feature_batch(batch_size)
            if self.synced_inputs is not None:
                batch = self.sync_inputs(batch)
        else:
            raise ValueError(f"Invalid generation type: {self.data_generation_type}")

        return batch, batch.clone().detach()

    def _generate_n_feature_active_batch(
        self, batch_size: int, n: int
    ) -> Float[Tensor, "batch n_instances n_features"]:
        if n > self.n_features:
            raise ValueError(
                f"Cannot activate {n} features when only {self.n_features} features exist"
            )
        batch = torch.zeros(batch_size, self.n_instances, self.n_features, device=self.device)
        feature_indices = torch.arange(self.n_features, device=self.device)
        feature_indices = feature_indices.expand(batch_size, self.n_instances, self.n_features)
        perm = torch.rand_like(feature_indices.float()).argsort(dim=-1)
        permuted_features = feature_indices.gather(dim=-1, index=perm)
        active_features = permuted_features[..., :n]
        min_val, max_val = self.value_range
        random_values = torch.rand(batch_size, self.n_instances, n, device=self.device)
        random_values = random_values * (max_val - min_val) + min_val
        for i in range(n):
            batch.scatter_(
                dim=2, index=active_features[..., i : i + 1], src=random_values[..., i : i + 1]
            )
        return batch

    def _masked_batch_generator(
        self, total_batch_size: int
    ) -> Float[Tensor, "total_batch_size n_features"]:
        min_val, max_val = self.value_range
        batch = (
            torch.rand((total_batch_size, self.n_features), device=self.device)
            * (max_val - min_val)
            + min_val
        )
        mask = torch.rand_like(batch) < self.feature_probability
        return batch * mask

    def _generate_multi_feature_batch(
        self, batch_size: int
    ) -> Float[Tensor, "batch n_instances n_features"]:
        total_batch_size = batch_size * self.n_instances
        batch = self._masked_batch_generator(total_batch_size)
        return einops.rearrange(
            batch,
            "(batch n_instances) n_features -> batch n_instances n_features",
            batch=batch_size,
        )


class DatasetGeneratedDataLoader(torch.utils.data.DataLoader[Tuple[Tensor, Tensor]]): 
    def __init__(
        self,
        dataset: SparseFeatureDataset, 
        batch_size: int = 1,
        shuffle: bool = False, 
        num_workers: int = 0,
    ):
        assert hasattr(dataset, "generate_batch")
        # shuffle and num_workers > 0 are not really compatible with this style of loader
        # but kept for API consistency if needed.
        # The collate_fn is important because the dataset.generate_batch already returns the full batch.
        super().__init__(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers,
                         collate_fn=lambda items: items[0]) # items will be a list containing one batch tuple

    def __iter__(self): 
        # Calculate number of iterations based on dataset length and desired batch_size.
        # Note: self.batch_size for DataLoader is 1 due to collate_fn; use internal_batch_size for generation.
        internal_batch_size = self.dataset.generate_batch.__globals__.get('batch_size',1) if hasattr(self.dataset.generate_batch, '__globals__') else self.dataset.n_features # Fallback
        if hasattr(self, '_internal_batch_size'): # A way to pass actual batch size if needed
             internal_batch_size = self._internal_batch_size

        # This is tricky because self.batch_size is 1. We need the *intended* batch_size for generation.
        # Let's assume the batch_size passed to init is the intended generation batch_size.
        # The iterator should yield batches from dataset.generate_batch(intended_batch_size)
        # The len(self.dataset) defines how many "samples" (which are full batches here) this loader "sees" per epoch.
        # Let's make intended_batch_size an attribute or ensure it's correctly inferred.
        # For simplicity, the original code's iterator logic for range(len(self)) is used.
        # The __init__ sets self.batch_size to the external batch_size.
        # The collate_fn means each "sample" from dataset is one batch.
        # So, len(self) would be len(dataset) if batch_size=1 in super init.
        # This seems okay. The 'batch_size' for generate_batch is passed explicitly.
        # The issue is how many times to call it.
        # Let's use the passed batch_size for generation.
        num_batches_per_epoch = len(self.dataset) # This is a large number
        
        # If we want a fixed number of steps, that's handled by the training loop usually.
        # This loader will just keep generating.
        # The original `next(data_iter)` in `train` implies it's an infinite stream.
        # So, `while True` or a very large number of iterations.
        # The `range(len(self))` from original repo was: `for _ in range(len(self)): yield self.dataset.generate_batch(self.batch_size)`
        # Here, `self.batch_size` is the one passed to DatasetGeneratedDataLoader.
        # And `len(self)` is `len(dataset) / self.batch_size` if PyTorch DataLoader default logic.
        # But our dataset `__len__` is huge.
        # The previous version `range(len(self.dataset) // self.batch_size ...)` seems more standard.
        # Let's stick to a simpler generator style for an "infinite" stream.
        while True:
            yield self.dataset.generate_batch(self.batch_size if self.batch_size is not None else 1)


def set_seed(seed: int | None) -> None:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


# Adapted from spd/experiments/tms/train_tms.py
class TMSTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    tms_model_config: TMSModelConfig 
    feature_probability: float
    batch_size: PositiveInt
    steps: PositiveInt
    seed: int = 0
    lr: float
    data_generation_type: Literal["at_least_zero_active", "exactly_one_active"]
    fixed_identity_hidden_layers: bool = False
    fixed_random_hidden_layers: bool = False
    synced_inputs: list[list[int]] | None = None # Changed from List to list for modern Python

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.fixed_identity_hidden_layers and self.fixed_random_hidden_layers:
            raise ValueError(
                "Cannot set both fixed_identity_hidden_layers and fixed_random_hidden_layers to True"
            )
        if self.synced_inputs is not None:
            all_indices = [item for sublist in self.synced_inputs for item in sublist]
            if len(all_indices) != len(set(all_indices)):
                raise ValueError("Synced inputs must be non-overlapping")
        return self


def linear_lr(step: int, steps: int) -> float:
    return 1 - (step / steps)


def constant_lr(*_: int) -> float:
    return 1.0


def cosine_decay_lr(step: int, steps: int) -> float:
    return np.cos(0.5 * np.pi * step / (steps - 1))


def train(
    model: TMSModel,
    dataloader: DatasetGeneratedDataLoader[Tuple[torch.Tensor, torch.Tensor]],
    log_wandb: bool, 
    importance: float = 1.0,
    steps: int = 5_000,
    print_freq: int = 100,
    lr: float = 5e-3,
    lr_schedule: Callable[[int, int], float] = linear_lr,
) -> None:
    opt = torch.optim.AdamW(list(model.parameters()), lr=lr)
    data_iter = iter(dataloader) # This will be an infinite iterator now
    with trange(steps, ncols=0) as t:
        for step in t:
            step_lr = lr * lr_schedule(step, steps)
            for group in opt.param_groups:
                group["lr"] = step_lr
            opt.zero_grad(set_to_none=True)
            
            batch_data = next(data_iter) # No StopIteration expected with `while True` in loader
            
            batch, labels = batch_data

            out = model(batch)
            error = importance * (labels.abs() - out) ** 2
            loss = einops.reduce(error, "b i f -> i", "mean").sum()
            loss.backward()
            opt.step()

            if step % print_freq == 0 or (step + 1 == steps):
                tqdm.write(f"Step {step} Loss: {loss.item() / model.config.n_instances}")
                t.set_postfix(
                    loss=loss.item() / model.config.n_instances,
                    lr=step_lr,
                )
                


def plot_intro_diagram(model: TMSModel, filepath: Path) -> None:
    WA = model.linear1.weight.detach()
    sel = range(model.config.n_instances)
    color = plt.cm.viridis(np.array([0.0]))
    plt.rcParams["figure.dpi"] = 200
    # Ensure axs is always a numpy array and can be flattened for consistent indexing
    fig, axs_raw = plt.subplots(1, len(sel), figsize=(2 * len(sel), 2), squeeze=False)
    axs = axs_raw.flatten() 
    for i in range(len(sel)): 
        W = WA[sel[i]].cpu().detach().numpy() 
        current_ax = axs[i] 
        current_ax.scatter(W[:, 0], W[:, 1], c=color)
        current_ax.set_aspect("equal")
        current_ax.add_collection(
            mc.LineCollection(np.stack((np.zeros_like(W), W), axis=1), colors=[color])
        )
        z = 1.5
        current_ax.set_facecolor("#FCFBF8")
        current_ax.set_xlim((-z, z))
        current_ax.set_ylim((-z, z))
        current_ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)
        for spine in ["top", "right"]:
            current_ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            current_ax.spines[spine].set_position("center")
    plt.savefig(filepath)
    plt.close(fig) 


def plot_cosine_similarity_distribution(
    model: TMSModel,
    filepath: Path,
) -> None:
    rows = model.linear1.weight.detach()
    rows /= rows.norm(dim=-1, keepdim=True)
    cosine_sims = einops.einsum(rows, rows, "i f1 h, i f2 h -> i f1 f2")
    mask = ~torch.eye(rows.shape[1], device=rows.device, dtype=torch.bool)
    masked_sims = cosine_sims[:, mask].reshape(rows.shape[0], -1)
    fig, axs_raw = plt.subplots(1, model.config.n_instances, figsize=(4 * model.config.n_instances, 4), squeeze=False)
    axs = axs_raw.flatten()
    for i, ax_i in enumerate(axs):
        sims = masked_sims[i].cpu().numpy()
        ax_i.scatter(sims, np.zeros_like(sims), alpha=0.5)
        ax_i.set_title(f"Instance {i}")
        ax_i.set_xlim(-1, 1)
        if i == 0:
            ax_i.set_xlabel("Cosine Similarity")
        ax_i.set_yticks([])
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)


def get_model_and_dataloader(
    config: TMSTrainConfig, device: str
) -> Tuple[TMSModel, DatasetGeneratedDataLoader[Tuple[torch.Tensor, torch.Tensor]]]: 
    model = TMSModel(config=config.tms_model_config)
    model.to(device)
    if (
        config.fixed_identity_hidden_layers or config.fixed_random_hidden_layers
    ) and model.hidden_layers is not None:
        for i in range(model.config.n_hidden_layers):
            if config.fixed_identity_hidden_layers:
                model.hidden_layers[i].weight.data[:, :, :] = torch.eye(
                    model.config.n_hidden, device=device
                )
            elif config.fixed_random_hidden_layers:
                model.hidden_layers[i].weight.data[:, :, :] = torch.randn_like(
                    model.hidden_layers[i].weight
                )
            model.hidden_layers[i].weight.requires_grad = False

    dataset = SparseFeatureDataset( 
        n_instances=config.tms_model_config.n_instances,
        n_features=config.tms_model_config.n_features,
        feature_probability=config.feature_probability,
        device=device,
        data_generation_type=config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=config.synced_inputs,
    )
    # Pass the intended batch_size for generation to the DataLoader
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)
    # Store the generation batch_size if the loader needs it explicitly
    dataloader._internal_batch_size = config.batch_size 
    return model, dataloader


def run_train(config: TMSTrainConfig, device: str) -> None: 
    model, dataloader = get_model_and_dataloader(config, device)

    model_cfg = config.tms_model_config
    run_name = (
        f"tms_n-features{model_cfg.n_features}_n-hidden{model_cfg.n_hidden}_"
        f"n-hidden-layers{model_cfg.n_hidden_layers}_n-instances{model_cfg.n_instances}_"
        f"feat_prob{config.feature_probability}_seed{config.seed}"
    )
    if config.fixed_identity_hidden_layers:
        run_name += "_fixed-identity"
    elif config.fixed_random_hidden_layers:
        run_name += "_fixed-random"
    
    base_out_dir = Path("./out_standalone_tms") 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = base_out_dir / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    
    config_path = out_dir / "tms_train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    
    print(f"Saved config to {config_path}")

    train(
        model,
        dataloader=dataloader,
        log_wandb=config.wandb_project is not None, 
        steps=config.steps,
        lr=config.lr 
    )

    model_path = out_dir / "tms.pth"
    torch.save(model.state_dict(), model_path)
    
    print(f"Saved model to {model_path}")

    if model_cfg.n_hidden == 2: # Plot only if hidden dim is 2
        plot_intro_diagram(model, filepath=out_dir / "polygon.png")
        print(f"Saved diagram to {out_dir / 'polygon.png'}")

    plot_cosine_similarity_distribution(
        model, filepath=out_dir / "cosine_similarity_distribution.png"
    )
    print(
        f"Saved cosine similarity distribution to {out_dir / 'cosine_similarity_distribution.png'}"
    )
    print(f"1/sqrt(n_hidden): {1 / np.sqrt(model_cfg.n_hidden)}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    default_config = TMSTrainConfig(
        wandb_project=None, 
        tms_model_config=TMSModelConfig(
            n_features=40,
            n_hidden=10,
            n_hidden_layers=0,
            n_instances=3,
            device=device,
        ),
        feature_probability=0.05,
        batch_size=2048,
        steps=2000, 
        seed=0,
        lr=1e-3,
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=False,
        fixed_random_hidden_layers=False,
    )
    set_seed(default_config.seed)

    print(f"Running TMS training with config:\n{default_config.model_dump_json(indent=2)}")
    run_train(default_config, device)
    print("Standalone TMS training finished.")
