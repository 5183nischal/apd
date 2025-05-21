# Standalone script for applying APD to a trained TMS model.
#
# Usage:
#   python apply_apd_standalone.py /path/to/your_apd_config.yaml
#
# Description:
#   This script loads a pre-trained TMS model (trained by `train_tms_standalone.py`)
#   and applies the Attribution-based Parameter Decomposition (APD) method to it.
#   It is self-contained. APD model checkpoints and analysis plots are saved locally.
#
# Configuration:
#   - APD parameters are controlled by a YAML configuration file passed as a
#     command-line argument.
#   - An example structure for this YAML can be adapted from the `Config` Pydantic
#     model defined within this script (see `Config` and `TMSTaskConfig`).
#   - Key fields in the YAML config file:
#     - `task_config.pretrained_model_path`: Path to the `.pth` file of the trained TMS model.
#     - `task_config.pretrained_model_config_path`: Path to the `tms_train_config.yaml`
#       file corresponding to the trained TMS model.
#     - `C`: Number of APD components.
#     - `m`: Bottleneck dimension for APD components (optional).
#     - `topk`: If using top-k sparsity for attributions.
#     - `lp_sparsity_coeff` and `pnorm`: If using L_p sparsity.
#     - Loss coefficients (e.g., `param_match_coeff`, `topk_recon_coeff`, etc.).
#     - `lr`, `steps`, `batch_size` for the APD optimization.
#     - `seed`, `attribution_type`.
#
# Outputs:
#   - Saved APD model state_dict (e.g., `spd_model_STEP.pth`).
#   - Saved final APD configuration (`final_apd_config.yaml`).
#   - Copied target model and its training config.
#   - Analysis plots (component weights, attributions, etc.).
#   - These are saved in a timestamped subdirectory under `./out_standalone_apd/`.

import contextlib
import dataclasses
import datetime
import functools
import pathlib
import random
import sys # For a simple logger replacement
import yaml
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, Dict, List, Literal, NamedTuple, Optional, Tuple, TypeVar, Union

import einops
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.hooks as py_hooks # Renamed to avoid conflict with local hooks.py content
from jaxtyping import Bool, Float
from matplotlib import collections as mc
from pydantic import (BaseModel, ConfigDict, Field, NonNegativeFloat,
                      NonNegativeInt, PositiveFloat, PositiveInt,
                      model_validator)
from pydantic.v1.utils import deep_update # For Pydantic v1 compatibility in load_config if needed
from torch import Tensor
from tqdm import tqdm, trange

# --- Logger Replacement ---
class SimpleLogger:
    def info(self, msg):
        print(f"INFO: {msg}", file=sys.stdout)
    def warning(self, msg):
        print(f"WARNING: {msg}", file=sys.stderr)
    def error(self, msg):
        print(f"ERROR: {msg}", file=sys.stderr)

logger = SimpleLogger()

# --- Utility Functions (from spd/module_utils.py and spd/utils.py) ---

# From spd/module_utils.py
def get_nested_module_attr(module: nn.Module, access_string: str) -> Any:
    names = access_string.split(".")
    try:
        mod = functools.reduce(getattr, names, module)
    except AttributeError as err:
        raise AttributeError(f"{module} does not have nested attribute {access_string}") from err
    return mod

def collect_nested_module_attrs(
    module: nn.Module,
    attr_name: str,
    include_attr_name: bool = True,
) -> Dict[str, Tensor]:
    attributes: Dict[str, Tensor] = {}
    all_modules = module.named_modules()
    for name, submodule in all_modules:
        if hasattr(submodule, attr_name):
            submodule_attr = getattr(submodule, attr_name)
            if not isinstance(submodule_attr, Tensor):
                raise ValueError(
                    f"Attribute '{attr_name}' is not a tensor. "
                    f"Available modules: {[n for n, _ in all_modules]}"
                )
            key = name + "." + attr_name if include_attr_name else name
            attributes[key] = submodule_attr
    if not attributes:
        raise ValueError(
            f"No modules found with attribute '{attr_name}'. "
            f"Available modules: {[n for n, _ in all_modules]}"
        )
    return attributes

@torch.inference_mode()
def remove_grad_parallel_to_subnetwork_vecs(
    A: Float[Tensor, "... d_in m"], A_grad: Float[Tensor, "... d_in m"]
) -> None:
    parallel_component = einops.einsum(A_grad, A, "... d_in m, ... d_in m -> ... m")
    A_grad -= einops.einsum(parallel_component, A, "... m, ... d_in m -> ... d_in m")

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

# From spd/utils.py
ModelPath = Union[str, pathlib.Path] # From spd.types
Probability = float # From spd.types, assuming it's a float

def set_seed(seed: int | None) -> None:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

T_config = TypeVar("T_config", bound=BaseModel)
def load_config(config_path_or_obj: Union[pathlib.Path, str, T_config], config_model: type[T_config]) -> T_config:
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj
    if isinstance(config_path_or_obj, str):
        config_path_or_obj = pathlib.Path(config_path_or_obj)
    assert isinstance(config_path_or_obj, pathlib.Path), f"passed config is of invalid type {type(config_path_or_obj)}"
    assert config_path_or_obj.suffix == ".yaml", f"Config file {config_path_or_obj} must be a YAML file."
    assert config_path_or_obj.exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj) as f:
        config_dict = yaml.safe_load(f)
    return config_model(**config_dict)

DataGenerationType = Literal[
    "exactly_one_active", "exactly_two_active", "at_least_zero_active" # Simplified for TMS
]

class SparseFeatureDataset(torch.utils.data.Dataset[Tuple[Tensor, Tensor]]):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        feature_probability: float,
        device: str,
        data_generation_type: DataGenerationType = "at_least_zero_active",
        value_range: tuple[float, float] = (0.0, 1.0),
        synced_inputs: Optional[List[List[int]]] = None, # Changed from list to List for older typing compat
    ):
        self.n_instances = n_instances
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device
        self.data_generation_type = data_generation_type
        self.value_range = value_range
        self.synced_inputs = synced_inputs

    def __len__(self) -> int:
        return 2**20 # Effectively infinite for training loop

    def sync_inputs(self, batch: Tensor) -> Tensor: # Simplified type hints
        assert self.synced_inputs is not None
        # ... (rest of sync_inputs logic - assuming it's correct from original)
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


    def generate_batch(self, batch_size: int) -> Tuple[Tensor, Tensor]: # Simplified type hints
        number_map = {"exactly_one_active": 1, "exactly_two_active": 2} # Simplified
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

    def _generate_n_feature_active_batch(self, batch_size: int, n: int) -> Tensor:
        # ... (exact copy from train_tms_standalone.py)
        if n > self.n_features:
            raise ValueError(f"Cannot activate {n} features when only {self.n_features} features exist")
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
            batch.scatter_(dim=2, index=active_features[..., i : i + 1], src=random_values[..., i : i + 1])
        return batch

    def _masked_batch_generator(self, total_batch_size: int) -> Tensor:
        # ... (exact copy from train_tms_standalone.py)
        min_val, max_val = self.value_range
        batch = (torch.rand((total_batch_size, self.n_features), device=self.device) * (max_val - min_val) + min_val)
        mask = torch.rand_like(batch) < self.feature_probability
        return batch * mask
        
    def _generate_multi_feature_batch(self, batch_size: int) -> Tensor:
        # ... (exact copy from train_tms_standalone.py)
        total_batch_size = batch_size * self.n_instances
        batch = self._masked_batch_generator(total_batch_size)
        return einops.rearrange(batch, "(batch n_instances) n_features -> batch n_instances n_features", batch=batch_size)

Q_dl = TypeVar("Q_dl") # For DatasetGeneratedDataLoader
class DatasetGeneratedDataLoader(torch.utils.data.DataLoader[Q_dl]): # Copied from spd/utils.py
    def __init__(self, dataset: torch.utils.data.Dataset[Q_dl], batch_size: int = 1, shuffle: bool = False, num_workers: int = 0):
        assert hasattr(dataset, "generate_batch")
        # The collate_fn is important because the dataset.generate_batch already returns the full batch.
        super().__init__(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers, collate_fn=lambda items: items[0])
        self._intended_batch_size = batch_size


    def __iter__(self):
        while True:
            yield self.dataset.generate_batch(self._intended_batch_size) # type: ignore

def calc_recon_mse(output: Tensor, labels: Tensor, has_instance_dim: bool = False) -> Tensor:
    # ... (exact copy from spd/utils.py, simplified type hints)
    recon_loss = (output - labels) ** 2
    if recon_loss.ndim == 3:
        assert has_instance_dim
        recon_loss = einops.reduce(recon_loss, "b i f -> i", "mean")
    elif recon_loss.ndim == 2:
        recon_loss = recon_loss.mean()
    else:
        raise ValueError(f"Expected 2 or 3 dims in recon_loss, got {recon_loss.ndim}")
    return recon_loss
    
def get_lr_schedule_fn(lr_schedule: Literal["linear", "constant", "cosine", "exponential"], lr_exponential_halflife: Optional[PositiveFloat] = None) -> Callable[[int, int], float]:
    # ... (exact copy from spd/utils.py, logger replaced)
    if lr_schedule == "linear":
        return lambda step, steps: 1 - (step / steps)
    elif lr_schedule == "constant":
        return lambda *_: 1.0
    elif lr_schedule == "cosine":
        return lambda step, steps: 1.0 if steps == 1 else np.cos(0.5 * np.pi * step / (steps - 1))
    elif lr_schedule == "exponential":
        assert lr_exponential_halflife is not None
        halflife = lr_exponential_halflife
        gamma = 0.5 ** (1 / halflife)
        logger.info(f"Using exponential LR schedule with halflife {halflife} steps (gamma {gamma})")
        return lambda step, steps: gamma**step
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

def get_lr_with_warmup(step: int, steps: int, lr: float, lr_schedule_fn: Callable[[int, int], float], lr_warmup_pct: float) -> float:
    # ... (exact copy from spd/utils.py)
    warmup_steps = int(steps * lr_warmup_pct)
    if step < warmup_steps:
        return lr * (step / warmup_steps)
    return lr * lr_schedule_fn(step - warmup_steps, steps - warmup_steps)

# calc_grad_attributions and its helpers
def calc_grad_attributions(target_out: Tensor, pre_weight_acts: Dict[str, Tensor], post_weight_acts: Dict[str, Tensor], component_weights: Dict[str, Tensor], C: int) -> Tensor:
    # ... (exact copy from spd/utils.py, simplified type hints)
    post_weight_act_names = [k.removesuffix(".hook_post") for k in post_weight_acts]
    pre_weight_act_names = [k.removesuffix(".hook_pre") for k in pre_weight_acts]
    component_weight_names = list(component_weights.keys())
    assert set(post_weight_act_names) == set(pre_weight_act_names) == set(component_weight_names)

    attr_shape = target_out.shape[:-1] + (C,)
    attribution_scores: Tensor = torch.zeros(attr_shape, device=target_out.device, dtype=target_out.dtype)
    
    current_component_acts = {} # Renamed from component_acts to avoid conflict
    for param_name in pre_weight_act_names:
        current_component_acts[param_name] = einops.einsum(
            pre_weight_acts[param_name + ".hook_pre"].detach().clone(),
            component_weights[param_name],
            "... d_in, ... C d_in d_out -> ... C d_out",
        )
    out_dim = target_out.shape[-1]
    for feature_idx in range(out_dim):
        feature_attributions: Tensor = torch.zeros(attr_shape, device=target_out.device, dtype=target_out.dtype)
        grad_post_weight_acts: tuple[Tensor, ...] = torch.autograd.grad(
            target_out[..., feature_idx].sum(), list(post_weight_acts.values()), retain_graph=True
        )
        for i, param_name in enumerate(post_weight_act_names):
            feature_attributions += einops.einsum(
                grad_post_weight_acts[i],
                current_component_acts[param_name], # Use renamed var
                "... d_out ,... C d_out -> ... C",
            )
        attribution_scores += feature_attributions**2
    return attribution_scores

def calc_ablation_attributions(spd_model: 'SPDModel', batch: Tensor, out: Tensor) -> Tensor: # Forward ref SPDModel
    # ... (exact copy from spd/utils.py, simplified type hints)
    attr_shape = out.shape[:-1] + (spd_model.C,)
    has_instance_dim = len(out.shape) == 3
    attributions = torch.zeros(attr_shape, device=out.device, dtype=out.dtype)
    for subnet_idx in range(spd_model.C):
        stored_vals = spd_model.set_subnet_to_zero(subnet_idx, has_instance_dim)
        ablation_out, _, _ = spd_model(batch) # spd_model needs to be callable
        out_recon = ((out - ablation_out) ** 2).mean(dim=-1)
        attributions[..., subnet_idx] = out_recon
        spd_model.restore_subnet(subnet_idx, stored_vals, has_instance_dim)
    return attributions

def calc_activation_attributions(component_acts: Dict[str, Tensor]) -> Tensor:
    # ... (exact copy from spd/utils.py, simplified type hints)
    first_param = component_acts[next(iter(component_acts.keys()))]
    assert len(first_param.shape) in (3, 4) # (batch C d_out) or (batch n_instances C d_out)
    attribution_scores: Tensor = torch.zeros(first_param.shape[:-1], device=first_param.device, dtype=first_param.dtype)
    for param_matrix in component_acts.values():
        attribution_scores += param_matrix.pow(2).sum(dim=-1)
    return attribution_scores
    
def calculate_attributions(model: 'SPDModel', batch: Tensor, out: Tensor, target_out: Tensor, pre_weight_acts: Dict[str, Tensor], post_weight_acts: Dict[str, Tensor], component_acts: Dict[str, Tensor], attribution_type: Literal["ablation", "gradient", "activation"]) -> Tensor:
    # ... (exact copy from spd/utils.py, simplified type hints, forward ref SPDModel)
    attributions = None
    if attribution_type == "ablation":
        attributions = calc_ablation_attributions(spd_model=model, batch=batch, out=out)
    elif attribution_type == "gradient":
        component_weights = collect_nested_module_attrs(model, attr_name="component_weights", include_attr_name=False)
        attributions = calc_grad_attributions(
            target_out=target_out,
            pre_weight_acts=pre_weight_acts,
            post_weight_acts=post_weight_acts,
            component_weights=component_weights,
            C=model.C,
        )
    elif attribution_type == "activation":
        attributions = calc_activation_attributions(component_acts=component_acts)
    else:
        raise ValueError(f"Invalid attribution type: {attribution_type}")
    return attributions

def calc_topk_mask(attribution_scores: Tensor, topk: float, batch_topk: bool) -> Tensor:
    # ... (exact copy from spd/utils.py, simplified type hints)
    batch_size = attribution_scores.shape[0]
    # Ensure topk calculation is integer for indices
    effective_topk = int(topk * batch_size) if batch_topk else int(topk)

    if batch_topk:
        original_shape = attribution_scores.shape
        # Reshape: (batch, ..., C) -> (..., batch * C) if ... exists, or (batch, C) -> (1, batch*C)
        if attribution_scores.ndim > 2: # Has instance dim
            attribution_scores_reshaped = einops.rearrange(attribution_scores, "b i C -> i (b C)")
        else: # No instance dim
            attribution_scores_reshaped = einops.rearrange(attribution_scores, "b C -> 1 (b C)")
    else:
        attribution_scores_reshaped = attribution_scores

    topk_indices = attribution_scores_reshaped.topk(effective_topk, dim=-1).indices
    topk_mask_flat = torch.zeros_like(attribution_scores_reshaped, dtype=torch.bool)
    topk_mask_flat.scatter_(dim=-1, index=topk_indices, value=True)

    if batch_topk:
        if attribution_scores.ndim > 2: # Has instance dim
            topk_mask = einops.rearrange(topk_mask_flat, "i (b C) -> b i C", b=batch_size)
        else: # No instance dim
            topk_mask = einops.rearrange(topk_mask_flat, "1 (b C) -> b C", b=batch_size)
    else:
        topk_mask = topk_mask_flat
    return topk_mask

def replace_deprecated_param_names(params: Dict[str, Tensor], name_map: Dict[str, str]) -> Dict[str, Tensor]:
    # ... (exact copy from spd/utils.py)
    for k in list(params.keys()):
        for old_name, new_name in name_map.items():
            if old_name in k:
                params[k.replace(old_name, new_name)] = params[k]
                del params[k]
    return params

def collect_subnetwork_attributions(spd_model: 'SPDModel', target_model: 'HookedRootModule', device: str, n_instances: Optional[int] = None) -> Tensor:
    # ... (exact copy from spd/utils.py, forward refs)
    test_batch = torch.eye(spd_model.n_features, device=device)
    if n_instances is not None:
        test_batch = einops.repeat(test_batch, "batch n_features -> batch n_instances n_features", n_instances=n_instances)
    
    target_cache_filter = lambda k: k.endswith((".hook_pre", ".hook_post"))
    target_out, target_cache = target_model.run_with_cache(test_batch, names_filter=target_cache_filter)
    
    component_weights = collect_nested_module_attrs(spd_model, attr_name="component_weights", include_attr_name=False)
    
    attribution_scores = calc_grad_attributions(
        target_out=target_out,
        component_weights=component_weights,
        pre_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_pre")},
        post_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_post")},
        C=spd_model.C,
    )
    return attribution_scores

# --- Hooking Infrastructure (from spd/hooks.py) ---
@dataclasses.dataclass
class LensHandle:
    hook: py_hooks.RemovableHandle
    is_permanent: bool = False
    context_level: int | None = None

NamesFilter = Union[Callable[[str], bool], Sequence[str], str, None]
T_HookPoint = TypeVar("T_HookPoint", bound=Tensor)

@runtime_checkable # type: ignore
class _HookFunctionProtocol(Callable[[T_HookPoint, 'HookPoint'], Any]): ... # Forward ref HookPoint

HookFunction = _HookFunctionProtocol

class HookPoint(nn.Module):
    # ... (exact copy from spd/hooks.py, py_hooks used for RemovableHandle)
    def __init__(self):
        super().__init__()
        self.fwd_hooks: List[LensHandle] = []
        self.bwd_hooks: List[LensHandle] = []
        self.ctx = {}
        self.name: Optional[str] = None

    def add_perma_hook(self, hook: HookFunction, dir: Literal["fwd", "bwd"] = "fwd") -> None:
        self.add_hook(hook, dir=dir, is_permanent=True)

    def add_hook(self, hook: HookFunction, dir: Literal["fwd", "bwd"] = "fwd", is_permanent: bool = False, level: Optional[int] = None, prepend: bool = False) -> None:
        def full_hook(module: torch.nn.Module, module_input: Any, module_output: Any):
            if dir == "bwd": module_output = module_output[0]
            return hook(module_output, hook=self)
        
        if isinstance(hook, functools.partial): full_hook.__name__ = f"partial({hook.func.__repr__()},...)"
        else: full_hook.__name__ = hook.__repr__()

        if dir == "fwd":
            pt_handle = self.register_forward_hook(full_hook)
            _internal_hooks = self._forward_hooks
            visible_hooks = self.fwd_hooks
        elif dir == "bwd":
            pt_handle = self.register_full_backward_hook(full_hook)
            _internal_hooks = self._backward_hooks
            visible_hooks = self.bwd_hooks
        else: raise ValueError(f"Invalid direction {dir}")
        
        handle = LensHandle(pt_handle, is_permanent, level)
        if prepend:
            _internal_hooks.move_to_end(handle.hook.id, last=False) # type: ignore
            visible_hooks.insert(0, handle)
        else: visible_hooks.append(handle)

    def remove_hooks(self, dir: Literal["fwd", "bwd", "both"] = "fwd", including_permanent: bool = False, level: Optional[int] = None) -> None:
        def _remove_hooks(handles: List[LensHandle]) -> List[LensHandle]:
            output_handles = []
            for handle in handles:
                if (including_permanent or (not handle.is_permanent)) and (level is None or handle.context_level == level):
                    handle.hook.remove()
                else: output_handles.append(handle)
            return output_handles
        if dir == "fwd" or dir == "both": self.fwd_hooks = _remove_hooks(self.fwd_hooks)
        if dir == "bwd" or dir == "both": self.bwd_hooks = _remove_hooks(self.bwd_hooks)
        if dir not in ["fwd", "bwd", "both"]: raise ValueError(f"Invalid direction {dir}")

    def clear_context(self): self.ctx = {}
    def forward(self, x: T_HookPoint) -> T_HookPoint: return x
    def layer(self):
        if self.name is None: raise ValueError("Name cannot be None")
        return int(self.name.split(".")[1])

class HookedRootModule(nn.Module):
    # ... (exact copy from spd/hooks.py, logger replaced)
    name: Optional[str]
    mod_dict: Dict[str, nn.Module]
    hook_dict: Dict[str, HookPoint]

    def __init__(self, *args: Any):
        super().__init__()
        self.is_caching = False
        self.context_level = 0

    def setup(self):
        self.mod_dict = {}
        self.hook_dict = {}
        for name, module in self.named_modules():
            if name == "": continue
            module.name = name # type: ignore
            self.mod_dict[name] = module
            if isinstance(module, HookPoint): self.hook_dict[name] = module

    def hook_points(self): return self.hook_dict.values()
    def remove_all_hook_fns(self, direction: Literal["fwd", "bwd", "both"] = "both", including_permanent: bool = False, level: Optional[int] = None):
        for hp in self.hook_points(): hp.remove_hooks(direction, including_permanent=including_permanent, level=level)
    def clear_contexts(self):
        for hp in self.hook_points(): hp.clear_context()
    def reset_hooks(self, clear_contexts: bool = True, direction: Literal["fwd", "bwd", "both"] = "both", including_permanent: bool = False, level: Optional[int] = None):
        if clear_contexts: self.clear_contexts()
        self.remove_all_hook_fns(direction, including_permanent, level=level)
        self.is_caching = False

    def check_and_add_hook(self, hook_point: HookPoint, hook_point_name: str, hook: HookFunction, dir: Literal["fwd", "bwd"] = "fwd", is_permanent: bool = False, level: Optional[int] = None, prepend: bool = False):
        self.check_hooks_to_add(hook_point, hook_point_name, hook, dir=dir, is_permanent=is_permanent, prepend=prepend)
        hook_point.add_hook(hook, dir=dir, is_permanent=is_permanent, level=level, prepend=prepend)
    def check_hooks_to_add(self, hook_point: HookPoint, hook_point_name: str, hook: HookFunction, dir: Literal["fwd", "bwd"] = "fwd", is_permanent: bool = False, prepend: bool = False): pass
    
    def add_hook(self, name: Union[str, Callable[[str], bool]], hook: HookFunction, dir: Literal["fwd", "bwd"] = "fwd", is_permanent: bool = False, level: Optional[int] = None, prepend: bool = False):
        if isinstance(name, str):
            hook_point = self.mod_dict[name]
            assert isinstance(hook_point, HookPoint)
            self.check_and_add_hook(hook_point, name, hook, dir=dir, is_permanent=is_permanent, level=level, prepend=prepend)
        else:
            for hook_point_name, hp in self.hook_dict.items():
                if name(hook_point_name): self.check_and_add_hook(hp, hook_point_name, hook, dir=dir, is_permanent=is_permanent, level=level, prepend=prepend)
    
    def add_perma_hook(self, name: Union[str, Callable[[str], bool]], hook: HookFunction, dir: Literal["fwd", "bwd"] = "fwd"):
        self.add_hook(name, hook, dir=dir, is_permanent=True)

    @contextlib.contextmanager
    def hooks(self, fwd_hooks: List[Tuple[str, Callable]] = [], bwd_hooks: List[Tuple[str, Callable]] = [], reset_hooks_end: bool = True, clear_contexts: bool = False):
        try:
            self.context_level += 1
            for name, hook in fwd_hooks: self.add_hook(name, hook, dir="fwd", level=self.context_level)
            for name, hook in bwd_hooks: self.add_hook(name, hook, dir="bwd", level=self.context_level)
            yield self
        finally:
            if reset_hooks_end: self.reset_hooks(clear_contexts, including_permanent=False, level=self.context_level)
            self.context_level -= 1
            
    def run_with_hooks(self, *model_args: Any, fwd_hooks: List[Tuple[str, Callable]] = [], bwd_hooks: List[Tuple[str, Callable]] = [], reset_hooks_end: bool = True, clear_contexts: bool = False, **model_kwargs: Any):
        if len(bwd_hooks) > 0 and reset_hooks_end: logger.warning("WARNING: Hooks will be reset at the end of run_with_hooks. This removes the backward hooks before a backward pass can occur.")
        with self.hooks(fwd_hooks, bwd_hooks, reset_hooks_end, clear_contexts) as hooked_model:
            return hooked_model.forward(*model_args, **model_kwargs)

    def get_caching_hooks(self, names_filter: NamesFilter = None, incl_bwd: bool = False, device: Optional[torch.device] = None, remove_batch_dim: bool = False, cache: Optional[Dict[str, Tensor]] = None) -> Tuple[Dict[str, Tensor], List[Tuple[str, Callable]], List[Tuple[str, Callable]]]:
        if cache is None: cache = {}
        if names_filter is None: names_filter_fn = lambda name: True
        elif isinstance(names_filter, str): names_filter_fn = lambda name: name == names_filter
        elif isinstance(names_filter, list): names_filter_fn = lambda name: name in names_filter # type: ignore
        elif callable(names_filter): names_filter_fn = names_filter
        else: raise ValueError("names_filter must be a string, list of strings, or function")
        
        self.is_caching = True
        def save_hook(tensor: Tensor, hook: HookPoint, is_backward: bool = False):
            if hook.name is None: raise RuntimeError("Hook should have been provided a name")
            hook_name = hook.name
            if is_backward: hook_name += "_grad"
            resid_stream = tensor.to(device) if device is not None else tensor
            if remove_batch_dim: resid_stream = resid_stream[0]
            cache[hook_name] = resid_stream # type: ignore

        fwd_hooks: List[Tuple[str, Callable]] = []
        bwd_hooks: List[Tuple[str, Callable]] = []
        for name, _ in self.hook_dict.items():
            if names_filter_fn(name):
                fwd_hooks.append((name, functools.partial(save_hook, is_backward=False)))
                if incl_bwd: bwd_hooks.append((name, functools.partial(save_hook, is_backward=True)))
        return cache, fwd_hooks, bwd_hooks

    def run_with_cache(self, *model_args: Any, names_filter: NamesFilter = None, device: Optional[torch.device] = None, remove_batch_dim: bool = False, incl_bwd: bool = False, reset_hooks_end: bool = True, clear_contexts: bool = False, **model_kwargs: Any):
        cache_dict, fwd_hooks, bwd_hooks = self.get_caching_hooks(names_filter, incl_bwd, device, remove_batch_dim=remove_batch_dim)
        with self.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, reset_hooks_end=reset_hooks_end, clear_contexts=clear_contexts): # type: ignore
            model_out = self(*model_args, **model_kwargs)
            if incl_bwd:
                if not isinstance(model_out, Tensor) or model_out.ndim != 0 : # type: ignore
                    logger.warning(f"Output for incl_bwd is not a scalar tensor: {model_out}. Gradients may be unexpected.")
                model_out.backward() # type: ignore
        return model_out, cache_dict
        
# --- Model Components (from spd/models/components.py) ---
class Linear(nn.Module): # Uses local HookPoint
    # ... (exact copy from train_tms_standalone.py, but with HookPoint)
    def __init__(self, d_in: int, d_out: int, n_instances: Optional[int] = None, init_type: Literal["kaiming_uniform", "xavier_normal"] = "kaiming_uniform", init_scale: float = 1.0):
        super().__init__()
        shape = (n_instances, d_in, d_out) if n_instances is not None else (d_in, d_out)
        self.weight = nn.Parameter(torch.empty(shape))
        init_param_(self.weight, scale=init_scale, init_type=init_type)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        x = self.hook_pre(x)
        out = einops.einsum(x, self.weight, "batch ... d_in, ... d_in d_out -> batch ... d_out")
        out = self.hook_post(out)
        return out

class TransposedLinear(nn.Module): # Uses local HookPoint
    # ... (exact copy from train_tms_standalone.py, but with HookPoint and correct forward)
    def __init__(self, original_weight: nn.Parameter):
        super().__init__()
        self.register_buffer("original_weight", original_weight, persistent=False)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
    @property
    def weight(self) -> Tensor: return einops.rearrange(self.original_weight, "... d_in d_out -> ... d_out d_in")
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        x = self.hook_pre(x)
        out = einops.einsum(x, self.weight, "batch ... d_out, ... d_out d_in -> batch ... d_in")
        out = self.hook_post(out)
        return out

class LinearComponent(nn.Module): # Uses local HookPoint
    # ... (exact copy from spd/models/components.py)
    def __init__(self, d_in: int, d_out: int, C: int, n_instances: Optional[int] = None, init_type: Literal["kaiming_uniform", "xavier_normal"] = "kaiming_uniform", init_scale: float = 1.0, m: Optional[int] = None):
        super().__init__()
        self.n_instances = n_instances
        self.C = C
        self.m = min(d_in, d_out) if m is None else m
        shape_A = (n_instances, C, d_in, self.m) if n_instances is not None else (C, d_in, self.m)
        shape_B = (n_instances, C, self.m, d_out) if n_instances is not None else (C, self.m, d_out)
        self.A = nn.Parameter(torch.empty(shape_A))
        self.B = nn.Parameter(torch.empty(shape_B))
        self.hook_pre = HookPoint()
        self.hook_component_acts = HookPoint()
        self.hook_post = HookPoint()
        init_param_(self.A, scale=init_scale, init_type=init_type)
        init_param_(self.B, scale=init_scale, init_type=init_type)
    @property
    def component_weights(self) -> Tensor: return einops.einsum(self.A, self.B, "... C d_in m, ... C m d_out -> ... C d_in d_out")
    @property
    def weight(self) -> Tensor: return einops.einsum(self.A, self.B, "... C d_in m, ... C m d_out -> ... d_in d_out")
    def forward(self, x: Tensor, topk_mask: Optional[Bool[Tensor, "batch ... C"]] = None) -> Tensor: # type: ignore
        x = self.hook_pre(x)
        inner_acts = einops.einsum(x, self.A, "batch ... d_in, ... C d_in m -> batch ... C m")
        if topk_mask is not None:
            assert topk_mask.shape == inner_acts.shape[:-1]
            inner_acts = einops.einsum(inner_acts, topk_mask, "batch ... C m, batch ... C -> batch ... C m")
        component_acts = einops.einsum(inner_acts, self.B, "batch ... C m, ... C m d_out -> batch ... C d_out")
        self.hook_component_acts(component_acts)
        out = einops.einsum(component_acts, "batch ... C d_out -> batch ... d_out")
        out = self.hook_post(out)
        return out
        
class TransposedLinearComponent(LinearComponent): # Uses local HookPoint
    # ... (exact copy from spd/models/components.py)
    def __init__(self, original_A: nn.Parameter, original_B: nn.Parameter):
        nn.Module.__init__(self) # Call nn.Module init directly
        self.n_instances, self.C, _, self.m = original_A.shape
        self.hook_pre = HookPoint()
        self.hook_component_acts = HookPoint()
        self.hook_post = HookPoint()
        self.register_buffer("original_A", original_A, persistent=False)
        self.register_buffer("original_B", original_B, persistent=False)
    @property
    def A(self) -> Tensor: return einops.rearrange(self.original_B, "... C m d_out -> ... C d_out m")
    @property
    def B(self) -> Tensor: return einops.rearrange(self.original_A, "... C d_in m -> ... C m d_in")
    # component_weights and weight properties are inherited and work due to new A and B.

# --- Base SPDModel (from spd/models/base.py) ---
class SPDModel(HookedRootModule): # Inherits local HookedRootModule
    # ... (exact copy from spd/models/base.py)
    def set_subnet_to_zero(self, subnet_idx: int, has_instance_dim: bool) -> Dict[str, Tensor]:
        stored_vals = {}
        for attr_name in ["A", "B"]:
            params = collect_nested_module_attrs(self, attr_name)
            for param_name, param in params.items():
                if self.parent_is_transposed_linear(param_name): continue
                if has_instance_dim:
                    stored_vals[param_name] = param.data[:, subnet_idx, :, :].detach().clone()
                    param.data[:, subnet_idx, :, :] = 0.0
                else:
                    stored_vals[param_name] = param.data[subnet_idx, :, :].detach().clone()
                    param.data[subnet_idx, :, :] = 0.0
        return stored_vals

    def restore_subnet(self, subnet_idx: int, stored_vals: Dict[str, Tensor], has_instance_dim: bool) -> None:
        for name, val in stored_vals.items():
            param = get_nested_module_attr(self, name)
            if has_instance_dim: param.data[:, subnet_idx, :, :] = val
            else: param.data[subnet_idx, :, :] = val

    def set_As_to_unit_norm(self) -> None:
        params = collect_nested_module_attrs(self, "A")
        for param_name, param in params.items():
            if not self.parent_is_transposed_linear(param_name):
                param.data /= param.data.norm(p=2, dim=-2, keepdim=True)

    def fix_normalized_adam_gradients(self) -> None:
        params = collect_nested_module_attrs(self, "A")
        for param_name, param in params.items():
            if not self.parent_is_transposed_linear(param_name):
                assert param.grad is not None
                remove_grad_parallel_to_subnetwork_vecs(param.data, param.grad)

    def parent_is_transposed_linear(self, param_name: str) -> bool:
        parent_module_name = ".".join(param_name.split(".")[:-1])
        parent_module = get_nested_module_attr(self, parent_module_name)
        return isinstance(parent_module, TransposedLinearComponent)

# --- TMS Model Definitions (from spd/experiments/tms/models.py) ---
class TMSModelConfig(BaseModel): # Copied from train_tms_standalone.py
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_instances: PositiveInt
    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt
    device: str

def _tms_forward(x: Tensor, linear1: Union[Linear, LinearComponent], linear2: Union[TransposedLinear, TransposedLinearComponent], b_final: Tensor, topk_mask: Optional[Tensor] = None, hidden_layers: Optional[nn.ModuleList] = None) -> Tensor:
    # ... (exact copy from spd/experiments/tms/models.py, uses local components)
    hidden = linear1(x, topk_mask=topk_mask) if isinstance(linear1, LinearComponent) else linear1(x)
    if hidden_layers is not None:
        for layer in hidden_layers:
            hidden = layer(hidden, topk_mask=topk_mask) if isinstance(layer, LinearComponent) else layer(hidden)
    out_pre_relu = (linear2(hidden, topk_mask=topk_mask) if isinstance(linear2, TransposedLinearComponent) else linear2(hidden)) + b_final
    out = F.relu(out_pre_relu)
    return out

class TMSModel(HookedRootModule): # Inherits local HookedRootModule
    # ... (adapted from spd/experiments/tms/models.py)
    def __init__(self, config: TMSModelConfig):
        super().__init__()
        self.config = config
        self.linear1 = Linear(d_in=config.n_features, d_out=config.n_hidden, n_instances=config.n_instances, init_type="xavier_normal")
        self.linear2 = TransposedLinear(self.linear1.weight)
        self.b_final = nn.Parameter(torch.zeros((config.n_instances, config.n_features)))
        self.hidden_layers = None
        if config.n_hidden_layers > 0:
            self.hidden_layers = nn.ModuleList()
            for _ in range(config.n_hidden_layers):
                self.hidden_layers.append(Linear(d_in=config.n_hidden, d_out=config.n_hidden, n_instances=config.n_instances, init_type="xavier_normal"))
        self.setup() # HookedRootModule setup

    def forward(self, x: Tensor, **_: Any) -> Tensor:
        return _tms_forward(x=x, linear1=self.linear1, linear2=self.linear2, b_final=self.b_final, hidden_layers=self.hidden_layers)

    @classmethod
    def from_pretrained(cls, model_path: str, config_path: str) -> Tuple['TMSModel', Dict[str, Any]]: # Forward ref
        logger.info(f"Loading TMSModel from local paths: model_path='{model_path}', config_path='{config_path}'")
        with open(config_path) as f:
            tms_train_config_dict = yaml.safe_load(f)
        
        # The train_config_dict from train_tms.py contains tms_model_config nested.
        # If the provided config_path points directly to a TMSModelConfig yaml, adapt this.
        # Assuming it's the output from train_tms_standalone.py (TMSTrainConfig)
        if "tms_model_config" not in tms_train_config_dict:
             # Try to load as if it's TMSModelConfig directly
            try:
                tms_model_config_dict_for_model = tms_train_config_dict 
                tms_config = TMSModelConfig(**tms_model_config_dict_for_model)
                # In this case, the loaded dict IS the model config dict, so we need to wrap it for the return value
                # to match original structure if other parts of APD script expect `tms_train_config_dict['tms_model_config']`
                # However, for standalone, it's simpler if tms_train_config_dict *is* the one with `tms_model_config` key.
                # Let's assume the config_path is for TMSTrainConfig format.
                # If it was just TMSModelConfig, the caller would need to prepare tms_train_config_dict.
                raise KeyError("Assuming config_path points to a TMSTrainConfig yaml with a 'tms_model_config' key.")

            except KeyError: # If it's not TMSTrainConfig format.
                 raise ValueError(f"Config file at {config_path} does not contain 'tms_model_config' key. Please provide a config file as saved by train_tms_standalone.py (TMSTrainConfig format), or a direct TMSModelConfig yaml.")

        tms_config = TMSModelConfig(**tms_train_config_dict["tms_model_config"])
        tms = cls(config=tms_config)
        
        params = torch.load(model_path, weights_only=True, map_location=tms_config.device) # Use device from config
        params = replace_deprecated_param_names(params, {"W": "linear1.weight"}) # From original
        tms.load_state_dict(params)
        logger.info("TMSModel loaded successfully from local files.")
        return tms, tms_train_config_dict


class TMSSPDModelConfig(BaseModel): # Copied from spd/experiments/tms/models.py
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_instances: PositiveInt
    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt
    C: Optional[PositiveInt] = None # Changed from PositiveInt | None
    bias_val: float
    device: str
    m: Optional[PositiveInt] = None # Changed from PositiveInt | None

class TMSSPDModel(SPDModel): # Inherits local SPDModel
    # ... (exact copy from spd/experiments/tms/models.py, uses local components)
    def __init__(self, config: TMSSPDModelConfig):
        super().__init__()
        self.config = config
        self.n_instances = config.n_instances 
        self.n_features = config.n_features 
        self.C = config.C if config.C is not None else config.n_features
        self.bias_val = config.bias_val
        self.m = min(config.n_features, config.n_hidden) + 1 if config.m is None else config.m

        self.linear1 = LinearComponent(d_in=config.n_features, d_out=config.n_hidden, n_instances=config.n_instances, init_type="xavier_normal", init_scale=1.0, C=self.C, m=self.m)
        self.linear2 = TransposedLinearComponent(self.linear1.A, self.linear1.B)
        bias_data = (torch.zeros((config.n_instances, config.n_features), device=config.device) + config.bias_val)
        self.b_final = nn.Parameter(bias_data)
        self.hidden_layers = None
        if config.n_hidden_layers > 0:
            self.hidden_layers = nn.ModuleList(
                [LinearComponent(d_in=config.n_hidden, d_out=config.n_hidden, n_instances=config.n_instances, init_type="xavier_normal", init_scale=1.0, C=self.C, m=self.m) for _ in range(config.n_hidden_layers)]
            )
        self.setup() # HookedRootModule setup
    def forward(self, x: Tensor, topk_mask: Optional[Tensor] = None) -> Tensor:
        return _tms_forward(x=x, linear1=self.linear1, linear2=self.linear2, b_final=self.b_final, hidden_layers=self.hidden_layers, topk_mask=topk_mask)

# --- Core APD Logic (from spd/run_spd.py) ---
class TMSTaskConfig(BaseModel): # Copied from spd/run_spd.py
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["tms"] = "tms"
    feature_probability: Probability
    train_bias: bool
    bias_val: float
    data_generation_type: DataGenerationType = "at_least_zero_active"
    pretrained_model_path: ModelPath 
    # Add new field for local target model config path
    pretrained_model_config_path: ModelPath 


class Config(BaseModel): # Copied from spd/run_spd.py, renamed ResidualMLPTaskConfig to avoid defining it
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: Optional[str] = None # Will be None for standalone
    wandb_run_name: Optional[str] = None
    wandb_run_name_prefix: str = ""
    seed: int = 0
    topk: Optional[PositiveFloat] = None
    batch_topk: bool = True
    exact_topk: bool = False
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    image_freq: Optional[PositiveInt] = None
    image_on_first_step: bool = True
    slow_images: bool = False # Not used if no W&B
    save_freq: Optional[PositiveInt] = None
    lr: PositiveFloat
    out_recon_coeff: Optional[NonNegativeFloat] = None
    act_recon_coeff: Optional[NonNegativeFloat] = None
    param_match_coeff: Optional[NonNegativeFloat] = 1.0
    topk_recon_coeff: Optional[NonNegativeFloat] = None
    schatten_coeff: Optional[NonNegativeFloat] = None
    schatten_pnorm: Optional[NonNegativeFloat] = None
    lp_sparsity_coeff: Optional[NonNegativeFloat] = None
    distil_from_target: bool = False
    pnorm: Optional[PositiveFloat] = None
    C: PositiveInt
    m: Optional[PositiveInt] = None
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = "constant"
    lr_exponential_halflife: Optional[PositiveFloat] = None
    lr_warmup_pct: Probability = 0.0
    sparsity_loss_type: Literal["jacobian"] = "jacobian" # Only jacobian was used in original
    unit_norm_matrices: bool = False
    attribution_type: Literal["gradient", "ablation", "activation"] = "gradient"
    task_config: TMSTaskConfig # Simplified, only TMSTaskConfig

    DEPRECATED_CONFIG_KEYS: ClassVar[List[str]] = ["topk_param_attrib_coeff", "orthog_coeff", "hardcode_topk_mask_step", "pnorm_end", "topk_l2_coeff", "spd_type", "sparsity_warmup_pct"]
    RENAMED_CONFIG_KEYS: ClassVar[Dict[str, str]] = {"topk_act_recon_coeff": "act_recon_coeff"}
    
    @model_validator(mode="before")
    def handle_deprecated_config_keys(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        # ... (exact copy from spd/run_spd.py, logger replaced)
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
    def validate_model(self) -> 'Config': # Using Self requires Python 3.9+ and from typing import Self
        # ... (exact copy from spd/run_spd.py, logger replaced)
        if self.topk is not None:
            if self.batch_topk:
                if not (self.batch_size * self.topk).is_integer(): logger.warning(f"batch_size * topk={self.batch_size * self.topk} is not an integer, will round down")
            else:
                if not self.topk.is_integer(): raise ValueError("topk must be an integer when not using batch_topk")
        if not self.topk_recon_coeff and not self.lp_sparsity_coeff: logger.warning("Neither topk_recon_coeff nor lp_sparsity_coeff is set")
        if self.topk_recon_coeff is not None: assert self.topk is not None, "topk must be set if topk_recon_coeff is set"
        if self.lp_sparsity_coeff is not None: assert self.pnorm is not None, "pnorm must be set if lp_sparsity_coeff is set"
        if self.topk is None: assert self.topk_recon_coeff is None, "topk_recon_coeff is not None but topk is"
        if (self.param_match_coeff is not None and self.param_match_coeff > 0 and self.out_recon_coeff is not None and self.out_recon_coeff > 0):
            logger.warning("Both param_match_coeff and out_recon_coeff are > 0. It's typical to only set one.")
        msg = "is 0, you may wish to instead set it to null to avoid calculating the loss"
        if self.topk_recon_coeff == 0: logger.warning(f"topk_recon_coeff {msg}")
        if self.lp_sparsity_coeff == 0: logger.warning(f"lp_sparsity_coeff {msg}")
        if self.param_match_coeff == 0: logger.warning(f"param_match_coeff {msg}")
        if self.lr_schedule == "exponential": assert self.lr_exponential_halflife is not None, "lr_exponential_halflife must be set if lr_schedule is exponential"
        if self.schatten_coeff is not None: assert self.schatten_pnorm is not None, "schatten_pnorm must be set if schatten_coeff is set"
        return self

# Loss functions from spd/run_spd.py
def _calc_param_mse(params1: Dict[str, Tensor], params2: Dict[str, Tensor], n_params: int, device: str) -> Tensor:
    # ... (exact copy from spd/run_spd.py)
    param_match_loss = torch.tensor(0.0, device=device)
    for name in params1: param_match_loss = param_match_loss + ((params2[name] - params1[name]) ** 2).sum(dim=(-2, -1))
    return param_match_loss / n_params

def calc_param_match_loss(param_names: List[str], target_model: HookedRootModule, spd_model: SPDModel, n_params: int, device: str) -> Tensor:
    # ... (exact copy from spd/run_spd.py)
    target_params = {}
    spd_params = {}
    for param_name in param_names:
        target_params[param_name] = get_nested_module_attr(target_model, param_name + ".weight")
        spd_params[param_name] = get_nested_module_attr(spd_model, param_name + ".weight")
    return _calc_param_mse(params1=target_params, params2=spd_params, n_params=n_params, device=device)

def calc_lp_sparsity_loss(out: Tensor, attributions: Tensor, step_pnorm: float) -> Tensor:
    # ... (exact copy from spd/run_spd.py)
    d_model_out = out.shape[-1]
    attributions = attributions / d_model_out
    lp_sparsity_loss_per_k = (attributions.abs() + 1e-16) ** (step_pnorm * 0.5)
    return lp_sparsity_loss_per_k

def calc_schatten_loss(As: Dict[str, Tensor], Bs: Dict[str, Tensor], mask: Tensor, p: float, n_params: int, device: str) -> Tensor:
    # ... (exact copy from spd/run_spd.py)
    assert As.keys() == Bs.keys(), "As and Bs must have the same keys"
    n_instances = mask.shape[1] if mask.ndim == 3 else None
    accumulate_shape = (n_instances,) if n_instances is not None else ()
    schatten_penalty = torch.zeros(accumulate_shape, device=device) # type: ignore
    batch_size = mask.shape[0]
    for name in As:
        A, B = As[name], Bs[name]
        S_A = einops.einsum(A, A, "... C d_in m, ... C d_in m -> ... C m")
        S_B = einops.einsum(B, B, "... C m d_out, ... C m d_out -> ... C m")
        S_AB = S_A * S_B
        S_AB_topk = einops.einsum(S_AB, mask, "... C m, batch ... C -> batch ... C m")
        schatten_penalty = schatten_penalty + ((S_AB_topk + 1e-16) ** (0.5 * p)).sum(dim=(0, -2, -1))
    return schatten_penalty / n_params / batch_size

def calc_act_recon(target_post_weight_acts: Dict[str, Tensor], layer_acts: Dict[str, Tensor]) -> Tensor:
    # ... (exact copy from spd/run_spd.py)
    assert target_post_weight_acts.keys() == layer_acts.keys(), f"Layer keys must match: {target_post_weight_acts.keys()} != {layer_acts.keys()}"
    device = next(iter(layer_acts.values())).device
    total_act_dim = 0
    loss = torch.zeros(1, device=device) if len(next(iter(layer_acts.values())).shape) == 2 else torch.zeros(next(iter(layer_acts.values())).shape[1], device=device) # handle instance dim
    
    for layer_name in target_post_weight_acts:
        total_act_dim += target_post_weight_acts[layer_name].shape[-1]
        error = ((target_post_weight_acts[layer_name] - layer_acts[layer_name]) ** 2).sum(dim=-1)
        loss = loss + error
    return (loss / total_act_dim).mean(dim=0)


# optimize function from spd/run_spd.py
def optimize(model: SPDModel, config: Config, device: str, dataloader: torch.utils.data.DataLoader, target_model: HookedRootModule, param_names: List[str], plot_results_fn: Optional[Callable[..., Dict[str, plt.Figure]]] = None, out_dir: Optional[pathlib.Path] = None) -> None:
    # ... (exact copy from spd/run_spd.py, W&B calls removed, logger replaced)
    model.to(device=device)
    target_model.to(device=device)
    has_instance_dim = hasattr(model, "n_instances")
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)
    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)
    n_params = 0
    for param_name in param_names: n_params += get_nested_module_attr(target_model, param_name + ".weight").numel()
    if has_instance_dim: n_params = n_params / model.n_instances # type: ignore

    epoch = 0
    data_iter = iter(dataloader)
    for step in tqdm(range(config.steps + 1), ncols=0):
        if config.unit_norm_matrices:
            assert isinstance(model, SPDModel), "Can only norm matrices in SPDModel instances"
            model.set_As_to_unit_norm()
        step_lr = get_lr_with_warmup(step=step, steps=config.steps, lr=config.lr, lr_schedule_fn=lr_schedule_fn, lr_warmup_pct=config.lr_warmup_pct)
        for group in opt.param_groups: group["lr"] = step_lr
        opt.zero_grad(set_to_none=True)
        try: batch = next(data_iter)[0]
        except StopIteration:
            logger.info(f"Epoch {epoch} finished, starting new epoch") # Use logger
            epoch += 1; data_iter = iter(dataloader); batch = next(data_iter)[0]
        batch = batch.to(device=device)
        
        target_cache_filter = lambda k: k.endswith((".hook_pre", ".hook_post"))
        target_out, target_cache = target_model.run_with_cache(batch, names_filter=target_cache_filter)
        spd_cache_filter = lambda k: k.endswith((".hook_post", ".hook_component_acts"))
        out, spd_cache = model.run_with_cache(batch, names_filter=spd_cache_filter)

        out_recon_loss = calc_recon_mse(out, target_out, has_instance_dim)
        param_match_loss = None
        if config.param_match_coeff is not None: param_match_loss = calc_param_match_loss(param_names=param_names, target_model=target_model, spd_model=model, n_params=n_params, device=device) # type: ignore
        
        post_weight_acts = {k: v for k, v in target_cache.items() if k.endswith("hook_post")}
        attributions = calculate_attributions(model=model, batch=batch, out=out, target_out=target_out, pre_weight_acts={k: v for k, v in target_cache.items() if k.endswith("hook_pre")}, post_weight_acts=post_weight_acts, component_acts={k: v for k, v in spd_cache.items() if k.endswith("hook_component_acts")}, attribution_type=config.attribution_type)
        
        lp_sparsity_loss_per_k = None
        if config.lp_sparsity_coeff is not None:
            assert config.pnorm is not None
            lp_sparsity_loss_per_k = calc_lp_sparsity_loss(out=out, attributions=attributions, step_pnorm=config.pnorm)

        out_topk, schatten_loss, topk_recon_loss, topk_mask, layer_acts_topk = None, None, None, None, None
        if config.topk is not None:
            topk_attrs: Tensor = attributions[..., :-1] if config.distil_from_target else attributions
            if config.exact_topk:
                assert config.batch_topk and hasattr(model, "n_instances") and model.n_instances == 1 # type: ignore
                exact_topk_val = ((batch != 0).sum() / batch.shape[0]).item() # type: ignore
                topk_mask = calc_topk_mask(topk_attrs, exact_topk_val, batch_topk=True)
            else: topk_mask = calc_topk_mask(topk_attrs, config.topk, batch_topk=config.batch_topk)
            if config.distil_from_target:
                last_subnet_mask = torch.ones((*topk_mask.shape[:-1], 1), dtype=torch.bool, device=device) # type: ignore
                topk_mask = torch.cat((topk_mask, last_subnet_mask), dim=-1) # type: ignore
            out_topk, topk_spd_cache = model.run_with_cache(batch, names_filter=spd_cache_filter, topk_mask=topk_mask)
            layer_acts_topk = {k: v for k, v in topk_spd_cache.items() if k.endswith("hook_post")}
            if config.topk_recon_coeff is not None:
                assert out_topk is not None
                topk_recon_loss = calc_recon_mse(out_topk, target_out, has_instance_dim)
        
        act_recon_loss = None
        if config.act_recon_coeff is not None:
            # Simplified: No ResidualMLPTaskConfig specific logic for standalone
            act_recon_layer_acts = layer_acts_topk if layer_acts_topk is not None else {k: v for k, v in spd_cache.items() if k.endswith("hook_post")}
            act_recon_loss = calc_act_recon(target_post_weight_acts=post_weight_acts, layer_acts=act_recon_layer_acts)

        if config.schatten_coeff is not None:
            current_mask = topk_mask if topk_mask is not None else lp_sparsity_loss_per_k # type: ignore
            assert current_mask is not None
            schatten_pnorm_val = config.schatten_pnorm if config.schatten_pnorm is not None else 1.0
            schatten_loss = calc_schatten_loss(As=collect_nested_module_attrs(model, attr_name="A", include_attr_name=False), Bs=collect_nested_module_attrs(model, attr_name="B", include_attr_name=False), mask=current_mask, p=schatten_pnorm_val, n_params=n_params, device=device) # type: ignore

        lp_sparsity_loss = None
        if lp_sparsity_loss_per_k is not None: lp_sparsity_loss = lp_sparsity_loss_per_k.sum(dim=-1).mean(dim=0)

        loss_terms = {"param_match_loss": (param_match_loss, config.param_match_coeff), "out_recon_loss": (out_recon_loss, config.out_recon_coeff), "lp_sparsity_loss": (lp_sparsity_loss, config.lp_sparsity_coeff), "topk_recon_loss": (topk_recon_loss, config.topk_recon_coeff), "act_recon_loss": (act_recon_loss, config.act_recon_coeff), "schatten_loss": (schatten_loss, config.schatten_coeff)}
        loss = torch.tensor(0.0, device=device)
        for loss_name, (loss_term, coeff) in loss_terms.items():
            if coeff is not None:
                assert loss_term is not None, f"{loss_name} is None but coeff is not"
                loss = loss + coeff * loss_term.mean()

        if step % config.print_freq == 0:
            tqdm.write(f"Step {step}\nTotal loss: {loss.item()}\nlr: {step_lr}")
            for loss_name, (val, _) in loss_terms.items():
                if val is not None: tqdm.write(f"{loss_name}: {val.mean().item() if val.numel() > 1 else val.item()}")
        
        if plot_results_fn is not None and config.image_freq is not None and step % config.image_freq == 0 and (step > 0 or config.image_on_first_step) and out_dir is not None:
            fig_dict = plot_results_fn(model=model, target_model=target_model, step=step, out_dir=out_dir, device=device, config=config, topk_mask=topk_mask, batch=batch)
            # Plots are saved to disk by plot_results_fn, no W&B log needed.

        if ((config.save_freq is not None and step % config.save_freq == 0 and step > 0) or step == config.steps) and out_dir is not None:
            torch.save(model.state_dict(), out_dir / f"spd_model_{step}.pth")
            tqdm.write(f"Saved model to {out_dir / f'spd_model_{step}.pth'}")

        if step != config.steps:
            loss.backward()
            if config.unit_norm_matrices: model.fix_normalized_adam_gradients() # type: ignore
            opt.step()

# --- Main Script and Plotting (from spd/experiments/tms/tms_decomposition.py) ---
def get_run_name(config: Config, tms_model_config: TMSModelConfig) -> str: # Simplified
    # ... (copy from tms_decomposition.py, remove W&B specific naming)
    run_suffix = f"C{config.C}_topk{config.topk if config.topk is not None else 'None'}"
    run_suffix += f"_seed{config.seed}"
    run_suffix += f"_lr{config.lr:.0e}"
    run_suffix += f"_nfeat{tms_model_config.n_features}_nhid{tms_model_config.n_hidden}"
    return config.wandb_run_name_prefix + run_suffix # wandb_run_name_prefix can be empty

# Plotting functions (copied from tms_decomposition.py, logger replaced)
def plot_A_matrix(x: torch.Tensor, pos_only: bool = False) -> plt.Figure:
    # ... (exact copy)
    n_instances = x.shape[0]
    fig, axs_raw = plt.subplots(1, n_instances, figsize=(2.5 * n_instances, 2), squeeze=False, sharey=True)
    axs = axs_raw.flatten()
    cmap = "Blues" if pos_only else "RdBu"
    for i in range(n_instances):
        ax = axs[i]
        instance_data = x[i, :, :].detach().cpu().float().numpy()
        max_abs_val = np.abs(instance_data).max() if np.abs(instance_data).max() > 1e-6 else 1.0 # Avoid div by zero for empty
        vmin = 0 if pos_only else -max_abs_val
        im = ax.matshow(instance_data, vmin=vmin, vmax=max_abs_val, cmap=cmap)
        ax.xaxis.set_ticks_position("bottom")
        if i == 0: ax.set_ylabel("k", rotation=0, labelpad=10, va="center")
        else: ax.set_yticks([])
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("n_features")
    plt.subplots_adjust(wspace=0.1, bottom=0.15, top=0.9)
    return fig
    
def plot_subnetwork_attributions_multiple_instances(attribution_scores: Tensor, out_dir: pathlib.Path, step: Optional[int]) -> plt.Figure:
    # ... (exact copy, tqdm.write used)
    n_instances = attribution_scores.shape[1]
    fig, axes_raw = plt.subplots(1, n_instances, figsize=(5 * n_instances, 5), constrained_layout=True, squeeze=False)
    axes = axes_raw.flatten()
    images = []
    for idx, ax in enumerate(axes):
        instance_scores = attribution_scores[:, idx, :]
        im = ax.matshow(instance_scores.detach().cpu().numpy(), aspect="auto", cmap="Reds")
        images.append(im)
        for r_idx in range(instance_scores.shape[0]): # row index
            for c_idx in range(instance_scores.shape[1]): # col index
                ax.text(c_idx, r_idx, f"{instance_scores[r_idx, c_idx]:.2f}", ha="center", va="center", color="black", fontsize=8) # Smaller font
        ax.set_xlabel("Subnetwork Index")
        if idx == 0: ax.set_ylabel("Batch Index")
        ax.set_title(f"Instance {idx}")
    if images: # Ensure colorbar is added only if images exist
        norm = plt.Normalize(vmin=attribution_scores.min().item(), vmax=attribution_scores.max().item())
        for im_obj in images: im_obj.set_norm(norm) # Use im_obj to avoid conflict
        fig.colorbar(images[0], ax=axes.tolist()) # Pass list of axes
    fig.suptitle(f"Subnetwork Attributions (Step {step})")
    filename = f"subnetwork_attributions_s{step}.png" if step is not None else "subnetwork_attributions.png"
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight"); plt.close(fig)
    tqdm.write(f"Saved subnetwork attributions to {out_dir / filename}")
    return fig

def plot_subnetwork_attributions_statistics_multiple_instances(topk_mask: Tensor, out_dir: pathlib.Path, step: Optional[int]) -> plt.Figure:
    # ... (exact copy, tqdm.write used)
    n_instances = topk_mask.shape[1]
    fig, axes_raw = plt.subplots(1, n_instances, figsize=(5 * n_instances, 5), constrained_layout=True, squeeze=False)
    axes = axes_raw.flatten()
    for instance_idx in range(n_instances):
        ax = axes[instance_idx]
        instance_mask_sum = topk_mask[:, instance_idx].sum(dim=1).cpu().detach().numpy() # Sum over C dim
        min_val, max_val = int(instance_mask_sum.min().item()), int(instance_mask_sum.max().item())
        bins = list(range(min_val, max_val + 2))
        counts, _ = np.histogram(instance_mask_sum, bins=bins)
        bars = ax.bar(bins[:-1], counts, align="center", width=0.8)
        ax.set_xticks(bins[:-1]); ax.set_xticklabels([str(b) for b in bins[:-1]])
        ax.set_title(f"Instance {instance_idx}")
        if instance_idx == 0: ax.set_ylabel("Count")
        ax.set_xlabel("Number of active subnetworks")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    fig.suptitle(f"Active subnetworks per instance (batch_size={topk_mask.shape[0]})")
    filename = f"subnetwork_attributions_statistics_s{step}.png" if step is not None else "subnetwork_attributions_statistics.png"
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight"); plt.close(fig)
    tqdm.write(f"Saved subnetwork attributions statistics to {out_dir / filename}")
    return fig
    
def plot_component_weights(model: TMSSPDModel, step: int, out_dir: pathlib.Path, **_) -> plt.Figure:
    # ... (exact copy, tqdm.write used)
    component_weights = model.linear1.component_weights
    n_instances, C, _, _ = component_weights.shape
    fig, axs_raw = plt.subplots(C, n_instances, figsize=(2 * n_instances, 2 * C), constrained_layout=True, squeeze=False)
    axs = axs_raw # Keep 2D structure
    for i in range(n_instances):
        instance_max_abs = np.abs(component_weights[i].detach().cpu().numpy()).max()
        instance_max_abs = instance_max_abs if instance_max_abs > 1e-6 else 1.0
        for j in range(C):
            ax = axs[j, i]
            param = component_weights[i, j].detach().cpu().numpy()
            ax.matshow(param, cmap="RdBu", vmin=-instance_max_abs, vmax=instance_max_abs)
            ax.set_xticks([])
            if i == 0: ax.set_ylabel(f"k={j}", rotation=0, ha="right", va="center")
            if j == C - 1: ax.set_xlabel(f"Inst {i}", rotation=45, ha="right")
    fig.suptitle(f"Component Weights (Step {step})")
    fig.savefig(out_dir / f"component_weights_{step}.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    tqdm.write(f"Saved component weights to {out_dir / f'component_weights_{step}.png'}")
    return fig

def plot_batch_frequencies(frequencies: Tensor, xlabel: str, ax: plt.Axes, batch_size: int, title: Optional[str] = None) -> None:
    # ... (exact copy)
    n_instances, C = frequencies.shape
    for instance_idx in range(n_instances):
        bars = ax.bar(np.arange(C) + instance_idx * (C + 1), frequencies[instance_idx].detach().cpu().numpy(), align="center", width=0.8, label=f"Instance {instance_idx}")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    ax.set_xlabel(xlabel); ax.set_ylabel(f"Activation Count (batch_size={batch_size})")
    if title: ax.set_title(title)
    all_ticks, all_labels = [], []
    for i in range(n_instances):
        ticks = np.arange(C) + i * (C + 1)
        all_ticks.extend(ticks); all_labels.extend([str(j) for j in range(C)])
    ax.set_xticks(all_ticks); ax.set_xticklabels(all_labels)

def plot_batch_statistics(batch: Tensor, topk_mask: Tensor, out_dir: pathlib.Path, step: Optional[int]) -> Dict[str, plt.Figure]:
    # ... (exact copy, tqdm.write used)
    active_input_feats = (batch != 0).sum(dim=0) # Sum over batch dim
    topk_activations = topk_mask.sum(dim=0) # Sum over batch dim
    fig = plt.figure(figsize=(15, 10)); gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0]); plot_batch_frequencies(active_input_feats, "Input feature index", ax1, batch.shape[0], "Input feature frequencies across batch")
    ax2 = fig.add_subplot(gs[1]); plot_batch_frequencies(topk_activations, "Component index", ax2, batch.shape[0], "Component frequencies across batch")
    y_lims = [ax.get_ylim() for ax in [ax1, ax2]]; y_max = max(y_lims[0][1], y_lims[1][1])
    for ax in [ax1, ax2]: ax.set_ylim(0, y_max)
    filename = f"batch_statistics_s{step}.png" if step is not None else "batch_statistics.png"
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight"); plt.close(fig)
    tqdm.write(f"Saved batch statistics to {out_dir / filename}")
    return {"batch_statistics": fig}

def make_plots(model: TMSSPDModel, target_model: TMSModel, step: int, out_dir: pathlib.Path, device: str, config: Config, topk_mask: Optional[Tensor], batch: Tensor, **_) -> Dict[str, plt.Figure]:
    # ... (exact copy from tms_decomposition.py, logger replaced)
    plots = {}
    if model.hidden_layers is not None: logger.warning("Only plotting the W matrix params and not the hidden layers.") # Use logger
    plots["component_weights"] = plot_component_weights(model, step, out_dir)
    if config.topk is not None:
        assert topk_mask is not None; assert isinstance(config.task_config, TMSTaskConfig)
        n_instances = model.config.n_instances if hasattr(model, "config") else model.n_instances # type: ignore
        attribution_scores = collect_subnetwork_attributions(spd_model=model, target_model=target_model, device=device, n_instances=n_instances)
        plots["subnetwork_attributions"] = plot_subnetwork_attributions_multiple_instances(attribution_scores=attribution_scores, out_dir=out_dir, step=step)
        plots["subnetwork_attributions_statistics"] = plot_subnetwork_attributions_statistics_multiple_instances(topk_mask=topk_mask, out_dir=out_dir, step=step)
        batch_stat_plots = plot_batch_statistics(batch, topk_mask, out_dir, step)
        plots.update(batch_stat_plots)
    return plots

def save_target_model_info(out_dir: pathlib.Path, tms_model: TMSModel, tms_model_train_config_dict: Dict[str, Any]) -> None: # Simplified
    # ... (copy from tms_decomposition.py, W&B calls removed)
    torch.save(tms_model.state_dict(), out_dir / "tms.pth")
    with open(out_dir / "tms_train_config.yaml", "w") as f: yaml.dump(tms_model_train_config_dict, f, indent=2)
    logger.info(f"Saved target model and its config to {out_dir}")


# Main script execution part
def main_apd(config_path: str) -> None: # Renamed from main to avoid conflict if this file is imported
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load APD config (Config Pydantic model)
    apd_config = load_config(pathlib.Path(config_path), Config)
    logger.info(f"Loaded APD config from {config_path}")

    task_config = apd_config.task_config
    assert isinstance(task_config, TMSTaskConfig), "This script is for TMS task config."

    set_seed(apd_config.seed)
    logger.info(f"APD Config: {apd_config.model_dump_json(indent=2)}")

    # Load pretrained target TMSModel using adapted from_pretrained
    # TMSTaskConfig now needs pretrained_model_config_path
    if not hasattr(task_config, 'pretrained_model_config_path') or task_config.pretrained_model_config_path is None:
        # Infer config path if it's in the same directory as the model path
        inferred_config_path = pathlib.Path(task_config.pretrained_model_path).parent / "tms_train_config.yaml"
        if not inferred_config_path.exists():
            raise ValueError("pretrained_model_config_path not specified in TMSTaskConfig and could not be inferred (expected tms_train_config.yaml in same dir as model).")
        logger.info(f"pretrained_model_config_path not set, inferred as: {inferred_config_path}")
        target_model_train_config_path = inferred_config_path
    else:
        target_model_train_config_path = pathlib.Path(task_config.pretrained_model_config_path)

    target_model, target_model_train_config_dict = TMSModel.from_pretrained(
        model_path=str(task_config.pretrained_model_path),
        config_path=str(target_model_train_config_path)
    )
    target_model = target_model.to(device)
    target_model.eval()

    run_name = get_run_name(config=apd_config, tms_model_config=target_model.config)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    # Define output directory relative to script execution or a fixed location
    base_out_dir = pathlib.Path("./out_standalone_apd")
    out_dir = base_out_dir / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputs will be saved to: {out_dir}")

    with open(out_dir / "final_apd_config.yaml", "w") as f: # Renamed to avoid conflict
        yaml.dump(apd_config.model_dump(mode="json"), f, indent=2)
    
    # Save target model info (already loaded, just save a copy in APD output dir for completeness)
    save_target_model_info(
        out_dir=out_dir,
        tms_model=target_model,
        tms_model_train_config_dict=target_model_train_config_dict
    )

    tms_spd_model_config = TMSSPDModelConfig(
        **target_model.config.model_dump(mode="json"), # Use TMSModelConfig from loaded target
        C=apd_config.C,
        m=apd_config.m,
        bias_val=task_config.bias_val,
        device=device # ensure device is consistent
    )
    spd_model = TMSSPDModel(config=tms_spd_model_config)
    spd_model.b_final.data[:] = target_model.b_final.data.clone()
    if not task_config.train_bias:
        spd_model.b_final.requires_grad = False

    param_names = ["linear1", "linear2"]
    if spd_model.hidden_layers is not None:
        for i in range(len(spd_model.hidden_layers)): # type: ignore
            param_names.append(f"hidden_layers.{i}")

    # Use synced_inputs from the loaded target model's training config
    synced_inputs_from_target_config = target_model_train_config_dict.get("synced_inputs")

    dataset = SparseFeatureDataset(
        n_instances=target_model.config.n_instances,
        n_features=target_model.config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
        data_generation_type=task_config.data_generation_type, # type: ignore
        value_range=(0.0, 1.0),
        synced_inputs=synced_inputs_from_target_config,
    )
    # Pass the batch_size from apd_config to the dataloader for generation
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=apd_config.batch_size)
    # Store the generation batch_size for the loader's __iter__ method
    dataloader._intended_batch_size = apd_config.batch_size


    optimize(
        model=spd_model,
        config=apd_config,
        device=device,
        dataloader=dataloader,
        target_model=target_model,
        param_names=param_names,
        out_dir=out_dir,
        plot_results_fn=make_plots,
    )
    logger.info(f"APD optimization finished. Results in {out_dir}")

if __name__ == "__main__":
    fire.Fire(main_apd)
