# APD - Attribution-based Parameter Decomposition
Code used in the paper [Interpretability in Parameter Space: Minimizing
Mechanistic Description Length with
Attribution-based Parameter Decomposition](https://publications.apolloresearch.ai/apd)

Weights and Bias report accompanying the paper: https://api.wandb.ai/links/apollo-interp/h5ekyxm7

Note: previously called Sparse Parameter Decomposition (SPD). The package name will remain as `spd`
for now, but the repository has been renamed to `apd`.

## Installation
From the root of the repository, run one of

```bash
make install-dev  # To install the package, dev requirements and pre-commit hooks
make install  # To just install the package (runs `pip install -e .`)
```

## Usage
Place your wandb information in a .env file. You can use the .env.example file as an example.

The repository consists of several `experiments`, each of which containing scripts to train target
models and run APD.
- `spd/experiments/tms` - Toy model of superposition
- `spd/experiments/resid_mlp` - Toy model of compressed computation and toy model of distributed
  representations

Deprecated:
- `spd/experiments/piecewise` - Handcoded gated function model. Use [this](117284172497ca420f22c29cef3ddcd5e4bcceb8) commit if you need to use
  this experiment.

### Train a target model
All experiments require training a target model. Look for the `train_*.py` script in the experiment
directory. Your trained model will be saved locally and uploaded to wandb.

### Run APD
APD can be run by executing any of the `*_decomposition.py` scripts defined in the experiment
subdirectories. A config file is required for each experiment, which can be found in the same
directory. For example:
```bash
python spd/experiments/tms/tms_decomposition.py spd/experiments/tms/tms_topk_config.yaml
```
will run SPD on TMS with the config file `tms_topk_config.yaml` (which is the main config file used
for the TMS experiments in the paper).

Wandb sweep files are also provided in the experiment subdirectories, and can be run with e.g.:
```bash
wandb sweep spd/experiments/tms/tms_sweep_config.yaml
```

All experiments call the `optimize` function in `spd/run_spd.py`, which contains the main APD logic.

### Analyze results
Experiments contain `*_interp.py` scripts which generate the plots used in the paper.

## Running TMS Locally (Without W&B)

This section describes how to run the Toy Model of Superposition (TMS) experiment, including training the target model and running APD, entirely locally without requiring Weights & Biases.

### Prerequisites

*   Python (version as specified in `pyproject.toml`, e.g., 3.10 or higher).
*   `pip` for installing packages.
*   Install the necessary requirements by running `make install` or `pip install -e .` from the root of the repository.

### 1. Training the TMS Model Locally

The script `spd/experiments/tms/train_tms.py` is used to train the TMS target model. By default, it is configured to run locally, with W&B logging disabled.

To start local training, execute:
```bash
python spd/experiments/tms/train_tms.py
```

This will:
*   Use a default `TMSTrainConfig`. The script now defaults to the following configuration for local runs:
    ```python
    config = TMSTrainConfig(
        wandb_project=None,  # Disables W&B logging
        tms_model_config=TMSModelConfig(
            n_features=40,
            n_hidden=10,
            n_hidden_layers=0,
            n_instances=3,
            device="cuda" if torch.cuda.is_available() else "cpu", # Or your preferred device
        ),
        feature_probability=0.05,
        batch_size=2048,
        steps=2000, # Note: for a well-trained model, prefer 20_000 or more
        seed=0,
        lr=1e-3,
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=False,
        fixed_random_hidden_layers=False,
    )
    ```
*   Save the trained model checkpoint (e.g., `tms.pth`) and its corresponding training configuration (`tms_train_config.yaml`) into a timestamped subdirectory within `spd/experiments/tms/out/`. For example: `spd/experiments/tms/out/tms_n-features40_n-hidden10_n-hidden-layers0_n-instances3_feat_prob0.05_seed0_YYYYMMDD_HHMMSS_ms/`. Note down this path, as you'll need it for the APD step.

### 2. Running APD Decomposition Locally

Once the TMS target model is trained locally, you can run APD decomposition using `spd/experiments/tms/tms_decomposition.py`. This script requires a YAML configuration file.

For local runs, use `spd/experiments/tms/local_tms_topk_config.yaml`.

**Crucially, you must update this YAML file before running APD:**
1.  Open `spd/experiments/tms/local_tms_topk_config.yaml`.
2.  Locate the `task_config.pretrained_model_path` field.
3.  Change its value from the placeholder (`"./out/tms_model_placeholder/tms.pth"`) to the actual file path of the `tms.pth` checkpoint you saved in the previous training step. For example:
    ```yaml
    task_config:
      task_name: tms
      # ... other parameters ...
      pretrained_model_path: "./out/tms_n-features40_n-hidden10_n-hidden-layers0_n-instances3_feat_prob0.05_seed0_20231027_123456_789/tms.pth"
      # ... other parameters ...
    ```
    (Remember to use the correct relative path from the repository root, or an absolute path).

After updating the config file, run the APD decomposition:
```bash
python spd/experiments/tms/tms_decomposition.py spd/experiments/tms/local_tms_topk_config.yaml
```

APD outputs, including model checkpoints (e.g., `spd_model_*.pth`) and any generated plots, will be saved locally into a new timestamped subdirectory within `spd/experiments/tms/out/`.

### 3. Managing Hyperparameters

*   **TMS Model Training**:
    *   When running `spd/experiments/tms/train_tms.py` directly, default hyperparameters are set within the `if __name__ == "__main__":` block of the script. You can modify the `TMSTrainConfig` object instantiation there for different local runs.
    *   If you import and use `TMSTrainConfig` or the training functions as a library in your own scripts, you can define the configuration programmatically.

*   **APD Decomposition**:
    *   Hyperparameters for the APD process are defined in the YAML configuration file (e.g., `spd/experiments/tms/local_tms_topk_config.yaml`). You can edit this file to change APD settings like `topk`, `param_match_coeff`, `steps`, `lr`, etc.

## Development

Suggested extensions and settings for VSCode/Cursor are provided in `.vscode/`. To use the suggested
settings, copy `.vscode/settings-example.json` to `.vscode/settings.json`.

There are various `make` commands that may be helpful

```bash
make check  # Run pre-commit on all files (i.e. pyright, ruff linter, and ruff formatter)
make type  # Run pyright on all files
make format  # Run ruff linter and formatter on all files
make test  # Run tests that aren't marked `slow`
make test-all  # Run all tests
```