# Weights & Biases (wandb) Setup

## Installation

```bash
pip install wandb
```

## First-time login

```bash
wandb login
```

Paste your API key from https://wandb.ai/authorize when prompted.

## Basic usage

Just run training as normal — wandb starts automatically:

```bash
python -m src.train --config configs/default.yaml
```

Metrics are logged to the `bridge-anchors` project by default.

## Offline mode

For environments without internet access:

```bash
WANDB_MODE=offline python -m src.train --config configs/default.yaml
```

Logs are saved locally and can be synced later with `wandb sync`.

## Disabling wandb

For quick tests where you don't need logging:

```bash
WANDB_MODE=disabled python -m src.train --config configs/default.yaml
```

Training proceeds normally with no logging overhead. This is also the fallback behavior if wandb is not installed.

## Viewing results

1. Go to https://wandb.ai → select project "bridge-anchors"
2. The runs table shows all experiments with key metrics (best_mean_recall, params, etc.)
3. Click a run to see training curves (loss, mR, lr over epochs)

## Comparing runs

1. Select multiple runs in the runs table (checkboxes)
2. Click "Compare" to see overlaid training curves
3. Use the parallel coordinates chart for hyperparameter analysis

## Environment variables reference

| Variable | Description | Example |
|----------|-------------|---------|
| `WANDB_PROJECT` | Override project name | `WANDB_PROJECT=my-project` |
| `WANDB_NAME` | Override run name | `WANDB_NAME=my-run` |
| `WANDB_MODE` | `online` / `offline` / `disabled` | `WANDB_MODE=disabled` |
| `WANDB_ENTITY` | Team/org name | `WANDB_ENTITY=my-team` |

## Config file

The project name can also be set in `configs/default.yaml`:

```yaml
logging:
  wandb_project: bridge-anchors
```

The environment variable `WANDB_PROJECT` takes precedence over the config file.

## Graceful degradation

If wandb is not installed, training runs normally with a warning message. No metrics are logged, but all other functionality (checkpoints, console output) works unchanged.
