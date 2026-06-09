# Math LLM

[![CI](https://github.com/brendanlong/math-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/brendanlong/math-llm/actions/workflows/ci.yml)

A minimal transformer model for learning basic arithmetic operations with chain-of-thought reasoning, starting with single-digit addition and scaling to multi-digit problems.

Currently the model learns to mimic human-understandable reasoning, but the goal is to make it learn to reason without instructions, and without RL, to try to learn something from how the model "naturally" learns to reason and if we can understand it.

The current chain-of-thought shows recursive pairwise addition with digits reversed (least-significant first, which is easier for a left-to-right model):

```text
<begin>99+21=<think>99+12=021</think>120<end>
<begin>3+5+2=<think>3+5+2=8+2=01</think>10<end>
```

Inside `<think>` the operands and intermediate results are digit-reversed (`99+12=021` means 99+21=120); the final answer after `</think>` is in normal digit order.

## Features

- **Multiple architectures**: Standard transformer, Universal Transformer (weight sharing), Feedback Transformer, Mamba-style SSM (selective state space, no attention)
- **Positional encodings**: Learned, Sinusoidal, RoPE (Rotary), PoPE (Polar)
- **Attention variants**: Standard softmax, softmax1 (quiet attention with abstention)
- **Layer norm placement**: Pre-LN or post-LN (`layer_norm_type` in config)
- **Optimizers**: AdamW, ADOPT, Muon
- **Attention visualization**: Export attention weights for analysis with BertViz/Ecco
- **Mechanistic interpretability**: Logit lens, attention patterns, and causal tracing via `scripts/trace_model.py`

## Quick Start

```bash
# Setup development environment (installs dependencies with uv)
./setup.sh

# Generate training data
python scripts/generate_data.py

# Train the model
python scripts/train.py --config config/standard-small.yaml

# Test the model (config auto-detected from checkpoint directory)
python scripts/evaluate.py --checkpoint checkpoints/standard-small/

# Try the model interactively
python scripts/interactive.py --checkpoint checkpoints/standard-small/
```

All inference scripts accept `--checkpoint` as either a checkpoint file
(`model.safetensors`) or a directory containing one; the model config is
auto-detected from `model_config.yaml` next to the checkpoint.

## Project Structure

```text
math-llm/
├── config/                    # Model configuration YAML files
│   ├── standard-small.yaml    # Base config with learned positional encoding
│   ├── standard-small-{sinusoidal,rope,pope}.yaml
│   ├── standard-small-{softmax1,pope-softmax1}.yaml
│   ├── standard-small-rope-preln.yaml      # Pre-LN variant
│   ├── standard-small-rope-preln-2L8H.yaml # 2-layer / 8-head variant
│   ├── universal-*.yaml       # Universal Transformer (weight sharing)
│   ├── feedback-*.yaml        # Feedback Transformer
│   ├── ssm-*.yaml             # Mamba-style SSM
│   └── ...                    # xsmall/medium/large variants
├── scripts/
│   ├── generate_data.py       # Data generation
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── interactive.py         # Interactive inference
│   ├── visualize_attention.py # Attention visualization (BertViz)
│   ├── trace_model.py         # Mechanistic interpretability analyses
│   ├── benchmark.py           # Training throughput benchmark
│   ├── benchmark_dataloader.py# Data loading benchmark
│   └── compare_benchmarks.py  # Compare two benchmark runs
├── notebooks/
│   └── attention_visualization.ipynb  # Interactive attention analysis
├── src/
│   ├── config.py              # Configuration loading and checkpoint resolution
│   ├── model.py               # Transformer architectures
│   ├── ssm.py                 # Mamba-style SSM architecture
│   ├── tokenizer.py           # Custom tokenizer
│   ├── data.py                # Data loading utilities
│   ├── generation.py          # Data generation utilities
│   ├── training.py            # Trainer callbacks and metrics
│   ├── optimizers.py          # Optimizer factory (AdamW/ADOPT/Muon)
│   ├── activation_stats.py    # Activation statistics collection
│   ├── types.py               # Shared type definitions
│   └── utils.py               # Logging, device, and model loading helpers
├── tests/                     # pytest test suite
├── data/                      # Generated datasets (gitignored)
├── checkpoints/               # Model checkpoints (gitignored)
└── logs/                      # Training logs (gitignored)
```

## Usage

### Data Generation

```bash
python scripts/generate_data.py --num-examples 100000 --max-digits 2

# Generate reversed-format data (digits reversed, no CoT)
python scripts/generate_data.py --num-examples 100000 --max-digits 3 --reversed-format
```

Splits are leakage-free: all copies of a duplicated expression are assigned to
the same split, so val/test never contain expressions seen in training.
Generation output depends only on the seed and parameters, not on the number
of workers.

#### Data Generation Arguments

- `--num-examples`: Number of examples to generate - default: `10000`
- `--max-digits`: Maximum digits per operand - default: `2`
- `--max-operands`: Maximum operands per expression - default: `3`
- `--seed`: Random seed - default: `42`
- `--output-dir`: Output directory - default: `data`
- `--train-ratio`: Fraction of data for training - default: `0.8`
- `--val-ratio`: Fraction of data for validation - default: `0.1`
- `--num-workers`: Worker processes for generation - default: CPU count
- `--no-include-cot`: Disable chain-of-thought reasoning
- `--fixed-length-cot`: Pad CoT to fixed length with `<noop>` tokens
- `--reversed-format`: Reverse digit order in operands and result (e.g., `8+21=02` for `8+12=20`). Automatically disables CoT.
- `--zero-pad`: Zero-pad all numbers in each example to the same width

### Training

The training script supports comprehensive configuration with colored logging and W&B integration:

```bash
# Train with standard small model
python scripts/train.py --config config/standard-small.yaml

# Train with different model configuration
python scripts/train.py \
  --config config/standard-medium.yaml \
  --batch-size 64 \
  --learning-rate 2e-4 \
  --num-epochs 20 \
  --fp16 \
  --data-dir data \
  --output-dir checkpoints/experiment1

# Training without W&B logging
python scripts/train.py --config config/standard-small.yaml --no-wandb

# Resume from checkpoint
python scripts/train.py --config config/standard-small.yaml --output-dir checkpoints/experiment1 --resume
```

#### Training Arguments

**Model Configuration:**

- `--config`: Path to model configuration YAML file (required)
- `--max-length`: Maximum sequence length - default: longest example length from dataset metadata

**Training Hyperparameters:**

- `--batch-size`: Training batch size - default: `32`
- `--eval-batch-size`: Evaluation batch size - default: `64`
- `--learning-rate`: Learning rate - default: `1e-3`
- `--weight-decay`: Weight decay - default: `0.01`
- `--num-epochs`: Number of training epochs - default: `10`
- `--warmup-steps`: Learning rate warmup steps - default: `500`
- `--optimizer`: Optimizer (`adamw`, `adopt`, `muon`) - default: `adamw`

**Data and I/O:**

- `--data-dir`: Directory with train/val/test JSONL files - default: `data`
- `--output-dir`: Checkpoint output directory - default: `checkpoints/<config-name>/`
- `--num-workers`: Data loading workers - default: `4`

**Logging and Checkpointing:**

- `--save-steps`: Save checkpoint every N steps - default: `1000`
- `--eval-steps`: Evaluate every N steps - default: `1000`
- `--logging-steps`: Log metrics every N steps - default: `100`
- `--wandb-group`: W&B group name for organizing related runs
- `--track-activation-stats`: Track activation statistics (kurtosis, outliers) during training

**System Options:**

- `--fp16`: Enable mixed precision training
- `--no-wandb`: Disable Weights & Biases logging
- `--no-torch-compile`: Disable torch.compile
- `--profile`: Run a short profiling pass with torch.profiler instead of training
- `--resume`: Resume training from existing checkpoint in output directory
- `--seed`: Random seed for reproducibility - default: `42`

#### Model Sizes

| Size | Parameters | Layers | Hidden Size | Heads | Feed-Forward |
|------|------------|--------|-------------|-------|--------------|
| Small | ~1M | 4 | 256 | 4 | 512 |
| Medium | ~5M | 6 | 512 | 8 | 1024 |
| Large | ~10M | 8 | 512 | 8 | 2048 |

#### Output Files

Training generates several output files in the checkpoint directory:

- `model.safetensors`: Final trained model weights
- `model_config.yaml`: Model configuration (copied from input config)
- `training_config.json`: Complete training configuration
- `test_results.json`: Final evaluation metrics
- `logs/training.log`: Detailed training logs with timestamps

### Evaluation

The evaluation script computes exact match accuracy and token-level accuracy on test data.
The model configuration is automatically detected from the checkpoint directory (`model_config.yaml`).

```bash
# Basic evaluation on test set (config auto-detected)
python scripts/evaluate.py --checkpoint checkpoints/standard-small/

# Evaluate specific checkpoint file
python scripts/evaluate.py --checkpoint checkpoints/standard-small/checkpoint-1000/model.safetensors

# Evaluate specific data file
python scripts/evaluate.py \
  --checkpoint checkpoints/standard-small/ \
  --data-path data/custom_test.jsonl

# Override config for older checkpoints without model_config.yaml
python scripts/evaluate.py \
  --checkpoint old_checkpoints/model.bin \
  --config config/standard-small.yaml \
  --batch-size 128

# Save results to file
python scripts/evaluate.py \
  --checkpoint checkpoints/standard-small/ \
  --output-file results.json
```

#### Evaluation Arguments

**Required:**

- `--checkpoint`: Path to model checkpoint file or directory containing one

**Model Configuration:**

- `--config`: Path to model configuration YAML file (auto-detected from checkpoint dir if not specified)

**Data Options:**

- `--data-path`: Specific evaluation data file (overrides `--data-dir`)
- `--data-dir`: Directory containing test.jsonl - default: `data`

**Evaluation Settings:**

- `--batch-size`: Evaluation batch size - default: `64`
- `--max-length`: Maximum sequence length - default: `128`
- `--output-file`: Save results to JSON file
- `--activation-stats`: Compute and save activation statistics
- `--activation-stats-batches`: Max batches for activation stats - default: all

**System:**

- `--device`: Device to use (`cuda`, `cpu`, `auto`) - default: `auto`

#### Metrics

- **Exact Match Accuracy**: Percentage of completely correct arithmetic answers (greedy decoding, `<think>` sections excluded from comparison)
- **Token-Level Accuracy**: Per-token prediction accuracy during teacher forcing
- **Number of Examples**: Total examples evaluated

### Interactive Inference

Test your trained model interactively by providing arithmetic expressions and seeing the model's completions.
The model configuration is automatically detected from the checkpoint directory (`model_config.yaml`).

```bash
# Basic interactive session (config auto-detected)
python scripts/interactive.py --checkpoint checkpoints/standard-small/

# Step-by-step mode showing top-5 next-token probabilities
python scripts/interactive.py --checkpoint checkpoints/standard-small/ --mode probability

# Override config for older checkpoints
python scripts/interactive.py \
  --checkpoint checkpoints/standard-small/checkpoint-1000/model.safetensors \
  --config config/standard-medium.yaml \
  --max-new-tokens 15
```

#### Interactive Arguments

**Required:**

- `--checkpoint`: Path to model checkpoint file or directory containing one

**Model Configuration:**

- `--config`: Path to model configuration YAML file (auto-detected from checkpoint dir if not specified)

**Generation Settings:**

- `--max-new-tokens`: Maximum tokens to generate - default: `512`
- `--mode`: `normal` (default) or `probability` (step-by-step with top-5 predictions)

**System:**

- `--device`: Device to use (`cuda`, `cpu`, `auto`) - default: `auto`

The session uses greedy decoding and displays per-token confidence; type 'quit' or 'exit' to end the session, or use Ctrl+C/Ctrl+D.

### Visualizing Attention with BertViz

Visualize attention patterns from trained models using [BertViz](https://github.com/jessevig/bertviz).

```bash
# Generate HTML visualizations
python scripts/visualize_attention.py --checkpoint checkpoints/standard-small-pope --output-dir viz/

# With custom input expression
python scripts/visualize_attention.py --checkpoint checkpoints/standard-small-pope --input "99+21=" --output-dir viz/

# Generate completion first, then visualize attention on full sequence
python scripts/visualize_attention.py --checkpoint checkpoints/standard-small-pope --input "99+21=" --generate --output-dir viz/
```

Opens the generated HTML files in a browser to see:

- **Head View**: Detailed attention patterns for individual heads in each layer
- **Model View**: Bird's-eye view of attention across all layers and heads

For interactive exploration, use the Jupyter notebook:

```bash
jupyter lab notebooks/attention_visualization.ipynb
```

### Mechanistic Interpretability Tracing

Run logit lens, attention pattern, and causal tracing analyses on a trained standard-architecture model:

```bash
python scripts/trace_model.py \
  --checkpoint checkpoints/standard-small-rope-preln/ \
  --prompt "<begin>10+9=" \
  --output-dir traces/
```

This produces matplotlib heatmaps and a text summary showing how predictions evolve through layers, what each attention head attends to, and which (layer, position) activations are causally important.

## Model Details

- **Architecture**: Small transformer decoder (1M-10M parameters), plus Universal/Feedback/SSM variants
- **Vocabulary**: 17 tokens (digits 0-9, `+`, `=`, `<end>`, `<think>`, `</think>`, `<noop>`, `<begin>`)
- **Context Length**: 128 tokens (sufficient for chain-of-thought reasoning)
- **Task**: Next-token prediction on arithmetic expressions with reasoning
- **Formats** (every example starts with `<begin>`):
  - Simple: `"<begin>3+5=8<end>"`
  - With reasoning: `"<begin>99+21=<think>99+12=021</think>120<end>"` (digits inside `<think>` are reversed)
  - Reversed: `"<begin>8+21=02<end>"` (digits reversed for easier left-to-right processing)

### Positional Encodings

| Type | Description | Config Value |
|------|-------------|--------------|
| **Learned** | Trainable position embeddings (default) | `learned` |
| **Sinusoidal** | Fixed sin/cos embeddings from "Attention is All You Need" | `sinusoidal` |
| **RoPE** | Rotary Position Embeddings - rotates Q/K based on position | `rope` |
| **PoPE** | Polar Position Embeddings - separates magnitude (content) from phase (position) | `pope` |

### Softmax Variants

| Variant | Description | Config Value |
|---------|-------------|--------------|
| **Standard** | Traditional softmax attention | `standard` |
| **Softmax1** | Adds +1 to denominator, allowing attention heads to "abstain" when they have no useful information | `softmax1` |

Softmax1 implements "Quiet Attention" from Evan Miller's "[Attention Is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html)".

### Attention Visualization

Models support returning attention weights for visualization with tools like BertViz or Ecco:

```python
from src.model import create_model_from_config
from src.config import load_config

config = load_config("config/standard-small.yaml")
model = create_model_from_config(config)

# Get attention weights
result = model(input_ids, labels=labels, output_attentions=True)
attentions = result["attentions"]  # tuple of (batch, heads, seq, seq) per layer

# Each tensor shows what each position attends to
for layer_idx, attn in enumerate(attentions):
    print(f"Layer {layer_idx}: {attn.shape}")
```

## Hardware Requirements

- **GPU**: CUDA-capable (tested on RTX 3060 with 8GB VRAM)
- **Memory**: 8GB+ system RAM recommended
- **Storage**: ~1GB for datasets and checkpoints

## Monitoring

Training progress is logged to Weights & Biases. Key metrics:

- Token-level accuracy
- Loss curves
- Learning rate schedules
- Optional activation statistics (with `--track-activation-stats`)

Exact match accuracy is computed by `scripts/evaluate.py` after training.

## Results

### Positional Encoding Comparison (standard-small, 5 epochs)

| Model | Positional Encoding | Softmax | Token Accuracy |
|-------|---------------------|---------|----------------|
| standard-small-pope | PoPE | standard | **99.93%** |
| standard-small-sinusoidal | sinusoidal | standard | 99.92% |
| standard-small-rope | RoPE | standard | 99.90% |
| standard-small-pope-softmax1 | PoPE | softmax1 | 99.85% |
| standard-small | learned | standard | 92.54% |
| standard-small-softmax1 | learned | softmax1 | 92.55% |

**Key findings:**

- All positional encodings (sinusoidal, RoPE, PoPE) dramatically outperform learned embeddings on this task
- PoPE, sinusoidal, and RoPE perform essentially identically (~99.9%)
- The softmax1 variant doesn't significantly impact accuracy on this task

Note: these results predate the train/test leakage fix in data splitting
(duplicated expressions could previously appear in both train and test), so
absolute numbers are likely optimistic; relative comparisons should still hold.

See [DESIGN.md](DESIGN.md) for architectural details and design decisions.
