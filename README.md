# Math LLM

A minimal transformer model for learning basic arithmetic operations, starting with single-digit addition.

## Quick Start

```bash
# Setup development environment
./setup.sh

# Generate training data
python scripts/generate_data.py

# Train the model
python scripts/train.py

# Test the model
python scripts/evaluate.py
```

## Project Structure

```text
math-llm/
├── scripts/
│   ├── generate_data.py    # Data generation
│   ├── train.py           # Training script
│   └── evaluate.py        # Evaluation script
├── src/
│   ├── model.py           # Model architecture
│   ├── tokenizer.py       # Custom tokenizer
│   └── data.py            # Data loading utilities
├── data/                  # Generated datasets
├── checkpoints/           # Model checkpoints
└── logs/                  # Training logs
```

## Usage

### Data Generation

```bash
python scripts/generate_data.py --num-examples 100000 --max-digits 2
```

### Training

The training script supports comprehensive configuration with colored logging and W&B integration:

```bash
# Basic training with default settings
python scripts/train.py

# Custom configuration
python scripts/train.py \
  --model-size medium \
  --batch-size 64 \
  --learning-rate 2e-4 \
  --num-epochs 20 \
  --fp16 \
  --data-dir data \
  --output-dir checkpoints/experiment1

# Training without W&B logging
python scripts/train.py --no-wandb

# Resume from checkpoint
python scripts/train.py --output-dir checkpoints/experiment1
```

#### Training Arguments

**Model Configuration:**

- `--model-size`: Model size (`small`, `medium`, `large`) - default: `small`
- `--max-length`: Maximum sequence length - default: `32`

**Training Hyperparameters:**

- `--batch-size`: Training batch size - default: `32`
- `--eval-batch-size`: Evaluation batch size - default: `64`
- `--learning-rate`: Learning rate - default: `1e-4`
- `--weight-decay`: Weight decay - default: `0.01`
- `--num-epochs`: Number of training epochs - default: `10`
- `--warmup-steps`: Learning rate warmup steps - default: `500`

**Data and I/O:**

- `--data-dir`: Directory with train/val/test JSON files - default: `data`
- `--output-dir`: Checkpoint output directory - default: `checkpoints`
- `--num-workers`: Data loading workers - default: `4`

**Logging and Checkpointing:**

- `--save-steps`: Save checkpoint every N steps - default: `1000`
- `--eval-steps`: Evaluate every N steps - default: `1000`
- `--logging-steps`: Log metrics every N steps - default: `100`

**System Options:**

- `--fp16`: Enable mixed precision training
- `--no-wandb`: Disable Weights & Biases logging
- `--seed`: Random seed for reproducibility - default: `42`

#### Model Sizes

| Size | Parameters | Layers | Hidden Size | Heads | Feed-Forward |
|------|------------|--------|-------------|-------|--------------|
| Small | ~1M | 4 | 256 | 4 | 512 |
| Medium | ~5M | 6 | 512 | 8 | 1024 |
| Large | ~10M | 8 | 512 | 8 | 2048 |

#### Output Files

Training generates several output files in the checkpoint directory:

- `pytorch_model.bin`: Final trained model weights
- `training_config.json`: Complete training configuration
- `test_results.json`: Final evaluation metrics
- `logs/training.log`: Detailed training logs with timestamps

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

## Model Details

- **Architecture**: Small transformer decoder (1M-10M parameters)
- **Vocabulary**: 12 tokens (digits 0-9, +, =, <end>)
- **Task**: Next-token prediction on arithmetic expressions
- **Format**: `"operand1+operand2=result<end>"`

## Hardware Requirements

- **GPU**: CUDA-capable (tested on RTX3060 with 12GB VRAM)
- **Memory**: 8GB+ system RAM recommended
- **Storage**: ~1GB for datasets and checkpoints

## Monitoring

Training progress is logged to Weights & Biases. Key metrics:

- Exact match accuracy (complete answer correctness)
- Token-level accuracy
- Loss curves
- Learning rate schedules

## Results

Results will be documented here as training progresses.

See [DESIGN.md](DESIGN.md) for architectural details and design decisions.
