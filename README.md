# Math LLM

A minimal transformer model for learning basic arithmetic operations with chain-of-thought reasoning, starting with single-digit addition and scaling to multi-digit problems.

Currently the model learns to mimic human-understable reasoning, but the goal is to make it learn to reason without instructions, and without RL, to try to learn something from how the model "naturally" learns to reason and if we can understand it.

The current CoT looks like this:

```text
➤ Enter expression: 99+21=
💭 Generating completion for: 99+21=
✨ Model output: 99+21=<think>9+1=10 9+2+1=11 0+1=1 1+0=1</think>110<end>
🤔 Chain of thought:
  99+21=
  🧮 Think:
    9+1=10 9+2+1=11 0+1=1 1+0=1
  110<end>
✅ Answer: 99+21=110
```

## Quick Start

```bash
# Setup development environment
./setup.sh

# Install package in development mode (required for imports to work)
pip install -e .

# Generate training data
python scripts/generate_data.py

# Train the model
python scripts/train.py --config config/standard-small.yaml

# Test the model (config auto-detected from checkpoint directory)
python scripts/evaluate.py --checkpoint checkpoints/model.safetensors

# Try the model interactively
python scripts/interactive.py --checkpoint checkpoints/model.safetensors
```

## Project Structure

```text
math-llm/
├── config/                # Model configuration YAML files
│   ├── standard-small.yaml
│   ├── standard-medium.yaml
│   ├── universal-small.yaml
│   └── ...
├── scripts/
│   ├── generate_data.py    # Data generation
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── interactive.py     # Interactive inference
├── src/
│   ├── config.py          # Configuration loading
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

# Generate reversed-format data (digits reversed, no CoT)
python scripts/generate_data.py --num-examples 100000 --max-digits 3 --reversed-format
```

#### Data Generation Arguments

- `--num-examples`: Number of examples to generate - default: `100000`
- `--max-digits`: Maximum digits per operand - default: `3`
- `--max-operands`: Maximum operands per expression - default: `3`
- `--seed`: Random seed - default: `42`
- `--output-dir`: Output directory - default: `data`
- `--no-include-cot`: Disable chain-of-thought reasoning
- `--fixed-length-cot`: Pad CoT to fixed length with `<noop>` tokens
- `--reversed-format`: Reverse digit order in operands and result (e.g., `8+21=02` for `8+12=20`). Automatically disables CoT.

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
- `--max-length`: Maximum sequence length - default: `128`

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
- `model_config.yaml`: Model configuration (copied from input config)
- `training_config.json`: Complete training configuration
- `test_results.json`: Final evaluation metrics
- `logs/training.log`: Detailed training logs with timestamps

### Evaluation

The evaluation script computes exact match accuracy and token-level accuracy on test data.
The model configuration is automatically detected from the checkpoint directory (`model_config.yaml`).

```bash
# Basic evaluation on test set (config auto-detected)
python scripts/evaluate.py --checkpoint checkpoints/model.safetensors

# Evaluate specific checkpoint
python scripts/evaluate.py --checkpoint checkpoints/checkpoint-1000/model.safetensors

# Evaluate specific data file
python scripts/evaluate.py \
  --checkpoint checkpoints/model.safetensors \
  --data-path data/custom_test.jsonl

# Override config for older checkpoints without model_config.yaml
python scripts/evaluate.py \
  --checkpoint old_checkpoints/model.bin \
  --config config/standard-small.yaml \
  --batch-size 128

# Save results to file
python scripts/evaluate.py \
  --checkpoint checkpoints/model.safetensors \
  --output-file results.json
```

#### Evaluation Arguments

**Required:**

- `--checkpoint`: Path to model checkpoint file

**Model Configuration:**

- `--config`: Path to model configuration YAML file (auto-detected from checkpoint dir if not specified)

**Data Options:**

- `--data-path`: Specific evaluation data file (overrides `--data-dir`)
- `--data-dir`: Directory containing test.jsonl - default: `data`

**Evaluation Settings:**

- `--batch-size`: Evaluation batch size - default: `64`
- `--max-length`: Maximum sequence length - default: `128`
- `--output-file`: Save results to JSON file

**System:**

- `--device`: Device to use (`cuda`, `cpu`, `auto`) - default: `auto`

#### Metrics

- **Exact Match Accuracy**: Percentage of completely correct arithmetic answers
- **Token-Level Accuracy**: Per-token prediction accuracy during teacher forcing
- **Number of Examples**: Total examples evaluated

### Interactive Inference

Test your trained model interactively by providing arithmetic expressions and seeing the model's completions.
The model configuration is automatically detected from the checkpoint directory (`model_config.yaml`).

```bash
# Basic interactive session (config auto-detected)
python scripts/interactive.py --checkpoint checkpoints/model.safetensors

# Override config for older checkpoints
python scripts/interactive.py \
  --checkpoint checkpoints/checkpoint-1000/model.safetensors \
  --config config/standard-medium.yaml \
  --max-new-tokens 15
```

#### Interactive Arguments

**Required:**

- `--checkpoint`: Path to model checkpoint file

**Model Configuration:**

- `--config`: Path to model configuration YAML file (auto-detected from checkpoint dir if not specified)

**Generation Settings:**

- `--max-new-tokens`: Maximum tokens to generate - default: `512`

**System:**

- `--device`: Device to use (`cuda`, `cpu`, `auto`) - default: `auto`

#### Usage Examples

The interactive script accepts various input formats:

```text
➤ Enter expression: 3+5=
✨ Model completion: '3+5=' → '3+5=8<end>'

➤ Enter expression: 12+34=
✨ Model completion: '12+34=' → '12+34=46<end>'

➤ Enter expression: 7+
✨ Model completion: '7+' → '7+3=10<end>'
```

**Input Validation:**

- Only accepts digits (0-9), plus sign (+), and equals sign (=)
- Provides helpful error messages for invalid input
- Type 'quit' or 'exit' to end the session, or use Ctrl+C/Ctrl+D

## Model Details

- **Architecture**: Small transformer decoder (1M-10M parameters)
- **Vocabulary**: 17 tokens (digits 0-9, +, =, <end>, <think>, </think>, <noop>, <begin>)
- **Context Length**: 128 tokens (sufficient for chain-of-thought reasoning)
- **Task**: Next-token prediction on arithmetic expressions with reasoning
- **Formats**:
  - Simple: `"3+5=8<end>"`
  - With reasoning: `"658+189=<think>8+9=17 5+8+1=14 6+1+1=8</think>847<end>"`
  - Reversed: `"8+21=02<end>"` (digits reversed for easier left-to-right processing)

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
