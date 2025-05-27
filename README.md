# Math LLM

A minimal transformer model for learning basic arithmetic operations with chain-of-thought reasoning, starting with single-digit addition and scaling to multi-digit problems.

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

# Try the model interactively
python scripts/interactive.py --checkpoint checkpoints/model.safetensors
```

## Project Structure

```text
math-llm/
├── scripts/
│   ├── generate_data.py    # Data generation
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── interactive.py     # Interactive inference
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

# Enable CoT-agnostic training (masks chain-of-thought content in loss)
python scripts/train.py --cot-agnostic
```

#### Training Arguments

**Model Configuration:**

- `--model-size`: Model size (`small`, `medium`, `large`) - default: `small`
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
- `--cot-agnostic`: Enable CoT-agnostic training mode (masks chain-of-thought content in loss)

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

The evaluation script computes exact match accuracy and token-level accuracy on test data:

```bash
# Basic evaluation on test set (final model)
python scripts/evaluate.py --checkpoint checkpoints/model.safetensors

# Evaluate specific checkpoint
python scripts/evaluate.py --checkpoint checkpoints/checkpoint-1000/model.safetensors

# Evaluate specific data file
python scripts/evaluate.py \
  --checkpoint checkpoints/model.safetensors \
  --data-path data/custom_test.json

# Custom model size and batch size
python scripts/evaluate.py \
  --checkpoint checkpoints/model.safetensors \
  --model-size medium \
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

- `--model-size`: Model size (`small`, `medium`, `large`) - default: `small`

**Data Options:**

- `--data-path`: Specific evaluation data file (overrides `--data-dir`)
- `--data-dir`: Directory containing test.json - default: `data`

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

Test your trained model interactively by providing arithmetic expressions and seeing the model's completions:

```bash
# Basic interactive session
python scripts/interactive.py --checkpoint checkpoints/model.safetensors

# Use specific model size and generation settings
python scripts/interactive.py \
  --checkpoint checkpoints/checkpoint-1000/model.safetensors \
  --model-size medium \
  --temperature 0.2 \
  --max-new-tokens 15
```

#### Interactive Arguments

**Required:**

- `--checkpoint`: Path to model checkpoint file

**Model Configuration:**

- `--model-size`: Model size (`small`, `medium`, `large`) - default: `small`

**Generation Settings:**

- `--max-new-tokens`: Maximum tokens to generate - default: `20`
- `--temperature`: Sampling temperature (lower = more deterministic) - default: `0.1`

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
- **Vocabulary**: 18 tokens (digits 0-9, +, =, <end>, <think_digit>, </think_digit>, <think_multi>, </think_multi>, \n)
- **Context Length**: 128 tokens (sufficient for chain-of-thought reasoning)
- **Task**: Next-token prediction on arithmetic expressions with reasoning
- **Formats**:
  - Simple: `"3+5=8<end>"`
  - With reasoning: `"658+189=<think_digit>\n8+9=17\n5+8+1=14\n6+1+1=8</think_digit>847<end>"`

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
