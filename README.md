# Math LLM

A minimal transformer model for learning basic arithmetic operations, starting with single-digit addition.

## Quick Start

```bash
# Install dependencies
uv sync

# Generate training data
python scripts/generate_data.py

# Train the model
python scripts/train.py

# Test the model
python scripts/evaluate.py
```

## Project Structure

```
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
```bash
python scripts/train.py --model-size small --batch-size 32 --learning-rate 1e-4
```

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