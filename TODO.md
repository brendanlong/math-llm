# Implementation Roadmap

## Phase 1: Project Setup

- [x] Create documentation (README, DESIGN, CONTRIBUTING, CLAUDE)
- [x] Initialize Python project with `uv`
- [x] Setup dependencies (torch, transformers, datasets, wandb)
- [x] Configure development tools (ruff, pyright, pre-commit)
- [x] Create directory structure

## Phase 2: Core Components

- [x] **Custom tokenizer** (`src/tokenizer.py`)
  - Character-level tokenizer for digits, +, =, <end>
  - Vocabulary size: 13 tokens
  - Encode/decode methods

- [x] **Data generation** (`scripts/generate_data.py`)
  - Generate arithmetic expressions: "a+b=c<end>"
  - Start with single-digit addition
  - Save to disk for reproducibility
  - Train/val/test splits (80/10/10)

- [x] **Model architecture** (`src/model.py`)
  - Small transformer decoder (~1M-10M parameters)
  - 4-8 layers, 256-512 hidden size
  - Compatible with HuggingFace Transformers

## Phase 3: Training Infrastructure

- [x] **Data loading** (`src/data.py`)
  - PyTorch Dataset and DataLoader
  - Proper tokenization and padding
  - Efficient batching

- [x] **Training script** (`scripts/train.py`)
  - HuggingFace Trainer integration
  - W&B logging setup
  - Checkpoint saving every 1000 steps
  - Mixed precision training

- [x] **Evaluation** (`scripts/evaluate.py`)
  - Exact match accuracy
  - Token-level accuracy
  - Inference on test set

## Phase 4: Testing & Validation

- [ ] **Unit tests** (`tests/`)
  - Test tokenizer encode/decode
  - Test data generation
  - Test model forward pass

- [ ] **End-to-end validation**
  - Generate small dataset (1k examples)
  - Train tiny model (100k parameters)
  - Verify training loop works
  - Check W&B logging

## Phase 5: Scaling & Optimization

- [ ] **Scale up data**
  - 100k+ training examples
  - Multi-digit addition support
  - Larger number ranges

- [ ] **Model optimization**
  - Tune hyperparameters
  - Optimize for RTX3060 memory
  - Performance profiling

- [ ] **Advanced evaluation**
  - Accuracy by problem difficulty
  - Error analysis
  - Generalization to unseen ranges

## Phase 6: Extensions (Future)

- [ ] Support subtraction, multiplication, division
- [ ] Multi-step arithmetic problems
- [ ] Chain-of-thought reasoning
- [ ] Interactive inference demo

## Current Priority

Phase 4: Next priority is unit tests (`tests/`) for core components.
