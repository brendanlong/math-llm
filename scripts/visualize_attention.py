"""Visualize attention patterns from trained arithmetic models using BertViz.

This script loads a trained model and generates interactive HTML visualizations
of attention patterns for a given input expression.

Usage:
    python scripts/visualize_attention.py --checkpoint checkpoints/standard-small-pope
    python scripts/visualize_attention.py --checkpoint checkpoints/standard-small-pope --input "99+21="
    python scripts/visualize_attention.py --checkpoint checkpoints/standard-small-pope --output-dir viz/
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from src.config import load_config
from src.model import BaseModel, FeedbackTransformerModel, create_model_from_config
from src.tokenizer import tokenizer

BEGIN_TOKEN = "<begin>"


def ensure_begin_token(text: str) -> str:
    """Prepend <begin> token to text if not already present.

    Args:
        text: Input text

    Returns:
        Text with <begin> token prepended if not already present
    """
    if not text.startswith(BEGIN_TOKEN):
        return BEGIN_TOKEN + text
    return text


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize attention patterns from trained models"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., checkpoints/standard-small-pope)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="12+34=",
        help="Input expression to visualize (default: '12+34=')",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate output and visualize the full sequence",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate when --generate is set (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save HTML visualizations (default: prints to stdout)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="both",
        choices=["head", "model", "both"],
        help="Which view to generate (default: both)",
    )
    return parser.parse_args()


def load_model(checkpoint_dir: str, device: str = "auto") -> tuple[BaseModel, str]:
    """Load a model from a checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        device: Device to load model on

    Returns:
        Tuple of (model, device_str)
    """
    checkpoint_path = Path(checkpoint_dir)
    config_path = checkpoint_path / "model_config.yaml"
    weights_path = checkpoint_path / "model.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(config_path)
    model = create_model_from_config(config)

    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_dir}")
    print(f"  Architecture: {config.architecture}")
    print(f"  Positional encoding: {config.positional_encoding}")
    print(f"  Softmax variant: {config.softmax_variant}")
    print(f"  Layers: {config.n_layers}, Heads: {config.n_heads}")
    print(f"  Device: {device}")

    return model, device


def get_attention_weights(
    model: BaseModel, input_text: str, device: str
) -> tuple[tuple[torch.Tensor, ...], list[str]]:
    """Get attention weights for an input expression.

    Args:
        model: Loaded model
        input_text: Input expression
        device: Device model is on

    Returns:
        Tuple of (attention_weights, tokens)
    """
    input_text = ensure_begin_token(input_text)
    token_ids = tokenizer.encode(input_text)
    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    tokens_result = tokenizer.convert_ids_to_tokens(token_ids)
    tokens: list[str] = (
        tokens_result if isinstance(tokens_result, list) else [tokens_result]
    )

    with torch.no_grad():
        outputs = model(inputs, labels=inputs, output_attentions=True)

    if isinstance(outputs, dict) and "attentions" in outputs:
        attention = outputs["attentions"]
        assert isinstance(attention, tuple)
        return attention, tokens
    else:
        raise ValueError(
            "Model did not return attention weights. "
            "Ensure output_attentions=True is supported."
        )


def generate_and_get_attention(
    model: BaseModel, input_text: str, device: str, max_new_tokens: int = 50
) -> tuple[tuple[torch.Tensor, ...], list[str], str]:
    """Generate output and get attention weights for the full sequence.

    Args:
        model: Loaded model
        input_text: Input expression
        device: Device model is on
        max_new_tokens: Maximum tokens to generate

    Returns:
        Tuple of (attention_weights, tokens, generated_text)
    """
    input_text = ensure_begin_token(input_text)
    input_token_ids = tokenizer.encode(input_text)
    inputs = torch.tensor([input_token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            inputs, max_new_tokens=max_new_tokens, temperature=0.1
        )

    generated_token_ids = generated[0].tolist()
    generated_text = tokenizer.decode(generated_token_ids)
    print(f"Generated: {generated_text}")

    with torch.no_grad():
        outputs = model(generated, labels=generated, output_attentions=True)

    if isinstance(outputs, dict) and "attentions" in outputs:
        attention = outputs["attentions"]
        assert isinstance(attention, tuple)
        tokens_result = tokenizer.convert_ids_to_tokens(generated_token_ids)
        tokens: list[str] = (
            tokens_result if isinstance(tokens_result, list) else [tokens_result]
        )
        return attention, tokens, generated_text
    else:
        raise ValueError(
            "Model did not return attention weights. "
            "Ensure output_attentions=True is supported."
        )


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Import bertviz here to give helpful error if not installed
    try:
        from bertviz import head_view, model_view
    except ImportError:
        print("Error: bertviz is not installed.")
        print("Install with: pip install bertviz")
        print("Or install dev dependencies: pip install -e '.[dev]'")
        return

    model, device = load_model(args.checkpoint, args.device)

    # Check if model supports attention output
    if isinstance(model, FeedbackTransformerModel):
        print(
            "\nWarning: FeedbackTransformerModel does not support output_attentions. "
            "Attention visualization is not available for this architecture."
        )
        return

    print(f"\nInput: {args.input}")

    if args.generate:
        attention, tokens, _generated_text = generate_and_get_attention(
            model, args.input, device, args.max_new_tokens
        )
    else:
        attention, tokens = get_attention_weights(model, args.input, device)

    print(f"Tokens: {tokens}")
    print(f"Number of layers: {len(attention)}")
    print(f"Attention shape per layer: {attention[0].shape}")

    # Generate visualizations
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.view in ("head", "both"):
            html_head = head_view(attention, tokens, html_action="return")
            if html_head is not None and html_head.data is not None:
                head_path = output_dir / "attention_head_view.html"
                with open(head_path, "w") as f:
                    f.write(str(html_head.data))
                print(f"\nSaved head view to: {head_path}")

        if args.view in ("model", "both"):
            html_model = model_view(attention, tokens, html_action="return")
            if html_model is not None and html_model.data is not None:
                model_path = output_dir / "attention_model_view.html"
                with open(model_path, "w") as f:
                    f.write(str(html_model.data))
                print(f"Saved model view to: {model_path}")

        print("\nOpen the HTML files in a browser to view the visualizations.")
    else:
        print(
            "\nNo --output-dir specified. Use --output-dir to save HTML visualizations."
        )
        print(
            "Example: python scripts/visualize_attention.py --checkpoint checkpoints/standard-small-pope --output-dir viz/"
        )


if __name__ == "__main__":
    main()
