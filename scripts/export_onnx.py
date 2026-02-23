#!/usr/bin/env python3
"""Export a BookGPT model checkpoint to ONNX format.

The ONNX file is saved in the same directory as the model.pt file.

Usage:
    python scripts/export_onnx.py data/models/pretrained/calculus_made_easy/best/model.pt
    python scripts/export_onnx.py data/models/pretrained/calculus_made_easy/best/model.pt --opset 17
    python scripts/export_onnx.py data/models/pretrained/calculus_made_easy/best/model.pt --seq-len 128

The exported model takes input_ids (int64, shape [batch, seq_len]) and returns
logits (float32, shape [batch, seq_len, vocab_size]).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookgpt.model.gpt2 import GPT2


def export_to_onnx(
    model_pt_path: str,
    opset_version: int = 17,
    seq_len: int | None = None,
):
    """Export a model.pt checkpoint to ONNX format.

    Args:
        model_pt_path: Path to model.pt file. Expects config.json in the same directory.
        opset_version: ONNX opset version (default 17).
        seq_len: Sequence length for the dummy input. Defaults to model's context_length.
    """
    model_pt_path = Path(model_pt_path).resolve()

    if not model_pt_path.exists():
        print(f"Error: {model_pt_path} not found")
        sys.exit(1)

    if model_pt_path.name != "model.pt":
        print(f"Warning: expected file named 'model.pt', got '{model_pt_path.name}'")

    model_dir = model_pt_path.parent
    config_path = model_dir / "config.json"

    if not config_path.exists():
        print(f"Error: config.json not found in {model_dir}")
        print("Both model.pt and config.json must be in the same directory.")
        sys.exit(1)

    # Load model on CPU for export
    print(f"Loading model from {model_dir}...")
    model = GPT2.from_pretrained(model_dir, device="cpu")
    model.eval()

    # Determine sequence length
    if seq_len is None:
        seq_len = model.config.context_length
    seq_len = min(seq_len, model.config.context_length)

    # Create dummy input
    dummy_input = torch.randint(
        0, model.config.vocab_size, (1, seq_len), dtype=torch.long
    )

    # ONNX output path — same directory as model.pt
    onnx_path = model_dir / "model.onnx"

    print(f"Exporting to ONNX (opset={opset_version}, seq_len={seq_len})...")
    print(f"  Input shape:  [batch, {seq_len}] (int64)")
    print(f"  Output shape: [batch, {seq_len}, {model.config.vocab_size}] (float32)")

    # We need a wrapper that only returns logits (not the loss tuple)
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            logits, _ = self.model(input_ids)
            return logits

    wrapper = OnnxWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(onnx_path),
        opset_version=opset_version,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
    )

    # Verify the export
    onnx_size = onnx_path.stat().st_size / (1024 * 1024)
    print(f"\nONNX model saved: {onnx_path}")
    print(f"  Size: {onnx_size:.1f} MB")

    # Optional: verify with onnxruntime if available
    try:
        import onnxruntime as ort
        import numpy as np

        print("\nVerifying with ONNX Runtime...")
        session = ort.InferenceSession(str(onnx_path))

        # Run inference with dummy input
        ort_inputs = {"input_ids": dummy_input.numpy()}
        ort_outputs = session.run(None, ort_inputs)
        ort_logits = ort_outputs[0]

        # Compare with PyTorch output
        with torch.no_grad():
            pt_logits = wrapper(dummy_input).numpy()

        max_diff = np.abs(pt_logits - ort_logits).max()
        mean_diff = np.abs(pt_logits - ort_logits).mean()
        print(f"  Max  diff (PyTorch vs ONNX): {max_diff:.6f}")
        print(f"  Mean diff (PyTorch vs ONNX): {mean_diff:.6f}")

        if max_diff < 1e-4:
            print("  ✓ Verification passed — outputs match")
        else:
            print(f"  ⚠ Outputs differ by up to {max_diff:.4f} (may be floating point precision)")

    except ImportError:
        print("\nInstall onnxruntime to verify: pip install onnxruntime")
        print("Skipping verification.")

    # Also verify with onnx if available
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model structure is valid")
    except ImportError:
        pass
    except Exception as e:
        print(f"  ⚠ ONNX validation warning: {e}")

    print(f"\nDone! ONNX model at: {onnx_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export BookGPT model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model_pt",
        type=str,
        help="Path to model.pt file (config.json must be in the same directory)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length for export (default: model's context_length)",
    )
    args = parser.parse_args()

    export_to_onnx(
        model_pt_path=args.model_pt,
        opset_version=args.opset,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
