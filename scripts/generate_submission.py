"""Generate competition-compliant submission.csv for Kaggle MAGFiLO filament segmentation.

Submission format:
  filament_id,segmentation_rle
  <image_stem>_<instance_index>,<COCO_RLE_string>

Each row represents one predicted filament instance with a 2048x2048 mask encoded
using pycocotools.mask.encode (LEB128 COCO string).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import create_model
from src.training.postprocessing import postprocess_probability


def encode_mask_to_rle(binary_mask: np.ndarray) -> str:
    """Encode a binary (H, W) uint8 mask into a COCO RLE string."""
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    encoded = mask_util.encode(fortran_mask)
    return encoded["counts"].decode("ascii")


def decode_rle_to_mask(rle_string: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    """Decode a COCO RLE string back to a binary (H, W) uint8 mask."""
    rle_dict = {"size": [height, width], "counts": rle_string.encode("ascii")}
    return mask_util.decode(rle_dict)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, float, dict[str, Any]]:
    """Load model architecture, weights, and recommended threshold from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    model_name = model_config.get("name")
    if not model_name:
        raise ValueError(f"Checkpoint at {checkpoint_path} missing model name in config")

    options = {k: v for k, v in model_config.items() if k != "name"}
    model = create_model(model_name, **options)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    threshold = float(checkpoint.get("threshold", 0.5))
    return model, threshold, config


def predict_probability_map(
    model: torch.nn.Module,
    image_path: str | Path,
    input_resolution: int = 512,
    target_size: tuple[int, int] = (2048, 2048),
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Run model on resized test image and return bilinearly upsampled probability map."""
    with Image.open(image_path) as raw_img:
        grayscale = raw_img.convert("L")

    # Resize to model training resolution (e.g. 512x512)
    resized = grayscale.resize((input_resolution, input_resolution), Image.Resampling.BILINEAR)
    image_tensor = torch.from_numpy(
        np.asarray(resized, dtype=np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, dict):
            seg_logits = output["segmentation"]
        else:
            seg_logits = output
        prob_512 = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()

    # Bilinear upsample continuous probability map to target resolution (2048x2048)
    prob_img = Image.fromarray(prob_512)
    prob_target = np.asarray(prob_img.resize(target_size, Image.Resampling.BILINEAR))
    return prob_target


def generate_submission(
    checkpoints: Sequence[str | Path],
    test_dir: str | Path,
    output_csv: str | Path,
    threshold: float | None = None,
    input_resolution: int = 512,
    target_size: tuple[int, int] = (2048, 2048),
    min_instance_size: int = 32,
    postprocessing: str = "connected_components",
    device: str = "cpu",
) -> Path:
    """Predict filament instances for all images in test_dir and write submission.csv."""
    torch_device = torch.device(device)
    test_path = Path(test_dir)
    if not test_path.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    image_paths = sorted(
        p for p in test_path.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff"}
    )
    if not image_paths:
        raise ValueError(f"No valid image files found in {test_path}")

    # Load model(s)
    models: list[torch.nn.Module] = []
    thresholds: list[float] = []
    print(f"Loading {len(checkpoints)} checkpoint(s)...")
    for cp_path in checkpoints:
        model, cp_thresh, _ = load_model_from_checkpoint(cp_path, torch_device)
        models.append(model)
        thresholds.append(cp_thresh)
        print(f"  Loaded {cp_path} (default threshold = {cp_thresh:.4f})")

    chosen_threshold = threshold if threshold is not None else float(np.mean(thresholds))
    print(f"\nInference Settings:")
    print(f"  Test Images:       {len(image_paths)}")
    print(f"  Input Resolution:  {input_resolution}x{input_resolution}")
    print(f"  Target Mask Size:  {target_size[0]}x{target_size[1]}")
    print(f"  Threshold:         {chosen_threshold:.4f}")
    print(f"  Min Instance Size: {min_instance_size} px (at target resolution)")
    print(f"  Device:            {device}")
    print(f"  Output CSV:        {output_csv}\n")

    out_file = Path(output_csv)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []
    total_filaments = 0

    for img_path in tqdm(image_paths, desc="Predicting test images"):
        stem = img_path.stem

        # Average probability map across ensemble
        probs = [
            predict_probability_map(
                m, img_path,
                input_resolution=input_resolution,
                target_size=target_size,
                device=torch_device,
            )
            for m in models
        ]
        avg_prob = np.mean(probs, axis=0)

        # Connected component instance extraction
        instance_map = postprocess_probability(
            avg_prob,
            threshold=chosen_threshold,
            method=postprocessing,
            min_instance_size=min_instance_size,
        )

        unique_instances = [int(x) for x in np.unique(instance_map) if x > 0]
        for instance_idx, inst_id in enumerate(unique_instances, start=1):
            filament_id = f"{stem}_{instance_idx}"
            binary_mask = (instance_map == inst_id)
            rle_string = encode_mask_to_rle(binary_mask)
            rows.append((filament_id, rle_string))
            total_filaments += 1

    # Write submission.csv
    with out_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)

    print(f"\n✅ Submission successfully written to {out_file}")
    print(f"  Total test images:      {len(image_paths)}")
    print(f"  Total filaments found:  {total_filaments}")
    print(f"  Average per image:      {total_filaments / len(image_paths):.2f}")
    print(f"  File size:              {out_file.stat().st_size / (1024 * 1024):.2f} MB")

    # Sanity check: decode first prediction if any
    if rows:
        sample_id, sample_rle = rows[0]
        decoded = decode_rle_to_mask(sample_rle, *target_size)
        print(f"  Verification check: '{sample_id}' successfully decoded back to "
              f"mask of shape {decoded.shape} with {np.sum(decoded)} foreground pixels.")

    return out_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate competition submission.csv with COCO RLE encoded filament masks."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "runs/boundary_aware_unet/boundary-aware-unet/corrected-data-live-seed45/best_model.pt"
        ],
        help="One or more model checkpoint paths. If multiple are given, probability predictions are ensembled.",
    )
    parser.add_argument(
        "--test-dir",
        default="data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images",
        help="Path to folder containing raw test images (.jpeg / .png).",
    )
    parser.add_argument(
        "--output-csv",
        default="submission.csv",
        help="Output CSV path (default: submission.csv).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Detection probability threshold (default: uses optimal validation threshold from checkpoint).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Model input resolution (default: 512).",
    )
    parser.add_argument(
        "--min-instance-size",
        type=int,
        default=32,
        help="Minimum instance area in pixels at 2048x2048 (default: 32).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device: 'cpu' or 'cuda' (default: cpu).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_submission(
        checkpoints=args.checkpoints,
        test_dir=args.test_dir,
        output_csv=args.output_csv,
        threshold=args.threshold,
        input_resolution=args.resolution,
        min_instance_size=args.min_instance_size,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
