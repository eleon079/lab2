import argparse
import os
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HF_HOME = os.getenv("HF_HOME", ".hf_cache")
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", f"{HF_HOME}/hub")
HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE", f"{HF_HOME}/datasets")
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data/processed"))
HF_DATASET_REPO = os.getenv("HF_DATASET_NAME", "blanchon/INRIA-Aerial-Image-Labeling")
HF_MAX_SAMPLES = int(os.getenv("HF_MAX_SAMPLES", "1000"))

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ.setdefault("XDG_CACHE_HOME", ".cache")

Path(HF_HOME).mkdir(parents=True, exist_ok=True)
Path(HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)
Path(HF_DATASETS_CACHE).mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)

from huggingface_hub import snapshot_download
import numpy as np
from PIL import Image


def find_dataset_root(snapshot_path: Path) -> Path:
    candidates = [
        snapshot_path / "data" / "train" / "images",
        snapshot_path / "train" / "images",
    ]
    for c in candidates:
        if c.exists():
            return c.parent.parent if c.parts[-3:] == ("train", "images") else c.parent
    raise FileNotFoundError(
        f"Could not find expected train/images folder inside snapshot: {snapshot_path}"
    )


def gather_pairs(dataset_root: Path):
    train_images_dir = dataset_root / "train" / "images"
    train_masks_dir = dataset_root / "train" / "gt"

    if not train_images_dir.exists():
        raise FileNotFoundError(f"Missing train images folder: {train_images_dir}")
    if not train_masks_dir.exists():
        raise FileNotFoundError(f"Missing train masks folder: {train_masks_dir}")

    image_paths = sorted(train_images_dir.glob("*.tif"))
    pairs = []

    for image_path in image_paths:
        mask_path = train_masks_dir / image_path.name
        if mask_path.exists():
            pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError("No matching image/mask pairs were found in train/images and train/gt.")

    return pairs


def copy_pairs(pairs, output_dir: Path, split_name: str):
    img_dir = output_dir / split_name / "images"
    mask_dir = output_dir / split_name / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying {split_name} split to {img_dir.parent} ...")

    for idx, (image_path, mask_path) in enumerate(pairs):
        out_name = f"{split_name}_{idx:05d}.png"

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Ensure mask is binary 0/255
        mask_arr = np.array(mask)
        mask_arr = (mask_arr > 0).astype(np.uint8) * 255
        mask = Image.fromarray(mask_arr, mode="L")

        image.save(img_dir / out_name)
        mask.save(mask_dir / out_name)

        if (idx + 1) % 25 == 0:
            print(f"  copied {idx + 1} {split_name} samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default=HF_DATASET_REPO)
    parser.add_argument("--output-dir", type=str, default=str(DATA_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=HF_MAX_SAMPLES)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    random.seed(args.seed)

    print(f"Downloading dataset snapshot from Hugging Face: {args.dataset_name}")
    print(f"HF cache: {HF_HUB_CACHE}")
    print(f"Output dir: {output_dir}")
    print(f"Max samples: {args.max_samples}")

    snapshot_path = Path(
        snapshot_download(
            repo_id=args.dataset_name,
            repo_type="dataset",
            cache_dir=HF_HUB_CACHE,
        )
    )

    print(f"Snapshot path: {snapshot_path}")

    dataset_root = snapshot_path
    if (snapshot_path / "data").exists():
        dataset_root = snapshot_path / "data"

    pairs = gather_pairs(dataset_root)
    random.shuffle(pairs)

    if args.max_samples > 0:
        pairs = pairs[: min(len(pairs), args.max_samples)]

    n = len(pairs)
    if n < 3:
        raise RuntimeError("Not enough samples to create train/val/test splits.")

    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    print(f"Final sizes -> train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

    copy_pairs(train_pairs, output_dir, "train")
    copy_pairs(val_pairs, output_dir, "val")
    copy_pairs(test_pairs, output_dir, "test")

    print("Dataset prepared successfully.")
    print(
        {
            "dataset": args.dataset_name,
            "snapshot_path": str(snapshot_path),
            "source": "train/images + train/gt from Hugging Face dataset snapshot",
            "train": len(train_pairs),
            "val": len(val_pairs),
            "test": len(test_pairs),
        }
    )


if __name__ == "__main__":
    main()