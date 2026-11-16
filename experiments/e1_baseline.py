"""E1 baseline (no generative model): attribute classifier trained on real images only.
It is scored on attribute combinations that were never seen during training, which is the reference the diffusion pipeline (E2) has to beat. 
Protocol: experiment E1 in docs/02_cadrage_metier_roi_architecture.pdf, section 8.
Run from the repository root:
    uv run python -m experiments.e1_baseline
"""

import csv
import struct
import time
import zipfile
from itertools import pairwise
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from configurations import FACTORS, N_CONFIGS, SHAPE, decode

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mpi3d_real.npz"
RESULTS = ROOT / "experiments" / "results.csv"
CHECKPOINT = ROOT / "checkpoints" / "e1_classifier.pt"
EXPERIMENT = "E1"
SEED = 0
POSITIONS = 40 * 40  # arm positions recorded for every configuration
IMAGES_PER_CONFIG = 100  # positions sampled per configuration (out of 1600)
VAL_FRACTION = 0.1
EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

# --- data ---

def open_images(path: Path) -> np.memmap:
    """Memory-map the `images` array inside the .npz, so the 12 GB never enter RAM."""
    with zipfile.ZipFile(path) as archive:
        entry = archive.getinfo("images.npy")
    if entry.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{path.name} is compressed and cannot be memory-mapped")
    with path.open("rb") as file:
        file.seek(entry.header_offset + 26)
        name_length, extra_length = struct.unpack("<HH", file.read(4))
        file.seek(entry.header_offset + 30 + name_length + extra_length)
        version = np.lib.format.read_magic(file)
        read_header = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, _, dtype = read_header(file)
        pixels_start = file.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=pixels_start)

def load_subset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Read IMAGES_PER_CONFIG random arm positions for each configuration.
    Returns the images (N, 64, 64, 3) and their factor indices (N, 5).
    """
    images = open_images(DATA)
    positions = rng.permuted(np.tile(np.arange(POSITIONS), (N_CONFIGS, 1)), axis=1)
    positions = np.sort(positions[:, :IMAGES_PER_CONFIG], axis=1)  # ascending: sequential reads
    index = (np.arange(N_CONFIGS)[:, None] * POSITIONS + positions).ravel()
    labels = np.repeat([decode(c) for c in range(N_CONFIGS)], IMAGES_PER_CONFIG, axis=0)
    return np.asarray(images[index]), labels

def held_out_mask(labels: np.ndarray) -> np.ndarray:
    """Test combinations: the diagonal of the color x shape grid (white cone, green cube, ...).
    Every color and every shape stays in training; only their pairing is new at test time.
    """
    return labels[:, 0] == labels[:, 1]

# --- model ---

class AttributeNet(nn.Module):
    """Shared convolutional trunk with one linear head per factor."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        channels = [3, width, 2 * width, 4 * width, 8 * width]
        stages = []
        for c_in, c_out in pairwise(channels):
            stages += [
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
        self.trunk = nn.Sequential(*stages, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.heads = nn.ModuleList(nn.Linear(channels[-1], n) for n in SHAPE)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self.trunk(x)
        return [head(features) for head in self.heads]

def as_input(images: torch.Tensor) -> torch.Tensor:
    """uint8 (N, H, W, 3) -> float (N, 3, H, W) in [0, 1]."""
    return images.permute(0, 3, 1, 2).float() / 255

# --- training and scoring ---

def train(model: AttributeNet, images: torch.Tensor, labels: torch.Tensor) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for batch in torch.randperm(len(images), device=images.device).split(BATCH_SIZE):
            logits = model(as_input(images[batch]))
            loss = sum(F.cross_entropy(out, labels[batch, k]) for k, out in enumerate(logits))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch)
        print(f"epoch {epoch:2d}/{EPOCHS}  loss {running / len(images):.4f}")

@torch.no_grad()
def evaluate(model: AttributeNet, images: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Accuracy per factor, plus `all`: every factor right on the same image."""
    model.eval()
    correct = torch.cat(
        [
            torch.stack([out.argmax(1) for out in model(as_input(images[batch]))], dim=1)
            == labels[batch]
            for batch in torch.arange(len(images), device=images.device).split(1024)
        ]
    )
    scores = dict(zip(FACTORS, correct.float().mean(0).tolist(), strict=True))
    scores["all"] = correct.all(1).float().mean().item()
    return scores

def save_results(rows: list[dict]) -> None:
    """Write this experiment's rows to the shared results file, replacing its previous run."""
    kept = []
    if RESULTS.exists():
        with RESULTS.open(newline="") as file:
            kept = [row for row in csv.DictReader(file) if row["experiment"] != EXPERIMENT]
    with RESULTS.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(kept + rows)

# --- experiment ---

def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    images, labels = load_subset(rng)
    print(f"{len(images)} images loaded in {time.perf_counter() - start:.0f} s")
    held_out = held_out_mask(labels)
    seen = rng.permutation(np.flatnonzero(~held_out))
    n_val = int(VAL_FRACTION * len(seen))
    splits = {"train": seen[n_val:], "seen": seen[:n_val], "held_out": np.flatnonzero(held_out)}
    splits = {name: torch.from_numpy(index).to(device) for name, index in splits.items()}
    images = torch.from_numpy(images).to(device)
    labels = torch.from_numpy(labels).to(device)
    model = AttributeNet().to(device)
    start = time.perf_counter()
    train(model, images[splits["train"]], labels[splits["train"]])
    print(f"trained on {len(splits['train'])} images in {time.perf_counter() - start:.0f} s")
    rows = []
    for split in ("seen", "held_out"):
        index = splits[split]
        for factor, accuracy in evaluate(model, images[index], labels[index]).items():
            rows.append(
                {
                    "experiment": EXPERIMENT,
                    "split": split,
                    "factor": factor,
                    "accuracy": round(accuracy, 4),
                    "n_images": len(index),
                    "seed": SEED,
                    "status": "measured",
                }
            )
            print(f"{split:9s} {factor:17s} {accuracy:.1%}")
    save_results(rows)
    CHECKPOINT.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"results -> {RESULTS.relative_to(ROOT)}, weights -> {CHECKPOINT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()