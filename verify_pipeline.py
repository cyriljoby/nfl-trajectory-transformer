"""
verify_pipeline.py — Phase 1 completion gate.

Two checks, both required before building the model:
  (A) MASKS & SHAPES are correct. The #1 bug source is masking, so we assert
      that mask sums equal true sequence lengths and that padded slots are zero.
  (B) NORMALIZATION is sane by eye: plot one play's trajectories in raw field
      coordinates vs the ego-relative normalized frame the model actually sees.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import (
    FIELD_LENGTH, FIELD_WIDTH, FEATURE_NAMES, N_FEATURES, N_STATIC,
    PerPlayerDataset, collate_perplayer, load_records,
)

ART = Path(__file__).parent / "artifacts"


def check_masks(records):
    """(A) Build a real batch and assert the masks match the true lengths."""
    ds = PerPlayerDataset(records, condition_on_ball=True)
    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_perplayer)
    batch = next(iter(loader))

    B, T_max, F = batch.inputs.shape
    print("=== (A) shapes & masks ===")
    print(f"inputs      {tuple(batch.inputs.shape)}  (B, T_max, N_FEATURES={N_FEATURES})")
    print(f"input_mask  {tuple(batch.input_mask.shape)}")
    print(f"targets     {tuple(batch.targets.shape)}  (B, H_max, 2)")
    print(f"target_mask {tuple(batch.target_mask.shape)}")
    print(f"static      {tuple(batch.static.shape)}  (B, N_STATIC={N_STATIC})")

    assert F == N_FEATURES and batch.static.shape[1] == N_STATIC

    # History is LEFT-padded: every real frame must be contiguous at the end.
    for i in range(B):
        m = batch.input_mask[i]
        n = int(m.sum())
        assert m[T_max - n:].all() and not m[: T_max - n].any(), "history mask not left-packed"
        # Padded (masked-out) input rows must be exactly zero.
        assert batch.inputs[i, ~m].abs().sum() == 0, "nonzero data under input pad"

    # Targets are RIGHT-padded: real frames contiguous at the start, count==horizon.
    for i in range(B):
        m = batch.target_mask[i]
        h = int(batch.horizon[i])
        assert int(m.sum()) == h, "target_mask sum != horizon"
        assert m[:h].all() and not m[h:].any(), "target mask not right-packed"
        assert batch.targets[i, ~m].abs().sum() == 0, "nonzero data under target pad"

    print("OK: input_mask is left-packed, target_mask sum == horizon, pads are zero.")

    # Ablation toggle: goal features must vanish when conditioning is off.
    ds_no = PerPlayerDataset(records, condition_on_ball=False)
    feat = ds_no[0]["feat"]
    gi = [FEATURE_NAMES.index("goal_dx"), FEATURE_NAMES.index("goal_dy")]
    assert np.allclose(feat[:, gi], 0.0), "goal features not zeroed in ablation"
    assert not np.allclose(ds[0]["feat"][:, gi], 0.0), "goal features unexpectedly zero"
    print("OK: ball-conditioning toggle zeros goal features for the ablation.\n")
    return ds


def plot_play(records):
    """(B) Plot one play raw (field coords) vs normalized (ego frame)."""
    # Pick a play with a clear targeted receiver for an interpretable picture.
    rec = max(records, key=lambda r: int(r["scored"].sum()))
    bx, by = rec["recon"][0, 2], rec["recon"][0, 3]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: RAW standardized field coordinates -----------------------
    axL.set_title(f"Raw field coords (standardized)  game {rec['game_id']} play {rec['play_id']}")
    for ai in range(len(rec["inputs"])):
        x_throw, y_throw = rec["recon"][ai, 0], rec["recon"][ai, 1]
        feat = rec["inputs"][ai]
        # Reconstruct absolute field x,y from the ego displacement features.
        xs = feat[:, 0] + x_throw
        ys = feat[:, 1] + y_throw
        scored = rec["scored"][ai]
        axL.plot(xs, ys, "-", lw=2 if scored else 0.8,
                 color="C3" if scored else "0.7", alpha=0.9 if scored else 0.6)
        axL.plot(xs[-1], ys[-1], "o", ms=6 if scored else 3,
                 color="C3" if scored else "0.7")  # throw position
        if scored and rec["targets"][ai].shape[0] > 0:
            tx = rec["targets"][ai][:, 0] + x_throw
            ty = rec["targets"][ai][:, 1] + y_throw
            axL.plot(tx, ty, "--", lw=2, color="C0")  # future (target)
    axL.plot(bx, by, "*", ms=18, color="gold", mec="k", label="ball landing")
    axL.set_xlim(0, FIELD_LENGTH); axL.set_ylim(0, FIELD_WIDTH)
    axL.set_xlabel("x (yd)"); axL.set_ylabel("y (yd)")
    axL.legend(loc="upper right"); axL.set_aspect("equal")

    # ---- Right: NORMALIZED ego frame for the scored players -------------
    axR.set_title("Normalized ego frame (what the model sees): throw frame at origin")
    for ai in np.where(rec["scored"])[0]:
        feat = rec["inputs"][ai]
        x_throw, y_throw = rec["recon"][ai, 0], rec["recon"][ai, 1]
        axR.plot(feat[:, 0], feat[:, 1], "-", lw=2, color="C3", alpha=0.7)  # history
        tgt = rec["targets"][ai]
        axR.plot(tgt[:, 0], tgt[:, 1], "--", lw=2, color="C0")             # future
        # Goal vector in the ego frame: where the ball lands relative to throw.
        axR.plot(bx - x_throw, by - y_throw, "*", ms=14, color="gold", mec="k")
    axR.plot(0, 0, "ko", ms=6, label="throw frame (origin)")
    axR.axhline(0, color="0.85", lw=0.5); axR.axvline(0, color="0.85", lw=0.5)
    axR.set_xlabel("dx from throw (yd)"); axR.set_ylabel("dy from throw (yd)")
    axR.legend(loc="upper right"); axR.set_aspect("equal")

    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "phase1_raw_vs_normalized.png"
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"=== (B) plot ===\nsaved -> {out}")
    print("Eyeball: history (red) should flow INTO the throw dot; future (blue dashed)")
    print("should continue from it toward the gold ball-landing star.\n")


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("processed/records_w01.pkl")
    records = load_records(path)
    print(f"loaded {len(records)} plays from {path}\n")
    check_masks(records)
    plot_play(records)
    print("Phase 1 gate complete.")
