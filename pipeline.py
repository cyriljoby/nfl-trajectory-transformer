"""
pipeline.py — NFL Big Data Bowl 2026 trajectory-forecasting data pipeline.

Responsibilities (Phase 1):
  1. JOIN   per-week input (pre-throw tracking) with output (post-throw target
            positions) on game_id + play_id + nfl_id + frame_id.
  2. NORMALIZE so the model sees motion, not absolute field position:
            - standardize play direction with a 180 deg rotation,
            - express positions as displacement from the throw frame (ego origin),
            - encode angles as sin/cos,
            - add a per-frame goal vector toward the ball-landing spot.
  3. TENSORIZE into padded, masked tensors. Input history length VARIES (9-74
            frames) and output horizon VARIES (5-94 frames); BOTH axes are padded
            and masked. A wrong mask silently corrupts the loss and the reported
            error, so masks are built explicitly here and asserted in the verifier.

The processed unit is a *play*: every record holds all tracked agents (for the
cross-agent interaction model) plus a per-agent `scored` flag. The per-player
(independent) model simply iterates the scored agents.  Loss/metric are only
ever computed on scored agents (player_to_predict == True).
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# NFL field geometry (yards). Length includes both end zones; width is 160 ft.
FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3

DATA_DIR = Path(__file__).parent / "nfl-big-data-bowl-2026-prediction" / "train"
PROCESSED_DIR = Path(__file__).parent / "processed"

# Roles seen in the data; one-hot order is fixed for reproducibility.
ROLES = ["Targeted Receiver", "Defensive Coverage", "Other Route Runner", "Passer"]
ROLE_TO_IDX = {r: i for i, r in enumerate(ROLES)}

# Per-frame input feature layout (documented so model.py and the verifier agree).
#   0 dx_from_throw   x displacement from the player's last input (throw) frame
#   1 dy_from_throw   y displacement from the throw frame
#   2 s              speed (yd/s); a magnitude, rotation-invariant
#   3 a              acceleration (yd/s^2); magnitude, rotation-invariant
#   4 sin_dir        direction of motion, sin/cos to avoid wraparound
#   5 cos_dir
#   6 sin_o          body orientation, sin/cos
#   7 cos_o
#   8 goal_dx        vector to ball landing: ball_land_x - x   (goal conditioning)
#   9 goal_dy        ball_land_y - y
FEATURE_NAMES = [
    "dx_from_throw", "dy_from_throw", "s", "a",
    "sin_dir", "cos_dir", "sin_o", "cos_o",
    "goal_dx", "goal_dy",
]
N_FEATURES = len(FEATURE_NAMES)
# Indices of the goal-conditioning features (zeroed for the no-ball ablation).
GOAL_FEATURE_IDX = [FEATURE_NAMES.index("goal_dx"), FEATURE_NAMES.index("goal_dy")]

# Per-agent static feature layout:
#   last-frame position relative to ball landing (locates agent in a SHARED
#   frame for cross-agent attention), height, weight, role one-hot, side.
N_STATIC = 2 + 1 + 1 + len(ROLES) + 1


def parse_height(h: str) -> float:
    """'6-1' -> inches (73.0)."""
    feet, inches = h.split("-")
    return float(feet) * 12.0 + float(inches)


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #

def _standardize_direction(df: pd.DataFrame, dir_col_pairs) -> pd.DataFrame:
    """
    Make offense always move toward +x via a 180 deg rotation about field center.

    A 180 deg rotation (x->L-x, y->W-y, angle->angle+180) is chosen over a pure
    x-axis reflection because rotation PRESERVES the play's handedness: a route
    breaking to the receiver's left stays a left break. A reflection would mirror
    left/right, turning each play into a different (non-physical) play.

    `dir_col_pairs` is a list of (x_col, y_col) position columns to flip; angle
    columns 'dir'/'o' are flipped if present.
    """
    left = df["play_direction"].values == "left"
    if not left.any():
        return df
    for xc, yc in dir_col_pairs:
        df.loc[left, xc] = FIELD_LENGTH - df.loc[left, xc].values
        df.loc[left, yc] = FIELD_WIDTH - df.loc[left, yc].values
    for ac in ("dir", "o"):
        if ac in df.columns:
            df.loc[left, ac] = (df.loc[left, ac].values + 180.0) % 360.0
    return df


def _load_week(week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and direction-standardize one week's input and output frames."""
    wk = f"{week:02d}"
    inp = pd.read_csv(DATA_DIR / f"input_2023_w{wk}.csv")
    out = pd.read_csv(DATA_DIR / f"output_2023_w{wk}.csv")

    # Standardize the input frame: player position, the ball-landing goal, angles.
    inp = _standardize_direction(inp, [("x", "y"), ("ball_land_x", "ball_land_y")])

    # The output file carries no play_direction column, so propagate it from the
    # input (constant per play) before flipping the future positions identically.
    pd_map = inp.drop_duplicates(["game_id", "play_id"]).set_index(
        ["game_id", "play_id"]
    )["play_direction"]
    out = out.join(pd_map, on=["game_id", "play_id"])
    out = _standardize_direction(out, [("x", "y")])
    return inp, out


def build_records(weeks) -> list[dict]:
    """
    Process the given weeks into per-play records.

    Each record:
        game_id, play_id : ints (game_id kept for the game-level split)
        inputs   : list[np.ndarray (T_i, N_FEATURES)]   per agent, throw-frame last
        targets  : list[np.ndarray (H_i, 2)]            future displacement (dx,dy)
        static   : np.ndarray (n_agents, N_STATIC)
        scored   : np.ndarray (n_agents,) bool          player_to_predict
        nfl_id   : list[int]
        recon    : np.ndarray (n_agents, 5)  [x_throw, y_throw, bx, by, is_left]
                   stored only to map predictions back to field coords at submission
                   time; NOT used for the (translation/rotation-invariant) metric.
    """
    records: list[dict] = []
    for week in weeks:
        inp, out = _load_week(week)

        # Precompute angle sin/cos once (vectorized) on the whole week.
        dir_rad = np.deg2rad(inp["dir"].values)
        o_rad = np.deg2rad(inp["o"].values)
        inp = inp.assign(
            sin_dir=np.sin(dir_rad), cos_dir=np.cos(dir_rad),
            sin_o=np.sin(o_rad), cos_o=np.cos(o_rad),
        )

        out_groups = dict(tuple(out.groupby(["game_id", "play_id"])))

        for (game_id, play_id), play in inp.groupby(["game_id", "play_id"]):
            oplay = out_groups.get((game_id, play_id))
            if oplay is None:
                continue  # no targets for this play; skip
            out_by_player = dict(tuple(oplay.groupby("nfl_id")))

            inputs, targets, statics, scored, nfl_ids, recon = [], [], [], [], [], []
            bx = play["ball_land_x"].iloc[0]
            by = play["ball_land_y"].iloc[0]
            is_left = float(play["play_direction"].iloc[0] == "left")

            for nfl_id, pl in play.groupby("nfl_id"):
                pl = pl.sort_values("frame_id")
                x = pl["x"].values
                y = pl["y"].values
                x_throw, y_throw = x[-1], y[-1]  # last input frame == the throw

                feat = np.stack([
                    x - x_throw,                 # dx_from_throw
                    y - y_throw,                 # dy_from_throw
                    pl["s"].values,
                    pl["a"].values,
                    pl["sin_dir"].values, pl["cos_dir"].values,
                    pl["sin_o"].values, pl["cos_o"].values,
                    bx - x,                      # goal_dx (vector to ball landing)
                    by - y,                      # goal_dy
                ], axis=1).astype(np.float32)

                is_scored = bool(pl["player_to_predict"].iloc[0])

                # Targets only exist for scored players; store displacement from
                # the throw frame so the model predicts motion, not field position.
                if is_scored and nfl_id in out_by_player:
                    o = out_by_player[nfl_id].sort_values("frame_id")
                    tgt = np.stack([o["x"].values - x_throw,
                                    o["y"].values - y_throw], axis=1).astype(np.float32)
                else:
                    is_scored = False
                    tgt = np.zeros((0, 2), dtype=np.float32)

                role_oh = np.zeros(len(ROLES), dtype=np.float32)
                role_oh[ROLE_TO_IDX[pl["player_role"].iloc[0]]] = 1.0
                static = np.concatenate([
                    [x_throw - bx, y_throw - by],                       # shared-frame loc
                    [parse_height(pl["player_height"].iloc[0]) / 12.0],  # height (ft)
                    [pl["player_weight"].iloc[0] / 100.0],               # weight (cwt)
                    role_oh,
                    [1.0 if pl["player_side"].iloc[0] == "Offense" else 0.0],
                ]).astype(np.float32)

                inputs.append(feat)
                targets.append(tgt)
                statics.append(static)
                scored.append(is_scored)
                nfl_ids.append(int(nfl_id))
                recon.append([x_throw, y_throw, bx, by, is_left])

            if not any(scored):
                continue  # nothing to learn from / score on this play

            records.append({
                "game_id": int(game_id),
                "play_id": int(play_id),
                "inputs": inputs,
                "targets": targets,
                "static": np.stack(statics),
                "scored": np.array(scored, dtype=bool),
                "nfl_id": nfl_ids,
                "recon": np.array(recon, dtype=np.float32),
            })
    return records


def save_records(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_records(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Datasets + collate (padding & masking happen here)
# --------------------------------------------------------------------------- #

@dataclass
class Batch:
    """One padded/masked batch for the per-player model."""
    inputs: torch.Tensor       # (B, T_max, N_FEATURES)
    input_mask: torch.Tensor   # (B, T_max) bool, True = real history frame
    targets: torch.Tensor      # (B, H_max, 2)
    target_mask: torch.Tensor  # (B, H_max) bool, True = real future frame
    static: torch.Tensor       # (B, N_STATIC)
    horizon: torch.Tensor      # (B,) int, true output length per sample
    meta: list                 # list of (game_id, play_id, nfl_id)


class PerPlayerDataset(Dataset):
    """
    One item == one scored agent (player_to_predict == True).

    This is the primary unit for the independent (per-player) model: predict a
    player's future trajectory from its own pre-throw history + goal condition.
    `condition_on_ball=False` zeros the goal-vector features for the Phase-4
    ball-conditioning ablation.
    """

    def __init__(self, records: list[dict], condition_on_ball: bool = True):
        self.condition_on_ball = condition_on_ball
        self.items = []  # (record_idx, agent_idx)
        for ri, rec in enumerate(records):
            for ai, sc in enumerate(rec["scored"]):
                if sc:
                    self.items.append((ri, ai))
        self.records = records

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        ri, ai = self.items[idx]
        rec = self.records[ri]
        feat = rec["inputs"][ai].copy()
        if not self.condition_on_ball:
            feat[:, GOAL_FEATURE_IDX] = 0.0  # ablation: hide the ball-landing goal
        return {
            "feat": feat,
            "tgt": rec["targets"][ai],
            "static": rec["static"][ai],
            "meta": (rec["game_id"], rec["play_id"], rec["nfl_id"][ai]),
        }


def collate_perplayer(samples) -> Batch:
    """
    Pad variable-length history and horizon to the per-BATCH max and build masks.

    Padding to the batch max (not a global 74/94) keeps tensors small; the masks
    mark which frames are real so padding never enters attention or the loss.
    """
    B = len(samples)
    T_max = max(s["feat"].shape[0] for s in samples)
    H_max = max(s["tgt"].shape[0] for s in samples)

    inputs = torch.zeros(B, T_max, N_FEATURES)
    input_mask = torch.zeros(B, T_max, dtype=torch.bool)
    targets = torch.zeros(B, H_max, 2)
    target_mask = torch.zeros(B, H_max, dtype=torch.bool)
    static = torch.zeros(B, N_STATIC)
    horizon = torch.zeros(B, dtype=torch.long)
    meta = []

    for i, s in enumerate(samples):
        # History is LEFT-padded so the throw frame (the most recent, most
        # informative frame) lands at the same final index T_max-1 for every
        # sample — convenient if the model reads the last position directly.
        t = s["feat"].shape[0]
        inputs[i, T_max - t:] = torch.from_numpy(s["feat"])
        input_mask[i, T_max - t:] = True

        h = s["tgt"].shape[0]
        targets[i, :h] = torch.from_numpy(s["tgt"])
        target_mask[i, :h] = True
        horizon[i] = h

        static[i] = torch.from_numpy(s["static"])
        meta.append(s["meta"])

    return Batch(inputs, input_mask, targets, target_mask, static, horizon, meta)


# --------------------------------------------------------------------------- #
# CLI: build the processed records
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build processed BDB2026 records.")
    p.add_argument("--weeks", type=int, nargs="+", default=list(range(1, 19)))
    p.add_argument("--out", type=Path, default=PROCESSED_DIR / "records.pkl")
    args = p.parse_args()

    print(f"Processing weeks {args.weeks} ...")
    recs = build_records(args.weeks)
    n_scored = sum(int(r["scored"].sum()) for r in recs)
    print(f"  plays: {len(recs)}   scored agents: {n_scored}")
    save_records(recs, args.out)
    print(f"  saved -> {args.out}")
