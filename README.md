# NFL Trajectory Forecasting — Goal-Conditioned Temporal Transformer

A from-scratch PyTorch temporal transformer for **goal-conditioned multi-agent
trajectory forecasting**: given player tracking data up to the moment a pass is
thrown, predict where the targeted receiver and defensive-coverage players move
*while the ball is in the air*, conditioned on the known ball-landing location.

Data: [NFL Big Data Bowl 2026](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction).

> **Status:** in development. This README documents the design and the rigor
> choices; held-out numbers and plots are filled in as phases complete. No
> results are reported here until they are real.

---

## Problem

At the instant a quarterback releases the ball, each pass play freezes into a
forecasting problem: where will the contesting players be when the ball comes
down? The model receives every tracked player's pre-throw motion and the ball's
future landing spot, and must predict the future `(x, y)` trajectory of the
**scored players** (the targeted receiver and defensive-coverage defenders) for a
variable horizon until the ball lands.

- **Inputs (per player, per frame):** position `x, y`, speed `s`, acceleration
  `a`, motion direction `dir`, orientation `o`, plus static player attributes.
- **Goal condition:** `ball_land_x`, `ball_land_y` — present on every play.
- **Targets:** future `(x, y)` for players with `player_to_predict == True`.
- **Metric:** displacement error (RMSE, in yards) over valid future frames of
  scored players only.

The data is genuinely hard: **variable input history** (≈9–74 frames) and
**variable output horizon** (5–94 frames), 12–13 agents per play, two teams,
and interaction effects on contested catches.

---

## Approach

A temporal transformer encoder over each player's pre-throw frame sequence,
conditioned on the ball-landing location, decoded to a future trajectory.

Two model variants are built and compared:

1. **Independent (per-player):** each scored player's future is predicted from
   its own history + the goal condition. Strong, simple baseline.
2. **Interaction (cross-agent attention):** adds attention across the agents on
   the field so defenders and receivers can influence each other's predicted
   paths — the hypothesis being that interaction matters most on contested balls.

### Two axes, two treatments
- **Time axis** is an *ordered sequence* → **sinusoidal positional encoding** over
  frames.
- **Agent axis** is a *permutation-invariant set* → **no positional encoding**;
  agents are attended as an unordered set.

---

## Engineering rigor (the point of the project)

These are deliberate correctness and methodology choices, each defensible in
detail:

- **Attention implemented from scratch.** Scaled dot-product attention
  `softmax(QKᵀ/√d_k)V` and multi-head attention are hand-written — no
  `nn.TransformerEncoder` or `nn.MultiheadAttention`.
- **Game-level train/val split** on `game_id` (no game in both sets) to prevent
  play- and temporal-leakage. The split is seeded and saved to disk.
- **Explicit masking on both variable axes** — padded input history *and* padded
  output horizon are masked so neither padding nor invalid frames corrupt the
  loss or the reported error.
- **Overfit-one-batch gate.** Before any full training run, the model must drive a
  single batch to ~0 error — the fastest proof that masking and normalization are
  correct.
- **Scored-players-only loss/metric.** Loss and RMSE are computed exclusively on
  `player_to_predict == True`.
- **Honest reporting.** Error is real displacement RMSE in yards. Nothing is
  rounded down or relabeled as "accuracy."

---

## Pipeline

1. **Join** per-week input (pre-throw) ↔ output (post-throw) on
   `game_id + play_id + nfl_id + frame_id` across all 18 weeks.
2. **Normalize:** flip by `play_direction` so offense always moves one way;
   convert to ball-relative / line-of-scrimmage-relative coordinates; encode
   angles as `sin/cos` of `dir` and `o`.
3. **Tensorize:** build padded, masked input and output tensors.

**Per-frame feature vector:**
`[x, y, s, a, sin(dir), cos(dir), sin(o), cos(o)]` + ball-landing relative offset.

---

## Planned findings

- **Interaction vs. independent:** does cross-agent attention reduce RMSE, and is
  the gain concentrated on contested catches?
- **Ball-conditioning ablation:** RMSE with vs. without the `ball_land` goal
  condition (toggled via config).
- **Error vs. horizon:** how displacement error grows as the prediction horizon
  lengthens.

*(Numbers and plots added as Phase 3+ complete.)*

---

## Repository layout

```
pipeline.py     data loading, join, normalization, padding + masking
model.py        from-scratch attention, temporal transformer, both variants
train.py        game-level split, training loop, masked RMSE evaluation
```

The dataset and the Kaggle evaluation harness are **not** committed (see
`.gitignore`); download the data from the competition page.

---

## Reproducing

```bash
# 1. Place the competition data under nfl-big-data-bowl-2026-prediction/
# 2. Create the environment (PyTorch, polars/pandas, matplotlib)
# 3. Run the pipeline, then training
python train.py
```

Everything is seeded and the game-level split is saved for reproducibility.

---

## License

For educational and portfolio use.
