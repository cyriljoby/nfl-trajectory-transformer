CLAUDE.md — NFL Trajectory Forecasting (BDB 2026)

What this project is

A from-scratch temporal transformer (PyTorch) for goal-conditioned multi-agent
trajectory forecasting: predict where the targeted receiver and defensive-coverage
players move while the ball is in the air, given their pre-throw motion and the known
ball landing location. Data: NFL Big Data Bowl 2026.

This is a real, defensible ML project for a SWE/AI/ML resume. Every line must survive
an interviewer asking "how does this work and why this way." Do not shortcut the hard
parts — they are the point.

Roughly 5 days of work. Finishing a defensible core beats a half-built ambitious version.


NON-NEGOTIABLES (do not shortcut)


Implement attention FROM SCRATCH. Hand-write scaled dot-product attention
(softmax(QKᵀ/√d_k)V) and multi-head attention. DO NOT use nn.TransformerEncoder,
nn.MultiheadAttention, or any prebuilt transformer block.
Temporal positional encoding (sinusoidal) over the frame axis — frames are an
ordered sequence, position carries meaning. If cross-agent attention is added, the
AGENT axis is a permutation-invariant set → NO positional encoding there. Two axes,
two treatments.
Game-level train/val split (game_id). No game in both. Prevents temporal/play
leakage. This is a deliberate rigor choice and an interview talking point.
Correct masking — the #1 bug source. Input history length VARIES (9–74 frames) and
output horizon VARIES (5–94 frames). Pad and mask BOTH. A wrong mask silently corrupts
the loss and the error metric. Verify masks explicitly; overfit one batch before
scaling (if you can't overfit one batch, there's a mask/normalization bug).
Honest numbers only. Report real displacement error (RMSE in yards). Never
fabricate or round the error down. The metric is regression in yards — report it as
such; do not dress it up as "accuracy."



Build order (phases — finishing is the priority; each gated by a completion check)


Phase 1 — pipeline: join input↔output, normalize by play_direction + ball-relative
coords, build padded/masked input & output tensors. Check: shapes correct + plays
plotted raw-vs-normalized and verified by eye.
Phase 2 — model + overfit one batch: from-scratch temporal transformer; prove
architecture + masking by overfitting a single batch to ~0 error before scaling.
Phase 3 — full training: game-level split, first honest RMSE on held-out games.
Phase 4 — findings (CUTTABLE): interaction-vs-independent baseline +
ball-conditioning ablation.
Phase 5 — polish: repo, README with number + error-vs-horizon plot, fill bullets.


If time runs short: cut Phase 4, never the core forecaster. Do NOT add
deployment/animation.
Safety valve: if masking fights you, drop cross-agent attention → per-player model is
a complete project; cross-agent attention becomes the stretch.


VERIFIED dataset facts (confirmed against the files — trust these)


train/input_2023_w[01-18].csv = pre-throw tracking; train/output_2023_w[01-18].csv
= post-throw target positions (x, y). Join on game_id+play_id+nfl_id+frame_id.
~819 plays/week × 18 ≈ ~14,000+ plays.
Input history per player: median 27 frames (9–74). Deep temporal signal.
Output horizon per player: median 10 frames (5–94) = num_frames_output. VARIABLE.
Scored players: player_to_predict == True (~Defensive Coverage + exactly one
Targeted Receiver per play). Only these are scored — compute loss/metric on them.
ball_land_x/ball_land_y: present on every row (no nulls) — the goal condition.
12–13 players/play on field (both sides) for multi-agent context.
play_direction ∈ {left, right} → flip so offense always moves one way.
x ∈ ~[1,120], y ∈ ~[1,53]. Normalize to ball-relative / LOS-relative.
Per-frame features: x, y, s, a, dir, o. Static: player_height/weight/position/role.
kaggle_evaluation/ API included → leaderboard submission is real.



Features & model defaults


Input features per frame: [x, y, s, a, sin(dir), cos(dir), sin(o), cos(o)] + relative
ball_land offset. (sin/cos on angles.)
Start with a PER-PLAYER model (each scored player's future from own history + ball_land)
→ get it working → THEN add cross-agent attention.
Defaults: d_model=128, 4–8 heads, 2–4 temporal layers, AdamW lr 3e-4, wd 1e-2.
Loss: masked MSE/RMSE on predicted (x,y) over valid output frames of scored players.


Conventions


Python 3, PyTorch. Separate pipeline / model / train modules.
Seed everything; save the game-level split to disk for reproducibility.
Clarity over cleverness — this code is read in interviews. Comment the WHY, especially
for masking and normalization.


Do NOT


Do not use prebuilt transformer/attention layers.
Do not use random (non-game) splits.
Do not compute loss/metric on non-scored players.
Do not fabricate or round down the error.
Do not add deployment/animation/overlays — out of scope.
Do not skip the overfit-one-batch check before full training.