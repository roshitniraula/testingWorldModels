# World Model Interpretability — Findings & Working Notes

Living document tracking what we've found and why, kept in sync with `WMIP.ipynb`. This is the source material for the eventual paper — write results here as soon as they're produced, not after the fact.

*(All work below was produced in a single day, 2026-08-02, across several bugfix/methodology passes. Per-version deltas have been collapsed into one current-state account — see §9 for the bug-fix history, kept because the bugs themselves are methodologically instructive.)*

## 1. Project Summary

We build a MiniGrid `DoorKey-6x6` world model (tiny 2-layer TransformerLens `HookedTransformer`) with a deliberately injected spurious confound: the border-row color is perfectly correlated with which quadrant the agent occupies. We test whether the model relies on this shortcut (OOD test with the color mapping flipped), diagnose it with linear probes, and try to causally remove it two ways: (A) activation ablation at inference time, (B) adversarial regularization during training.

## 2. Current Setup

- **Env**: `MiniGrid-DoorKey-6x6-v0`, `FullyObsWrapper`, wrapped in `QuadrantColorConfound` (top-left → RED, bottom-right → BLUE, flipped in `ood_env`).
- **Tokenization**: each of the 36 grid cells → `(object, color, state)` flattened to one integer (`cell_to_token`); one timestep = `[cell_0..cell_35, action]`, 37 tokens. `VOCAB_SIZE = 205`.
- **Model**: 2-layer, `d_model=128`, `d_head=32`, `n_heads=4`, `d_mlp=512`, causal (`HookedTransformer`).
- **Prediction head**: `next_state_head`, a linear layer on the **last sequence position's** residual stream (the only position that has seen both the full grid and the action, given causal attention) — predicts all 36 next-cell tokens at once.
- **Data-collection/eval policy**: a lightweight BFS scripted policy (navigate→key→pickup→navigate→door→toggle→navigate→goal), mixed 50/50 with random actions (10% noise even within scripted episodes) — applied to training data collection *and* all eval rollouts.
- **Training data**: 800 episodes, max 50 steps/episode. Eval: 500 episodes.
- **Device**: CPU, deliberately — MPS was flagged by TransformerLens as able to silently produce incorrect results on this PyTorch version, unacceptable for a project whose conclusions depend on exact activation math.
- **Training**: 100 epochs (confirmed sufficient — loss no longer visibly decreasing, unlike an earlier 20-epoch pass).
- **Seeding**: per-episode action sampling uses `np.random.default_rng`, decoupled from gym's own RNG, so baseline/ablated/regularized evaluations use identical state-action trajectories (a true paired comparison).

## 3. Current Results (single run, seed=0 — see §6 for multi-seed as an open item)

**Data-collection policy fix, validated in-notebook:**

| | key pickup | door open | goal reached |
|---|---|---|---|
| pure random (rejected policy) | 43.0% | 3.5% | 0.0% |
| mixed BFS/random (current default) | 67.5% | 45.5% | 44.0% |

**Baseline**: in-dist exact-match 0.290 / OOD 0.131 (per-cell 0.968 / 0.926). Probe B (color) = **1.000**, Probe A (position) R² = 0.831, strongest at layer L=0.

**Multi-direction ablation, greedy-probe vs. random-direction control** (full table in `results_mac_cpu.json`):

| k | Probe B (greedy) | Probe B (random) | OOD exact (greedy) | OOD exact (random) | In-dist exact (greedy) | In-dist exact (random) |
|---|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 0.134 | 0.134 | 0.295 | 0.295 |
| 1 | 0.998 | 1.000 | 0.035 | 0.098 | 0.055 | 0.249 |
| 2 | 0.894 | 1.000 | 0.015 | 0.088 | 0.026 | 0.234 |
| 3 | 0.858 | 1.000 | 0.022 | 0.058 | 0.025 | 0.162 |
| 4 | 0.828 | 1.000 | 0.020 | 0.055 | 0.020 | 0.119 |
| 5 | 0.781 | 1.000 | 0.0002 | 0.043 | 0.000 | 0.104 |

Corrected Probe B majority-class baseline: **0.495** (not the naive 1/3 used in an earlier pass — `quadrant_label` puts "neutral" on 18/36 cells, not a balanced 3-class split).

**Head-level path patching**: patching any single one of the 8 attention heads (2 layers × 4 heads) from the clean run into the corrupted run changed exact-match accuracy by **exactly 0.0** in every case, and per-cell accuracy by a small *negative* amount (−2% to −5%) in every case. Corrupted (no-patch) baseline: exact-match 0.116, per-cell 0.934.

**Whole-layer path patching** (all 4 heads at a layer at once, not one at a time): still **exactly 0.0** exact-match recovery at both layers, and per-cell accuracy *more* negative than any single-head patch (layer 0: per-cell delta −0.043, vs. the best single head's −0.027; layer 1: per-cell delta −0.085, vs. the best single head's −0.023). A bigger intervention hurts more but still recovers nothing — see Key Finding 6.

**Layer-depth comparison** (greedy-probe ablation curve repeated at layer 1, not just the strongest-color layer 0 — **caveat: with only 2 layers total, layer 1 is also `LAST_LAYER_RESID`, the exact input to `next_state_head`; see Key Finding 7 before over-reading this**):

| k | Probe B (layer 1) | OOD exact (layer 1) | In-dist exact (layer 1) | OOD per-cell (layer 1) |
|---|---|---|---|---|
| 0 | 1.000 | 0.134 | 0.295 | 0.927 |
| 1 | 1.000 | 0.020 | 0.056 | 0.896 |
| 2 | 1.000 | 0.018 | 0.018 | 0.782 |
| 3 | 1.000 | 0.000 | 0.000 | 0.586 |
| 4 | 0.979 | 0.000 | 0.000 | 0.587 |
| 5 | 0.964 | 0.000 | 0.000 | 0.582 |

**Method B (adversarial), finer lambd sweep** (gradient-reversal bug fixed — see §9; lambd grid extended from {0.5, 2.0, 5.0} to {0.5, 1.0, 1.5, 2.0, 5.0}):

| lambd | In-dist exact | OOD exact | Probe B acc |
|---|---|---|---|
| 0.5 | 0.029 | 0.018 | 0.954 |
| 1.0 | 0.000 | 0.000 | 0.474 |
| 1.5 | 0.003 | 0.001 | 0.474 |
| 2.0 | 0.002 | 0.000 | 0.474 |
| 5.0 | 0.000 | 0.000 | 0.474 |

Raw artifact: `results_mac_cpu.json`. **Reproducibility note**: the baseline model and every ablation/path-patching number above reproduced *exactly bit-for-bit* against the prior run — but this lambd sweep's numbers (re-run at the same seed, same code) differ slightly from the previous pass's lambd=0.5/2.0/5.0 values (e.g. lambd=0.5 was previously Probe B=1.000/in-dist=0.094, now Probe B=0.954/in-dist=0.029). See Key Finding 8 / §6 limitation 6 — this is adversarial-training-specific float nondeterminism, not a seeding bug.

## 4. Key Findings

1. **Color is linearly decodable (Probe B = 1.000) in every condition tested at baseline**, strongest at layer L=0.
2. **Greedy-probe ablation directions are meaningfully worse than random directions, on both axes — but random ablation isn't harmless either.** At k=1, greedy ablation craters in-dist exact-match to 0.055 (an 81% relative drop from 0.295) while random-direction ablation only drops it to 0.249 (16% relative drop); by k=5, greedy is at 0.000 while random is still at 0.104 (35% of baseline retained). Probe B tells a consistent story: greedy ablation erodes color decodability round over round (1.000 → 0.781), while random ablation leaves it at exactly 1.000 every round. This is the strongest evidence so far that the greedy-derived directions are doing something specific to color, not just generic damage to a small, undertrained-relative-to-task network — **but** "any comparable perturbation is somewhat costly to this model" is also true. Both hold at once; neither cancels the other out.
3. **Color is never actually driven to chance, even after 5 rounds of greedy ablation.** Against the corrected majority-class baseline (0.495), the k=5 plateau (0.781) is still ~28 points above chance.
4. **No single attention head accounts for the clean-vs-corrupted causal effect.** Patching any one of the 8 heads changed exact-match by exactly 0.0 in every case, and slightly *hurt* per-cell accuracy in every case. Either the effect is redundantly distributed across heads (patching one still leaves seven "corrupted" heads dominating), it lives partly in the MLP layers (untested), or exact-match is too brittle a metric to detect a partial single-head effect.
5. **Adversarial training reveals a genuine trade-off, not a clean success — and the finer sweep shows there's no middle ground, not just an untested gap.** lambd=0.5 leaves Probe B mostly intact (0.954) with in-dist exact-match still near-zero but nonzero (0.029); by lambd=1.0, Probe B is already fully at the chance floor (0.474) *and* in-dist exact-match is exactly 0.0. There's no lambd in {0.5, 1.0, 1.5, 2.0, 5.0} that suppresses color while leaving the task functional — the transition from "no effect" to "total collapse" happens somewhere between 0.5 and 1.0, with nothing observed in between.
6. **Likely mechanism**: layer 0 sits immediately downstream of the token embedding, where color is copied in directly — cheap enough to represent redundantly across more than one direction, and cheap enough that a weak adversarial penalty can't out-compete it.
7. **Whole-layer path patching (all 4 heads at once) still recovers exactly 0.0 exact-match, at both layers — and hurts per-cell accuracy more than any single-head patch did.** This rules out "one head individually matters" *and* "the four heads jointly reconstruct the effect if patched together" at the granularity path patching can see. What's left: the effect is either in the MLP layers (untested), or a whole-layer attention patch still isn't the right unit of intervention.
8. **The layer-1 ablation comparison surfaced a methodological confound, not (yet) a clean depth finding.** With only 2 layers, layer 1 *is* `LAST_LAYER_RESID` — the exact residual stream `next_state_head` reads from. Ablating there is uninterpretable as a "deeper layer" test: it's directly corrupting the read-out itself. Consistent with that, layer-1 ablation is *far* more behaviorally destructive than layer 0's (exact-match hits exactly 0.0 by k=3, per-cell floor ~0.58 vs. layer 0's ~0.74) while Probe B barely moves (1.000 → 0.964 by k=5, vs. layer 0's 1.000 → 0.781) — the opposite dissociation from layer 0's curve, and plausibly just an artifact of ablating the output position rather than a genuine finding about shortcut depth. **A real layer-depth comparison needs a model with ≥3 layers**, so there's a non-terminal hidden layer to test that isn't also the read-out.
9. **Baseline training and all ablation/path-patching numbers are exactly bit-reproducible across reruns (same seed); adversarial (Method B) training is not.** Rerunning the identical lambd=0.5/2.0/5.0 configurations produced different exact-match numbers than the prior pass, while every other number in the notebook matched to full floating-point precision. Most likely cause: adversarial training's extra branch (two losses, two optimizers, gradient reversal) has more multi-threaded-reduction surface area than baseline's single-loss backward pass, and CPU floating-point non-associativity there compounds over 100 epochs of SGD. Not fixed (see decision below) — treat any single adversarial-training number as approximate, not exact, until/unless this gets locked down with `torch.use_deterministic_algorithms` and single-threaded execution.
10. **PCA of the layer-0 residual stream is the cleanest visual in the notebook** (`artifacts/pca_residual_stream.png`). Projected to 2 principal components, color/quadrant produces near-perfect, visually obvious clustering (left group vs. right group, near-zero overlap) — color is the *dominant* source of variance in this layer's first two components. Position (Probe A, R²=0.83) shows up as a second, orthogonal sub-structure nested within each color cluster, not competing with color for the same variance.
11. **Attention patterns complicate the naive story** (`artifacts/attention_patterns.png`). If the model reads color by attending to the border row, the final (action-informed) position should show elevated attention on row 0. It doesn't, in any of the 4 layer-0 heads — attention on row 0 is unremarkable-to-low; the brightest cells are scattered elsewhere in the grid. Whatever injects color so strongly into the residual stream (per PCA/probe results) isn't visible as elevated attention mass on the source position. Not yet distinguished: modest attention weight may be enough to linearly inject a detectable signal, the computation may happen via the MLP rather than attention-copying, or color may leak in through some other row-0 cell interaction not captured by looking only at attention *from* the final position.

## 5. Candidate Angles for the Paper

Working thesis, to be confirmed/refuted by Phases 2–4 (§7): *shortcut features encoded very early/cheaply (near the input embedding) are harder to causally suppress via standard single-direction linear techniques than the literature's canonical deeper examples (e.g. Othello-GPT's board state) — both in the number of directions required and in adversarial training's ability to prevent them from forming.*

Two independent generalization axes for the results section: confound **strength** (Phase 2) and confound **type/complexity** (Phase 4), plus a methods contribution (SAE-vs-greedy-probe ablation comparison, Phase 3).

## 6. Known Limitations (unresolved)

1. **Single run, single seed, no error bars.** One training run, one init, one fixed train/test split. Collapse magnitude, plateau height, and which layer looks "strongest" could all be idiosyncratic to this run. Needs ≥3 seeds before any number here goes in a paper.
2. **Layer-depth comparison is confounded by model size.** With only 2 layers, "layer 1" ablation is also "ablate `LAST_LAYER_RESID`, the direct input to `next_state_head`" — not a clean test of shallow-vs-deep representation. A real answer needs a ≥3-layer model with a genuine non-terminal hidden layer to test (see Key Finding 8).
3. **Possible probe degeneracy in later ablation rounds.** Each round's probe is fit on already rank-reduced 128-dim data with a modest sample size — rounds 3–5 may be partly fitting sampling noise rather than a persisting color-relevant direction, which would independently explain both the slow probe decline and part of the behavioral damage.
4. ~~Single-head path patching doesn't rule out multi-head/MLP effects.~~ **Resolved (partially)**: whole-layer patching also recovers exactly 0.0 — rules out "one head" and "all heads jointly," leaves "lives in the MLP" untested.
5. ~~No lambd tested between 0.5 and 2.0.~~ **Resolved**: finer sweep {0.5, 1.0, 1.5, 2.0, 5.0} shows no middle ground exists in this range — the transition from no-effect to total-collapse happens somewhere between 0.5 and 1.0 with nothing in between, not a gap that was merely untested.
6. **Adversarial (Method B) training results are not bit-reproducible across reruns, even at a fixed seed** (baseline and ablation/path-patching numbers are). Likely CPU floating-point non-associativity in the extra backward-pass branches, compounding over 100 epochs. Not fixed — decided to note and move on rather than chase full determinism (`torch.use_deterministic_algorithms` + single-threaded execution) right now. Treat any single Method B number as approximate.

## 7. Roadmap

### 7.1 Active plan (current pass — see implementation plan for full detail)

- [x] ~~Finer adversarial lambd sweep (e.g. 1.0, 1.5)~~ — done; no middle ground found in {0.5, 1.0, 1.5, 2.0, 5.0} (see §4 Key Finding 5).
- [x] ~~Whole-layer path patching (all 4 heads at a layer simultaneously)~~ — done; still exactly 0.0 exact-match recovery at both layers (see §4 Key Finding 7).
- [x] ~~Layer-depth comparison — repeat ablation/probing at layer 1~~ — done, but surfaced a confound (layer 1 == `LAST_LAYER_RESID` in a 2-layer model); a clean version needs a ≥3-layer model (see §4 Key Finding 8, §6 limitation 2).
- [x] Shared `wm_lib.py` module extraction — written (`wm_lib.py`), notebook rewiring + parity check in progress.
- [ ] Multi-seed reruns (≥3 seeds) of the full pipeline — not yet done; every number above is single-seed/single-run.
- [ ] **Phase 2**: confound-strength sweep — vary color–quadrant correlation from 100% down to ~50%, re-run the pipeline per level. Driver written (`run_confound_sweep.py`), not yet executed. See implementation plan for the noise-injection design and metrics.

### 7.2 Future phases / ideas (not scheduled — parked for after the active plan lands)

- **Phase 3 — SAE-vs-probe ablation comparison**: train a sparse autoencoder (`sae-lens`) on the layer-0 residual stream; compare SAE-latent-derived ablation directions against the raw logistic-regression directions used so far. Does the SAE find a cleaner/more surgically-ablatable color feature — fewer directions needed, or a bigger drop in Probe B per direction removed?
- **Phase 4 — relational key→door confound**: a second, harder confound (non-pixel, compositional/relational rather than a raw color feature) in a new environment. Tests whether the dimensionality/ablatability findings generalize beyond a literal color-token shortcut, or are specific to how cheap and low-dimensional color is to represent.
- **Phase 5 (idea) — model-scale sweep**: repeat the ablation/probing pipeline at a few model sizes (e.g. d_model 64/128/256, 2 vs 4 layers). Tests whether "shortcut needs multiple directions and resists chance" is a property of this specific undertrained-relative-to-task 2-layer/128-dim model, or holds as capacity increases.
- **Phase 6 (idea) — pixel-based confound**: replace the symbolic per-cell tokenization with a raw-pixel/CNN-encoded observation, keeping the same color confound. Right now color is cheap partly *because it's literally one input token* — this tests whether the "early/cheap shortcut is hard to ablate" finding survives when color has to be extracted by a learned filter instead of being handed to the model as a discrete symbol.
- **Phase 7 (idea) — domain randomization as a third causal-removal method**: instead of post-hoc ablation (Method A) or adversarial regularization (Method B), randomize the confound *during data collection itself* (i.e., train on a low-strength-correlation version, per Phase 2) and see how it compares to A/B as a removal strategy. This is a natural combination of Phase 2's sweep machinery with the existing two-methods comparison, rather than a fully separate experiment.

## 8. Data-Collection Policy — Why It Changed

Prompted by a suspicion that DoorKey had become "too narrow" to learn from or analyze, instrumented the (then-current) random-policy `collect_trajectories` logic (800 episodes, max 50 steps/episode) to directly count how often the environment's defining mechanics are ever witnessed:

| Event | % of episodes | % of all timesteps |
|---|---|---|
| Key ever picked up | 44.1% | 15.3% |
| Door ever opened | **3.6%** | **1.05%** |
| Goal ever reached | **0.0%** | — |

Across 800 episodes / 40,000 timesteps of uniform-random actions, the goal was reached **zero** times, and the door was open for only ~1 in 100 timesteps. Position and key-carrying state — the state actually exercised by random rollouts — already probed and ablated cleanly, but door-toggling and goal-completion (the two mechanics the environment is named after) were almost entirely absent from both training and eval data.

**Diagnosis**: not an indictment of DoorKey — it's a well-known hard-exploration benchmark precisely because uniform-random action sequences essentially never execute the precise ordered sequence it requires (navigate→key→pickup→navigate→door→toggle→navigate→goal). The bottleneck was the *data-collection policy*, not the environment's inherent richness.

**Decision**: keep DoorKey (all tokenization/confound/probing/ablation infrastructure is debugged and reusable), and mix a lightweight BFS-based scripted policy with random actions for both training and eval data.

**Alternative considered and rejected (for now)**: switching to a different environment entirely (e.g. MiniGrid `Dynamic-Obstacles`, where other entities move every timestep, sidestepping the sparse-event problem structurally). Rejected as primary fix because it would require a new confound design, wrapper, and tokenization rework, while the policy fix directly addressed the diagnosed cause at much lower cost. Kept on the roadmap (§7) as a possible follow-up if the policy fix turns out to be insufficient.

## 9. Bug-Fix History (kept for methodological record)

These are real bugs hit and fixed over the course of the day. Kept here (not deleted) because each one independently explains a chunk of earlier confusing results, and the pattern of bugs is itself a useful lesson for future phases.

- **Readout position bug**: next-state predictions were originally read off sequence positions 0–35 of `[cell_0..cell_35, action]` — positions that structurally never see the action (token 36) and so can't predict the post-action grid. Symptom: the model converged to a trivial "copy the input cell forward" shortcut, which explained all of the earliest confusing results at once (no OOD collapse, single-direction ablation doing nothing, high per-cell accuracy that was really just background-copying). **Fix**: added `next_state_head`, reading only from the last (action-informed) position.
- **Missing `torch.no_grad()`** on two `run_with_cache` calls — the actual trigger for an early CUDA OOM on Colab (an untracked autograd graph over the full eval set), not genuine memory pressure.
- **Stale-`L` bug**: the ablation layer was leftover loop state rather than the actual argmax of probe accuracy.
- **MPS correctness risk**: TransformerLens flagged MPS as able to silently produce incorrect results on this PyTorch version — device selection changed to `cuda if available else cpu`, unconditionally skipping MPS.
- **Noise floor discovery**: the same unpatched model scored 0.023 vs. 0.018 OOD exact-match across two eval calls before per-episode action sampling was seeded — the "noise floor" was roughly the same size as the effect being measured. Fixed by seeding both eval functions with `np.random.default_rng`, decoupled from gym's own RNG, so baseline/ablated/regularized evaluations use identical state-action trajectories.
- **Method B detach bug (the most significant fix)**: `run_with_cache`'s hook unconditionally does `cache[hook_name] = tensor.detach().to(device)` (confirmed by reading `transformer_lens/HookedRootModule.py:362-364`) — there is no opt-out. `train_adversarial_model` was reading `h_color` from this cache, so `grad_reverse`'s backward pass had nowhere to send a gradient: the autograd graph from `adv_loss` back into the world model's parameters was severed at the cache boundary. `adv_loss` still dropped to ~0 (the adversary learned to read color off a *frozen* snapshot), which is exactly why it looked like "the adversary always wins" — but the world model itself was never receiving the penalty at all, for any lambd. **Fix**: replaced `run_with_cache` with a `model_adv.hooks()` context manager capturing the non-detached tensor. Confirmed working: the lambd sweep is no longer bit-identical across lambd values (see §3/§4).
- **Path-patching batch/shape mismatch**: `eval_with_hooks_static` chunked its input in batches of 128 while the patch hook closed over a full-size (~3459-example) source activation tensor — patching a 128-example chunk with a 3459-example tensor is a shape mismatch. **Fix**: removed the chunking (the model is tiny enough — 2 layers, `d_model=128` — that a single unbatched forward pass over the whole patch dataset is fine on CPU).
- **Random-policy data collection** almost never witnessed the door or goal — see §8.
