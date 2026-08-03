"""
Phase 2 -- confound-strength sweep (see findings.md Sec 7.1 / the implementation
plan at /Users/roshit/.claude/plans/modular-zooming-manatee.md).

Varies QuadrantColorConfound's correlation strength from 1.0 (today's fully
deterministic confound) down to 0.5 (color carries zero information about
quadrant), crossed with 3 random seeds, to test whether the project's
central finding -- the shortcut needs multiple ablation directions and
resists adversarial suppression -- is a property of the confound itself or
an artifact of the single maximally-easy p=1.0 setting tested so far.

The strength=1.0 column of this sweep, across all 3 seeds, IS the
multi-seed rerun of the current baseline (see plan Step 5) -- not a
separate effort.

This is a long-running (multi-hour, CPU-only) job. Run it in the background,
e.g.:
    nohup ./wm-interp/bin/python3 run_confound_sweep.py > confound_sweep.log 2>&1 &

Results are written to results_confound_sweep_mac_cpu.json after EVERY grid
point completes (not just at the end), and already-completed (strength,
seed) pairs found in that file are skipped on restart -- a crash or sleep
interruption only costs the in-flight grid point, not the whole run.
"""

import json
import time
import datetime
import numpy as np
import torch

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper

import wm_lib as wl

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

STRENGTHS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
SEEDS = [0, 1, 2]

N_TRAIN_EPISODES = 800
N_PROBE_EPISODES = 200          # for fitting Probe A/B (mirrors WMIP.ipynb's eval_data)
N_BASELINE_EVAL_EPISODES = 300  # in-dist/OOD rollout eval (trimmed from WMIP.ipynb's 500 -- see plan)
N_ABLATION_EVAL_EPISODES = 150  # per round of the multi-direction ablation curve (x2 conditions x k_max+1 rounds)
K_MAX = 5
N_EPOCHS = 100
ADV_LAMBD = 2.0   # the lowest lambd that measurably suppressed Probe B at strength=1.0 -- see plan for why
                  # only one lambd (not the full {0.5,1,1.5,2,5} sweep) runs per grid point

SCRIPTED_POLICY_PROB = 0.5
SCRIPTED_NOISE_PROB = 0.1

OUT_PATH = "results_confound_sweep_mac_cpu.json"


def make_envs(strength):
    train_env = wl.QuadrantColorConfound(
        FullyObsWrapper(gym.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")),
        flip=False, strength=strength,
    )
    ood_env = wl.QuadrantColorConfound(
        FullyObsWrapper(gym.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")),
        flip=True, strength=strength,
    )
    return train_env, ood_env


def run_one_grid_point(strength, seed, device):
    # Derive every seed0 offset from `seed` so a "seed" varies BOTH model
    # init and the actual training/eval trajectories, not just weight init
    # (see plan Step 6 "Seeds" -- this project's existing collect_trajectories
    # calls used hardcoded literal offsets, which wouldn't give genuine
    # multi-seed variance on their own).
    base = seed * 100_000
    seed0_train = base + 0
    seed0_probe = base + 50_000
    seed0_rollout = base + 10_000

    train_env, ood_env = make_envs(strength)

    train_data = wl.collect_trajectories(
        train_env, n_episodes=N_TRAIN_EPISODES, seed0=seed0_train,
        scripted_policy_prob=SCRIPTED_POLICY_PROB, scripted_noise_prob=SCRIPTED_NOISE_PROB)
    eval_data = wl.collect_trajectories(
        train_env, n_episodes=N_PROBE_EPISODES, seed0=seed0_probe,
        scripted_policy_prob=SCRIPTED_POLICY_PROB, scripted_noise_prob=SCRIPTED_NOISE_PROB)

    cfg = wl.make_model_config(device=device)

    print(f"[strength={strength} seed={seed}] training baseline...")
    model, next_state_head = wl.train_baseline_model(cfg, train_data, n_epochs=N_EPOCHS, seed=seed, log_every=25)

    in_dist = wl.evaluate_rollout_accuracy_dual(
        model, next_state_head, train_env, cfg, n_episodes=N_BASELINE_EVAL_EPISODES,
        scripted_policy_prob=SCRIPTED_POLICY_PROB, scripted_noise_prob=SCRIPTED_NOISE_PROB, seed0=seed0_rollout)
    ood = wl.evaluate_rollout_accuracy_dual(
        model, next_state_head, ood_env, cfg, n_episodes=N_BASELINE_EVAL_EPISODES,
        scripted_policy_prob=SCRIPTED_POLICY_PROB, scripted_noise_prob=SCRIPTED_NOISE_PROB, seed0=seed0_rollout)
    print(f"[strength={strength} seed={seed}] baseline in-dist={in_dist['exact_match_acc']:.4f} "
          f"OOD={ood['exact_match_acc']:.4f}")

    # --- Probing ---
    tokens = wl.build_input_tokens(eval_data).to(cfg.device)
    xy_positions = np.array([t["agent_pos"] for t in eval_data])
    color_labels = np.array([wl.quadrant_label(p) for p in xy_positions])

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: "resid_post" in n)
    acc_color, r2_xy, probes_B = wl.fit_probes_all_layers(cache, cfg, xy_positions, color_labels)
    L = int(np.argmax(acc_color))
    probe_B = probes_B[L]
    h_np = cache["resid_post", L][:, -1, :].detach().cpu().numpy()
    chance_baseline = wl.majority_class_baseline(color_labels)
    print(f"[strength={strength} seed={seed}] strongest color layer L={L} "
          f"(Probe B acc={acc_color[L]:.4f}, chance={chance_baseline:.4f})")

    # --- Single-direction ablation ---
    HOOK_NAME = f"blocks.{L}.hook_resid_post"
    color_direction = probe_B.coef_[0] / np.linalg.norm(probe_B.coef_[0])
    color_direction_t = torch.tensor(color_direction, dtype=torch.float32, device=cfg.device)
    single_hook = wl.make_multi_direction_hook([color_direction_t])

    ood_single = wl.evaluate_rollout_accuracy_with_hook(
        model, next_state_head, ood_env, HOOK_NAME, single_hook, cfg,
        n_episodes=N_ABLATION_EVAL_EPISODES, scripted_policy_prob=SCRIPTED_POLICY_PROB)
    indist_single = wl.evaluate_rollout_accuracy_with_hook(
        model, next_state_head, train_env, HOOK_NAME, single_hook, cfg,
        n_episodes=N_ABLATION_EVAL_EPISODES, scripted_policy_prob=SCRIPTED_POLICY_PROB)

    # --- Multi-direction ablation curve: greedy-probe vs. random-direction control ---
    random_rng = np.random.default_rng(seed + 999)
    greedy_rounds = wl.run_ablation_curve(
        model, next_state_head, ood_env, train_env, tokens, cfg, color_labels,
        get_next_direction=lambda probe_k, h_current: probe_k.coef_[0] / np.linalg.norm(probe_k.coef_[0]),
        label=f"greedy s={strength} seed={seed}", hook_name=HOOK_NAME, h_init=h_np,
        k_max=K_MAX, n_eval_episodes=N_ABLATION_EVAL_EPISODES, scripted_policy_prob=SCRIPTED_POLICY_PROB)
    random_rounds = wl.run_ablation_curve(
        model, next_state_head, ood_env, train_env, tokens, cfg, color_labels,
        get_next_direction=lambda probe_k, h_current: wl.random_unit_direction(h_current.shape[1], random_rng),
        label=f"random s={strength} seed={seed}", hook_name=HOOK_NAME, h_init=h_np,
        k_max=K_MAX, n_eval_episodes=N_ABLATION_EVAL_EPISODES, scripted_policy_prob=SCRIPTED_POLICY_PROB)

    # --- Method B: one adversarial run at ADV_LAMBD ---
    print(f"[strength={strength} seed={seed}] training adversarial (lambd={ADV_LAMBD})...")
    model_adv, head_adv, _ = wl.train_adversarial_model(
        cfg, train_data, lambd=ADV_LAMBD, resid_layer=L, n_epochs=N_EPOCHS, seed=seed, log_every=25)
    in_dist_adv = wl.evaluate_rollout_accuracy_dual(
        model_adv, head_adv, train_env, cfg, n_episodes=N_ABLATION_EVAL_EPISODES,
        scripted_policy_prob=SCRIPTED_POLICY_PROB, seed0=seed0_rollout)
    ood_adv = wl.evaluate_rollout_accuracy_dual(
        model_adv, head_adv, ood_env, cfg, n_episodes=N_ABLATION_EVAL_EPISODES,
        scripted_policy_prob=SCRIPTED_POLICY_PROB, seed0=seed0_rollout)
    with torch.no_grad():
        _, cache_adv = model_adv.run_with_cache(tokens, names_filter=lambda n: "resid_post" in n)
    h_adv = cache_adv["resid_post", L][:, -1, :].detach().cpu().numpy()
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    Xtr, Xte, ctr, cte = train_test_split(h_adv, color_labels, test_size=0.2, random_state=0)
    adv_probe_acc = accuracy_score(cte, LogisticRegression(max_iter=1000).fit(Xtr, ctr).predict(Xte))

    return {
        "strength": strength,
        "seed": seed,
        "strongest_layer_L": L,
        "probe_B_majority_class_baseline": chance_baseline,
        "baseline": {"in_dist": in_dist, "ood": ood, "probe_B_acc": acc_color[L], "probe_A_r2": r2_xy[L]},
        "single_direction_ablation": {
            "ood": ood_single, "in_dist_control": indist_single,
        },
        "multi_direction_ablation": {"k_max": K_MAX, "greedy_rounds": greedy_rounds, "random_rounds": random_rounds},
        "adversarial": {
            "lambd": ADV_LAMBD, "in_dist": in_dist_adv, "ood": ood_adv, "probe_B_acc": adv_probe_acc,
        },
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    try:
        with open(OUT_PATH) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

    done = {(r["strength"], r["seed"]) for r in results}

    grid = [(s, seed) for s in STRENGTHS for seed in SEEDS]
    print(f"Grid: {len(grid)} points ({len(STRENGTHS)} strengths x {len(SEEDS)} seeds), "
          f"{len(done)} already completed")

    for strength, seed in grid:
        if (strength, seed) in done:
            print(f"skipping strength={strength} seed={seed} (already in {OUT_PATH})")
            continue

        t0 = time.time()
        print(f"\n=== strength={strength} seed={seed} ===")
        result = run_one_grid_point(strength, seed, device)
        result["elapsed_sec"] = time.time() - t0
        result["timestamp"] = datetime.datetime.now().isoformat()
        results.append(result)

        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"=== strength={strength} seed={seed} done in {result['elapsed_sec']:.0f}s "
              f"-- checkpointed to {OUT_PATH} ===")

    print(f"\nSweep complete: {len(results)} grid points written to {OUT_PATH}")


if __name__ == "__main__":
    main()
