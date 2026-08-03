"""
Shared world-model-interpretability infrastructure for the DoorKey-6x6
quadrant-color-confound project (see findings.md).

Extracted from WMIP.ipynb so that a second driver -- run_confound_sweep.py,
plus any future experiment notebook -- can call the same tokenization,
data-collection, training, probing, ablation, and path-patching code instead
of copy-pasting notebook cells. WMIP.ipynb imports from this module for all
of the functions below; only narrative markdown, top-level driver calls, and
plotting stay in the notebook.

Colab users: upload this file alongside WMIP.ipynb (two-file upload) --
see README.md.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset

import gymnasium as gym
from collections import deque, Counter

from transformer_lens import HookedTransformer, HookedTransformerConfig

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


# ---------------------------------------------------------------------------
# Tokenization constants (fixed for MiniGrid-DoorKey-6x6-v0)
# ---------------------------------------------------------------------------

N_OBJECTS = 11        # len(IDX_TO_OBJECT)
N_COLOR_SLOTS = 6     # 6 real MiniGrid colors
N_STATES = 3           # 0=open/unused, 1=closed, 2=locked -- max observed state value + 1
N_ACTIONS = 7           # env.action_space.n
GRID_CELLS = 36         # 6x6 grid, flattened

CELL_VOCAB_SIZE = N_OBJECTS * N_COLOR_SLOTS * N_STATES
ACTION_TOKEN_OFFSET = CELL_VOCAB_SIZE   # actions start right after all possible cell tokens
VOCAB_SIZE = CELL_VOCAB_SIZE + N_ACTIONS


def cell_to_token(obj, color, state):
    """Flatten one (object, color, state) triple into a single integer token."""
    return (obj * N_COLOR_SLOTS * N_STATES) + (color * N_STATES) + state


def action_to_token(action):
    return ACTION_TOKEN_OFFSET + action


def flatten_grid_to_tokens(grid):
    """(H, W, 3) grid -> flat list of cell tokens, one per cell."""
    H, W, _ = grid.shape
    tokens = []
    for y in range(H):
        for x in range(W):
            obj, color, state = grid[y, x]
            tokens.append(cell_to_token(int(obj), int(color), int(state)))
    return tokens


def build_input_tokens(data):
    seqs = [flatten_grid_to_tokens(t["obs"]) + [action_to_token(t["action"])] for t in data]
    return torch.tensor(seqs, dtype=torch.long)


def build_token_dataset(data):
    input_seqs, target_seqs = [], []
    for transition in data:
        obs_tokens = flatten_grid_to_tokens(transition["obs"])
        act_token = action_to_token(transition["action"])
        next_obs_tokens = flatten_grid_to_tokens(transition["next_obs"])
        input_seqs.append(obs_tokens + [act_token])
        target_seqs.append(next_obs_tokens)
    input_ids = torch.tensor(input_seqs, dtype=torch.long)
    target_ids = torch.tensor(target_seqs, dtype=torch.long)
    return TensorDataset(input_ids, target_ids)


def build_token_dataset_with_labels(data):
    input_seqs, target_seqs, color_seqs = [], [], []
    for transition in data:
        obs_tokens = flatten_grid_to_tokens(transition["obs"])
        act_token = action_to_token(transition["action"])
        next_obs_tokens = flatten_grid_to_tokens(transition["next_obs"])
        input_seqs.append(obs_tokens + [act_token])
        target_seqs.append(next_obs_tokens)
        color_seqs.append(quadrant_label(transition["agent_pos"]))
    input_ids = torch.tensor(input_seqs, dtype=torch.long)
    target_ids = torch.tensor(target_seqs, dtype=torch.long)
    color_labels_t = torch.tensor(color_seqs, dtype=torch.long)
    return TensorDataset(input_ids, target_ids, color_labels_t)


def quadrant_label(pos, W=6, H=6):
    x, y = pos
    if x < W // 2 and y < H // 2:
        return 0   # top_left
    elif x >= W // 2 and y >= H // 2:
        return 1   # bottom_right
    else:
        return 2   # neutral / dead zone


# ---------------------------------------------------------------------------
# Env: quadrant-color confound, with a tunable correlation strength
# ---------------------------------------------------------------------------

class QuadrantColorConfound(gym.ObservationWrapper):
    """
    Paints a border strip of the observation with a color that is a
    (by default deterministic) function of the agent's quadrant.
    flip=False reproduces the training scheme; flip=True is the OOD test
    condition.

    strength (default 1.0, fully backward-compatible): the probability,
    drawn fresh once per episode, that this episode's border color
    actually follows the quadrant rule at all. With probability
    (1 - strength), the episode instead gets a color drawn independently
    of the true quadrant (uncorrelated noise) -- see findings.md Phase 2
    ("confound-strength sweep") for why this is a per-episode draw, not a
    per-timestep one. strength=1.0 is exactly the original behavior (the
    coin-flip branch is unreachable, since draws are in [0, 1)).

    seed, if given, reseeds the internal RNG on every reset(seed=...) call
    that itself passes a seed -- this makes the per-episode
    confound-following draw reproducible in exactly the way the rest of
    this project's data-collection seeding already works (seed0 + episode
    index), rather than depending on wrapper-construction-time state.
    """

    RED, BLUE, NEUTRAL = 0, 2, 5

    def __init__(self, env, flip=False, strength=1.0, seed=None):
        super().__init__(env)
        self.flip = flip
        self.strength = strength
        self._rng = np.random.default_rng(seed)
        self._confound_following = True

    def reset(self, *, seed=None, **kwargs):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._confound_following = self._rng.random() < self.strength
        return super().reset(seed=seed, **kwargs)

    def observation(self, obs):
        grid = obs["image"].copy()
        H, W, _ = grid.shape
        ax, ay = self.env.unwrapped.agent_pos
        top_left = ax < W // 2 and ay < H // 2
        bottom_right = ax >= W // 2 and ay >= H // 2

        color = self.NEUTRAL
        if top_left:
            color = self.BLUE if self.flip else self.RED
        elif bottom_right:
            color = self.RED if self.flip else self.BLUE

        if not self._confound_following and (top_left or bottom_right):
            color = self._rng.choice([self.RED, self.BLUE])   # confound destroyed this episode

        grid[0, :, 1] = color   # overwrite the color channel of row 0
        obs["image"] = grid
        return obs


# ---------------------------------------------------------------------------
# Scripted (BFS) policy, mixed with random actions -- see findings.md Sec 8
# ---------------------------------------------------------------------------

DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}   # 0=right,1=down,2=left,3=up


def _dir_to(delta):
    for d, vec in DIR_VEC.items():
        if vec == delta:
            return d
    return None


def _find_all(grid, obj_type):
    """All (x, y) grid coordinates holding an object of the given type."""
    return [(x, y) for x in range(grid.width) for y in range(grid.height)
            if (cell := grid.get(x, y)) is not None and cell.type == obj_type]


def _bfs_path(grid, start, is_passable, target):
    """Shortest 4-connected grid path from start to target, or None if unreachable."""
    if start == target:
        return [start]
    visited = {start}
    parent = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == target:
            break
        cx, cy = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cx + dx, cy + dy)
            if not (0 <= nxt[0] < grid.width and 0 <= nxt[1] < grid.height) or nxt in visited:
                continue
            if not is_passable(nxt):
                continue
            visited.add(nxt)
            parent[nxt] = cur
            q.append(nxt)
    if target not in visited:
        return None
    path = [target]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def scripted_action(unwrapped_env, phase):
    """One step of a 3-phase (key -> door -> goal) greedy BFS planner.

    Returns (action, next_phase). action is None when the planner has
    nothing useful to propose -- the caller should fall back to a random
    action in that case.
    """
    grid = unwrapped_env.grid
    ax, ay = unwrapped_env.agent_pos
    agent_dir = unwrapped_env.agent_dir

    if phase == "key" and unwrapped_env.carrying is not None:
        phase = "door"
    if phase == "door":
        doors = _find_all(grid, "door")
        if doors and grid.get(*doors[0]).is_open:
            phase = "goal"

    if phase == "key":
        targets = _find_all(grid, "key")
        interact_action = 3   # pickup
    elif phase == "door":
        targets = _find_all(grid, "door")
        interact_action = 5   # toggle
    elif phase == "goal":
        targets = _find_all(grid, "goal")
        interact_action = None   # just walk onto it
    else:
        return None, phase

    if not targets:
        return None, phase
    target = targets[0]

    def passable(cell_xy):
        if cell_xy == (ax, ay):
            return True
        cell = grid.get(*cell_xy)
        if cell is None:
            return True
        if cell_xy == target and interact_action is not None:
            return False   # can't walk onto the key/door cell itself -- must stand adjacent
        return cell.can_overlap()

    if interact_action is None:
        bfs_target_candidates = [target]
    else:
        tx, ty = target
        bfs_target_candidates = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cx, cy = tx + dx, ty + dy
            if 0 <= cx < grid.width and 0 <= cy < grid.height:
                c = grid.get(cx, cy)
                if c is None or c.can_overlap():
                    bfs_target_candidates.append((cx, cy))
        if not bfs_target_candidates:
            return None, phase

    best_path = None
    for cand in bfs_target_candidates:
        p = _bfs_path(grid, (ax, ay), passable, cand)
        if p is not None and (best_path is None or len(p) < len(best_path)):
            best_path = p
    if best_path is None:
        return None, phase

    if len(best_path) == 1:
        if interact_action is None:
            return None, phase   # already standing on the goal cell
        fx, fy = target
        desired_dir = _dir_to((fx - ax, fy - ay))
        if desired_dir is None:
            return None, phase
        if agent_dir == desired_dir:
            return interact_action, phase
        diff = (desired_dir - agent_dir) % 4
        return (0 if diff == 3 else 1), phase   # 0=turn left, 1=turn right (shorter turn)

    nxt = best_path[1]
    desired_dir = _dir_to((nxt[0] - ax, nxt[1] - ay))
    if agent_dir == desired_dir:
        return 2, phase   # forward
    diff = (desired_dir - agent_dir) % 4
    return (0 if diff == 3 else 1), phase


def choose_action(env, rng, use_scripted, phase, scripted_noise_prob=0.1):
    """Wraps scripted_action with noise-injection and random fallback --
    written once, reused by data collection and every eval function."""
    if use_scripted and rng.random() > scripted_noise_prob:
        action, phase = scripted_action(env.unwrapped, phase)
        if action is not None:
            return action, phase
    return int(rng.integers(env.action_space.n)), phase


def collect_trajectories(env, n_episodes=500, max_steps=50, seed0=0,
                          scripted_policy_prob=0.5, scripted_noise_prob=0.1):
    data = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed0 + ep)
        rng = np.random.default_rng(seed0 + ep)   # decoupled from gym's own RNG, for reproducibility
        use_scripted = rng.random() < scripted_policy_prob
        phase = "key"
        for t in range(max_steps):
            pos_before = env.unwrapped.agent_pos
            action, phase = choose_action(env, rng, use_scripted, phase, scripted_noise_prob)
            next_obs, reward, terminated, truncated, info = env.step(action)
            pos_after = env.unwrapped.agent_pos
            data.append({
                "obs": obs["image"] if isinstance(obs, dict) else obs,
                "action": action,
                "next_obs": next_obs["image"] if isinstance(next_obs, dict) else next_obs,
                "agent_pos": pos_before,
                "next_agent_pos": pos_after,
            })
            obs = next_obs
            if terminated or truncated:
                break
    return data


def event_coverage(env, policy_scripted_prob, n_episodes=200, seed0=90_000, scripted_noise_prob=0.1):
    """Instrumentation used to diagnose the random-policy exploration
    bottleneck (findings.md Sec 8) and to confirm the scripted-policy mix
    actually fixes it."""
    key_c, door_c, goal_c = 0, 0, 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed0 + ep)
        rng = np.random.default_rng(seed0 + ep)
        use_scripted = rng.random() < policy_scripted_prob
        phase = "key"
        unwrapped = env.unwrapped
        key_seen = door_seen = goal_seen = False
        for t in range(50):
            action, phase = choose_action(env, rng, use_scripted, phase, scripted_noise_prob)
            obs, reward, terminated, truncated, info = env.step(action)
            if unwrapped.carrying is not None:
                key_seen = True
            doors = _find_all(unwrapped.grid, "door")
            if doors and unwrapped.grid.get(*doors[0]).is_open:
                door_seen = True
            if terminated and reward > 0:
                goal_seen = True
            if terminated or truncated:
                break
        key_c += int(key_seen); door_c += int(door_seen); goal_c += int(goal_seen)
    return {
        "key_pickup_rate": key_c / n_episodes,
        "door_open_rate": door_c / n_episodes,
        "goal_reach_rate": goal_c / n_episodes,
    }


# ---------------------------------------------------------------------------
# Model config, training, prediction head
# ---------------------------------------------------------------------------

def make_model_config(device, n_layers=2, d_model=128, n_ctx=48, d_head=32,
                       n_heads=4, d_mlp=512, d_vocab=VOCAB_SIZE):
    # MPS deliberately excluded by callers (see WMIP.ipynb / findings.md):
    # TransformerLens warns MPS can silently produce incorrect results on
    # some ops, unacceptable for a project whose conclusions hinge on exact
    # activation math.
    return HookedTransformerConfig(
        n_layers=n_layers, d_model=d_model, n_ctx=n_ctx, d_head=d_head,
        n_heads=n_heads, d_mlp=d_mlp, d_vocab=d_vocab,
        act_fn="gelu", normalization_type="LN", device=device,
    )


def last_layer_resid_name(cfg):
    return f"blocks.{cfg.n_layers - 1}.hook_resid_post"


def predict_next_grid_logits(model, head, input_ids, cfg):
    """input_ids: (batch, 37) -> logits: (batch, GRID_CELLS, CELL_VOCAB_SIZE)"""
    resid_name = last_layer_resid_name(cfg)
    _, cache = model.run_with_cache(input_ids, names_filter=resid_name)
    h_last = cache[resid_name][:, -1, :]   # (batch, d_model) -- action-informed
    return head(h_last).view(-1, GRID_CELLS, CELL_VOCAB_SIZE)


def train_baseline_model(cfg, train_data, n_epochs=100, seed=0, batch_size=64, lr=3e-4, log_every=10):
    """Freshly initializes a world model + next_state_head and trains on
    train_data. Returns (model, next_state_head)."""
    torch.manual_seed(seed)
    model = HookedTransformer(cfg)
    next_state_head = nn.Linear(cfg.d_model, GRID_CELLS * CELL_VOCAB_SIZE).to(cfg.device)

    train_ds = build_token_dataset(train_data)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(list(model.parameters()) + list(next_state_head.parameters()), lr=lr)

    for epoch in range(n_epochs):
        total_loss = 0.0
        for input_ids, target_ids in loader:
            input_ids, target_ids = input_ids.to(cfg.device), target_ids.to(cfg.device)
            pred_logits = predict_next_grid_logits(model, next_state_head, input_ids, cfg)
            loss = torch.nn.functional.cross_entropy(
                pred_logits.reshape(-1, CELL_VOCAB_SIZE), target_ids.reshape(-1)
            )
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch}: loss {total_loss / len(loader):.4f}")

    return model, next_state_head


# ---------------------------------------------------------------------------
# Rollout evaluation
# ---------------------------------------------------------------------------

def init_history(obs):
    return obs["image"] if isinstance(obs, dict) else obs


def update_history(history, action, next_obs):
    return next_obs["image"] if isinstance(next_obs, dict) else next_obs


def true_next_tokens(next_obs):
    next_grid = next_obs["image"] if isinstance(next_obs, dict) else next_obs
    return np.array(flatten_grid_to_tokens(next_grid))


def model_predict_next(model, head, history, action, cfg):
    input_tokens = flatten_grid_to_tokens(history) + [action_to_token(action)]
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=cfg.device)
    with torch.no_grad():
        pred_logits = predict_next_grid_logits(model, head, input_ids, cfg)
    return pred_logits[0].argmax(dim=-1).cpu().numpy()


def evaluate_rollout_accuracy_dual(model, head, env, cfg, n_episodes=500,
                                    scripted_policy_prob=0.5, scripted_noise_prob=0.1,
                                    max_steps=30, seed0=10_000):
    exact_correct, exact_total = 0, 0
    cell_correct, cell_total = 0, 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed0 + ep)
        rng = np.random.default_rng(seed0 + ep)
        use_scripted = rng.random() < scripted_policy_prob
        phase = "key"
        history = init_history(obs)
        for t in range(max_steps):
            action, phase = choose_action(env, rng, use_scripted, phase, scripted_noise_prob)
            pred_tokens = model_predict_next(model, head, history, action, cfg)
            next_obs, reward, terminated, truncated, info = env.step(action)
            true_tokens = true_next_tokens(next_obs)

            exact_correct += int(np.array_equal(pred_tokens, true_tokens))
            exact_total += 1
            cell_correct += int((pred_tokens == true_tokens).sum())
            cell_total += len(true_tokens)

            history = update_history(history, action, next_obs)
            if terminated or truncated:
                break

    return {
        "exact_match_acc": exact_correct / exact_total,
        "per_cell_acc": cell_correct / cell_total,
    }


def evaluate_rollout_accuracy_with_hook(model, head, env, hook_name, hook_fn, cfg,
                                        n_episodes=500, scripted_policy_prob=0.5,
                                        scripted_noise_prob=0.1, max_steps=30, seed0=10_000):
    exact_correct, exact_total = 0, 0
    cell_correct, cell_total = 0, 0
    resid_name = last_layer_resid_name(cfg)

    captured = {}
    def capture_last_resid(resid, hook):
        captured["h"] = resid[:, -1, :]
        return resid

    hooks = [(hook_name, hook_fn), (resid_name, capture_last_resid)]

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed0 + ep)
        rng = np.random.default_rng(seed0 + ep)
        use_scripted = rng.random() < scripted_policy_prob
        phase = "key"
        history = init_history(obs)
        for t in range(max_steps):
            action, phase = choose_action(env, rng, use_scripted, phase, scripted_noise_prob)

            input_tokens = flatten_grid_to_tokens(history) + [action_to_token(action)]
            input_ids = torch.tensor([input_tokens], dtype=torch.long, device=cfg.device)
            with torch.no_grad():
                model.run_with_hooks(input_ids, fwd_hooks=hooks)
                pred_logits = head(captured["h"]).view(1, GRID_CELLS, CELL_VOCAB_SIZE)
            pred_tokens = pred_logits[0].argmax(dim=-1).cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(action)
            true_tokens = true_next_tokens(next_obs)

            exact_correct += int(np.array_equal(pred_tokens, true_tokens))
            exact_total += 1
            cell_correct += int((pred_tokens == true_tokens).sum())
            cell_total += len(true_tokens)

            history = update_history(history, action, next_obs)
            if terminated or truncated:
                break

    return {
        "exact_match_acc": exact_correct / exact_total,
        "per_cell_acc": cell_correct / cell_total,
    }


def eval_with_hooks_static(model, next_state_head, cfg, tokens_batch, target_batch, extra_hooks):
    """Single-pass (no env rollout) accuracy check -- used by path patching.
    Deliberately not chunked (see findings.md Sec 9 batch/shape-mismatch bug):
    a patch hook closes over a fixed-size source activation tensor, so
    chunking here would require re-slicing that source per chunk too --
    this model is tiny enough that a single unbatched CPU forward pass is
    simpler and fine."""
    resid_name = last_layer_resid_name(cfg)
    captured = {}
    def capture_last(resid, hook):
        captured["h"] = resid[:, -1, :]
        return resid
    with torch.no_grad():
        model.run_with_hooks(tokens_batch, fwd_hooks=list(extra_hooks) + [(resid_name, capture_last)])
        pred_logits = next_state_head(captured["h"]).view(-1, GRID_CELLS, CELL_VOCAB_SIZE)
    pred_tokens = pred_logits.argmax(dim=-1)
    exact_match = (pred_tokens == target_batch).all(dim=-1).float().mean().item()
    per_cell = (pred_tokens == target_batch).float().mean().item()
    return exact_match, per_cell


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def fit_probes_all_layers(cache, cfg, xy_positions, color_labels):
    """Fits Probe A (Ridge, true (x,y) position) and Probe B (LogisticRegression,
    background color) at every layer's final-position residual stream.
    Returns (acc_color, r2_xy, probes_B) -- one entry per layer, plus the
    fitted Probe B objects (needed later to derive ablation directions)."""
    acc_color, r2_xy, probes_B = [], [], []
    for Li in range(cfg.n_layers):
        h_L = cache["resid_post", Li][:, -1, :].detach().cpu().numpy()

        Xtr, Xte, ytr, yte = train_test_split(h_L, xy_positions, test_size=0.2, random_state=0)
        r2_xy.append(r2_score(yte, Ridge().fit(Xtr, ytr).predict(Xte)))

        Xtr, Xte, ctr, cte = train_test_split(h_L, color_labels, test_size=0.2, random_state=0)
        probe_Li = LogisticRegression(max_iter=1000).fit(Xtr, ctr)
        acc_color.append(accuracy_score(cte, probe_Li.predict(Xte)))
        probes_B.append(probe_Li)

    return acc_color, r2_xy, probes_B


def majority_class_baseline(color_labels):
    counts = Counter(color_labels.tolist())
    return max(counts.values()) / len(color_labels)


# ---------------------------------------------------------------------------
# Ablation (Method A)
# ---------------------------------------------------------------------------

def make_multi_direction_hook(directions):
    """Projects out several unit directions at once from the residual
    stream. An empty list is a no-op (identity) hook -- a single-direction
    ablation is just make_multi_direction_hook([direction])."""
    def ablation_hook(resid, hook):
        out = resid
        for d in directions:
            proj = (out @ d).unsqueeze(-1) * d
            out = out - proj
        return out
    return ablation_hook


def run_with_ablation_and_capture(model, tokens, hook_name, hook_fn):
    """Runs the model with hook_fn attached at hook_name and returns the
    post-hook final-position residual stream (used to re-probe for the
    next round of ablation)."""
    captured = {}
    def combined_hook(resid, hook):
        out = hook_fn(resid, hook)
        captured["h"] = out[:, -1, :].detach()
        return out
    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=[(hook_name, combined_hook)])
    return captured["h"].cpu().numpy()


def random_unit_direction(d_model, rng):
    """A random direction in activation space, normalized like a probe
    direction -- the control condition for the ablation loop below."""
    v = rng.normal(size=d_model)
    return v / np.linalg.norm(v)


def run_ablation_curve(model, next_state_head, ood_env, train_env, tokens, cfg,
                        color_labels, get_next_direction, label, hook_name, h_init,
                        k_max=5, n_eval_episodes=200, scripted_policy_prob=0.5):
    """Shared driver for both the greedy (probe-derived) and random-direction
    control ablation curves -- same evaluation, same k_max, only the source
    of each round's direction differs.

    get_next_direction(probe_k, h_current) -> direction (np.ndarray, d_model,)
    """
    directions = []
    rounds = []
    h_current = h_init

    for k in range(0, k_max + 1):
        Xtr, Xte, ctr, cte = train_test_split(h_current, color_labels, test_size=0.2, random_state=0)
        probe_k = LogisticRegression(max_iter=1000).fit(Xtr, ctr)
        acc_k = accuracy_score(cte, probe_k.predict(Xte))

        hook_fn = make_multi_direction_hook(directions)
        ood_k = evaluate_rollout_accuracy_with_hook(
            model, next_state_head, ood_env, hook_name, hook_fn, cfg,
            n_episodes=n_eval_episodes, scripted_policy_prob=scripted_policy_prob)
        indist_k = evaluate_rollout_accuracy_with_hook(
            model, next_state_head, train_env, hook_name, hook_fn, cfg,
            n_episodes=n_eval_episodes, scripted_policy_prob=scripted_policy_prob)

        rounds.append({
            "k": k,
            "probe_B_acc": acc_k,
            "ood_exact_match": ood_k["exact_match_acc"],
            "indist_exact_match": indist_k["exact_match_acc"],
            "ood_per_cell": ood_k["per_cell_acc"],
            "indist_per_cell": indist_k["per_cell_acc"],
        })
        print(f"[{label}] k={k}: Probe B acc={acc_k:.4f}  OOD exact={ood_k['exact_match_acc']:.4f} "
              f"(per-cell {ood_k['per_cell_acc']:.4f})  In-dist exact={indist_k['exact_match_acc']:.4f} "
              f"(per-cell {indist_k['per_cell_acc']:.4f})")

        if k == k_max:
            break

        d_k = get_next_direction(probe_k, h_current)
        directions.append(torch.tensor(d_k, dtype=torch.float32, device=cfg.device))

        refreshed_hook = make_multi_direction_hook(directions)
        h_current = run_with_ablation_and_capture(model, tokens.to(cfg.device), hook_name, refreshed_hook)

    return rounds


# ---------------------------------------------------------------------------
# Path patching
# ---------------------------------------------------------------------------

def make_head_patch_hook(head_idx, source_z):
    """Patches one attention head's z-output (pre-projection) with the value
    it took on a different (clean) run, leaving every other head untouched."""
    def patch_hook_fn(z, hook):
        z = z.clone()
        z[:, :, head_idx, :] = source_z[:, :, head_idx, :]
        return z
    return patch_hook_fn


def make_layer_patch_hook(source_z):
    """Patches ALL heads' z-output at a layer at once (whole-layer patch),
    unlike make_head_patch_hook above which patches a single head."""
    def patch_hook_fn(z, hook):
        return source_z
    return patch_hook_fn


# ---------------------------------------------------------------------------
# Method B: adversarial (gradient-reversal) regularization
# ---------------------------------------------------------------------------

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


N_COLOR_CLASSES = 3   # top_left, bottom_right, neutral -- matches quadrant_label()


def train_adversarial_model(cfg, train_data, lambd, resid_layer, n_epochs=100, seed=0,
                             batch_size=64, lr=3e-4, adv_lr=1e-3, log_every=10):
    """Freshly initializes a world model + adversary and trains with a
    gradient-reversal penalty of strength lambd against color-decodability
    at resid_layer. Returns (model_adv, next_state_head_adv, color_adversary).

    IMPORTANT: uses a plain forward hook (model_adv.hooks(...)) to capture
    the residual stream, NOT run_with_cache -- run_with_cache unconditionally
    .detach()es every cached tensor (TransformerLens's HookedRootModule),
    which silently severs the autograd graph and makes lambd a no-op. See
    findings.md Sec 9 for the full bug history."""
    torch.manual_seed(seed)
    model_adv = HookedTransformer(cfg)
    next_state_head_adv = nn.Linear(cfg.d_model, GRID_CELLS * CELL_VOCAB_SIZE).to(cfg.device)
    color_adversary = nn.Linear(cfg.d_model, N_COLOR_CLASSES).to(cfg.device)
    opt_adv = torch.optim.AdamW(list(model_adv.parameters()) + list(next_state_head_adv.parameters()), lr=lr)
    adv_opt = torch.optim.AdamW(color_adversary.parameters(), lr=adv_lr)

    resid_name = f"blocks.{resid_layer}.hook_resid_post"
    last_resid_name = last_layer_resid_name(cfg)

    train_ds_adv = build_token_dataset_with_labels(train_data)
    loader_adv = DataLoader(train_ds_adv, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        total_task_loss, total_adv_loss = 0.0, 0.0
        for input_ids, target_ids, color_labels_batch in loader_adv:
            input_ids = input_ids.to(cfg.device)
            target_ids = target_ids.to(cfg.device)
            color_labels_batch = color_labels_batch.to(cfg.device)

            captured = {}
            def capture_hook(resid, hook):
                captured[hook.name] = resid
                return resid

            with model_adv.hooks(fwd_hooks=[(resid_name, capture_hook), (last_resid_name, capture_hook)]):
                model_adv(input_ids)

            h_color = captured[resid_name][:, -1, :]
            h_last = captured[last_resid_name][:, -1, :]

            pred_logits = next_state_head_adv(h_last).view(-1, GRID_CELLS, CELL_VOCAB_SIZE)
            task_loss = torch.nn.functional.cross_entropy(
                pred_logits.reshape(-1, CELL_VOCAB_SIZE), target_ids.reshape(-1))
            adv_logits = color_adversary(grad_reverse(h_color, lambd=lambd))
            adv_loss = torch.nn.functional.cross_entropy(adv_logits, color_labels_batch)

            loss = task_loss + adv_loss
            opt_adv.zero_grad(); adv_opt.zero_grad()
            loss.backward()
            opt_adv.step(); adv_opt.step()

            total_task_loss += task_loss.item()
            total_adv_loss += adv_loss.item()

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"  lambd={lambd} epoch {epoch}: task_loss {total_task_loss/len(loader_adv):.4f}  "
                  f"adv_loss {total_adv_loss/len(loader_adv):.4f}")

    return model_adv, next_state_head_adv, color_adversary
