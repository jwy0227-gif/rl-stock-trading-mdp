import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None


ACTION_NAMES = ("buy", "hold", "sell")
BUY, HOLD, SELL = 0, 1, 2
TIE_PRIORITY = (HOLD, BUY, SELL)


@dataclass(frozen=True)
class StockTradingMDP:
    horizon: int = 20
    s0: float = 100.0
    u: float = 1.1
    d: float = 0.9
    p: float = 0.5
    gamma: float = 1.0

    def states(self):
        return [(t, k, h) for t in range(self.horizon + 1) for k in range(t + 1) for h in (0, 1)]

    def nonterminal_states(self):
        return [(t, k, h) for t in range(self.horizon) for k in range(t + 1) for h in (0, 1)]

    def valid_actions(self, state):
        _, _, h = state
        if h == 0:
            return (HOLD, BUY)
        return (HOLD, SELL)

    def next_holding(self, h, action):
        if action == BUY:
            return 1
        if action == SELL:
            return 0
        return h

    def transitions(self, state, action):
        t, k, h = state
        if t >= self.horizon:
            return ()

        new_h = self.next_holding(h, action)
        up_reward = math.log(self.u) if new_h == 1 else 0.0
        down_reward = math.log(self.d) if new_h == 1 else 0.0
        return (
            (self.p, (t + 1, k + 1, new_h), up_reward),
            (1.0 - self.p, (t + 1, k, new_h), down_reward),
        )

    def initial_state(self):
        return (0, 0, 0)

    def price(self, t, k):
        return self.s0 * (self.u ** k) * (self.d ** (t - k))


def make_state_index(env):
    states = env.states()
    return states, {s: i for i, s in enumerate(states)}


def tie_breaking_argmax(values, actions):
    best_value = max(values[a] for a in actions)
    tied = {a for a in actions if np.isclose(values[a], best_value, atol=1e-12)}
    for action in TIE_PRIORITY:
        if action in tied:
            return action
    return actions[0]


def value_iteration(env):
    states, state_to_idx = make_state_index(env)
    n_states = len(states)
    q = np.full((n_states, len(ACTION_NAMES)), np.nan)
    v = np.zeros(n_states)
    policy = np.full(n_states, -1, dtype=int)

    for t in range(env.horizon - 1, -1, -1):
        for k in range(t + 1):
            for h in (0, 1):
                state = (t, k, h)
                idx = state_to_idx[state]
                for action in env.valid_actions(state):
                    total = 0.0
                    for prob, next_state, reward in env.transitions(state, action):
                        total += prob * (reward + env.gamma * v[state_to_idx[next_state]])
                    q[idx, action] = total
                policy[idx] = tie_breaking_argmax(q[idx], env.valid_actions(state))
                v[idx] = q[idx, policy[idx]]

    return states, state_to_idx, q, v, policy


def epsilon_value(schedule, episode, total_episodes):
    kind = schedule["kind"]
    if kind == "constant":
        return schedule["eps"]
    if kind == "linear":
        frac = episode / max(1, total_episodes - 1)
        return schedule["start"] + frac * (schedule["end"] - schedule["start"])
    if kind == "exponential":
        rate = schedule["decay"]
        return schedule["end"] + (schedule["start"] - schedule["end"]) * math.exp(-rate * episode)
    raise ValueError(f"Unknown epsilon schedule: {kind}")


def alpha_value(schedule, visits):
    kind = schedule["kind"]
    if kind == "constant":
        return schedule["alpha"]
    if kind == "harmonic":
        return schedule["scale"] / (schedule["offset"] + visits)
    if kind == "sqrt":
        return schedule["scale"] / math.sqrt(schedule["offset"] + visits)
    raise ValueError(f"Unknown alpha schedule: {kind}")


def alpha_values(schedule, visit_counts):
    kind = schedule["kind"]
    visit_counts = visit_counts.astype(float)
    if kind == "constant":
        return np.full_like(visit_counts, schedule["alpha"], dtype=float)
    if kind == "harmonic":
        return schedule["scale"] / (schedule["offset"] + visit_counts)
    if kind == "sqrt":
        return schedule["scale"] / np.sqrt(schedule["offset"] + visit_counts)
    raise ValueError(f"Unknown alpha schedule: {kind}")


def choose_action(q_row, valid_actions, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.choice(valid_actions))
    return tie_breaking_argmax(q_row, valid_actions)


def sample_transition(env, state, action, rng):
    t, k, h = state
    new_h = env.next_holding(h, action)
    if rng.random() < env.p:
        reward = math.log(env.u) if new_h == 1 else 0.0
        return (t + 1, k + 1, new_h), reward
    reward = math.log(env.d) if new_h == 1 else 0.0
    return (t + 1, k, new_h), reward


def q_learning(env, state_to_idx, episodes, epsilon_schedule, alpha_schedule, seed, checkpoints):
    rng = np.random.default_rng(seed)
    n_states = len(state_to_idx)
    q = np.zeros((n_states, len(ACTION_NAMES)))
    visits = np.zeros((n_states, len(ACTION_NAMES)), dtype=np.int64)
    snapshots = {}

    checkpoint_set = set(checkpoints)
    for episode in range(1, episodes + 1):
        state = env.initial_state()
        eps = epsilon_value(epsilon_schedule, episode - 1, episodes)

        while state[0] < env.horizon:
            idx = state_to_idx[state]
            valid = env.valid_actions(state)
            action = choose_action(q[idx], valid, eps, rng)
            visits[idx, action] += 1

            next_state, reward = sample_transition(env, state, action, rng)
            next_idx = state_to_idx[next_state]
            if next_state[0] == env.horizon:
                target = reward
            else:
                next_valid = env.valid_actions(next_state)
                target = reward + env.gamma * max(q[next_idx, a] for a in next_valid)

            alpha = alpha_value(alpha_schedule, visits[idx, action])
            q[idx, action] += alpha * (target - q[idx, action])
            state = next_state

        if episode in checkpoint_set:
            snapshots[episode] = q.copy()

    return q, snapshots


def greedy_policy_from_q(env, states, q):
    policy = np.full(len(states), -1, dtype=int)
    v = np.zeros(len(states))
    for idx, state in enumerate(states):
        if state[0] == env.horizon:
            continue
        valid = env.valid_actions(state)
        policy[idx] = tie_breaking_argmax(q[idx], valid)
        v[idx] = q[idx, policy[idx]]
    return v, policy


def metrics(env, states, q_vi, v_vi, policy_vi, q_ql):
    nonterminal = [i for i, s in enumerate(states) if s[0] < env.horizon]
    valid_pairs = [(i, a) for i in nonterminal for a in env.valid_actions(states[i])]
    q_error = max(abs(q_vi[i, a] - q_ql[i, a]) for i, a in valid_pairs)
    v_ql, policy_ql = greedy_policy_from_q(env, states, q_ql)
    v_error = max(abs(v_vi[i] - v_ql[i]) for i in nonterminal)
    mismatch = sum(policy_vi[i] != policy_ql[i] for i in nonterminal) / len(nonterminal)
    return q_error, v_error, mismatch


def precompute_arrays(env, states, state_to_idx):
    n_states = len(states)
    valid_actions = np.full((n_states, 2), -1, dtype=np.int64)
    next_up = np.full((n_states, len(ACTION_NAMES)), -1, dtype=np.int64)
    next_down = np.full((n_states, len(ACTION_NAMES)), -1, dtype=np.int64)
    reward_up = np.zeros((n_states, len(ACTION_NAMES)))
    reward_down = np.zeros((n_states, len(ACTION_NAMES)))

    for idx, state in enumerate(states):
        actions = env.valid_actions(state) if state[0] < env.horizon else ()
        for j, action in enumerate(actions):
            valid_actions[idx, j] = action
            transitions = env.transitions(state, action)
            if transitions:
                _, up_state, up_reward = transitions[0]
                _, down_state, down_reward = transitions[1]
                next_up[idx, action] = state_to_idx[up_state]
                next_down[idx, action] = state_to_idx[down_state]
                reward_up[idx, action] = up_reward
                reward_down[idx, action] = down_reward
    return valid_actions, next_up, next_down, reward_up, reward_down


def q_learning_many(env, states, state_to_idx, episodes, epsilon_schedule, alpha_schedule, seeds, checkpoints):
    n_runs = len(seeds)
    n_states = len(states)
    q = np.zeros((n_runs, n_states, len(ACTION_NAMES)))
    visits = np.zeros((n_runs, n_states, len(ACTION_NAMES)), dtype=np.int64)
    rngs = [np.random.default_rng(seed) for seed in seeds]
    valid_actions, next_up, next_down, reward_up, reward_down = precompute_arrays(env, states, state_to_idx)
    state_time = np.array([s[0] for s in states])
    init_idx = state_to_idx[env.initial_state()]
    snapshots = {}
    run_idx = np.arange(n_runs)

    chunk_size = 5_000
    for chunk_start in range(1, episodes + 1, chunk_size):
        chunk_len = min(chunk_size, episodes - chunk_start + 1)
        explore_random = np.stack([rng.random((chunk_len, env.horizon)) for rng in rngs])
        action_random = np.stack([rng.integers(0, 2, size=(chunk_len, env.horizon)) for rng in rngs])
        up_random = np.stack([rng.random((chunk_len, env.horizon)) for rng in rngs])

        for local_ep in range(chunk_len):
            episode = chunk_start + local_ep
            state_idx = np.full(n_runs, init_idx, dtype=np.int64)
            eps = epsilon_value(epsilon_schedule, episode - 1, episodes)

            for step in range(env.horizon):
                valid = valid_actions[state_idx]
                q_valid = np.take_along_axis(q[run_idx, state_idx], valid, axis=1)
                greedy_pos = np.argmax(q_valid, axis=1)
                greedy_action = valid[run_idx, greedy_pos]

                explore = explore_random[:, local_ep, step] < eps
                random_action = valid[run_idx, action_random[:, local_ep, step]]
                action = np.where(explore, random_action, greedy_action)

                visits[run_idx, state_idx, action] += 1
                up_move = up_random[:, local_ep, step] < env.p
                next_idx = np.where(up_move, next_up[state_idx, action], next_down[state_idx, action])
                reward = np.where(up_move, reward_up[state_idx, action], reward_down[state_idx, action])

                next_valid = valid_actions[next_idx]
                q_next_valid = np.take_along_axis(q[run_idx, next_idx], next_valid, axis=1)
                max_next = np.max(q_next_valid, axis=1)
                terminal = state_time[next_idx] == env.horizon
                target = reward + np.where(terminal, 0.0, env.gamma * max_next)

                alpha = alpha_values(alpha_schedule, visits[run_idx, state_idx, action])
                q[run_idx, state_idx, action] += alpha * (target - q[run_idx, state_idx, action])
                state_idx = next_idx

            if episode in checkpoints:
                snapshots[episode] = q.copy()

    return q, snapshots


def experiment_grid():
    eps_schedules = {
        "eps_linear_1_to_001": {"kind": "linear", "start": 1.0, "end": 0.01},
        "eps_exp_slow": {"kind": "exponential", "start": 1.0, "end": 0.01, "decay": 3.0e-5},
        "eps_exp_fast": {"kind": "exponential", "start": 1.0, "end": 0.01, "decay": 2.0e-4},
    }
    alpha_schedules = {
        "alpha_const_01": {"kind": "constant", "alpha": 0.1},
        "alpha_sqrt": {"kind": "sqrt", "scale": 0.7, "offset": 1.0},
        "alpha_harmonic": {"kind": "harmonic", "scale": 1.0, "offset": 10.0},
    }
    return eps_schedules, alpha_schedules


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_line_svg(path, series_by_name, title, y_label, width=900, height=520):
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 55, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    xs = sorted({x for series in series_by_name.values() for x in series})
    ys = [y for series in series_by_name.values() for y in series.values()]
    y_min, y_max = min(ys), max(ys)
    if np.isclose(y_min, y_max):
        y_max = y_min + 1.0
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    def sx(x):
        return margin_left + (x - xs[0]) / (xs[-1] - xs[0]) * plot_w

    def sy(y):
        return margin_top + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="22">{svg_escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_h}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top+plot_h}" x2="{margin_left+plot_w}" y2="{margin_top+plot_h}" stroke="#333"/>',
        f'<text x="22" y="{margin_top+plot_h/2}" transform="rotate(-90 22 {margin_top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="14">{svg_escape(y_label)}</text>',
        f'<text x="{margin_left+plot_w/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="14">Episodes</text>',
    ]
    for t in np.linspace(y_min, y_max, 5):
        y = sy(t)
        parts.append(f'<line x1="{margin_left-5}" y1="{y:.1f}" x2="{margin_left+plot_w}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{t:.3f}</text>')
    for x in xs:
        px = sx(x)
        parts.append(f'<text x="{px:.1f}" y="{margin_top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="11">{x//1000}k</text>')
    for idx, (name, series) in enumerate(series_by_name.items()):
        color = colors[idx % len(colors)]
        points = " ".join(f"{sx(x):.1f},{sy(series[x]):.1f}" for x in xs if x in series)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
        ly = margin_top + 20 + idx * 20
        parts.append(f'<line x1="{width-250}" y1="{ly}" x2="{width-225}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width-218}" y="{ly+4}" font-family="Arial" font-size="12">{svg_escape(name)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_policy_svg(path, env, states, policy_a, policy_b=None, title="Policy"):
    cell = 34
    gap = 6
    label_w = 95
    width = label_w + (env.horizon + 1) * (cell + gap) + 30
    height = 90 + 2 * (env.horizon + 1) * (cell + gap)
    arrows = {BUY: "B", HOLD: "H", SELL: "S", -1: ""}
    colors = {BUY: "#4c78a8", HOLD: "#59a14f", SELL: "#e15759", -1: "#f7f7f7"}
    state_to_idx = {s: i for i, s in enumerate(states)}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20">{svg_escape(title)}</text>',
        '<text x="20" y="58" font-family="Arial" font-size="13">Each row is t; columns are k. Top block h=0 cash, bottom block h=1 stock.</text>',
    ]
    y0 = 78
    for block, h in enumerate((0, 1)):
        block_y = y0 + block * (env.horizon + 1) * (cell + gap)
        parts.append(f'<text x="15" y="{block_y+22}" font-family="Arial" font-size="14" font-weight="bold">h={h}</text>')
        for t in range(env.horizon + 1):
            y = block_y + t * (cell + gap)
            parts.append(f'<text x="{label_w-12}" y="{y+22}" text-anchor="end" font-family="Arial" font-size="11">t={t}</text>')
            for k in range(t + 1):
                idx = state_to_idx[(t, k, h)]
                action = int(policy_a[idx])
                x = label_w + k * (cell + gap)
                stroke = "#222"
                stroke_w = "1"
                if policy_b is not None and t < env.horizon and action != int(policy_b[idx]):
                    stroke = "#000"
                    stroke_w = "3"
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{colors[action]}" stroke="{stroke}" stroke-width="{stroke_w}"/>')
                parts.append(f'<text x="{x+cell/2}" y="{y+22}" text-anchor="middle" font-family="Arial" font-size="14" fill="white">{arrows[action]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_value_svg(path, env, states, values, title="Optimal value function"):
    cell = 34
    gap = 6
    label_w = 95
    width = label_w + (env.horizon + 1) * (cell + gap) + 30
    height = 90 + 2 * (env.horizon + 1) * (cell + gap)
    state_to_idx = {s: i for i, s in enumerate(states)}
    finite_vals = [values[state_to_idx[(t, k, h)]] for t in range(env.horizon) for k in range(t + 1) for h in (0, 1)]
    vmin, vmax = min(finite_vals), max(finite_vals)

    def color(v):
        z = 0.0 if np.isclose(vmin, vmax) else (v - vmin) / (vmax - vmin)
        r = int(49 + z * 180)
        g = int(105 + z * 80)
        b = int(160 - z * 110)
        return f"#{r:02x}{g:02x}{b:02x}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20">{svg_escape(title)}</text>',
        '<text x="20" y="58" font-family="Arial" font-size="13">Each row is t; columns are k. Top block h=0 cash, bottom block h=1 stock.</text>',
    ]
    y0 = 78
    for block, h in enumerate((0, 1)):
        block_y = y0 + block * (env.horizon + 1) * (cell + gap)
        parts.append(f'<text x="15" y="{block_y+22}" font-family="Arial" font-size="14" font-weight="bold">h={h}</text>')
        for t in range(env.horizon + 1):
            y = block_y + t * (cell + gap)
            parts.append(f'<text x="{label_w-12}" y="{y+22}" text-anchor="end" font-family="Arial" font-size="11">t={t}</text>')
            for k in range(t + 1):
                idx = state_to_idx[(t, k, h)]
                v = values[idx]
                x = label_w + k * (cell + gap)
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color(v)}" stroke="#222" stroke-width="1"/>')
                parts.append(f'<text x="{x+cell/2}" y="{y+21}" text-anchor="middle" font-family="Arial" font-size="8" fill="white">{v:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def font(size, bold=False):
    if ImageFont is None:
        return None
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
        # macOS / Linux fallbacks
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_center(draw, xy, text, fill, fnt):
    bbox = draw.textbbox((0, 0), text, font=fnt)
    x, y = xy
    draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2), text, fill=fill, font=fnt)


def report_canvas(width=1800, height=1100):
    return Image.new("RGB", (width, height), "white"), ImageDraw.Draw(Image.new("RGB", (1, 1)))


def value_color(v, vmin, vmax):
    if np.isclose(vmin, vmax):
        z = 0.5
    else:
        z = (v - vmin) / (vmax - vmin)
    # Teal -> light gray -> coral, readable in print.
    if z < 0.5:
        a = z / 0.5
        c0 = np.array([45, 120, 135])
        c1 = np.array([238, 241, 237])
    else:
        a = (z - 0.5) / 0.5
        c0 = np.array([238, 241, 237])
        c1 = np.array([198, 82, 80])
    c = (1 - a) * c0 + a * c1
    return tuple(int(x) for x in c)


def write_error_png(path, series_by_name, title="Mean max Q-error vs episodes", width=1800, height=1050):
    if Image is None:
        return
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(44, True)
    label_font = font(28, False)
    tick_font = font(24, False)
    legend_font = font(23, False)

    left, right, top, bottom = 150, 80, 120, 150
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = sorted({x for series in series_by_name.values() for x in series})
    ys = [y for series in series_by_name.values() for y in series.values()]
    y_min, y_max = min(ys), max(ys)
    y_pad = (y_max - y_min) * 0.12 if y_max > y_min else 0.01
    y_min = max(0, y_min - y_pad)
    y_max = y_max + y_pad
    x_min, x_max = math.log10(xs[0]), math.log10(xs[-1])

    def sx(x):
        return left + (math.log10(x) - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return top + (y_max - y) / (y_max - y_min) * plot_h

    draw.text((width / 2, 48), title, anchor="mm", fill="#1f2933", font=title_font)
    for i in range(6):
        yv = y_min + (y_max - y_min) * i / 5
        y = sy(yv)
        draw.line((left, y, left + plot_w, y), fill="#e5e7eb", width=2)
        draw.text((left - 16, y), f"{yv:.3f}", anchor="rm", fill="#4b5563", font=tick_font)
    draw.line((left, top, left, top + plot_h), fill="#111827", width=3)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#111827", width=3)
    for x in xs:
        px = sx(x)
        draw.line((px, top + plot_h, px, top + plot_h + 9), fill="#111827", width=2)
        draw.text((px, top + plot_h + 42), f"{x//1000}k", anchor="mm", fill="#4b5563", font=tick_font)
    draw.text((left + plot_w / 2, height - 55), "Episodes (log scale)", anchor="mm", fill="#111827", font=label_font)
    draw.text((48, top + plot_h / 2), "Q-error", anchor="mm", fill="#111827", font=label_font)

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]

    def x_label(x):
        return f"{x//1000}k" if x >= 1000 else str(x)

    def pretty_name(name):
        replacements = {
            "eps_linear_1_to_001": "linear eps",
            "eps_exp_fast": "fast exp eps",
            "eps_exp_slow": "slow exp eps",
            "alpha_harmonic": "harmonic alpha",
            "alpha_const_01": "constant alpha",
            "alpha_sqrt": "sqrt alpha",
        }
        label = name
        for old, new in replacements.items():
            label = label.replace(old, new)
        return label.replace("__", " + ")

    # Redraw x tick labels after helper definitions.
    draw.rectangle((left - 20, top + plot_h + 10, left + plot_w + 20, top + plot_h + 75), fill="white")
    for x in xs:
        px = sx(x)
        draw.line((px, top + plot_h, px, top + plot_h + 9), fill="#111827", width=2)
        draw.text((px, top + plot_h + 42), x_label(x), anchor="mm", fill="#4b5563", font=tick_font)

    for idx, (name, series) in enumerate(series_by_name.items()):
        color = colors[idx % len(colors)]
        pts = [(sx(x), sy(series[x])) for x in xs if x in series]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=7, joint="curve")
        for px, py in pts:
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill="white", outline=color, width=4)
        ly = top + 24 + idx * 46
        lx = width - 610
        draw.line((lx, ly, lx + 55, ly), fill=color, width=8)
        draw.text((lx + 70, ly), pretty_name(name), anchor="lm", fill="#1f2933", font=legend_font)
    img.save(path, dpi=(300, 300))


def draw_triangular_panel(draw, env, state_to_idx, values, policy, h, origin, tile, mode, vmin=None, vmax=None, compare_policy=None):
    ox, oy = origin
    title_font = font(30, True)
    tiny_font = font(18, False)
    tile_font = font(20, True)
    panel_title = "Cash states (h=0)" if h == 0 else "Stock states (h=1)"
    draw.text((ox, oy - 48), panel_title, anchor="lm", fill="#111827", font=title_font)
    palette = {
        HOLD: (71, 139, 86),
        BUY: (67, 111, 184),
        SELL: (202, 87, 76),
        -1: (235, 238, 242),
    }
    labels = {HOLD: "H", BUY: "B", SELL: "S", -1: ""}
    for t in range(env.horizon + 1):
        y = oy + t * tile
        draw.text((ox - 14, y + tile / 2), str(t), anchor="rm", fill="#6b7280", font=tiny_font)
        for k in range(t + 1):
            x = ox + k * tile
            idx = state_to_idx[(t, k, h)]
            if mode == "value":
                fill = (219, 234, 254) if np.isclose(vmin, vmax) else value_color(values[idx], vmin, vmax)
                text = ""
            else:
                action = int(policy[idx])
                fill = palette[action]
                text = labels[action]
            draw.rounded_rectangle((x + 2, y + 2, x + tile - 3, y + tile - 3), radius=5, fill=fill, outline="#ffffff", width=2)
            if compare_policy is not None and t < env.horizon and int(policy[idx]) != int(compare_policy[idx]):
                draw.rectangle((x + 3, y + 3, x + tile - 4, y + tile - 4), outline="#111827", width=4)
            if text:
                text_center(draw, (x + tile / 2, y + tile / 2), text, "white", tile_font)
    draw.text((ox + tile * 10, oy + tile * 21 + 35), "k = number of up moves", anchor="mm", fill="#6b7280", font=tiny_font)


def write_value_png(path, env, states, values, title="Optimal value function V*(s)", width=1800, height=1080):
    if Image is None:
        return
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(44, True)
    label_font = font(25, False)
    state_to_idx = {s: i for i, s in enumerate(states)}
    finite_vals = [values[state_to_idx[(t, k, h)]] for t in range(env.horizon) for k in range(t + 1) for h in (0, 1)]
    vmin, vmax = min(finite_vals), max(finite_vals)
    draw.text((width / 2, 52), title, anchor="mm", fill="#1f2933", font=title_font)
    if np.isclose(vmin, vmax):
        subtitle = f"All states have the same optimal value: V*(s) = {vmin:.4f}."
    else:
        subtitle = "Two compact triangular heatmaps; rows are t and columns are k."
    draw.text((width / 2, 94), subtitle, anchor="mm", fill="#4b5563", font=label_font)
    draw_triangular_panel(draw, env, state_to_idx, values, None, 0, (140, 170), 34, "value", vmin, vmax)
    draw_triangular_panel(draw, env, state_to_idx, values, None, 1, (950, 170), 34, "value", vmin, vmax)
    lx, ly, lw, lh = 520, 990, 760, 28
    if np.isclose(vmin, vmax):
        draw.rounded_rectangle((lx, ly - 10, lx + lw, ly + 56), radius=12, fill="#eff6ff", outline="#2563eb", width=3)
        draw.text((lx + lw / 2, ly + 23), "Uniform optimal value: V*(s) = 0.0000", anchor="mm", fill="#1e3a8a", font=label_font)
    else:
        for i in range(lw):
            val = vmin + (vmax - vmin) * i / max(1, lw - 1)
            draw.line((lx + i, ly, lx + i, ly + lh), fill=value_color(val, vmin, vmax))
        draw.rectangle((lx, ly, lx + lw, ly + lh), outline="#374151", width=2)
        draw.text((lx, ly + 55), f"lower V*: {vmin:.4f}", anchor="lm", fill="#4b5563", font=label_font)
        draw.text((lx + lw, ly + 55), f"higher V*: {vmax:.4f}", anchor="rm", fill="#4b5563", font=label_font)
    img.save(path, dpi=(300, 300))


def write_policy_png(path, env, states, policy, compare_policy=None, title="Optimal policy pi*(s)", width=1800, height=1050):
    if Image is None:
        return
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(44, True)
    label_font = font(25, False)
    legend_font = font(25, True)
    state_to_idx = {s: i for i, s in enumerate(states)}
    draw.text((width / 2, 52), title, anchor="mm", fill="#1f2933", font=title_font)
    subtitle = "B = buy, H = hold, S = sell. Rows are t and columns are k."
    if compare_policy is not None:
        subtitle += " Black borders mark mismatches."
    draw.text((width / 2, 94), subtitle, anchor="mm", fill="#4b5563", font=label_font)
    draw_triangular_panel(draw, env, state_to_idx, None, policy, 0, (140, 165), 33, "policy", compare_policy=compare_policy)
    draw_triangular_panel(draw, env, state_to_idx, None, policy, 1, (950, 165), 33, "policy", compare_policy=compare_policy)
    legend = [(BUY, "#436fb8", "B buy"), (HOLD, "#478b56", "H hold"), (SELL, "#ca574c", "S sell")]
    for i, (_, color, label) in enumerate(legend):
        x = 590 + i * 210
        y = 955
        draw.rounded_rectangle((x, y, x + 44, y + 44), radius=7, fill=color)
        draw.text((x + 62, y + 22), label, anchor="lm", fill="#1f2933", font=legend_font)
    img.save(path, dpi=(300, 300))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    env = StockTradingMDP()
    states, state_to_idx, q_vi, v_vi, policy_vi = value_iteration(env)
    raw_checkpoints = [100, 500, 1_000, 5_000, 10_000, 25_000, 50_000, args.episodes]
    checkpoints = sorted({cp for cp in raw_checkpoints if cp <= args.episodes})
    eps_schedules, alpha_schedules = experiment_grid()

    detailed_rows = []
    summary_rows = []
    learning_curves = {}
    best = None

    for eps_name, eps_schedule in eps_schedules.items():
        for alpha_name, alpha_schedule in alpha_schedules.items():
            setting = f"{eps_name}__{alpha_name}"
            final_metrics = []
            checkpoint_metrics = {cp: [] for cp in checkpoints}
            best_q_for_setting = None

            seeds = list(range(args.seeds))
            q_all, snapshots = q_learning_many(
                env, states, state_to_idx, args.episodes, eps_schedule, alpha_schedule, seeds, checkpoints
            )

            for seed_pos, seed in enumerate(seeds):
                q_ql = q_all[seed_pos]
                q_error, v_error, mismatch = metrics(env, states, q_vi, v_vi, policy_vi, q_ql)
                final_metrics.append((q_error, v_error, mismatch))
                detailed_rows.append({
                    "setting": setting,
                    "epsilon_schedule": eps_name,
                    "alpha_schedule": alpha_name,
                    "seed": seed,
                    "q_error": q_error,
                    "value_error": v_error,
                    "policy_mismatch": mismatch,
                })
                for cp, q_snapshot in snapshots.items():
                    checkpoint_metrics[cp].append(metrics(env, states, q_vi, v_vi, policy_vi, q_snapshot[seed_pos])[0])
                if seed == 0:
                    best_q_for_setting = q_ql

            q_mean, q_std = mean_std([m[0] for m in final_metrics])
            v_mean, v_std = mean_std([m[1] for m in final_metrics])
            p_mean, p_std = mean_std([m[2] for m in final_metrics])
            row = {
                "setting": setting,
                "epsilon_schedule": eps_name,
                "alpha_schedule": alpha_name,
                "q_error_mean": q_mean,
                "q_error_std": q_std,
                "value_error_mean": v_mean,
                "value_error_std": v_std,
                "policy_mismatch_mean": p_mean,
                "policy_mismatch_std": p_std,
            }
            summary_rows.append(row)
            learning_curves[setting] = {cp: mean_std(checkpoint_metrics[cp])[0] for cp in checkpoints}
            if best is None or q_mean < best["q_error_mean"]:
                best = {**row, "q": best_q_for_setting}

    write_csv(
        outdir / "experiment_summary.csv",
        summary_rows,
        [
            "setting", "epsilon_schedule", "alpha_schedule",
            "q_error_mean", "q_error_std",
            "value_error_mean", "value_error_std",
            "policy_mismatch_mean", "policy_mismatch_std",
        ],
    )
    write_csv(
        outdir / "experiment_by_seed.csv",
        detailed_rows,
        ["setting", "epsilon_schedule", "alpha_schedule", "seed", "q_error", "value_error", "policy_mismatch"],
    )

    top_settings = sorted(summary_rows, key=lambda r: r["q_error_mean"])[:5]
    top_curve_data = {r["setting"]: learning_curves[r["setting"]] for r in top_settings}
    write_line_svg(outdir / "error_vs_episodes.svg", top_curve_data, "Q-error vs episodes", "Mean max Q-error")
    write_value_svg(outdir / "optimal_value_function.svg", env, states, v_vi)
    write_policy_svg(outdir / "optimal_policy.svg", env, states, policy_vi, title="Optimal policy from value iteration")
    write_error_png(outdir / "q_error_vs_episodes_report.png", top_curve_data)
    write_value_png(outdir / "optimal_value_function_report.png", env, states, v_vi)
    write_policy_png(outdir / "optimal_policy_report.png", env, states, policy_vi)
    _, best_policy = greedy_policy_from_q(env, states, best["q"])
    write_policy_svg(
        outdir / "policy_comparison_best.svg",
        env,
        states,
        policy_vi,
        policy_b=best_policy,
        title=f"VI policy; thick borders mark mismatch with best Q-learning ({best['setting']})",
    )
    write_policy_png(
        outdir / "policy_comparison_best_report.png",
        env,
        states,
        policy_vi,
        compare_policy=best_policy,
        title=f"VI policy vs best Q-learning ({best['setting']})",
    )

    with (outdir / "best_setting.txt").open("w", encoding="utf-8") as f:
        f.write(f"Best setting by mean Q-error: {best['setting']}\n")
        f.write(f"Q-error mean/std: {best['q_error_mean']:.6f} / {best['q_error_std']:.6f}\n")
        f.write(f"Value-error mean/std: {best['value_error_mean']:.6f} / {best['value_error_std']:.6f}\n")
        f.write(f"Policy mismatch mean/std: {best['policy_mismatch_mean']:.6f} / {best['policy_mismatch_std']:.6f}\n")

    print(f"Wrote results to {outdir.resolve()}")
    print(f"Best setting: {best['setting']} q_error_mean={best['q_error_mean']:.6f}")


if __name__ == "__main__":
    main()
