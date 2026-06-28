"""
train.py
========
Trains the from-scratch Actor-Critic agent on the custom BlockControlEnv and
produces the figures used in the report:

  1. learning_curve.png   - episode return vs. episode (raw + moving average)
  2. trajectory.png       - block position & velocity over time for a trained
                            vs. an untrained policy (shows it learns to centre)
  3. value_policy.png     - learned value function V(s) and greedy policy over
                            the continuous state space (function approximation)

Run:  python train.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from block_env import BlockControlEnv
from actor_critic import ActorCritic


# ---------------------------------------------------------------------------
def run_episode(env, agent, train=True, greedy=False, record=False):
    """Run one episode; optionally update the agent and/or record a trajectory."""
    s_raw = env.reset()
    s = env.normalize(s_raw)
    total_r, steps = 0.0, 0
    traj = []

    done = False
    while not done:
        a, _ = agent.act(s, greedy=greedy)
        s_next_raw, r, done, info = env.step(a)
        s_next = env.normalize(s_next_raw)

        if record:
            traj.append((s_raw[0], s_raw[1], a))

        if train:
            agent.update(s, a, r, s_next, done)

        s_raw, s = s_next_raw, s_next
        total_r += r
        steps += 1

    return total_r, steps, traj


# ---------------------------------------------------------------------------
def moving_average(x, w=50):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


# ---------------------------------------------------------------------------
def main():
    SEED = 0
    N_EPISODES = 1500

    env = BlockControlEnv(seed=SEED)
    agent = ActorCritic(env.state_dim, env.n_actions, hidden=64,
                        gamma=0.99, actor_lr=2e-3, critic_lr=5e-3,
                        entropy_beta=0.01, seed=SEED)

    # Baseline (random policy) trajectory BEFORE any training, for comparison.
    _, _, traj_before = run_episode(env, agent, train=False, greedy=False, record=True)

    returns = []
    print(f"Training actor-critic for {N_EPISODES} episodes...\n")
    for ep in range(1, N_EPISODES + 1):
        R, steps, _ = run_episode(env, agent, train=True)
        returns.append(R)
        if ep % 100 == 0:
            recent = np.mean(returns[-100:])
            print(f"  episode {ep:4d} | return {R:8.2f} | "
                  f"avg(last100) {recent:8.2f} | steps {steps:3d}")

    # Greedy evaluation after training.
    eval_returns = [run_episode(env, agent, train=False, greedy=True)[0]
                    for _ in range(50)]
    print(f"\nGreedy eval over 50 episodes: "
          f"mean {np.mean(eval_returns):.2f} +/- {np.std(eval_returns):.2f}")

    # Trained trajectory for comparison plot.
    _, _, traj_after = run_episode(env, agent, train=False, greedy=True, record=True)

    # ----------------------------------------------------------------- plots
    _plot_learning_curve(returns)
    _plot_trajectories(env, traj_before, traj_after)
    _plot_value_and_policy(env, agent)

    # Save metrics for the report.
    np.savez("metrics.npz",
             returns=np.array(returns),
             eval_mean=np.mean(eval_returns),
             eval_std=np.std(eval_returns))
    print("\nSaved: learning_curve.png, trajectory.png, value_policy.png, metrics.npz")


# ---------------------------------------------------------------------------
def _plot_learning_curve(returns):
    plt.figure(figsize=(8, 4.5))
    plt.plot(returns, color="#9ecae1", lw=0.8, label="episode return")
    ma = moving_average(returns, 50)
    plt.plot(range(49, 49 + len(ma)), ma, color="#08519c", lw=2,
             label="50-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (return)")
    plt.title("Actor-Critic learning curve on BlockControlEnv")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=130)
    plt.close()


def _plot_trajectories(env, traj_before, traj_after):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, traj, title in [
        (axes[0], traj_before, "Before training (random policy)"),
        (axes[1], traj_after, "After training (greedy policy)"),
    ]:
        if len(traj) == 0:
            continue
        t = np.arange(len(traj)) * env.dt
        xs = [p[0] for p in traj]
        vs = [p[1] for p in traj]
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.plot(t, xs, color="#08519c", lw=1.8, label="position x")
        ax.plot(t, vs, color="#e6550d", lw=1.2, alpha=0.8, label="velocity v")
        ax.axhline(env.x_max, color="red", lw=0.6, ls=":")
        ax.axhline(-env.x_max, color="red", lw=0.6, ls=":")
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("state value")
    axes[0].legend(loc="upper right", fontsize=8)
    plt.suptitle("Block position & velocity over an episode")
    plt.tight_layout()
    plt.savefig("trajectory.png", dpi=130)
    plt.close()


def _plot_value_and_policy(env, agent):
    # Grid over the continuous state space.
    xs = np.linspace(-env.x_max, env.x_max, 120)
    vs = np.linspace(-env.v_max, env.v_max, 120)
    V = np.zeros((len(vs), len(xs)))
    P = np.zeros((len(vs), len(xs)))
    for i, v in enumerate(vs):
        for j, x in enumerate(xs):
            s = env.normalize(np.array([x, v]))
            V[i, j] = agent.value(s)
            a, _ = agent.act(s, greedy=True)
            P[i, j] = a   # 0=left, 1=none, 2=right

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    im0 = axes[0].pcolormesh(xs, vs, V, shading="auto", cmap="viridis")
    axes[0].set_title("Learned value function V(s)")
    axes[0].set_xlabel("position x"); axes[0].set_ylabel("velocity v")
    fig.colorbar(im0, ax=axes[0], label="V(s)")

    im1 = axes[1].pcolormesh(xs, vs, P, shading="auto", cmap="coolwarm")
    axes[1].set_title("Greedy policy (0=left, 1=none, 2=right)")
    axes[1].set_xlabel("position x"); axes[1].set_ylabel("velocity v")
    fig.colorbar(im1, ax=axes[1], ticks=[0, 1, 2], label="action")

    plt.tight_layout()
    plt.savefig("value_policy.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
