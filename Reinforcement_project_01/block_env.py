"""
block_env.py
============
A custom-built reinforcement-learning environment (NO OpenAI Gym) whose
dynamics are defined purely by closed-form Newtonian equations of motion
with viscous damping.

Task
----
A block of mass `m` slides on a frictionless-but-damped 1-D surface.  At every
time step the agent applies one of three DISCRETE forces:

    action 0 -> push left   (-F)
    action 1 -> no force     (0)
    action 2 -> push right  (+F)

The agent's goal is to keep the block near the centre (position x = 0) with low
velocity.  This exercises a CONTINUOUS STATE SPACE  s = [x, v]  while keeping
the action space discrete, which is exactly what an actor-critic policy-gradient
method is designed for.

Equations of motion (continuous-time)
-------------------------------------
    m * x''  =  F_action  -  c * x'
                 ^applied      ^viscous damping (opposes motion)

We integrate this with a stable semi-implicit (symplectic) Euler scheme:

    a      = (F_action - c * v) / m
    v_next = v + a  * dt
    x_next = x + v_next * dt

Reward
------
A quadratic regulator-style cost (so this is essentially a discrete-action LQR):

    r = -( x^2  +  w_v * v^2  +  w_u * (F_action / F)^2 )

The agent therefore gets the *most* reward (closest to 0) when the block sits at
the centre, moving slowly, using little control effort.  An episode ends when
the block leaves the allowed region |x| > x_max (a failure, penalised) or after
`max_steps` steps.
"""

import numpy as np


class BlockControlEnv:
    """Closed-form 1-D block-on-a-surface control environment."""

    def __init__(self, seed: int | None = None):
        # ---- physical constants -------------------------------------------
        self.m = 1.0        # mass (kg)
        self.c = 0.30       # viscous damping coefficient (N per m/s)
        self.F = 10.0       # magnitude of the discrete push (N)
        self.dt = 0.05      # integration time step (s)

        # ---- episode / state bounds ---------------------------------------
        self.x_max = 2.4    # block fails if |x| exceeds this (m)
        self.v_max = 5.0    # used only for observation clipping / scaling
        self.max_steps = 200

        # ---- reward weights -----------------------------------------------
        self.w_v = 0.10     # penalty on velocity
        self.w_u = 0.01     # penalty on control effort

        # ---- spaces --------------------------------------------------------
        self.state_dim = 2          # continuous state  s = [x, v]
        self.n_actions = 3          # discrete actions  {left, none, right}
        self.action_to_force = {0: -self.F, 1: 0.0, 2: +self.F}

        self.rng = np.random.default_rng(seed)
        self.state = None
        self.steps = 0

    # ----------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Start a new episode from a small random offset near the centre."""
        x0 = self.rng.uniform(-0.5, 0.5)
        v0 = self.rng.uniform(-0.20, 0.20)
        self.state = np.array([x0, v0], dtype=np.float64)
        self.steps = 0
        return self.state.copy()

    # ----------------------------------------------------------------------
    def step(self, action: int):
        """Advance the closed-form dynamics by one time step.

        Returns (next_state, reward, done, info) -- the classic RL interface,
        implemented from scratch rather than imported from Gym.
        """
        assert action in (0, 1, 2), f"invalid action {action}"
        x, v = self.state
        force = self.action_to_force[action]

        # --- Newtonian update (semi-implicit Euler) ------------------------
        a = (force - self.c * v) / self.m
        v = v + a * self.dt
        x = x + v * self.dt

        self.state = np.array([x, v], dtype=np.float64)
        self.steps += 1

        # --- quadratic-regulator reward ------------------------------------
        reward = -(x ** 2 + self.w_v * v ** 2 + self.w_u * (force / self.F) ** 2)

        # --- termination ----------------------------------------------------
        out_of_bounds = abs(x) > self.x_max
        done = out_of_bounds or (self.steps >= self.max_steps)
        if out_of_bounds:
            reward -= 10.0   # extra penalty for letting the block escape

        info = {"out_of_bounds": out_of_bounds}
        return self.state.copy(), reward, done, info

    # ----------------------------------------------------------------------
    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Scale the continuous state to roughly [-1, 1] for the networks.

        Function approximators train far more reliably on normalised inputs;
        this does not change the physics, only the representation fed to the
        actor and critic.
        """
        x, v = state
        return np.array([x / self.x_max, v / self.v_max], dtype=np.float64)
