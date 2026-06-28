"""
actor_critic.py
===============
A one-step (online) Actor-Critic agent implemented entirely from scratch with
NumPy.  Every required course concept is implemented explicitly here:

  * CONTINUOUS STATE SPACE   - the networks take the real-valued state [x, v].
  * FUNCTION APPROXIMATION    - two one-hidden-layer MLPs approximate the
                                policy  pi(a|s)  (actor) and the value  V(s)
                                (critic).  No tables are used.
  * TEMPORAL-DIFFERENCE       - the critic is trained on the one-step TD target
                                y = r + gamma * V(s')   using the TD error
                                delta = y - V(s).
  * POLICY GRADIENT           - the actor is updated with the score-function
                                (REINFORCE-with-baseline) estimator, where the
                                TD error delta plays the role of the advantage:
                                    grad J = E[ delta * grad log pi(a|s) ].

The actor-critic update for one transition (s, a, r, s') is:

    delta      = r + gamma * V(s') * (1 - done)  -  V(s)      # TD error
    critic_loss = delta**2                                     # fit V toward TD target
    actor_loss  = -log pi(a|s) * delta  -  beta * H(pi(.|s))   # PG + entropy bonus

Backpropagation through both MLPs is written out by hand, and a small Adam
optimizer (also from scratch) keeps the updates stable.
"""

import numpy as np


# ===========================================================================
#  A tiny one-hidden-layer MLP with manual forward / backward passes
# ===========================================================================
class MLP:
    """y = W2 @ tanh(W1 @ x + b1) + b2.

    Stores the forward-pass activations so that `backward` can compute exact
    parameter gradients given the gradient of the loss w.r.t. the output.
    """

    def __init__(self, in_dim, hidden, out_dim, rng):
        # He/Xavier-ish initialisation for tanh hidden units.
        self.W1 = rng.standard_normal((hidden, in_dim)) * np.sqrt(1.0 / in_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((out_dim, hidden)) * np.sqrt(1.0 / hidden)
        self.b2 = np.zeros(out_dim)
        self.params = ["W1", "b1", "W2", "b2"]

    def forward(self, x):
        self._x = x
        self._z1 = self.W1 @ x + self.b1
        self._h = np.tanh(self._z1)
        self._y = self.W2 @ self._h + self.b2
        return self._y

    def backward(self, grad_y):
        """Given dL/dy, return a dict of parameter gradients."""
        grad_y = np.atleast_1d(grad_y)
        gW2 = np.outer(grad_y, self._h)
        gb2 = grad_y
        grad_h = self.W2.T @ grad_y
        grad_z1 = grad_h * (1.0 - self._h ** 2)          # tanh'(z) = 1 - tanh^2
        gW1 = np.outer(grad_z1, self._x)
        gb1 = grad_z1
        return {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}


# ===========================================================================
#  Adam optimizer (from scratch) for one MLP
# ===========================================================================
class Adam:
    def __init__(self, net: MLP, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.net, self.lr, self.b1, self.b2, self.eps = net, lr, b1, b2, eps
        self.m = {p: np.zeros_like(getattr(net, p)) for p in net.params}
        self.v = {p: np.zeros_like(getattr(net, p)) for p in net.params}
        self.t = 0

    def step(self, grads, scale=1.0):
        self.t += 1
        for p in self.net.params:
            g = grads[p] * scale
            self.m[p] = self.b1 * self.m[p] + (1 - self.b1) * g
            self.v[p] = self.b2 * self.v[p] + (1 - self.b2) * (g * g)
            mhat = self.m[p] / (1 - self.b1 ** self.t)
            vhat = self.v[p] / (1 - self.b2 ** self.t)
            update = self.lr * mhat / (np.sqrt(vhat) + self.eps)
            setattr(self.net, p, getattr(self.net, p) - update)


# ===========================================================================
#  Helper
# ===========================================================================
def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


# ===========================================================================
#  The Actor-Critic agent
# ===========================================================================
class ActorCritic:
    def __init__(self, state_dim, n_actions, hidden=64, gamma=0.99,
                 actor_lr=2e-3, critic_lr=5e-3, entropy_beta=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.n_actions = n_actions
        self.entropy_beta = entropy_beta

        # ACTOR  : state -> action logits  (policy pi(a|s) via softmax)
        self.actor = MLP(state_dim, hidden, n_actions, rng)
        # CRITIC : state -> scalar value estimate  V(s)
        self.critic = MLP(state_dim, hidden, 1, rng)

        self.actor_opt = Adam(self.actor, lr=actor_lr)
        self.critic_opt = Adam(self.critic, lr=critic_lr)

        self.rng = rng

    # ----------------------------------------------------------------------
    def policy(self, s):
        """Return the action-probability vector pi(.|s)."""
        return softmax(self.actor.forward(s))

    def act(self, s, greedy=False):
        """Sample an action from the policy (or take the argmax if greedy)."""
        probs = self.policy(s)
        if greedy:
            return int(np.argmax(probs)), probs
        return int(self.rng.choice(self.n_actions, p=probs)), probs

    def value(self, s):
        """Critic estimate V(s) (scalar)."""
        return float(self.critic.forward(s)[0])

    # ----------------------------------------------------------------------
    def update(self, s, a, r, s_next, done):
        """Single online actor-critic update on one transition.

        Returns the TD error (useful for logging).
        """
        # --- CRITIC forward passes & TD error ------------------------------
        v_s = self.critic.forward(s)[0]                 # V(s)   (keeps cache for backward)
        v_next = 0.0 if done else self.critic.forward(s_next)[0]   # V(s')
        td_target = r + self.gamma * v_next             # TD(0) target
        delta = td_target - v_s                         # TD error  (== advantage)

        # NOTE: v_next's forward overwrote the critic cache, so re-run the
        #       forward pass on s to restore the cache before backprop.
        v_s = self.critic.forward(s)[0]

        # --- CRITIC update : minimise delta^2  ->  dL/dV(s) = -2 * delta ----
        # (We treat td_target as a fixed target, the standard TD semi-gradient.)
        critic_grads = self.critic.backward(np.array([-2.0 * delta]))
        self.critic_opt.step(critic_grads)

        # --- ACTOR update : policy gradient with advantage = delta ---------
        logits = self.actor.forward(s)
        probs = softmax(logits)

        # Gradient of  -log pi(a|s)  w.r.t. logits is  (probs - onehot(a)).
        onehot = np.zeros(self.n_actions)
        onehot[a] = 1.0
        dlogp = probs - onehot                          # d(-log pi_a)/d logits

        # Policy-gradient term: scale by the advantage (delta is a constant here).
        grad_logits = dlogp * delta

        # Entropy bonus  +beta*H(pi)  encourages exploration.
        # dH/d logits_i = -probs_i * (log probs_i + H);  we subtract beta*dH.
        logp = np.log(probs + 1e-12)
        H = -np.sum(probs * logp)
        dH = -probs * (logp + H)
        grad_logits -= self.entropy_beta * dH

        actor_grads = self.actor.backward(grad_logits)
        self.actor_opt.step(actor_grads)

        return delta
