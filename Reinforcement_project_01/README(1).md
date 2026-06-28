# Reinforcement Learning Final Project

## An Actor-Critic Agent for a Custom Block-Centering Control Task

This project implements a custom reinforcement learning control task where an actor-critic agent learns to keep a horizontally moving block near the center of a flat surface. The block follows Newtonian motion with damping, and the agent chooses from three discrete force actions: push left, push right, or apply no force. The environment is built from closed-form equations instead of using OpenAI Gym.

## Project Description

The goal of this project is to train an agent that can stabilize a damped block around the center position, `x = 0`. The agent observes a continuous state made up of the block position and velocity, then selects a discrete action to control the block. An actor-critic method is used, where the actor learns the action policy and the critic estimates the value of each state using temporal-difference learning.

## Key Reinforcement Learning Concepts

- Continuous state space using position and velocity
- Discrete force actions: left, none, and right
- Function approximation using neural networks
- Temporal-difference learning for critic updates
- Policy-gradient learning for actor updates
- Custom closed-form environment dynamics

## Environment Formulation

### State Space

The state is continuous and represented as:

```text
s = [x, v]
```

where:

- `x` = block position
- `v` = block velocity

### Action Space

The agent selects one of three actions:

```text
0 = push left
1 = no force
2 = push right
```

These actions map to applied forces:

```text
{-F, 0, +F}
```

### Dynamics

The block motion is based on Newton's second law with damping:

```text
m * x_ddot = F_action - c * x_dot
```

The simulation updates velocity and position using a semi-implicit Euler method.

## Reward Function

The reward encourages the block to remain near the center with low velocity and minimal control effort:

```text
r = -(x^2 + w_v * v^2 + w_u * control_effort^2)
```

A higher reward is achieved when the block stays close to `x = 0`, moves slowly, and avoids unnecessary force.

## Actor-Critic Method

The project uses two neural network function approximators:

- **Actor network:** outputs action probabilities using a softmax policy.
- **Critic network:** estimates the state value `V(s)`.

The critic is trained using a one-step temporal-difference target, while the actor is updated using the TD error as an advantage estimate.

## Experimental Setup

| Parameter | Value |
|---|---:|
| Mass | 1.0 kg |
| Damping coefficient | 0.30 N·s/m |
| Force magnitude | 10.0 N |
| Time step | 0.05 s |
| Position bound | 2.4 m |
| Max steps per episode | 200 |
| Discount factor | 0.99 |
| Hidden units | 64 |
| Actor learning rate | 0.002 |
| Critic learning rate | 0.005 |
| Entropy bonus | 0.01 |
| Training episodes | 1500 |

## Results

The trained actor-critic agent successfully learned to keep the block near the center. The average return improved significantly from the beginning of training to the final episodes. During evaluation, the trained greedy policy completed full episodes without the block escaping the track.

The learned policy formed a near-linear switching behavior in the position-velocity plane. When the block moved too far or too fast in one direction, the agent applied the opposite force to stabilize it.

## Code Structure

```text
block_env.py        # Custom block-control environment
actor_critic.py     # Actor and critic neural networks with learning updates
train.py            # Training loop, evaluation, and figure generation
```

## How to Run

Install the required Python packages:

```bash
pip install numpy matplotlib
```

Run the training script:

```bash
python train.py
```

The script trains the agent, prints training progress, evaluates the greedy policy, and saves the generated result figures.

## Conclusion

This project demonstrates how an actor-critic reinforcement learning agent can solve a custom continuous-state control problem using closed-form physics equations. The final policy keeps the damped block near the center and shows the use of function approximation, temporal-difference learning, and policy gradients in a complete reinforcement learning system.
