# Gridworld MDP + Value Iteration

This project builds a Gymnasium-compatible Gridworld Markov Decision Process (MDP) and implements classic Dynamic Programming via Value Iteration. It includes utilities to evaluate policies, render value maps with best actions, visualize trajectory rollouts, and optionally record videos of agent behavior.

## Files

- `gridworld_mdp.py`: Main script containing the Gridworld environment, helper utilities (evaluation, rendering, trajectory collection), Value Iteration implementation, and example experiments.

## Requirements

- Python 3.x
- `gymnasium`
- `numpy`
- `matplotlib`
- `tqdm`
- `Pillow` (PIL)
- `imageio` (optional; visualization helpers)
- `moviepy` (optional; for video recording)
- `pyvirtualdisplay` + `ffmpeg` (optional; for headless video rendering)

## Setup

1. Ensure you have Python 3.x installed.
2. Install dependencies (example using pip):

    ```sh
    pip install gymnasium numpy matplotlib tqdm pillow imageio moviepy pyvirtualdisplay
    ```

    - For video recording, ensure `ffmpeg` is installed on your system (e.g., via your OS package manager).

3. No external datasets are required.

## How to Run

1. Open a terminal and navigate to this directory:

    ```sh
    cd "CS470 - Introduction to AI/Assignment 3/cs470_IsmayilovHuseyn_20220913"
    ```

2. Run the script:

    ```sh
    python gridworld_mdp.py
    ```

3. The script registers the Gridworld environment, runs Value Iteration, and shows matplotlib figures (value maps, trajectories, and error/return curves). Optional video helpers are included; enable and use them if your environment supports video rendering.

## Workflow

1. Define a 2D gridworld with obstacles, traps, stochastic transitions, rewards, and termination.
2. Register the environment with Gymnasium and instantiate it.
3. Run Value Iteration until the Bellman error drops below a threshold.
4. Evaluate the greedy policy derived from the value function and visualize value maps and sampled trajectories.
5. Repeat experiments under different stochasticity (epsilon) settings to compare behavior.

## Functions

- `eval_policy(env, policy, num_episodes)`: Runs a policy and returns per-episode returns.
- `save_video_of_model(env_name, model, suffix, num_episodes)`: Records a video of rollouts (optional).
- `play_video(filename)`: Utility to display a recorded video (optional).
- `render_value_map_with_action(env, Q, policy)`: Renders a heatmap of values with best actions overlaid.
- `collect_traj(env, policy, num_episodes)`: Collects trajectories (state sequences) under a policy.
- `plot_trajs(env, trajectories)`: Plots trajectory overlays on the grid.
- `plot_grid(env)`: Renders the grid with obstacles/traps and state indices.
- `plot_reward_error_graphs(ep_rews, errors)`: Plots mean episodic returns and Bellman errors over iterations.

## Classes

- `BaseGridEnv`: Base class defining the grid layout, state/action spaces, rendering, and helpers for coordinate transforms.
- `GridEnv`: Concrete environment implementing transition dynamics, rewards, and termination for the gridworld.
- `ValueIteration`: Dynamic Programming agent that computes the optimal value function and a greedy policy.

