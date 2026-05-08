"""
Meta-RL Training Script for ASGD-MetaBBO (Phase 6).

Trains the PPO outer controller over a distribution of optimization problems.

Training distribution:
  - BBOB 24 noiseless functions × D ∈ {10, 20} × 5 instances = 240 problems
  - For quick prototyping: use the 6 built-in test functions

Episode = one full optimization run.
Step = one generation within the run.
Action = (exploration_coeff, gd_beta, retrain_interval).

Run on GPU:
    python -m experiments.train_meta_rl --n_episodes 500 --device cuda

For quick testing (CPU, built-in functions only):
    python -m experiments.train_meta_rl --quick --device cpu
"""

import sys
import os
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meta_rl.environment import MetaBBOEnv
from src.meta_rl.ppo import get_policy, FixedPolicy
from benchmarks.functions import get_simple_problem, SIMPLE_FUNCTIONS

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def create_problem_distribution(use_bbob: bool = False):
    """
    Create the training problem distribution.

    Returns list of (func, dim, bounds, optimum, name) tuples.
    """
    problems = []

    # Built-in test functions (always available)
    for fname in SIMPLE_FUNCTIONS:
        for dim in [10, 20]:
            func, lb, ub, opt = get_simple_problem(fname, dim)
            problems.append({
                "func": func, "dim": dim,
                "bounds": (lb, ub), "optimum": opt,
                "name": f"{fname}_D{dim}",
            })

    if use_bbob:
        try:
            from benchmarks.functions import get_cec2022_suite
            for dim in [10, 20]:
                suite = get_cec2022_suite(dim)
                for name, info in suite.items():
                    problems.append({
                        "func": info["func"], "dim": dim,
                        "bounds": (info["lb"], info["ub"]),
                        "optimum": info["optimum"],
                        "name": f"{name}_D{dim}",
                    })
        except Exception as e:
            print(f"Warning: CEC2022 not available ({e}), using built-in only")

    return problems


def collect_rollout(env, policy, max_steps=200):
    """
    Collect one episode of experience.

    Returns rollout dict for PPO update.
    """
    state = env.reset()
    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

    for step in range(max_steps):
        if HAS_TORCH and hasattr(policy, 'get_action'):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_t, lp_t, val_t = policy.get_action(state_t)
            action = action_t.squeeze(0).numpy()
            log_prob = lp_t.item()
            value = val_t.item()
        else:
            action = policy.get_action(state)
            log_prob = 0.0
            value = 0.0

        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        dones.append(done)

        state = next_state
        if done:
            break

    # Get next value for GAE
    if HAS_TORCH and hasattr(policy, 'forward'):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            _, _, val_t = policy.forward(state_t)
        next_value = val_t.item()
    else:
        next_value = 0.0

    return {
        "states": states,
        "actions": actions,
        "log_probs": log_probs,
        "rewards": rewards,
        "values": values,
        "dones": dones,
        "next_value": next_value,
        "final_info": info if 'info' in dir() else {},
        "n_steps": len(states),
    }


def train_meta_rl(
    n_episodes: int = 200,
    problems: list = None,
    max_steps_per_episode: int = 200,
    max_evals_per_run: int = 5000,
    save_path: str = "checkpoints/meta_rl.pt",
    log_interval: int = 10,
    checkpoint_interval: int = 10,
    device: str = "cpu",
    resume: bool = True,
):
    """
    Main meta-RL training loop with power-outage protection.

    Features:
      - Auto-resume: detects existing checkpoint and continues
      - Periodic saves: every checkpoint_interval episodes
      - Log file: appends to checkpoints/training_log.txt
      - Saves full state: model, optimizer, episode, rewards, best
    """
    if problems is None:
        problems = create_problem_distribution(use_bbob=False)

    os.makedirs(os.path.dirname(save_path) or "checkpoints", exist_ok=True)
    log_file = os.path.join(os.path.dirname(save_path) or "checkpoints", "training_log.txt")

    def log(msg):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"Training distribution: {len(problems)} problems")
    log(f"Episodes: {n_episodes}, Max steps: {max_steps_per_episode}")
    log(f"FE budget per run: {max_evals_per_run}")
    log(f"Device: {device}")
    log("-" * 60)

    # Create policy
    state_dim = 64
    start_episode = 0
    episode_rewards = []
    episode_errors = []
    best_avg_reward = -float("inf")

    if HAS_TORCH:
        from src.meta_rl.ppo import ActorCritic, PPOTrainer
        policy = ActorCritic(state_dim=state_dim).to(device)
        trainer = PPOTrainer(policy)
        log(f"Using PPO with {sum(p.numel() for p in policy.parameters())} parameters")

        # --- AUTO-RESUME: load checkpoint if exists ---
        resume_path = save_path.replace(".pt", "_latest.pt")
        if resume and os.path.exists(resume_path):
            try:
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                policy.load_state_dict(ckpt["model_state"])
                trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
                start_episode = ckpt["episode"]
                episode_rewards = ckpt.get("episode_rewards", [])
                episode_errors = ckpt.get("episode_errors", [])
                best_avg_reward = ckpt.get("best_avg_reward", -float("inf"))
                log(f"*** RESUMED from episode {start_episode} (checkpoint: {resume_path}) ***")
                log(f"    Previous best avg reward: {best_avg_reward:.4f}")
                log(f"    Previous episodes completed: {len(episode_rewards)}")
            except Exception as e:
                log(f"Warning: could not load checkpoint ({e}), starting fresh")
                start_episode = 0
        else:
            log("Starting fresh training (no checkpoint found)")
    else:
        policy = FixedPolicy()
        trainer = None
        log("PyTorch unavailable — running with fixed policy (no learning)")

    # --- Training loop ---
    for ep in range(start_episode, n_episodes):
        rng = np.random.default_rng(ep)
        prob = problems[rng.integers(0, len(problems))]

        env = MetaBBOEnv(
            func=prob["func"],
            dim=prob["dim"],
            bounds=prob["bounds"],
            max_evals=max_evals_per_run,
            known_optimum=prob["optimum"],
            seed=ep * 7,
            surr_hidden=(32, 16),
            surr_epochs=10,
        )

        rollout = collect_rollout(env, policy, max_steps=max_steps_per_episode)

        total_reward = sum(rollout["rewards"])
        final_error = rollout["final_info"].get("error", float("inf"))
        episode_rewards.append(total_reward)
        episode_errors.append(final_error)

        # PPO update
        train_metrics = {}
        if trainer is not None and len(rollout["states"]) > 4:
            train_metrics = trainer.update(rollout)

        # --- Logging ---
        if (ep + 1) % log_interval == 0:
            recent_r = np.mean(episode_rewards[-log_interval:])
            recent_e = np.median(episode_errors[-log_interval:])
            msg = (f"  Episode {ep+1:>4}/{n_episodes} | "
                   f"reward={recent_r:>8.4f} | "
                   f"error={recent_e:>10.2e} | "
                   f"prob={prob['name']}")
            if train_metrics:
                msg += (f" | ploss={train_metrics['policy_loss']:.4f}"
                        f" vloss={train_metrics['value_loss']:.4f}")
            log(msg)

            # Save best model
            if HAS_TORCH and recent_r > best_avg_reward:
                best_avg_reward = recent_r
                torch.save({
                    "model_state": policy.state_dict(),
                    "episode": ep + 1,
                    "reward": best_avg_reward,
                }, save_path)

        # --- PERIODIC CHECKPOINT (power outage protection) ---
        if HAS_TORCH and (ep + 1) % checkpoint_interval == 0:
            resume_path = save_path.replace(".pt", "_latest.pt")
            torch.save({
                "model_state": policy.state_dict(),
                "optimizer_state": trainer.optimizer.state_dict(),
                "episode": ep + 1,
                "episode_rewards": episode_rewards,
                "episode_errors": episode_errors,
                "best_avg_reward": best_avg_reward,
            }, resume_path)

    # --- Final save ---
    if HAS_TORCH:
        resume_path = save_path.replace(".pt", "_latest.pt")
        torch.save({
            "model_state": policy.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "episode": n_episodes,
            "episode_rewards": episode_rewards,
            "episode_errors": episode_errors,
            "best_avg_reward": best_avg_reward,
        }, resume_path)

    log("\n" + "=" * 60)
    log("Training complete.")
    log(f"  Final avg reward (last 20): {np.mean(episode_rewards[-20:]):.4f}")
    log(f"  Final med error (last 20):  {np.median(episode_errors[-20:]):.2e}")
    if HAS_TORCH:
        log(f"  Best model: {save_path}")
        log(f"  Resume checkpoint: {resume_path}")
        log(f"  Log file: {log_file}")
    log("=" * 60)

    return policy


def main():
    parser = argparse.ArgumentParser(description="SDG-MetaBBO Meta-RL Training")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--max_evals", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--save", type=str, default="checkpoints/meta_rl.pt")
    parser.add_argument("--no_resume", action="store_true", help="Start fresh, ignore checkpoint")
    args = parser.parse_args()

    if args.quick:
        args.n_episodes = 20
        args.max_evals = 2000
        args.max_steps = 50

    problems = create_problem_distribution(use_bbob=not args.quick)

    train_meta_rl(
        n_episodes=args.n_episodes,
        problems=problems,
        max_steps_per_episode=args.max_steps,
        max_evals_per_run=args.max_evals,
        save_path=args.save,
        device=args.device,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
