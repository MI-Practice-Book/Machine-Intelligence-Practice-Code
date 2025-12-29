# code5_data_analysis.py
# 数据分析与处理：
# 5.1 状态分布分析
# 5.2 动作空间输出特性
# 5.3 奖励与学习过程关系
# 5.4 PPO Rollout 数据流转分析
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

ENV_ID = "Reacher-v4"
MODEL_PATH = "ppo_reacher_final"
OUT_DIR = "./figs_code5"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_EPISODES = 20
MAX_STEPS = 200


# =========================
# 收集 rollout 数据
# =========================
def collect_rollout_data(env, model):
    obs_list, act_list, rew_list = [], [], []

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        for t in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(reward)

            obs = next_obs
            if terminated or truncated:
                break

    return (
        np.array(obs_list),
        np.array(act_list),
        np.array(rew_list)
    )


# =========================
# 5.1 状态分布分析
# =========================
def analyze_state_distribution(obs):
    mean = obs.mean(axis=0)
    std = obs.std(axis=0)

    plt.figure()
    plt.bar(range(len(mean)), mean, yerr=std)
    plt.title("State Distribution (Mean ± Std)")
    plt.xlabel("State Dimension")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/state_distribution.png", dpi=200)
    plt.close()


# =========================
# 5.2 动作空间输出分析
# =========================
def analyze_action_distribution(actions):
    num_actions = actions.shape[1]

    for i in range(num_actions):
        plt.figure()
        plt.hist(actions[:, i], bins=50)
        plt.title(f"Action {i} Distribution")
        plt.xlabel("Action Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/action_{i}_distribution.png", dpi=200)
        plt.close()



# =========================
# 5.3 奖励与时间关系
# =========================
def analyze_reward_process(rewards):
    rewards = np.array(rewards)
    cumulative = np.cumsum(rewards)

    plt.figure()
    plt.plot(cumulative)
    plt.title("Cumulative Reward over Rollout Steps")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/reward_process.png", dpi=200)
    plt.close()


# =========================
# 5.4 PPO Rollout 数据流转示意
# =========================
def plot_rollout_flow():
    text = (
        "PPO Rollout Data Flow:\n\n"
        "State (obs)\n"
        "   ↓\n"
        "Policy Network (Actor)\n"
        "   ↓\n"
        "Action (continuous)\n"
        "   ↓\n"
        "Environment Step\n"
        "   ↓\n"
        "Reward, Next State\n"
        "   ↓\n"
        "Value Network (Critic)\n"
        "   ↓\n"
        "Advantage Estimation (GAE)\n"
        "   ↓\n"
        "Policy / Value Update\n"
    )

    plt.figure(figsize=(6, 6))
    plt.text(0.05, 0.95, text, va="top", fontsize=11)
    plt.axis("off")
    plt.title("PPO Rollout Buffer Data Flow")
    plt.savefig(f"{OUT_DIR}/rollout_flow.png", dpi=200)
    plt.close()


def main():
    env = gym.make(ENV_ID)
    model = PPO.load(MODEL_PATH)

    obs, acts, rews = collect_rollout_data(env, model)

    analyze_state_distribution(obs)
    analyze_action_distribution(acts)
    analyze_reward_process(rews)
    plot_rollout_flow()

    env.close()
    print("Code5 analysis finished. Figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
