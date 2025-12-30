# code5_data_analysis.py
# 数据分析与处理：
# 状态分布分析
# 动作空间输出特性
# 奖励与学习过程关系（TensorBoard: rollout/ep_rew_mean）


import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ENV_ID = "Reacher-v4"
MODEL_PATH = "ppo_reacher_final"

TB_LOG_DIR = "./ppo_reacher_tb/"

OUT_DIR = "./figs_code5"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_EPISODES = 20
MAX_STEPS = 200


# =========================
# 收集 rollout 数据（用于状态 / 动作分析）
# =========================
def collect_rollout_data(env, model):
    obs_list, act_list = [], []

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        for t in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, _, terminated, truncated, _ = env.step(action)

            obs_list.append(obs)
            act_list.append(action)

            obs = next_obs
            if terminated or truncated:
                break

    return np.array(obs_list), np.array(act_list)


# =========================
# 状态分布分析
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
# 奖励与学习过程关系
# 直接读取 TensorBoard: rollout/ep_rew_mean
# =========================
def analyze_reward_from_tensorboard(tb_root_dir):
    # 自动找到最新一次 PPO 训练目录（如 PPO_1 / PPO_2）
    subdirs = [
        os.path.join(tb_root_dir, d)
        for d in os.listdir(tb_root_dir)
        if os.path.isdir(os.path.join(tb_root_dir, d))
    ]
    if len(subdirs) == 0:
        raise RuntimeError("No TensorBoard log directory found.")

    tb_dir = sorted(subdirs)[-1]

    ea = EventAccumulator(tb_dir)
    ea.Reload()

    if "rollout/ep_rew_mean" not in ea.Tags()["scalars"]:
        raise RuntimeError("rollout/ep_rew_mean not found in TensorBoard logs.")

    events = ea.Scalars("rollout/ep_rew_mean")
    steps = [e.step for e in events]
    values = [e.value for e in events]

    plt.figure()
    plt.plot(steps, values)
    plt.xlabel("Training Timesteps")
    plt.ylabel("Episode Reward Mean")
    plt.title("rollout/ep_rew_mean (Training Curve)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/rollout_ep_rew_mean.png", dpi=200)
    plt.close()

# =========================
# 主流程
# =========================
def main():
    env = gym.make(ENV_ID)
    model = PPO.load(MODEL_PATH)

    # 状态 / 动作分析
    obs, acts = collect_rollout_data(env, model)
    analyze_state_distribution(obs)
    analyze_action_distribution(acts)

    analyze_reward_from_tensorboard(TB_LOG_DIR)

    env.close()
    print("Code5 analysis finished. Figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
