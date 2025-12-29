import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO

ENV_ID = "Reacher-v4"
MAX_STEPS = 300
FPS = 30

# =========================
# 工具函数：录制一条轨迹
# =========================
def record_episode(env, policy_fn, out_path):
    frames = []
    obs, _ = env.reset()

    for _ in range(MAX_STEPS):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(frame)

        if terminated or truncated:
            break

    imageio.mimsave(out_path, frames, fps=FPS)
    print(f"Saved video: {out_path}")


# =========================
# 1. 训练前：随机策略
# =========================
env_random = gym.make(ENV_ID, render_mode="rgb_array")

def random_policy(obs):
    return env_random.action_space.sample()

record_episode(
    env_random,
    random_policy,
    out_path="reacher_before_random.mp4"
)

env_random.close()


# =========================
# 2. 训练后：PPO 策略
# =========================
env_ppo = gym.make(ENV_ID, render_mode="rgb_array")

model = PPO.load("ppo_reacher_final")  # 你的训练模型

def ppo_policy(obs):
    action, _ = model.predict(obs, deterministic=True)
    return action

record_episode(
    env_ppo,
    ppo_policy,
    out_path="reacher_after_ppo.mp4"
)

env_ppo.close()
