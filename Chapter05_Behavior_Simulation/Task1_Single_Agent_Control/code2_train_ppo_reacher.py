# train_ppo_reacher.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

ENV_ID = "Reacher-v4"
TOTAL_TIMESTEPS = 1_000_000

def make_env():
    env = gym.make(ENV_ID)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    learning_rate=3e-4,
    tensorboard_log="./ppo_reacher_tb/"
)

eval_env = DummyVecEnv([make_env])

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=10_000,
    deterministic=True,
    render=False
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback
)

model.save("ppo_reacher_final")
env.close()
