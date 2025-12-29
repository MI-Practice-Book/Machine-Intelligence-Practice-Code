# check_env.py
import gymnasium as gym

env = gym.make("Reacher-v4", render_mode="rgb_array")

obs, info = env.reset()
print("obs shape:", obs.shape)
print("action space:", env.action_space)

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("reward:", reward)
    if terminated or truncated:
        break

env.close()
print("Environment OK")
