# code1_env_check.py
from smac.env import StarCraft2Env
import numpy as np

def main():
    env = StarCraft2Env(map_name="3m")  # 经典小队地图
    env_info = env.get_env_info()

    print("=== Environment Info ===")
    for k, v in env_info.items():
        print(f"{k}: {v}")

    env.reset()
    obs = env.get_obs()
    state = env.get_state()
    avail_actions = env.get_avail_actions()

    print("\n=== One Step Check ===")
    print("Number of agents:", len(obs))
    print("Obs dim (agent 0):", obs[0].shape)
    print("State dim:", state.shape)
    print("Available actions (agent 0):", avail_actions[0])

    actions = [np.random.choice(np.nonzero(a)[0]) for a in avail_actions]
    reward, terminated, info = env.step(actions)

    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Info:", info)

    env.close()
    print("\n[OK] SMAC environment check passed.")

if __name__ == "__main__":
    main()
