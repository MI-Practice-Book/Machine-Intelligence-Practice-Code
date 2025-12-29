import os
import sys
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYMARL_SRC = os.path.join(BASE_DIR, "pymarl-master", "src")
sys.path.append(PYMARL_SRC)

from smac.env import StarCraft2Env

MAP_NAME = "3m"
EPISODES = 2

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "pymarl-master",
    "results",
    "models",
    "qmix__2025-12-25_12-58-58",
    "42"
)

REPLAY_DIR_BEFORE = os.path.join(BASE_DIR, "replays_before")
REPLAY_DIR_AFTER  = os.path.join(BASE_DIR, "replays_after")

os.makedirs(REPLAY_DIR_BEFORE, exist_ok=True)
os.makedirs(REPLAY_DIR_AFTER, exist_ok=True)

def make_env(replay_dir):
    return StarCraft2Env(
        map_name=MAP_NAME,
        replay_dir=replay_dir
    )

def run_random_policy(env, episodes):
    print(">>> BEFORE: random policy")
    for ep in range(episodes):
        env.reset()
        terminated = False
        while not terminated:
            avail_actions = env.get_avail_actions()
            actions = [
                np.random.choice(np.nonzero(a)[0])
                for a in avail_actions
            ]
            _, terminated, _ = env.step(actions)
        print(f"[BEFORE] episode {ep + 1} finished")

def run_trained_policy(env, episodes, checkpoint_dir):
    print(">>> AFTER: trained QMIX policy")

    from controllers.basic_controller import BasicMAC
    from utils.dict2namedtuple import convert

    env_info = env.get_env_info()
    n_actions = env_info["n_actions"]

    scheme = {
        "obs": {
            "vshape": env_info["obs_shape"],
            "dtype": np.float32
        },
        "actions_onehot": {
            "vshape": (n_actions,),
            "dtype": np.float32
        },
        "actions": {
            "vshape": (1,),
            "dtype": np.int64
        },
    }

    args = {
        "n_agents": env_info["n_agents"],
        "n_actions": n_actions,
        "state_shape": env_info["state_shape"],

        "agent": "rnn",
        "agent_output_type": "q",
        "rnn_hidden_dim": 64,

        "obs_agent_id": True,
        "obs_last_action": True,

        "action_selector": "epsilon_greedy",
        "epsilon_start": 0.0,
        "epsilon_finish": 0.0,
        "epsilon_anneal_time": 1,

        "use_cuda": False,
    }
    args = convert(args)

    mac = BasicMAC(
        scheme=scheme,
        groups=None,
        args=args
    )

    mac.load_models(checkpoint_dir)

    for ep in range(episodes):
        env.reset()
        mac.init_hidden(batch_size=1)

        terminated = False
        t = 0
        while not terminated:
            obs = env.get_obs()
            avail_actions = env.get_avail_actions()

            actions = mac.select_actions(
                obs=obs,
                avail_actions=avail_actions,
                t_ep=t,
                test_mode=True   # 评估模式在这里生效
            )

            _, terminated, _ = env.step(actions)
            t += 1

        print(f"[AFTER] episode {ep + 1} finished")

if __name__ == "__main__":

    env_before = make_env(REPLAY_DIR_BEFORE)
    run_random_policy(env_before, EPISODES)
    env_before.close()

    env_after = make_env(REPLAY_DIR_AFTER)
    run_trained_policy(env_after, EPISODES, CHECKPOINT_DIR)
    env_after.close()

    print("✅ replay saved: before / after")
