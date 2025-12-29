# code4_record_episode.py
from smac.env import StarCraft2Env
import numpy as np

def run_and_save(map_name, save_replay, model=None):
    env = StarCraft2Env(
        map_name=map_name,
        replay_dir="replays" if save_replay else None,
        save_replay=save_replay
    )

    env.reset()
    terminated = False

    while not terminated:
        avail_actions = env.get_avail_actions()
        if model is None:
            actions = [np.random.choice(np.nonzero(a)[0]) for a in avail_actions]
        else:
            actions = model.select_actions(env)  # 伪接口，替换为 PyMARL runner

        reward, terminated, _ = env.step(actions)

    env.close()

if __name__ == "__main__":
    # 训练前（随机策略）
    run_and_save("3m", save_replay=True, model=None)

    # 训练后（加载已训练模型）
    # run_and_save("3m", save_replay=True, model=trained_model)
