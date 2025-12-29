# code3_visualize_training.py
# 可视化分析：Value Loss / Policy Loss / Evaluation Return Curve
# 依赖：pip install tensorboard matplotlib numpy

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator


TB_DIR = "./ppo_reacher_tb"   # 训练脚本里 tensorboard_log="./ppo_reacher_tb/"
EVAL_DIR = "./logs"          # EvalCallback 里 log_path="./logs/"

OUT_DIR = "./figs_code3"
os.makedirs(OUT_DIR, exist_ok=True)


def find_event_files(tb_dir: str):
    # SB3 的 event 文件通常在 tb_dir 下的子目录里（tb_log_name 默认是 PPO_1、PPO_2...）
    patterns = [
        os.path.join(tb_dir, "**", "events.out.tfevents.*"),
        os.path.join(tb_dir, "events.out.tfevents.*"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(files, key=os.path.getmtime)
    return files


def load_scalars_from_tb(tb_dir: str, tags):
    """
    从 TensorBoard event 文件中读取指定 tag 的标量序列。
    返回 dict[tag] = (steps, values)
    """
    event_files = find_event_files(tb_dir)
    if not event_files:
        raise FileNotFoundError(
            f"No tensorboard event files found under: {tb_dir}\n"
            f"请确认训练时开启了 tensorboard_log，并且已经跑过训练。"
        )

    # 取最新的一个 event 文件（通常就是最新一次训练）
    latest = event_files[-1]
    ea = event_accumulator.EventAccumulator(
        latest,
        size_guidance={
            event_accumulator.SCALARS: 0,  # 0 表示尽量读全量
        }
    )
    ea.Reload()

    available = set(ea.Tags().get("scalars", []))
    data = {}

    for tag in tags:
        if tag not in available:
            data[tag] = None
            continue
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.int64)
        vals = np.array([e.value for e in events], dtype=np.float32)
        data[tag] = (steps, vals)

    return data, latest, sorted(list(available))


def moving_average(y, window=50):
    if y is None or len(y) < window:
        return y
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(y, kernel, mode="valid")


def plot_and_save(x, y, title, xlabel, ylabel, out_path, smooth_window=0):
    plt.figure()
    if smooth_window and y is not None and len(y) >= smooth_window:
        y_s = moving_average(y, smooth_window)
        x_s = x[-len(y_s):]
        plt.plot(x_s, y_s)
    else:
        plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def load_eval_curve(eval_dir: str):
    """
    EvalCallback 默认会在 log_path 下写 evaluations.npz
    格式通常包含：
      - timesteps: shape (n_eval,)
      - results: shape (n_eval, n_episodes)  每次评估多条episode回报
      - ep_lengths: shape (n_eval, n_episodes)
    """
    npz_path = os.path.join(eval_dir, "evaluations.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Cannot find {npz_path}\n"
            f"请确认训练时使用了 EvalCallback，并且 log_path 设置为 {eval_dir}"
        )

    data = np.load(npz_path)
    timesteps = data["timesteps"]
    results = data["results"]  # 每次评估的多回合回报
    mean_return = results.mean(axis=1)
    std_return = results.std(axis=1)
    return timesteps, mean_return, std_return, npz_path


def main():
    # ====== 1) 从 TensorBoard 读取 loss ======
    # SB3 PPO 常见 tag（不同版本可能略有差异）
    # 重点取：train/value_loss, train/policy_gradient_loss（策略损失）
    tags = [
        "train/value_loss",
        "train/policy_gradient_loss",
    ]

    tb_data, latest_event_file, available_tags = load_scalars_from_tb(TB_DIR, tags)
    print("Using TB event file:", latest_event_file)

    # 如果 tag 缺失，给出可选提示（不直接崩）
    for tag in tags:
        if tb_data[tag] is None:
            print(f"[WARN] TB tag not found: {tag}")
            print("       Available scalar tags include (partial):")
            print("       ", ", ".join(available_tags[:30]))
            print("       如果你的版本使用了不同 tag，可在上面列表里找对应名字替换。")

    # ====== 2) 画 Value Loss ======
    if tb_data["train/value_loss"] is not None:
        steps, vals = tb_data["train/value_loss"]
        plot_and_save(
            steps, vals,
            title="Value Loss Curve (SB3 PPO)",
            xlabel="Environment Steps",
            ylabel="Value Loss",
            out_path=os.path.join(OUT_DIR, "value_loss.png"),
            smooth_window=50
        )

    # ====== 3) 画 Policy Loss（SB3 记录的是 policy_gradient_loss） ======
    if tb_data["train/policy_gradient_loss"] is not None:
        steps, vals = tb_data["train/policy_gradient_loss"]
        plot_and_save(
            steps, vals,
            title="Policy Loss Curve (Policy Gradient Loss, SB3 PPO)",
            xlabel="Environment Steps",
            ylabel="Policy Loss",
            out_path=os.path.join(OUT_DIR, "policy_loss.png"),
            smooth_window=50
        )

    # ====== 4) 画 Evaluation Return Curve（来自 EvalCallback 的 evaluations.npz） ======
    try:
        t, mean_r, std_r, npz_path = load_eval_curve(EVAL_DIR)
        print("Using eval file:", npz_path)

        plt.figure()
        plt.plot(t, mean_r)
        # 不指定颜色；用 fill_between 做误差带
        plt.fill_between(t, mean_r - std_r, mean_r + std_r, alpha=0.2)
        plt.title("Evaluation Return Curve (Mean ± Std)")
        plt.xlabel("Environment Steps")
        plt.ylabel("Evaluation Return")
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, "evaluation_return.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")
    except FileNotFoundError as e:
        print("[WARN]", e)

    print("\nDone. Figures saved under:", OUT_DIR)


if __name__ == "__main__":
    main()
