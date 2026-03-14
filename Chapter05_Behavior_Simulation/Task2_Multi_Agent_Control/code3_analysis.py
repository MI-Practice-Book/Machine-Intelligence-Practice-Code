import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ★ 关键：无 GUI 服务器必须加
import matplotlib.pyplot as plt
import matplotlib as mpl

# ===== 核心：Linux 中文配置 =====
# 设置字体为文泉驿正黑（Linux 原生支持）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# 可选：如果还想让图表更美观，可添加以下配置
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['font.size'] = 10

# ========= 路径配置 =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COUT_PATH = os.path.join(
    BASE_DIR,
    "pymarl-master/results/sacred/2/cout.txt"   # 改成你的 run id
)
OUT_DIR = os.path.join(BASE_DIR, "analysis_figs")

os.makedirs(OUT_DIR, exist_ok=True)

# ========= 解析 cout.txt =========
def parse_cout():
    pattern = re.compile(
        r"t_env:\s+(\d+)\s+\|\s+Episode:\s+(\d+).*?"
        r"battle_won_mean:\s+([\d\.]+).*?"
        r"dead_allies_mean:\s+([\d\.]+).*?"
        r"dead_enemies_mean:\s+([\d\.]+).*?"
        r"ep_length_mean:\s+([\d\.]+)",
        re.S
    )

    records = []
    with open(COUT_PATH, "r") as f:
        text = f.read()

    for m in pattern.finditer(text):
        records.append({
            "t_env": int(m.group(1)),
            "episode": int(m.group(2)),
            "battle_won_mean": float(m.group(3)),
            "dead_allies_mean": float(m.group(4)),
            "dead_enemies_mean": float(m.group(5)),
            "ep_length_mean": float(m.group(6)),
        })

    if len(records) == 0:
        raise RuntimeError("No valid stats found in cout.txt")

    df = pd.DataFrame(records)
    print("[INFO] Parsed columns:", df.columns.tolist())
    print("[INFO] Total records:", len(df))
    return df


def main():
    df = parse_cout()

    # ===== 图 1：团队胜率 =====
    plt.figure(figsize=(6, 4))
    plt.plot(df["t_env"], df["battle_won_mean"])
    plt.xlabel("环境步数")
    plt.ylabel("胜率")
    plt.title("训练期间的团队胜率")
    plt.grid(True)
    path1 = os.path.join(OUT_DIR, "胜率.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print("[SAVED]", path1)

    # ===== 图 2：平均回合长度 =====
    plt.figure(figsize=(6, 4))
    plt.plot(df["t_env"], df["ep_length_mean"])
    plt.xlabel("环境步数")
    plt.ylabel("回合长度")
    plt.title("训练期间的回合长度")
    plt.grid(True)
    path2 = os.path.join(OUT_DIR, "回合长度.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print("[SAVED]", path2)

    # ===== 图 3：敌我损失对比 =====
    plt.figure(figsize=(6, 4))
    plt.plot(df["t_env"], df["dead_enemies_mean"], label="Dead Enemies", marker='^')
    plt.plot(df["t_env"], df["dead_allies_mean"], label="Dead Allies", marker='o')
    plt.xlabel("环境步数")
    plt.ylabel("数量")
    plt.title("战斗结果统计")
    plt.legend()
    plt.grid(True)
    path3 = os.path.join(OUT_DIR, "战斗结果.png")
    plt.savefig(path3, dpi=150)
    plt.close()
    print("[SAVED]", path3)

    print("\n[OK] All analysis figures saved to:")
    print(" ", OUT_DIR)


if __name__ == "__main__":
    main()
