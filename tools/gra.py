import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 全局默认绘图风格（对本脚本生成的所有图生效）
plt.rcParams["grid.linewidth"] = 0.1
plt.rcParams["axes.linewidth"] = 0.5

def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_actual = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_to_actual:
            return lower_to_actual[name.lower()]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot success rate curve from a TensorBoard-exported CSV.")
    parser.add_argument("--csv", default="tools/SAC_20251229_1510SAC_SAC_0.csv", help="Path to CSV file")
    parser.add_argument("--out", default="success_rate_simple.png", help="Output image path")
    parser.add_argument("--title", default="Success Rate", help="Plot title")
    parser.add_argument(
        "--metric",
        default="auto",
        choices=["auto", "success", "reward"],
        help="How to interpret Value: auto|success|reward",
    )
    parser.add_argument("--ylabel", default=None, help="Y-axis label override")
    parser.add_argument("--ymin", type=float, default=None, help="Y-axis min override")
    parser.add_argument("--ymax", type=float, default=None, help="Y-axis max override")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # 兼容 TensorBoard 导出的列名：Wall time, Step, Value
    step_col = _pick_column(df, ["step", "global_step"])
    value_col = _pick_column(df, ["value", "success", "success_rate", "reward"]) 

    if step_col is None:
        step_col = df.columns[0]
    if value_col is None:
        value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    steps = pd.to_numeric(df[step_col], errors="coerce").to_numpy()
    success = pd.to_numeric(df[value_col], errors="coerce").to_numpy()

    mask = np.isfinite(steps) & np.isfinite(success)
    steps = steps[mask]
    success = success[mask]

    # 转换为千步
    steps_k = steps / 1000.0

    # 自动判断是否是“成功率”数据，并据此决定是否转百分比、是否固定 0-100
    y = success
    is_success_like = False
    if y.size > 0:
        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))
        if args.metric == "success":
            is_success_like = True
        elif args.metric == "reward":
            is_success_like = False
        else:
            # auto: 典型 success_rate 是 [0,1] 或 [0,100]
            is_success_like = (y_min >= 0.0) and (y_max <= 1.0 or y_max <= 100.0)

    if is_success_like and y.size > 0:
        # 如果是小数(0~1)，转换为百分比
        if float(np.nanmax(y)) <= 1.0:
            y = y * 100.0
        default_ylabel = "Success Rate (%)"
    else:
        default_ylabel = "Value"

    # 绘图
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps_k, y, "r-", linewidth=2.5)

    # 设置坐标轴
    plt.xlabel("Training Steps (k)")
    plt.ylabel(args.ylabel or default_ylabel)

    # X 轴刻度：根据最大步数自动生成（默认每 100k 一个大刻度）
    if steps_k.size > 0:
        max_k = float(np.nanmax(steps_k))
        max_tick = int(np.ceil(max_k / 100.0) * 100)
        ticks = list(range(0, max_tick + 1, 100))
        plt.xticks(ticks, ["0" if t == 0 else f"{t}k" for t in ticks])

    # Y 轴
    if args.ymin is not None or args.ymax is not None:
        plt.ylim(args.ymin, args.ymax)
    elif is_success_like:
        plt.yticks([0, 20, 40, 60, 80, 100])
        plt.ylim(0, 100)
    else:
        # reward / raw: 自动留白
        if y.size > 0:
            y_min = float(np.nanmin(y))
            y_max = float(np.nanmax(y))
            span = max(1e-9, y_max - y_min)
            pad = 0.05 * span
            plt.ylim(y_min - pad, y_max + pad)

    # 网格
    plt.grid(True, linestyle="--", alpha=0.1)

    # 标题
    plt.title(args.title)

    # 保存
    plt.tight_layout()
    plt.savefig(args.out, dpi=400)

    # 没有图形界面时，plt.show() 会阻塞；默认自动跳过
    has_display = bool(os.environ.get("DISPLAY"))
    if not args.no_show and has_display:
        plt.show()

    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# python3 tools/gra.py --csv 'tools/SAC_20251229_1510SAC_SAC_0 .csv' --out tools/reward.png --title 'Reward' --metric reward --no-show && ls -la tools/reward.png