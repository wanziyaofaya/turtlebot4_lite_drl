import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


# 全局默认绘图风格（对本脚本生成的所有图生效）
plt.rcParams["grid.linewidth"] = 0.1
plt.rcParams["axes.linewidth"] = 0.5

# 默认字号（不传命令行参数时也会固定使用这套）
DEFAULT_TITLE_SIZE = 18
DEFAULT_LABEL_SIZE = 16
DEFAULT_TICK_SIZE = 14
DEFAULT_LEGEND_SIZE = 14


def _configure_font(font: str) -> None:
    """Configure matplotlib font for Chinese text.

    If font is 'auto', it tries common CJK fonts available on Linux.
    If the requested/auto font isn't available, matplotlib may render squares.
    """
    plt.rcParams["axes.unicode_minus"] = False

    available = {f.name for f in font_manager.fontManager.ttflist}
    if font and font != "auto":
        plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
        return

    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
    ]
    chosen = next((c for c in candidates if c in available), None)
    if chosen is not None:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    else:
        print(
            "[WARN] 未检测到可用中文字体，中文可能显示为方块。"
            "建议安装 fonts-noto-cjk 或 fonts-wqy-zenhei。"
        )

def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_actual = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_to_actual:
            return lower_to_actual[name.lower()]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot training curves from one or more TensorBoard-exported CSV files.")
    parser.add_argument(
        "--font",
        type=str,
        default="auto",
        help="Font for Chinese text: 'auto' or a font name (e.g., 'Noto Sans CJK SC')",
    )
    parser.add_argument(
        "--csv",
        required=True,
        nargs="+",
        help="One or more CSV paths (TensorBoard export: Wall time,Step,Value)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels (same count as --csv). Default: inferred from filenames.",
    )
    parser.add_argument("--out", default="curve.png", help="Output image path")
    parser.add_argument("--title", default=None, help="Plot title (default: inferred)")
    parser.add_argument("--title-size", type=float, default=DEFAULT_TITLE_SIZE, help="Title font size")
    parser.add_argument("--label-size", type=float, default=DEFAULT_LABEL_SIZE, help="X/Y label font size")
    parser.add_argument("--tick-size", type=float, default=DEFAULT_TICK_SIZE, help="Tick label font size")
    parser.add_argument("--legend-size", type=float, default=DEFAULT_LEGEND_SIZE, help="Legend font size")
    parser.add_argument(
        "--metric",
        default="auto",
        choices=["auto", "success", "reward"],
        help="How to interpret Value: auto|success|reward",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Moving-average smoothing window (use 0 to disable).",
    )
    parser.add_argument(
        "--xmax-k",
        type=float,
        default=None,
        help="Max x-axis in k-steps (e.g., 500 means show up to 500k steps).",
    )
    parser.add_argument("--ylabel", default=None, help="Y-axis label override")
    parser.add_argument("--ymin", type=float, default=None, help="Y-axis min override")
    parser.add_argument("--ymax", type=float, default=None, help="Y-axis max override")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    args = parser.parse_args()

    _configure_font(args.font)

    if args.labels is not None and len(args.labels) != len(args.csv):
        parser.error("--labels must have the same number of items as --csv")

    csv_paths: list[str] = list(args.csv)
    labels: list[str]
    if args.labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in csv_paths]
    else:
        labels = list(args.labels)

    series_list: list[tuple[np.ndarray, np.ndarray, str]] = []
    global_steps_k_max = 0.0
    global_y_min = np.inf
    global_y_max = -np.inf

    for csv_path, label in zip(csv_paths, labels, strict=True):
        df = pd.read_csv(csv_path)

        # 兼容 TensorBoard 导出的列名：Wall time, Step, Value
        step_col = _pick_column(df, ["step", "global_step"])
        value_col = _pick_column(df, ["value", "success", "success_rate", "reward"])

        if step_col is None:
            step_col = df.columns[0]
        if value_col is None:
            value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        steps = pd.to_numeric(df[step_col], errors="coerce").to_numpy()
        values = pd.to_numeric(df[value_col], errors="coerce").to_numpy()

        mask = np.isfinite(steps) & np.isfinite(values)
        steps = steps[mask]
        values = values[mask]
        if steps.size == 0:
            continue

        # 按 step 排序，保证曲线连线顺序正确
        order = np.argsort(steps)
        steps = steps[order]
        values = values[order]

        # 转换为千步
        steps_k = steps / 1000.0

        # 截取到指定的最大步数（k steps）
        if args.xmax_k is not None:
            x_mask = steps_k <= float(args.xmax_k)
            steps_k = steps_k[x_mask]
            values = values[x_mask]
            if steps_k.size == 0:
                continue

        # 平滑：移动平均
        y = values
        if args.smooth and args.smooth > 1 and y.size > 0:
            y = pd.Series(y).rolling(window=args.smooth, min_periods=1).mean().to_numpy()

        global_steps_k_max = max(global_steps_k_max, float(np.nanmax(steps_k)))
        global_y_min = min(global_y_min, float(np.nanmin(y)))
        global_y_max = max(global_y_max, float(np.nanmax(y)))

        series_list.append((steps_k, y, label))

    if not series_list:
        raise SystemExit("No valid data points found in the provided CSV files.")

    # 自动判断是否是“成功率”数据，并据此决定是否转百分比、是否固定 0-100
    is_success_like = False
    if args.metric == "success":
        is_success_like = True
    elif args.metric == "reward":
        is_success_like = False
    else:
        # auto: 典型 success_rate 是 [0,1] 或 [0,100]
        is_success_like = (global_y_min >= 0.0) and (global_y_max <= 1.0 or global_y_max <= 100.0)

    if is_success_like:
        converted: list[tuple[np.ndarray, np.ndarray, str]] = []
        for steps_k, y, label in series_list:
            if float(np.nanmax(y)) <= 1.0:
                y = y * 100.0
            converted.append((steps_k, y, label))
        series_list = converted
        default_ylabel = "成功率(%)"
    else:
        default_ylabel = "奖励值" if args.metric == "reward" else "Value"

    # 绘图
    plt.figure(figsize=(9, 5))
    for steps_k, y, label in series_list:
        plt.plot(steps_k, y, linewidth=2.0, label=label)

    # X 轴范围
    x_max_for_plot = float(args.xmax_k) if args.xmax_k is not None else float(global_steps_k_max)
    if x_max_for_plot > 0:
        plt.xlim(0.0, x_max_for_plot)

    # 设置坐标轴
    plt.xlabel("步数", fontsize=float(args.label_size))
    plt.ylabel(args.ylabel or default_ylabel, fontsize=float(args.label_size))
    plt.tick_params(axis="both", which="major", labelsize=float(args.tick_size))

    # X 轴刻度：根据最大步数自动生成（默认每 100k 一个大刻度）
    if x_max_for_plot > 0:
        max_tick = int(np.ceil(x_max_for_plot / 100.0) * 100)
        ticks = list(range(0, max_tick + 1, 100))
        plt.xticks(ticks, ["0" if t == 0 else f"{t}k" for t in ticks])

    # Y 轴
    if args.ymin is not None or args.ymax is not None:
        plt.ylim(args.ymin, args.ymax)
    elif is_success_like:
        plt.yticks([0, 20, 40, 60, 80, 100])
        # 给 100% 留出顶部空间，避免曲线与边界重叠
        plt.ylim(0, 105)
        # 100% 参考线，让“到顶”更明显
        plt.gca().axhline(100, color="#666666", linestyle=":", linewidth=1.0, alpha=0.6, zorder=0)
    else:
        # reward / raw: 自动留白
        if np.isfinite(global_y_min) and np.isfinite(global_y_max):
            span = max(1e-9, global_y_max - global_y_min)
            pad = 0.05 * span
            plt.ylim(global_y_min - pad, global_y_max + pad)

    # 网格
    plt.grid(True, linestyle="--", alpha=0.1)

    # 标题
    if args.title is not None:
        title = args.title
    else:
        title = "成功率" if is_success_like else ("Training Curves" if len(series_list) > 1 else "Training Curve")
    plt.title(title, fontsize=float(args.title_size))

    if len(series_list) > 1:
        plt.legend(
            loc="best",
            frameon=False,
            fontsize=float(args.legend_size),
        )

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