
import argparse
import csv
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


@dataclass(frozen=True)
class Series:
	x: np.ndarray
	y: np.ndarray
	x_name: str
	y_name: str


def _read_csv_as_columns(path: str) -> tuple[list[str], dict[str, list[str]]]:
	with open(path, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		if reader.fieldnames is None:
			raise ValueError(f"CSV has no header: {path}")
		cols: dict[str, list[str]] = {name: [] for name in reader.fieldnames}
		for row in reader:
			for name in reader.fieldnames:
				cols[name].append(row.get(name, ""))
		return list(reader.fieldnames), cols


def _to_float_array(values: list[str]) -> np.ndarray:
	out = np.empty(len(values), dtype=float)
	for i, v in enumerate(values):
		try:
			out[i] = float(v)
		except (TypeError, ValueError):
			out[i] = np.nan
	return out


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
	lower_to_actual = {c.lower(): c for c in columns}
	for name in candidates:
		key = name.lower()
		if key in lower_to_actual:
			return lower_to_actual[key]
	return None


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
	if window <= 1 or y.size == 0:
		return y
	kernel = np.ones(window, dtype=float) / float(window)
	y_pad = np.pad(y, (window - 1, 0), mode="edge")
	return np.convolve(y_pad, kernel, mode="valid")


def _ema(y: np.ndarray, alpha: float) -> np.ndarray:
	if y.size == 0:
		return y
	alpha = float(alpha)
	if not (0.0 < alpha <= 1.0):
		return y
	out = np.empty_like(y, dtype=float)
	out[0] = float(y[0])
	for i in range(1, y.size):
		out[i] = alpha * float(y[i]) + (1.0 - alpha) * out[i - 1]
	return out


def load_series(csv_path: str, xcol: str = "auto", ycol: str = "auto") -> Series:
	columns, cols = _read_csv_as_columns(csv_path)

	if xcol == "auto":
		x_name = _pick_column(columns, ["step", "global_step", "wall time", "wall_time", "time"]) or columns[0]
	else:
		x_name = xcol if xcol in columns else (_pick_column(columns, [xcol]) or xcol)

	if ycol == "auto":
		y_name = _pick_column(columns, ["value", "reward", "success", "success_rate", "loss"]) or (
			columns[2] if len(columns) >= 3 else columns[-1]
		)
	else:
		y_name = ycol if ycol in columns else (_pick_column(columns, [ycol]) or ycol)

	if x_name not in cols:
		raise KeyError(f"xcol '{x_name}' not found in CSV columns: {columns}")
	if y_name not in cols:
		raise KeyError(f"ycol '{y_name}' not found in CSV columns: {columns}")

	x = _to_float_array(cols[x_name])
	y = _to_float_array(cols[y_name])

	mask = np.isfinite(x) & np.isfinite(y)
	x = x[mask]
	y = y[mask]

	return Series(x=x, y=y, x_name=x_name, y_name=y_name)


def plot_series(
	series: Series,
	out_path: str,
	title: str,
	metric: str = "auto",
	smooth: int = 0,
	smooth_method: str = "ma",
	ema_alpha: float = 0.1,
	xlabel: str | None = None,
	xunit: str = "auto",
	ylabel: str | None = None,
	ymin: float | None = None,
	ymax: float | None = None,
	no_show: bool = False,
) -> None:
	# style
	plt.rcParams["grid.linewidth"] = 0.1
	plt.rcParams["axes.linewidth"] = 0.5

	x = series.x
	y = series.y.astype(float, copy=False)

	x_name_l = series.x_name.lower()
	is_step_like = ("step" in x_name_l) or ("global" in x_name_l)

	# 你的 CSV 里 Step 是“训练轮数/epoch”，默认不应该 /1000。
	# xunit=auto: 只有当 Step 非常大(>=1e4)才用 k 展示，否则原样展示。
	max_x = float(np.nanmax(x)) if x.size > 0 else 0.0
	if xunit == "k":
		use_k = True
	elif xunit == "raw":
		use_k = False
	else:
		use_k = bool(is_step_like and max_x >= 1.0e4)

	x_plot = x / 1000.0 if use_k else x

	# auto success-rate heuristic
	is_success_like = False
	metric_l = metric.lower()
	y_name_l = series.y_name.lower()
	title_l = title.lower()
	is_mde_like = (metric_l == "mde") or ("mde" in y_name_l) or ("mde" in title_l)
	is_mse_like = (metric_l == "mse") or ("mse" in y_name_l) or ("mse" in title_l)
	is_rmse_like = (metric_l == "rmse") or ("rmse" in y_name_l) or ("rmse" in title_l)
	is_r2_like = (metric_l == "r2") or ("r2" in y_name_l) or ("r2" in title_l)
	if y.size > 0:
		y_min = float(np.nanmin(y))
		y_max = float(np.nanmax(y))
		if metric == "success":
			is_success_like = True
		elif metric_l in {"reward", "mde", "mse", "rmse", "r2"}:
			is_success_like = False
		else:
			is_success_like = (y_min >= 0.0) and (y_max <= 1.0 or y_max <= 100.0)

	if is_success_like and y.size > 0 and float(np.nanmax(y)) <= 1.0:
		y = y * 100.0

	if is_success_like:
		default_ylabel = "Success Rate (%)"
	elif metric_l == "reward":
		default_ylabel = "Reward"
	elif is_mde_like:
		default_ylabel = "MDE"
	elif is_rmse_like:
		default_ylabel = "RMSE"
	elif is_mse_like:
		default_ylabel = "MSE"
	elif is_r2_like:
		default_ylabel = "R2"
	else:
		default_ylabel = "Value"

	if smooth and smooth > 1 and y.size > 0:
		if smooth_method == "ema":
			y = _ema(y, alpha=ema_alpha)
		else:
			y = _moving_average(y, window=smooth)

	plt.figure(figsize=(8, 4.5))
	plt.plot(x_plot, y, "r-", linewidth=2.5)

	plt.title(title)
	if xlabel is not None:
		x_label = xlabel
	else:
		if use_k:
			x_label = "Training Steps (k)"
		elif is_step_like:
			x_label = "Training Epoch"
		else:
			x_label = series.x_name
	plt.xlabel(x_label)
	plt.ylabel(ylabel or default_ylabel)

	if x_plot.size > 0:
		if use_k:
			# 自适应刻度：尽量 6~8 个
			max_k = float(np.nanmax(x_plot))
			if max_k > 0:
				raw_step = max_k / 6.0
				pow10 = 10 ** np.floor(np.log10(raw_step))
				step = float(10 * pow10)
				for m in (1, 2, 5, 10):
					cand = float(m * pow10)
					if cand >= raw_step:
						step = cand
						break
				max_tick = float(np.ceil(max_k / step) * step)
				ticks = np.arange(0.0, max_tick + 0.5 * step, step)
				plt.xticks(ticks, ["0" if t == 0 else f"{int(t)}k" for t in ticks])
		else:
			# 轮数/episode：整数刻度
			ax = plt.gca()
			ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))

	if ymin is not None or ymax is not None:
		plt.ylim(ymin, ymax)
	elif is_success_like:
		plt.yticks([0, 20, 40, 60, 80, 100])
		plt.ylim(0, 100)
	else:
		if y.size > 0:
			y_min = float(np.nanmin(y))
			y_max = float(np.nanmax(y))
			span = max(1e-9, y_max - y_min)
			pad = 0.05 * span
			plt.ylim(y_min - pad, y_max + pad)

	plt.grid(True, linestyle="--", alpha=0.1)
	plt.tight_layout()
	plt.savefig(out_path, dpi=400)
	print(f"Saved: {out_path}")

	has_display = bool(os.environ.get("DISPLAY"))
	if not no_show and has_display:
		plt.show()


def main() -> int:
	parser = argparse.ArgumentParser(description="Plot a curve from a CSV (TensorBoard-exported compatible).")
	parser.add_argument("--csv", required=True, help="CSV path, e.g. tools/tabm_20260127_1729_4.csv")
	parser.add_argument("--out", default="tools/curve.png", help="Output image path")
	parser.add_argument("--title", default="Curve", help="Plot title")
	parser.add_argument("--xcol", default="auto", help="X column name (default: auto)")
	parser.add_argument("--ycol", default="auto", help="Y column name (default: auto)")
	parser.add_argument(
		"--metric",
		default="auto",
		choices=["auto", "success", "reward", "mde","mse","rmse","r2"],
		help="auto|success|reward|mde|mse|rmse|r2 (affects ylabel and y-limits)",
	)
	parser.add_argument("--smooth", type=int, default=0, help="Smoothing window (moving average). 0 disables")
	parser.add_argument("--smooth-method", choices=["ma", "ema"], default="ma", help="ma|ema")
	parser.add_argument("--ema-alpha", type=float, default=0.1, help="EMA alpha in (0,1]")
	parser.add_argument("--xlabel", default=None, help="Override x-axis label")
	parser.add_argument(
		"--xunit",
		choices=["auto", "raw", "k"],
		default="auto",
		help="X axis unit: auto (default), raw (no scaling), k (divide by 1000)",
	)
	parser.add_argument("--ylabel", default=None, help="Override y-axis label")
	parser.add_argument("--ymin", type=float, default=None, help="Y-axis min")
	parser.add_argument("--ymax", type=float, default=None, help="Y-axis max")
	parser.add_argument("--no-show", action="store_true", help="Do not open a window")
	args = parser.parse_args()

	series = load_series(args.csv, xcol=args.xcol, ycol=args.ycol)
	plot_series(
		series,
		out_path=args.out,
		title=args.title,
		metric=args.metric,
		smooth=args.smooth,
		smooth_method=args.smooth_method,
		ema_alpha=args.ema_alpha,
		xlabel=args.xlabel,
		xunit=args.xunit,
		ylabel=args.ylabel,
		ymin=args.ymin,
		ymax=args.ymax,
		no_show=args.no_show,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

