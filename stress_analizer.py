#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_jmeter_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = {
        "timeStamp",
        "elapsed",
        "success",
        "responseCode",
        "allThreads",
        "Latency",
        "Connect",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"В CSV не хватает колонок: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timeStamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Иногда success может приехать строкой "true"/"false"
    if df["success"].dtype == object:
        df["success"] = df["success"].astype(str).str.lower().eq("true")

    # На всякий случай responseCode приводим к строке
    df["responseCode"] = df["responseCode"].astype(str)

    # Время от начала теста в секундах
    t0 = df["timestamp"].min()
    df["seconds_from_start"] = (df["timestamp"] - t0).dt.total_seconds()

    return df


def aggregate_by_time(df: pd.DataFrame, freq: str = "10s") -> pd.DataFrame:
    """
    Агрегирует метрики по временным окнам.
    freq примеры: '5s', '10s', '30s'
    """
    grouped = (
        df.set_index("timestamp")
        .groupby(pd.Grouper(freq=freq))
        .apply(
            lambda g: pd.Series(
                {
                    "samples": len(g),
                    "throughput_rps": len(g) / pd.Timedelta(freq).total_seconds(),
                    "avg_elapsed": g["elapsed"].mean(),
                    "median_elapsed": g["elapsed"].median(),
                    "p90_elapsed": g["elapsed"].quantile(0.90),
                    "p95_elapsed": g["elapsed"].quantile(0.95),
                    "p99_elapsed": g["elapsed"].quantile(0.99),
                    "min_elapsed": g["elapsed"].min(),
                    "max_elapsed": g["elapsed"].max(),
                    "avg_threads": g["allThreads"].mean(),
                    "max_threads": g["allThreads"].max(),
                    "error_rate": (~g["success"]).mean() * 100,
                    "avg_latency": g["Latency"].mean(),
                    "avg_connect": g["Connect"].mean(),
                }
            )
        )
        .dropna()
        .reset_index()
    )

    if not grouped.empty:
        t0 = grouped["timestamp"].min()
        grouped["seconds_from_start"] = (
            grouped["timestamp"] - t0
        ).dt.total_seconds()

    return grouped


def save_plot(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранён график: {out_path}")


def plot_response_time_vs_load(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["allThreads"], df["elapsed"], s=8, alpha=0.35)

    # Линия по медиане elapsed для каждого значения нагрузки
    median_by_threads = (
        df.groupby("allThreads", as_index=False)["elapsed"].median()
        .sort_values("allThreads")
    )
    ax.plot(
        median_by_threads["allThreads"],
        median_by_threads["elapsed"],
        linewidth=2,
    )

    ax.set_title("Время отклика в зависимости от нагрузки")
    ax.set_xlabel("Количество активных потоков (allThreads)")
    ax.set_ylabel("Elapsed, ms")
    ax.grid(True, alpha=0.3)

    save_plot(fig, out_dir, "01_response_time_vs_load")


def plot_success_fail_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    ok = df[df["success"]]
    fail = df[~df["success"]]

    ax.scatter(ok["allThreads"], ok["elapsed"], s=8, alpha=0.25, label="Успех")
    ax.scatter(fail["allThreads"], fail["elapsed"], s=12, alpha=0.55, label="Ошибка")

    ax.set_title("Успешные и ошибочные запросы при разной нагрузке")
    ax.set_xlabel("Количество активных потоков (allThreads)")
    ax.set_ylabel("Elapsed, ms")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_plot(fig, out_dir, "02_success_fail_vs_load")


def plot_throughput_and_threads(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.plot(agg["seconds_from_start"], agg["throughput_rps"], label="Throughput (req/s)", color="blue")
    ax1.set_xlabel("Время с начала теста, сек")
    ax1.set_ylabel("Requests/sec")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(agg["seconds_from_start"], agg["avg_threads"], label="Средняя нагрузка (threads)", color="red")
    ax2.set_ylabel("Активные потоки")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Throughput и нагрузка во времени")
    save_plot(fig, out_dir, "03_throughput_and_threads_over_time")


def plot_error_rate_over_time(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(agg["seconds_from_start"], agg["error_rate"])
    ax.set_title("Процент ошибок по времени")
    ax.set_xlabel("Время с начала теста, сек")
    ax.set_ylabel("Error rate, %")
    ax.grid(True, alpha=0.3)

    save_plot(fig, out_dir, "04_error_rate_over_time")


def plot_percentiles_over_time(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(agg["seconds_from_start"], agg["median_elapsed"], label="Median")
    ax.plot(agg["seconds_from_start"], agg["p90_elapsed"], label="P90")
    ax.plot(agg["seconds_from_start"], agg["p95_elapsed"], label="P95")
    ax.plot(agg["seconds_from_start"], agg["p99_elapsed"], label="P99")

    ax.set_title("Перцентили времени отклика по времени")
    ax.set_xlabel("Время с начала теста, сек")
    ax.set_ylabel("Elapsed, ms")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_plot(fig, out_dir, "05_latency_percentiles_over_time")


def plot_avg_response_and_load(agg: pd.DataFrame, out_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.plot(agg["seconds_from_start"], agg["avg_elapsed"], label="Average elapsed", color="green")
    ax1.set_xlabel("Время с начала теста, сек")
    ax1.set_ylabel("Среднее время отклика, ms")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(agg["seconds_from_start"], agg["avg_threads"], label="Средняя нагрузка (threads)", color="orange")
    ax2.set_ylabel("Активные потоки")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Среднее время отклика и нагрузка во времени")
    save_plot(fig, out_dir, "06_avg_response_and_load_over_time")


def plot_response_codes(df: pd.DataFrame, out_dir: Path) -> None:
    code_counts = (
        df["responseCode"]
        .value_counts()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    code_counts.plot(kind="bar", ax=ax)
    ax.set_title("Распределение response codes")
    ax.set_xlabel("Response code")
    ax.set_ylabel("Количество запросов")
    ax.grid(True, axis="y", alpha=0.3)

    save_plot(fig, out_dir, "07_response_codes_distribution")


def write_summary(df: pd.DataFrame, agg: pd.DataFrame, out_dir: Path) -> None:
    summary_path = out_dir / "summary.txt"

    total = len(df)
    error_rate = (~df["success"]).mean() * 100
    avg_elapsed = df["elapsed"].mean()
    median_elapsed = df["elapsed"].median()
    p90 = df["elapsed"].quantile(0.90)
    p95 = df["elapsed"].quantile(0.95)
    p99 = df["elapsed"].quantile(0.99)
    max_threads = df["allThreads"].max()
    avg_threads = df["allThreads"].mean()
    avg_rps = agg["throughput_rps"].mean() if not agg.empty else 0
    peak_rps = agg["throughput_rps"].max() if not agg.empty else 0

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("ИТОГОВАЯ СВОДКА ПО STRESS-ТЕСТУ\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Всего запросов: {total}\n")
        f.write(f"Процент ошибок: {error_rate:.2f}%\n")
        f.write(f"Среднее время отклика: {avg_elapsed:.2f} ms\n")
        f.write(f"Медиана времени отклика: {median_elapsed:.2f} ms\n")
        f.write(f"P90: {p90:.2f} ms\n")
        f.write(f"P95: {p95:.2f} ms\n")
        f.write(f"P99: {p99:.2f} ms\n")
        f.write(f"Средняя нагрузка (allThreads): {avg_threads:.2f}\n")
        f.write(f"Максимальная нагрузка (allThreads): {max_threads}\n")
        f.write(f"Средний throughput: {avg_rps:.2f} req/s\n")
        f.write(f"Пиковый throughput: {peak_rps:.2f} req/s\n")

    print(f"Сохранена сводка: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Визуализация результатов стресс-теста JMeter"
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="Путь к stress-test-result.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("stress_charts"),
        help="Папка для сохранения графиков",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="10s",
        help="Размер временного окна для агрегации, например 5s / 10s / 30s",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df = load_jmeter_csv(args.csv)
    agg = aggregate_by_time(df, freq=args.window)

    plot_response_time_vs_load(df, args.out)
    plot_success_fail_scatter(df, args.out)
    plot_throughput_and_threads(agg, args.out)
    plot_error_rate_over_time(agg, args.out)
    plot_percentiles_over_time(agg, args.out)
    plot_avg_response_and_load(agg, args.out)
    plot_response_codes(df, args.out)
    write_summary(df, agg, args.out)

    print("\nГотово.")


if __name__ == "__main__":
    main()
