from __future__ import annotations

from pathlib import Path


def plot_rq2(result: dict, output: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    factors = (("creation_distance", ["<=4", "5-8", "9-16", ">16"], "Creation-to-termination distance"), ("obligation_count", ["1", "2", "3", "4", "5"], "Concurrent obligations"))
    colors = {"precision": "#2463a3", "recall": "#bc4a37", "exact_match": "#39805a"}
    labels = {"precision": "Precision", "recall": "Recall", "exact_match": "EM"}
    for axis, (factor, order, xlabel) in zip(axes, factors):
        available = [bucket for bucket in order if bucket in result[factor]]
        for key in colors:
            values = [100 * result[factor][bucket]["mean"][key] for bucket in available]
            axis.plot(available, values, marker="o", color=colors[key], label=labels[key])
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Performance (%)")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.savefig(output)
    plt.close(figure)


def plot_rq3(rows: list[dict], output: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_name = {row["model"]: row["metrics"] for row in rows}
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    for axis, key, label in zip(axes, ("precision", "recall", "exact_match"), ("Precision", "Recall", "EM")):
        sizes = [5000, 10000, 20000, 40000]
        axis.plot([size / 1000 for size in sizes], [100 * by_name[f"Qwen3-8B-{size}"][key] for size in sizes], marker="o", label="Scenario planning")
        axis.scatter([40], [100 * by_name["Qwen3-8B-direct-synthesis-40000"][key]], marker="s", label="Direct synthesis")
        axis.set_xlabel("Training examples (K)")
        axis.set_ylabel(label + " (%)")
        axis.set_xticks([5, 10, 20, 40])
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def plot_guidance(results: dict, recalls: dict, output: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6.4, 4), constrained_layout=True)
    for model, recall in recalls.items():
        x, y = 100 * recall, 100 * results[model]["delta_sec"]
        axis.scatter([x], [y])
        axis.annotate(model, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("ObligationBench Recall (%)")
    axis.set_ylabel("Change in joint functional/security pass rate (pp)")
    axis.grid(alpha=0.2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
