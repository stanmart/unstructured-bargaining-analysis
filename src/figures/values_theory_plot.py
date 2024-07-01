import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def shapley_value(Y: np.ndarray) -> np.ndarray:
    return 1 / 3 * Y + 1 / 3


def nucleolus(Y: np.ndarray) -> np.ndarray:
    nuc = np.ones_like(Y) * 1 / 3
    nuc[Y > 1 / 3] = Y[Y > 1 / 3]
    return nuc


def plot_values(
    num_samples: int = 100, multiplier: float = 100, treatments: list[float] = []
) -> Figure:
    Y = np.linspace(0, 1, num_samples)
    fig, ax = plt.subplots()

    ax.plot(Y * multiplier, shapley_value(Y) * multiplier, label="Shapley value")
    ax.plot(Y * multiplier, nucleolus(Y) * multiplier, label="Nucleolus")
    ax.plot(
        Y * multiplier,
        np.ones_like(Y) / 3 * multiplier,
        label="Equal split",
        linestyle="--",
    )

    for treatment in treatments:
        ax.axvline(treatment, color="black", linestyle=":")
        t = ax.text(
            treatment + multiplier * 0.012,
            multiplier * 0.99,
            f"Treatment $Y={treatment}$",
            rotation=-90,
            va="top",
        )
        t.set_bbox(
            dict(facecolor="white", alpha=0.5, boxstyle="round", edgecolor="white")
        )

    ax.set_xlabel("$Y$")
    ax.set_ylabel("Payoff of the big player")
    ax.legend()

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlim(0, multiplier)
    ax.set_ylim(0, multiplier)

    return fig


if __name__ == "__main__":
    fig = plot_values(num_samples=100, multiplier=100, treatments=[10, 30, 90])
    fig.set_size_inches(6, 4)
    fig.savefig(snakemake.output.figure, bbox_inches="tight", dpi=300)  # noqa F821 # type: ignore
