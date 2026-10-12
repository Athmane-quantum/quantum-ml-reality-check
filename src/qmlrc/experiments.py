"""Experiment definitions. Changing a value here changes what run_all reproduces."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    dataset: str  # "creditcard" or "synthetic"
    n_samples: int = 2000  # rows drawn per seed (without replacement)
    positive_fraction: float = 0.10  # fraud rate of the drawn subsample
    test_size: float = 0.30
    n_components: int | None = 8  # PCA features = qubits; None = keep all raw features
    n_folds: int = 3
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    families: tuple[str, ...] = ("classical", "quantum")
    vqc_steps: int = 150
    vqc_grid: dict | None = None
    description: str = ""


@dataclass(frozen=True)
class GeometryConfig:
    """Geometric difference (Huang et al.) of each quantum kernel model, at its CV-selected
    settings, against a list of classical kernels, on the training rows of every seed."""

    name: str
    benchmarks: tuple[str, ...]  # benchmark experiments whose kernel models are analysed
    models: tuple[str, ...] = ("QSVM-fidelity", "QSVM-projected", "QSVM-fidelity-sep", "QSVM-projected-sep")
    description: str = ""


@dataclass(frozen=True)
class ConcentrationConfig:
    name: str
    dataset: str
    n_samples: int = 300
    positive_fraction: float = 0.10
    qubits: tuple[int, ...] = (2, 4, 6, 8, 10)
    scales: tuple[float, ...] = (0.1, 0.3, 1.0)
    seed: int = 0
    description: str = field(default="")


CREDITCARD_Q8 = BenchmarkConfig(
    name="creditcard_q8",
    dataset="creditcard",
    description=(
        "Credit-card fraud (OpenML 1597), 2,000 rows per seed at 10% fraud, 70/30 split, "
        "8 PCA features (= 8 qubits) for every model."
    ),
)

CREDITCARD_FULL_CLASSICAL = BenchmarkConfig(
    name="creditcard_full_classical",
    dataset="creditcard",
    n_components=None,
    families=("classical",),
    description=(
        "Same rows and splits as creditcard_q8, but classical models see all 29 features. "
        "Not a like-for-like comparison: it shows what the 8-feature restriction costs."
    ),
)

CREDITCARD_Q8_SEPARABLE = replace(
    CREDITCARD_Q8,
    name="creditcard_q8_separable",
    families=("quantum-separable",),
    description=(
        "Entanglement ablation: the three quantum models with their entangling gates removed, "
        "on exactly the same rows, splits, folds and grids as creditcard_q8."
    ),
)

GEOMETRIC_DIFFERENCE = GeometryConfig(
    name="geometric_difference",
    benchmarks=("creditcard_q8", "creditcard_q8_separable"),
    description=(
        "Huang et al. geometric difference between each tuned quantum kernel and classical kernels, "
        "per seed, on the training rows of creditcard_q8."
    ),
)

KERNEL_CONCENTRATION = ConcentrationConfig(
    name="kernel_concentration",
    dataset="creditcard",
    description=(
        "Spread of fidelity and projected kernel values over 300 credit-card rows, "
        "as a function of qubit count and bandwidth."
    ),
)

FULL = [CREDITCARD_Q8, CREDITCARD_FULL_CLASSICAL, CREDITCARD_Q8_SEPARABLE, GEOMETRIC_DIFFERENCE, KERNEL_CONCENTRATION]


def quick(configs=FULL):
    """Tiny synthetic versions of the same experiments, for smoke tests (about a minute)."""
    out = []
    for cfg in configs:
        if isinstance(cfg, BenchmarkConfig):
            out.append(replace(cfg, dataset="synthetic", n_samples=300, seeds=(0,), vqc_steps=10,
                               vqc_grid={"scale": [0.3], "n_layers": [1]}))
        elif isinstance(cfg, ConcentrationConfig):
            out.append(replace(cfg, dataset="synthetic", n_samples=60, qubits=(2, 4)))
        else:
            out.append(cfg)
    return out
