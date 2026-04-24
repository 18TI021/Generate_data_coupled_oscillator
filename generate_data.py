from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import tqdm
except ModuleNotFoundError:  # pragma: no cover - environment dependent fallback
    tqdm = None


@dataclass
class SimulationResult:
    t: np.ndarray
    y: np.ndarray


def integrate_fixed_step_rk4(dynamics, y0: np.ndarray, t_eval: np.ndarray, args: tuple) -> SimulationResult:
    """Integrate an ODE on the supplied evaluation grid with classical RK4."""
    if t_eval.ndim != 1:
        raise ValueError("t_eval must be a one-dimensional array.")
    if len(t_eval) < 2:
        raise ValueError("t_eval must contain at least two time points.")

    states = np.empty((y0.shape[0], len(t_eval)), dtype=float)
    states[:, 0] = y0
    state = y0.astype(float, copy=True)

    for idx, t_now in enumerate(t_eval[:-1]):
        dt = float(t_eval[idx + 1] - t_now)
        if dt <= 0.0:
            raise ValueError("t_eval must be strictly increasing.")

        k1 = dynamics(t_now, state, *args)
        k2 = dynamics(t_now + 0.5 * dt, state + 0.5 * dt * k1, *args)
        k3 = dynamics(t_now + 0.5 * dt, state + 0.5 * dt * k2, *args)
        k4 = dynamics(t_now + dt, state + dt * k3, *args)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states[:, idx + 1] = state

    return SimulationResult(t=t_eval.copy(), y=states)


def coupled_duffing(
    t: float,
    y: np.ndarray,
    delta: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    omega: float,
    c: float,
) -> np.ndarray:
    """Coupled Duffing oscillator."""
    dy = np.zeros_like(y)
    system_num = y.shape[0] // 2
    for i in range(system_num):
        x = y[2 * i]
        v = y[2 * i + 1]
        dxdt = v
        dvdt = -delta[i] * v - alpha[i] * x - beta[i] * x**3 + gamma[i] * np.cos(omega * t)
        if i == 0:
            dvdt += c * (y[2] - x)
        elif i == system_num - 1:
            dvdt += c * (y[2 * (system_num - 2)] - x)
        else:
            dvdt += c * (y[2 * (i - 1)] + y[2 * (i + 1)] - 2 * x)
        dy[2 * i] = dxdt
        dy[2 * i + 1] = dvdt
    return dy


def simulate_coupled_duffing(
    delta: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    omega: float,
    c: float,
    y0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
):
    if not np.isclose(t_eval[0], t_span[0]) or not np.isclose(t_eval[-1], t_span[1]):
        raise ValueError("t_eval must match the provided t_span.")
    return integrate_fixed_step_rk4(
        coupled_duffing,
        y0,
        t_eval,
        args=(delta, alpha, beta, gamma, omega, c),
    )


def couple_vanderpol(
    t: float,
    y: np.ndarray,
    n_oscillators: int,
    c: float,
    mu: np.ndarray,
) -> np.ndarray:
    """Coupled Van der Pol oscillator."""
    dydt = np.zeros_like(y)
    for i in range(n_oscillators):
        x = y[2 * i]
        v = y[2 * i + 1]
        dxdt = v
        dvdt = mu[i] * (1 - x**2) * v - x
        if i > 0:
            dvdt += c * (y[2 * (i - 1)] - x)
        if i < n_oscillators - 1:
            dvdt += c * (y[2 * (i + 1)] - x)
        dydt[2 * i] = dxdt
        dydt[2 * i + 1] = dvdt
    return dydt


def simulate_coupled_vdp(
    n_oscillators: int,
    c: float,
    mu: np.ndarray,
    y0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
):
    if not np.isclose(t_eval[0], t_span[0]) or not np.isclose(t_eval[-1], t_span[1]):
        raise ValueError("t_eval must match the provided t_span.")
    return integrate_fixed_step_rk4(
        couple_vanderpol,
        y0,
        t_eval,
        args=(n_oscillators, c, mu),
    )


def prepare_output_dirs(base_dir: Path) -> None:
    for subdir in ("matrix", "data", "fig"):
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)


def epsilon_dir_name(epsilon: float) -> str:
    return f"epsilon{int(np.log10(epsilon))}"


def generate_duffing_case(
    output_root: Path,
    n_oscillators: int,
    c: float,
    seed: int,
    epsilon: float,
    sampling: int,
    dt: float,
    test_count: int,
    test_steps: int,
) -> None:
    if test_steps > sampling:
        raise ValueError(f"test_steps must be <= duffing sampling ({sampling}).")

    base_dir = (
        output_root
        / "couple_duffing"
        / f"N:{n_oscillators}"
        / f"c:{c}"
        / f"seed{seed}"
        / epsilon_dir_name(epsilon)
    )
    prepare_output_dirs(base_dir)

    rng = np.random.default_rng(seed=seed)
    delta = np.round(rng.uniform(0.1, 0.3, n_oscillators), 2)
    alpha = np.round(rng.uniform(-1.0, -0.5, n_oscillators), 2)
    beta = np.round(rng.uniform(0.5, 1.0, n_oscillators), 2)
    gamma = np.round(rng.uniform(0.0, 0.0, n_oscillators), 2)
    omega = 1.2
    y0 = rng.uniform(-1.5, 1.5, 2 * n_oscillators)

    t_span = (0.0, sampling * dt)
    t_eval = np.linspace(t_span[0], t_span[1], sampling)
    result = simulate_coupled_duffing(delta, alpha, beta, gamma, omega, c, y0, t_span, t_eval)

    x_train = result.y[:, :-1]
    y_train = result.y[:, 1:]
    np.savetxt(base_dir / "data" / "snapshot_x_train.txt", x_train)
    np.savetxt(base_dir / "data" / "snapshot_y_train.txt", y_train)

    for test_idx in range(test_count):
        rng_test = np.random.default_rng(test_idx + 31471)
        y0_test = y0 + rng_test.uniform(-0.1, 0.1, 2 * n_oscillators)
        test = simulate_coupled_duffing(delta, alpha, beta, gamma, omega, c, y0_test, t_span, t_eval)
        np.savetxt(base_dir / "data" / f"snapshot_x_test_{test_idx}.txt", test.y[:, :test_steps])


def generate_vdp_case(
    output_root: Path,
    n_oscillators: int,
    c: float,
    seed: int,
    epsilon: float,
    sampling: int,
    n_relax: int,
    dt: float,
    test_count: int,
    test_steps: int,
) -> None:
    if test_steps > sampling:
        raise ValueError(f"test_steps must be <= vdp sampling ({sampling}).")

    base_dir = (
        output_root
        / "couple_vdp"
        / f"N:{n_oscillators}"
        / f"c:{c}"
        / f"seed{seed}"
        / epsilon_dir_name(epsilon)
    )
    prepare_output_dirs(base_dir)

    total_sampling = sampling + n_relax

    rng = np.random.default_rng(seed)
    mu = np.round(rng.uniform(1.0, 1.5, n_oscillators), decimals=2)
    y0_x = np.round(rng.uniform(-pi / 2, pi / 2, n_oscillators), decimals=2)
    y0_v = np.round(rng.uniform(-1.0, 1.0, n_oscillators), decimals=2)

    y0 = np.zeros(2 * n_oscillators)
    for i in range(n_oscillators):
        y0[2 * i] = y0_x[i]
        y0[2 * i + 1] = y0_v[i]

    t_span = (0.0, dt * (total_sampling - 1))
    t_eval = np.linspace(t_span[0], t_span[1], total_sampling)
    result = simulate_coupled_vdp(n_oscillators, c, mu, y0, t_span, t_eval)

    x_data = result.y[:, n_relax:-1]
    y_data = result.y[:, n_relax + 1 :]
    np.savetxt(base_dir / "data" / "snapshot_x.txt", x_data)
    np.savetxt(base_dir / "data" / "snapshot_y.txt", y_data)

    test_end = n_relax + test_steps
    for test_idx in range(test_count):
        rng_test = np.random.default_rng(test_idx + 31471)
        y0_test = y0 + rng_test.normal(-0.1, 0.1, 2 * n_oscillators)
        result_test = simulate_coupled_vdp(n_oscillators, c, mu, y0_test, t_span, t_eval)
        np.savetxt(
            base_dir / "data" / f"snapshot_x_test_{test_idx}.txt",
            result_test.y[:, n_relax:test_end],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate training/test trajectories for coupled Duffing and Van der Pol systems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--system",
        choices=("duffing", "vdp", "both"),
        default="both",
        help="Which dataset family to generate.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("."), help="Root directory for outputs.")
    parser.add_argument("--N", type=int, default=3, help="Number of coupled oscillators.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)), help="Random seeds to generate.")
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[1.0e0],
        help="Epsilon values used in output directory naming.",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation time step.")
    parser.add_argument("--test-count", type=int, default=10, help="Number of test trajectories per case.")
    parser.add_argument("--test-steps", type=int, default=501, help="Saved length of each test trajectory.")
    parser.add_argument(
        "--duffing-couplings",
        type=float,
        nargs="+",
        default=[1.0],
        help="Coupling coefficients for Duffing data.",
    )
    parser.add_argument(
        "--duffing-sampling",
        type=int,
        default=2001,
        help="Number of sampled time points for Duffing trajectories.",
    )
    parser.add_argument(
        "--vdp-couplings",
        type=float,
        nargs="+",
        default=[1.0],
        help="Coupling coefficients for Van der Pol data.",
    )
    parser.add_argument(
        "--vdp-sampling",
        type=int,
        default=2001,
        help="Number of post-relaxation samples saved for Van der Pol trajectories.",
    )
    parser.add_argument(
        "--vdp-relax",
        type=int,
        default=500,
        help="Relaxation steps discarded before saving Van der Pol training data.",
    )
    return parser


def progress(iterable, desc: str):
    if tqdm is None:
        return iterable
    return tqdm.tqdm(iterable, desc=desc)


def run_duffing(args: argparse.Namespace) -> None:
    for c in progress(args.duffing_couplings, desc="Duffing"):
        for seed in args.seeds:
            for epsilon in args.epsilons:
                generate_duffing_case(
                    output_root=args.output_root,
                    n_oscillators=args.N,
                    c=c,
                    seed=seed,
                    epsilon=epsilon,
                    sampling=args.duffing_sampling,
                    dt=args.dt,
                    test_count=args.test_count,
                    test_steps=args.test_steps,
                )


def run_vdp(args: argparse.Namespace) -> None:
    for c in progress(args.vdp_couplings, desc="Van der Pol"):
        for seed in args.seeds:
            for epsilon in args.epsilons:
                generate_vdp_case(
                    output_root=args.output_root,
                    n_oscillators=args.N,
                    c=c,
                    seed=seed,
                    epsilon=epsilon,
                    sampling=args.vdp_sampling,
                    n_relax=args.vdp_relax,
                    dt=args.dt,
                    test_count=args.test_count,
                    test_steps=args.test_steps,
                )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.system in ("duffing", "both"):
            run_duffing(args)
        if args.system in ("vdp", "both"):
            run_vdp(args)
    except (ModuleNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
