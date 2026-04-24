# Coupled Oscillator Data Generation Scripts

This repository contains Python scripts for numerically integrating systems of coupled nonlinear oscillators and generating trajectory data for training and testing.

The repository includes the following two scripts:

- `generate_data_coupled_van_der_Pol.py`
- `generate_data_coupled_duffing.py`

Both scripts compute time evolution using `scipy.integrate.solve_ivp`.

---

## Requirements

```bash
pip install numpy scipy
```

Python libraries used:

- `numpy`
- `scipy`

---

## File Structure

```text
.
├── generate_data_coupled_van_der_Pol.py
├── generate_data_coupled_duffing.py
└── README.md
```

---

## 1. `generate_data_coupled_van_der_Pol.py`

### Overview

This script simulates a system of coupled Van der Pol oscillators. Each oscillator has a state variable `x_i` and a velocity variable `v_i`.

The state vector is arranged as follows:

```text
[x1, v1, x2, v2, ..., xN, vN]
```

### Model

The basic equations for each oscillator are:

```text
dx_i/dt = v_i

dv_i/dt = μ_i (1 - x_i^2) v_i - x_i + coupling
```

Here, `μ_i` controls the strength of the nonlinear damping term for oscillator `i`.

### Main Functions

#### `coupled_van_der_pol(t, y, mu, adj_matrix, c=0)`

Defines the right-hand side of the coupled Van der Pol oscillator system.

Arguments:

- `t`: Time.
- `y`: State vector `[x1, v1, x2, v2, ..., xN, vN]`.
- `mu`: Nonlinearity parameter for each oscillator.
- `adj_matrix`: Adjacency matrix representing the coupling structure between oscillators.
- `c`: Coupling coefficient used for small systems. The default value is `0`.

Returns:

- `dy`: Time derivative of the state vector.

#### `simulate_coupled_van_der_pol(mu, adj_matrix, y0, t_span, t_eval)`

Numerically integrates the coupled Van der Pol oscillator system using `solve_ivp`.

Arguments:

- `mu`: Nonlinearity parameter for each oscillator.
- `adj_matrix`: Adjacency matrix.
- `y0`: Initial state.
- `t_span`: Integration interval `(t_start, t_end)`.
- `t_eval`: Time points at which the solution is stored.

Returns:

- `OdeResult`: Result object returned by `solve_ivp`.

### Runtime Behavior

When the script is executed directly, it performs the following steps:

1. Sets the number of oscillators to `N = 3`.
2. Sets the number of samples to `sampling = 5001`.
3. Sets the random seed to `seed = 0`.
4. Constructs a nearest-neighbor adjacency matrix.
5. Randomly generates `μ` and the initial condition `y0`.
6. Simulates one training trajectory.
7. Simulates 100 test trajectories by adding noise to the initial condition.
8. Prints `Simulation completed.` when the simulation finishes.

---

## 2. `generate_data_coupled_duffing.py`

### Overview

This script simulates a system of coupled Duffing oscillators. As in the Van der Pol script, each oscillator has a state variable `x_i` and a velocity variable `v_i`.

The state vector is arranged as follows:

```text
[x1, v1, x2, v2, ..., xN, vN]
```

### Model

The basic equations for each oscillator are:

```text
dx_i/dt = v_i

dv_i/dt = -δ_i v_i - α_i x_i - β_i x_i^3 + coupling
```

The parameters have the following meanings:

- `δ_i`: Damping coefficient.
- `α_i`: Linear stiffness coefficient.
- `β_i`: Nonlinear stiffness coefficient.

### Main Functions

#### `coupled_duffing(t, y, delta, alpha, beta, adj_matrix, c=0)`

Defines the right-hand side of the coupled Duffing oscillator system.

Arguments:

- `t`: Time.
- `y`: State vector `[x1, v1, x2, v2, ..., xN, vN]`.
- `delta`: Damping coefficient for each oscillator.
- `alpha`: Linear stiffness coefficient for each oscillator.
- `beta`: Nonlinear stiffness coefficient for each oscillator.
- `adj_matrix`: Adjacency matrix representing the coupling structure between oscillators.
- `c`: Coupling coefficient used for small systems. The default value is `0`.

Returns:

- `dy`: Time derivative of the state vector.

#### `simulate_coupled_duffing(delta, alpha, beta, adj_matrix, y0, t_span, t_eval)`

Numerically integrates the coupled Duffing oscillator system using `solve_ivp`.

Arguments:

- `delta`: Damping coefficient for each oscillator.
- `alpha`: Linear stiffness coefficient for each oscillator.
- `beta`: Nonlinear stiffness coefficient for each oscillator.
- `adj_matrix`: Adjacency matrix.
- `y0`: Initial state.
- `t_span`: Integration interval `(t_start, t_end)`.
- `t_eval`: Time points at which the solution is stored.

Returns:

- `OdeResult`: Result object returned by `solve_ivp`.

### Runtime Behavior

When the script is executed directly, it performs the following steps:

1. Sets the number of oscillators to `N = 3`.
2. Sets the number of samples to `sampling = 5001`.
3. Sets the random seed to `seed = 0`.
4. Constructs a nearest-neighbor adjacency matrix.
5. Randomly generates `δ`, `α`, `β`, and the initial condition `y0`.
6. Simulates one training trajectory.
7. Simulates 100 test trajectories by adding noise to the initial condition.
8. Prints `Simulation completed.` when the simulation finishes.

---

## Coupling Structure

Both scripts use the following nearest-neighbor adjacency matrix when the number of oscillators is `N = 3`:

```python
adj_matrix = np.diag(np.ones(N - 1), k=1) + np.diag(np.ones(N - 1), k=-1)
```

For `N = 3`, this adjacency matrix is:

```text
[[0, 1, 0],
 [1, 0, 1],
 [0, 1, 0]]
```

This means that oscillator 1 is coupled to oscillator 2, and oscillator 2 is coupled to oscillator 3.

In the code, when the number of oscillators is 3 or larger, the coupling term is computed using `adj_matrix`. When the number of oscillators is less than 3, a simpler chain-style coupling based on the argument `c` is used instead.

---

## Simulation Settings

Both scripts use the following common settings:

```python
N = 3
sampling = 5001
dt = 0.01
t_span = (0, (sampling - 1) * dt)
t_eval = np.linspace(t_span[0], t_span[1], sampling)
```

With these settings, the system is simulated from time `0` to `50.0` with a time step of `0.01`.

---

## Usage

To run the Van der Pol oscillator simulation:

```bash
python generate_data_coupled_van_der_Pol.py
```

To run the Duffing oscillator simulation:

```bash
python generate_data_coupled_duffing.py
```

If the script finishes successfully, the following message is printed:

```text
Simulation completed.
```

---

## Notes

The current scripts do not save the simulation results to files. The results are stored in variables such as `result`, `result_test`, or `test`, but they are discarded when the script exits.

To save the results, add a command such as `numpy.save` or `numpy.savez`. For example:

```python
np.savez("van_der_pol_train.npz", t=result.t, y=result.y, mu=mu, y0=y0)
```

The `result.y` array returned by `solve_ivp` has shape `(2N, number_of_time_points)`. Its rows correspond to the following variables:

```text
x1, v1, x2, v2, ..., xN, vN
```

---

## Differences Between the Scripts

| Item | Van der Pol | Duffing |
|---|---|---|
| File name | `generate_data_coupled_van_der_Pol.py` | `generate_data_coupled_duffing.py` |
| Main nonlinearity | `μ(1 - x^2)v` | `βx^3` |
| Parameters | `mu` | `delta`, `alpha`, `beta` |
| Coupling term | `adj_matrix[i, j] * (x_j - x_i)` | `adj_matrix[i, j] * (x_j - x_i)^3` |
| Test initial-condition noise | Normal distribution | Uniform distribution |
| Training trajectory time interval | `0` to `50.0` | `0` to `50.0` |
| Test trajectory time interval | `0` to `50.0` | Approximately `0` to `10.0` |

---

## Possible Improvements

Possible improvements include:

1. Save simulation results as `.npz` or `.csv` files.
2. Allow parameters and initial conditions to be specified through command-line arguments.
3. Store training and test data explicitly as arrays.
4. Remove outdated references to `gamma` and `omega` from the Duffing script docstring.
5. Clarify that `c` is only used when `N < 3` and is not used in the default execution path.
6. Make the test trajectory length consistent between the Van der Pol and Duffing scripts.

---

