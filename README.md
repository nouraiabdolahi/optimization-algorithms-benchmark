# Optimization Algorithms Benchmark on Non-Convex Functions

First-order optimization algorithms (GD, Momentum, Adam) implemented from scratch in NumPy and evaluated on the 2D Rosenbrock function.

## Overview
This repository contains a simple benchmark comparing how standard first-order optimization methods navigate complex, ill-conditioned landscapes. The algorithms are tested on the Rosenbrock "banana" function to visualize convergence paths and oscillations.

## Algorithms Implemented
- **Vanilla Gradient Descent (GD):** Fixed learning rate updates along the negative gradient.
- **Gradient Descent with Momentum:** Accelerates convergence along flat valleys and dampens oscillations.
- **Adam:** Combines adaptive learning rates with first and second moment estimates.

## Mathematical Formulation
The benchmark optimizes the classic 2D Rosenbrock function:

$$f(x, y) = (a - x)^2 + b(y - x^2)^2$$

with standard parameters $a = 1$, $b = 100$, having a global minimum at $(1, 1)$ where $f(1, 1) = 0$.

Analytical gradients used for the updates:

$$\frac{\partial f}{\partial x} = -2(a - x) - 4bx(y - x^2)$$

$$\frac{\partial f}{\partial y} = 2b(y - x^2)$$

## Requirements
- Python 3.x
- `numpy`
- `matplotlib`

## Usage
Run the benchmark script:
python optimizer_benchmark.py

## Outputs
Console summary of final coordinates, function values, and iterations.
A generated contour plot saved as optimizer_trajectories.png.

## Project Structure
─ optimizer_benchmark.py       # Main benchmark script
─  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3aafc063-9845-40ab-b115-68cfee555b41" />
                               # Optimization trajectory plot
─ README.md                    # Documentation

## Notes
Convergence behavior strongly depends on hyperparameter selection (learning rate, momentum coefficients). This repository is intended for experimental and educational exploration.
