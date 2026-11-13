import numpy as np
import matplotlib.pyplot as plt


def rosenbrock(x, y, a=1.0, b=100.0):
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def rosenbrock_grad(x, y, a=1.0, b=100.0):
    grad_x = -2.0 * (a - x) - 4.0 * b * x * (y - x ** 2)
    grad_y = 2.0 * b * (y - x ** 2)
    return np.array([grad_x, grad_y])


def vanilla_gradient_descent(start_point, lr=0.001, max_iter=5000, tol=1e-4):
    point = np.asarray(start_point, dtype=float)
    history = [point.copy()]

    for _ in range(max_iter):
        grad = rosenbrock_grad(point[0], point[1])
        point = point - lr * grad
        history.append(point.copy())

        if np.linalg.norm(grad) < tol:
            break

    return np.array(history)


def momentum_descent(start_point, lr=0.001, beta=0.9, max_iter=5000, tol=1e-4):
    point = np.asarray(start_point, dtype=float)
    velocity = np.zeros_like(point)
    history = [point.copy()]

    for _ in range(max_iter):
        grad = rosenbrock_grad(point[0], point[1])
        velocity = beta * velocity + lr * grad
        point = point - velocity
        history.append(point.copy())

        if np.linalg.norm(grad) < tol:
            break

    return np.array(history)


def adam_descent(start_point, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=5000, tol=1e-4):
    point = np.asarray(start_point, dtype=float)
    m = np.zeros_like(point)
    v = np.zeros_like(point)
    history = [point.copy()]

    for t in range(1, max_iter + 1):
        grad = rosenbrock_grad(point[0], point[1])

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad ** 2)

        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        point = point - (lr / (np.sqrt(v_hat) + eps)) * m_hat
        history.append(point.copy())

        if np.linalg.norm(grad) < tol:
            break

    return np.array(history)


def report(name, path):
    final = path[-1]
    print(f"{name:<12} -> final (x, y) = ({final[0]:.5f}, {final[1]:.5f}), steps = {len(path) - 1}")


def main():
    start = [-1.5, 1.5]

    path_gd = vanilla_gradient_descent(start, lr=0.0008, max_iter=2500)
    path_mom = momentum_descent(start, lr=0.0008, beta=0.85, max_iter=2500)
    path_adam = adam_descent(start, lr=0.03, max_iter=2500)

    print("Optimization results:")
    report("Vanilla GD", path_gd)
    report("Momentum", path_mom)
    report("Adam", path_adam)

    x = np.linspace(-2.0, 2.0, 400)
    y = np.linspace(-1.0, 3.0, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    plt.figure(figsize=(9, 6))
    plt.contour(X, Y, Z, levels=np.logspace(0, 3, 25), cmap="gray_r", alpha=0.6)

    plt.plot(path_gd[:, 0], path_gd[:, 1], label="Vanilla GD", color="blue", linewidth=1.5)
    plt.plot(path_mom[:, 0], path_mom[:, 1], label="Momentum", color="green", linewidth=1.5)
    plt.plot(path_adam[:, 0], path_adam[:, 1], label="Adam", color="red", linewidth=1.8)

    plt.plot(start[0], start[1], "ro", label="Start (-1.5, 1.5)")
    plt.plot(1.0, 1.0, "k*", markersize=12, label="Global Minimum (1, 1)")

    plt.title("Optimization Trajectories on the Rosenbrock Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("optimizer_trajectories.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
