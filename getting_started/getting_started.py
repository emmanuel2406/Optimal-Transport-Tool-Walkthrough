import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from ott.geometry import pointcloud
from ott.solvers import linear

FIGURE_DIR = "getting_started/figures"

# Generate two 2-d point clouds of 7 and 11 points, respectively.
rngs = jax.random.split(jax.random.key(0), 2)
d, n_x, n_y = 2, 7, 11
x = jax.random.normal(rngs[0], (n_x, d))
y = jax.random.normal(rngs[1], (n_y, d)) + 0.5

def visualize_point_clubs(x, y):
    x_args = {"s": 100, "label": r"source $x$", "marker": "s", "edgecolor": "k"}
    y_args = {"s": 100, "label": r"target $y$", "edgecolor": "k", "alpha": 0.75}
    plt.figure(figsize=(9, 6))
    plt.scatter(x[:, 0], x[:, 1], **x_args)
    plt.scatter(y[:, 0], y[:, 1], **y_args)
    plt.legend()
    plt.savefig(f"{FIGURE_DIR}/point_clouds.png")

visualize_point_clubs(x, y)

"""
Store the ground cost between the two datasets
The default cost function is the squared Euclidean distance (SqEuclidean)
"""
geom = pointcloud.PointCloud(x, y, cost_fn=None)

"""
Sinkhorn algorithm (see Resource [1]) used to calculate the optimal coupling for `geom`. Wrapped in
convenience function `linear.solve()`
"""
solve_fn = jax.jit(linear.solve) # jitting => second time is faster (assuming similar shapes)
ot = solve_fn(geom)


def visualize_coupling(ot):
    plt.figure(figsize=(10, 6))
    plt.imshow(ot.matrix)
    plt.colorbar()
    plt.title("Optimal Coupling Matrix")
    plt.savefig(f"{FIGURE_DIR}/coupling.png")

visualize_coupling(ot)

"""
Sinkhorn output stores a lot of metadata, such as lower and upper bounds for the 
true OT-cost
"""
print(f"2-Wasserstein, lower bound = {ot.dual_cost:3f}, upper = {ot.primal_cost:3f}")


def reg_ot_cost(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """
    Define wrapper to calculate regularized OT cost for a pair of point clouds.
    """
    geom = pointcloud.PointCloud(x, y)
    ot = linear.solve(geom)
    return ot.reg_ot_cost


# Gradient function (no args)
r_ot = jax.value_and_grad(reg_ot_cost)
# grad_x essentially gives one step of gradient flow from x to y
cost, grad_x = r_ot(x, y)
assert grad_x.shape == x.shape


def gradient_flow(x: jnp.ndarray, y: jnp.ndarray, step: float = 2.0):
    """
    Step gives the learning rate (similar to SGD)
    Too big => overshoot, too small => slow convergence
    """
    x_args = {"s": 100, "label": r"source $x$", "marker": "s", "edgecolor": "k"}
    y_args = {"s": 100, "label": r"target $y$", "edgecolor": "k", "alpha": 0.75}
    x_t = x
    quiv_args = {"scale": 1, "angles": "xy", "scale_units": "xy", "width": 0.01}
    f, axes = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(12, 4))

    for iteration, ax in enumerate(axes):
        cost, grad_x = r_ot(x_t, y)
        ax.scatter(x_t[:, 0], x_t[:, 1], **x_args)
        ax.quiver(
            x_t[:, 0],
            x_t[:, 1],
            -step * grad_x[:, 0],
            -step * grad_x[:, 1],
            **quiv_args,
        )
        ax.scatter(y[:, 0], y[:, 1], **y_args)
        ax.set_title(f"iter: {iteration}, Reg OT cost: {cost:.3f}")
        x_t -= step * grad_x
    plt.savefig(f"{FIGURE_DIR}/gradient_flow.png")

gradient_flow(x, y, step=2.0)





