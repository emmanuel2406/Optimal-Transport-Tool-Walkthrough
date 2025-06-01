"""
What is an entropy-regularized optimal transport problem?
It is the same as the optimal transport problem, minimum energy to move a point cloud to another point cloud, but 
we add a penalty to the original objective to favor solutions with *high entropy*. This makes the problem smoother
and easier to solve, which gives tractability to larger datasets.
"""
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import ott
from ott.geometry import costs, pointcloud
from ott.solvers import linear

FIGURE_DIR = "geometry/figures"

def create_points(rng: jax.Array, n: int, m: int, d: int):
    rngs = jax.random.split(rng, 3)
    x = jax.random.normal(rngs[0], (n, d)) + 1
    y = jax.random.uniform(rngs[1], (m, d))
    return x, y


rng = jax.random.key(0)
n, m, d = 13, 17, 2
x, y = create_points(rng, n=n, m=m, d=d)
geom = pointcloud.PointCloud(x, y)

ot = linear.solve(geom)
# The out object contains many things, among which the regularized OT cost
print(
    " Sinkhorn has converged: ",
    ot.converged,
    "\n",
    "Error upon last iteration: ",
    ot.errors[(ot.errors > -1)][-1],
    "\n",
    "Sinkhorn required ",
    jnp.sum(ot.errors > -1),
    " iterations to converge. \n",
    "Entropy regularized OT cost: ",
    ot.ent_reg_cost,
    "\n",
    "OT cost (without entropy): ",
    jnp.sum(ot.matrix * ot.geom.cost_matrix),
)

def visualize_coupling(ot):
    # you can instantiate the OT matrix
    P = ot.matrix
    plt.imshow(P, cmap="Purples")
    plt.colorbar();
    plt.savefig(f"{FIGURE_DIR}/coupling.png")

visualize_coupling(ot)

def visualize_transport(ot):
    plott = ott.tools.plot.Plot()
    _ = plott(ot)
    plt.savefig(f"{FIGURE_DIR}/transport.png")

visualize_transport(ot)

# Optimization loop with fixed-length gradient descent
def optimize(
    x: jnp.ndarray,
    y: jnp.ndarray,
    num_iter: int = 300,
    dump_every: int = 5,
    learning_rate: float = 0.2,
    **kwargs,  # passed to the pointcloud.PointCloud geometry
):
    # Wrapper function that returns OT cost and OT output given a geometry.
    def reg_ot_cost(geom):
        out = linear.solve(geom)
        return out.reg_ot_cost, out

    # Apply jax.value_and_grad operator. Note that we make explicit that
    # we only wish to compute gradients w.r.t the first output,
    # using the has_aux flag. We also jit that function.
    reg_ot_cost_vg = jax.jit(jax.value_and_grad(reg_ot_cost, has_aux=True))

    # Run a naive, fixed stepsize, gradient descent on locations `x`.
    ots = []
    titles = []
    for i in range(0, num_iter + 1):
        geom = pointcloud.PointCloud(x, y, **kwargs)
        (reg_ot_cost, ot), geom_g = reg_ot_cost_vg(geom)
        assert ot.converged
        x = x - geom_g.x * learning_rate
        if i % dump_every == 0:
            ots.append(ot)
            titles.append(f"Iter {i}: Reg OT Cost: {reg_ot_cost}")
    return ots, titles

# Helper function to plot successively the optimal transports
def plot_ots(ots, titles, cost_type):
    fig = plt.figure(figsize=(8, 5))
    plott = ott.tools.plot.Plot(fig=fig, title=cost_type)
    anim = plott.animate(ots, frame_rate=4, titles=titles)
    filename = f"{FIGURE_DIR}/animation_{cost_type}.html"
    with open(filename, 'w') as f:
        f.write(anim.to_jshtml())
    plt.close()

# 2-Wasserstein gradient flow with squared Euclidean cost
plot_ots(*optimize(x, y, num_iter=100, epsilon=1e-2, cost_fn=costs.SqEuclidean()), cost_type="2-Wasserstein distance")

# 1-Wasserstein gradient flow with Euclidean cost
plot_ots(*optimize(x, y, num_iter=250, epsilon=1e-2, cost_fn=costs.Euclidean()), cost_type="1-Wasserstein distance")

# 0.5-Wasserstein gradient flow  with Euclidean cost
@jax.tree_util.register_pytree_node_class
class Custom(costs.TICost):
    """Custom, translation invariant cost, sqrt of Euclidean norm."""

    def h(self, z):
        return jnp.sqrt(jnp.abs(jnp.linalg.norm(z)))
plot_ots(*optimize(x, y, num_iter=400, epsilon=1e-2, cost_fn=Custom()), cost_type="0.5-Wasserstein distance")

#Cosine-Wasserstein gradient flow
plot_ots(*optimize(x, y, num_iter=400, epsilon=1e-2, cost_fn=costs.Cosine()), cost_type="Cosine-Wasserstein distance")


# Render the html in your browser to play the clips
# For larger powers of optimizations, it penalizes outliers more, and so 2-Wasserstein converges a lot faster in general
# When the powers are smaller, the points that have already converged start to jitter out of place
# Cosine-Wasserstein is simply not good, and applies a slow rotation until all points eventually get aligned

# Plotting utility
def plot_map(x, y, znew, tznew, forward: bool = True):
    plt.figure(figsize=(10, 8))
    if forward:
        label = r"$x_{\text{new}}$"
        tlabel = r"$T_{x\rightarrow y}(x_{\text{new}})$"
        marker_t = "X"
        marker_n = "o"
    else:
        label = r"$y_{\text{new}}$"
        tlabel = r"$T_{y\rightarrow x}(y_{\text{new}})$"
        marker_t = "o"
        marker_n = "X"

    plt.quiver(
        *znew.T,
        *(tznew - znew).T,
        color="k",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.007,
    )
    plt.scatter(*x.T, s=200, edgecolors="k", marker="o", label=r"$x$")
    plt.scatter(*y.T, s=200, edgecolors="k", marker="X", label=r"$y$")
    plt.scatter(*znew.T, s=200, edgecolors="k", marker=marker_n, label=label)
    plt.scatter(*tznew.T, s=150, edgecolors="k", marker=marker_t, label=tlabel)
    plt.legend(fontsize=22)
    if forward:
        plt.savefig(f"{FIGURE_DIR}/map_forward.png")
    else:
        plt.savefig(f"{FIGURE_DIR}/map_backward.png")

dual_potentials = ot.to_dual_potentials()
xnew, ynew = create_points(jax.random.key(1), 10, 10, d)

# Instead of learning hard transports such as the OT matrix, we learn soft transports from the sinkhorn algorithm
# This gives *entropic potential* learnt by Sinkhorn allows for generalization to new points

plot_map(x, y, xnew, dual_potentials.transport(xnew))
plot_map(x, y, ynew, dual_potentials.transport(ynew, forward=False), forward=False)
