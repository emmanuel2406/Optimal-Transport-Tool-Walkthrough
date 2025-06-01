"""
Two views of probability measures: either point clouds or histograms supported on a d-dimensional Cartesian grid
Use either `PointCloud` or `Grid` object from ott. 

Although point clouds are more interpretable, 
grids allow for more efficient fundamental operations (O(N^(1 + 1/d)) vs O(N^2)).
"""
import jax
import jax.numpy as jnp
import numpy as np
import timeit

from ott.geometry import costs, grid, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

FIGURE_DIR = "geometry/figures"


keys = jax.random.split(jax.random.key(0), 5)
# Cartesian product 5x6x7 possible points
grid_size = (5, 6, 7)

# Approach 1: Create grid with `x` argument
x = [
    jax.random.uniform(keys[0], (grid_size[0],)),
    jax.random.uniform(keys[1], (grid_size[1],)),
    jax.random.uniform(keys[2], (grid_size[2],)),
]

geom = grid.Grid(x=x)
a = jax.random.uniform(keys[3], grid_size)
b = jax.random.uniform(keys[4], grid_size)
a = a.ravel() / jnp.sum(a)  # Normalize to have unit total mass.
b = b.ravel() / jnp.sum(b)  # "

prob = linear_problem.LinearProblem(geom, a=a, b=b)
solver = sinkhorn.Sinkhorn()
out = solver(prob)

print(f"(1) Regularized OT cost = {out.reg_ot_cost}")

# Approach 2: Create grid with `grid_size` argument
geom = grid.Grid(grid_size=grid_size, epsilon=0.1)

# We recycle the same probability vectors
prob = linear_problem.LinearProblem(geom, a=a, b=b)
out = solver(prob)

print(f"(2) Regularized OT cost = {out.reg_ot_cost}")

# Both approaches assume sqEuclidean cost function by default.
# `cost_fns` argument allows us to give different cost functions per dimension.
@jax.tree_util.register_pytree_node_class
class MyCost(costs.CostFn):
    """An unusual cost function."""

    def norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x**3 + jnp.cos(x) ** 2, axis=-1)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return (
            self.norm(x)
            + self.norm(y)
            - jnp.sum(jnp.sin(x + 1) * jnp.sin(y)) * 2
        )
cost_fns = [MyCost(), costs.SqEuclidean(), MyCost()]  # 1 for each dimension.
geom = grid.Grid(grid_size=grid_size, cost_fns=cost_fns, epsilon=0.1)
prob = linear_problem.LinearProblem(geom, a=a, b=b)
out = solver(prob)

print(f"Mixed Cost Function Regularized OT cost = {out.reg_ot_cost}")


# Computational comparison of Grid vs Point Cloud
grid_size = (37, 29, 43)

keys = jax.random.split(jax.random.key(2), 2)
a = jax.random.uniform(keys[0], grid_size)
b = jax.random.uniform(keys[1], grid_size)
a = a.ravel() / jnp.sum(a)
b = b.ravel() / jnp.sum(b)

print("Total size of grid: ", jnp.prod(jnp.array(grid_size)))

# Instantiates Grid
geometry_grid = grid.Grid(grid_size=grid_size)
prob_grid = linear_problem.LinearProblem(geometry_grid, a=a, b=b)

execution_time = timeit.timeit(lambda: solver(prob_grid).reg_ot_cost.block_until_ready(), number=1)
print(f"Execution time for Grid: {execution_time} seconds")
out_grid = solver(prob_grid)
print(
    f"Regularized optimal transport cost using Grid = {out_grid.reg_ot_cost}\n"
)

# List all 3D points in cartesian product.
x, y, z = np.mgrid[0 : grid_size[0], 0 : grid_size[1], 0 : grid_size[2]]
xyz = jnp.stack(
    [
        jnp.array(x.ravel()) / jnp.maximum(1, grid_size[0] - 1),
        jnp.array(y.ravel()) / jnp.maximum(1, grid_size[1] - 1),
        jnp.array(z.ravel()) / jnp.maximum(1, grid_size[2] - 1),
    ]
).transpose()
# Instantiates PointCloud with `batch_size` argument.
# Computations require being run in batches, otherwise memory would
# overflow. This is achieved by setting `batch_size` to 1024.
geometry_pointcloud = pointcloud.PointCloud(xyz, xyz, batch_size=1024)
prob_pointcloud = linear_problem.LinearProblem(geometry_pointcloud, a=a, b=b)

execution_time = timeit.timeit(lambda: solver(prob_pointcloud).reg_ot_cost.block_until_ready(), number=1)
print(f"Execution time for PointCloud: {execution_time} seconds")
out_pointcloud = solver(prob_pointcloud)
print(
    f"Regularized optimal transport cost using PointCloud = {out_pointcloud.reg_ot_cost}"
)

