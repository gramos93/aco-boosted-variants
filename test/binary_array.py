import numpy as np
from mealpy import PermutationVar, WOA, Problem, ACOR, BinaryVar

# Define the graph representation
graph = np.array([
    [1, 3, 1],
    [1, 3, 1],
    [1, 1, 1],
])


class ShortestPathProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        self.eps = 1e10         # Penalty function for vertex with 0 connection
        super().__init__(bounds, minmax, **kwargs)

    def valid_path(self, grid, start, end):
        rows, cols = len(grid), len(grid[0])
        visited = [[False] * cols for _ in range(rows)]

        def dfs(row, col):
            if row < 0 or row >= rows or col < 0 or col >= cols or grid[row][col] == 0 or visited[row][col]:
                return

            visited[row][col] = True

            # Check if the destination is reached
            if (row, col) == end:
                return True

            # Explore adjacent cells
            if dfs(row + 1, col) or dfs(row - 1, col) or dfs(row, col + 1) or dfs(row, col - 1):
                return True

            return False

        # Start DFS from the source point
        return dfs(start[0], start[1])

    # Calculate the fitness of an individual
    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        path = x_decoded["placement_var"]
        path_2d = path.reshape((3, 3)).tolist()

        cost = 0
        dist = 0
        start_node = (0,0)
        end_node = (3,3)
        for i in range(len(path_2d)):
            for j in range(len(path_2d[0])):
                if path_2d[i][j] == 1:
                    cost += self.data[i][j]
                    dist += np.sqrt((i - end_node[0])**2 + (j - end_node[1])**2)
        if not self.valid_path(path_2d, start_node, end_node):
            return self.eps
        return cost + dist


num_nodes = len(graph)*len(graph)
#bounds = PermutationVar(valid_set=list(range(0, num_nodes)), name="path")
bounds = BinaryVar(n_vars=num_nodes, name="placement_var")
problem = ShortestPathProblem(bounds=bounds, minmax="min", data=graph)

model = WOA.OriginalWOA(epoch=200, pop_size=20)
model.solve(problem)

print(f"Best agent: {model.g_best}")                    # Encoded solution
print(f"Best solution: {model.g_best.solution}")        # Encoded solution
print(f"Best fitness: {model.g_best.target.fitness}")
print(np.array(model.g_best.solution).reshape((3, 3)))
