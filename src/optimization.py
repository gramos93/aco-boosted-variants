import numpy as np
from mealpy import BinaryVar, Problem
from mealpy.swarm_based.ACOR import OriginalACOR
from gridworld import GridWorld
from abc import ABC, abstractmethod
from typing import List

class SARProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        self.eps = 1e10
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
    
    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        path = x_decoded["placement_var"]
        path_2d = path.reshape(self.data['grid'].shape).tolist()

        cost = 0
        dist = 0
        start_node = (0,0)
        end_node = self.data['target']
        for i in range(len(path_2d)):
            for j in range(len(path_2d[i])):
                if path_2d[i][j] == 1:
                    cost += self.data['grid'][i][j]
        if not self.valid_path(path_2d, (0, 0), end_node):
            return self.eps
        return cost

class Strategy(OriginalACOR):
    # initialize the super class with epochs = 100 and pop_size = 3, for quick testing
    def __init__(self, epochs=100, pop_size=100):
        super().__init__(epochs, pop_size)

class SearchAlgorithm():
    def __init__(self, strategy: Strategy, gridworld: GridWorld) -> None:
        self._strategy = strategy
        self.size = gridworld.size
        #graph = self.grid_to_adjacency_matrix(gridworld.grid)
        num_tiles = self.size * self.size
        bounds = BinaryVar(n_vars=num_tiles, name="placement_var")
        data={
            'grid': gridworld.grid,
            'target': gridworld.target_location,
        }
        self._problem = SARProblem(bounds=bounds, minmax="min", data=data)
        self.metrics = {
            'fitness': [],
            'best_fitness': [],
            'best_solution': []
        }

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def log_metrics(self) -> None:
        pass

    def solve(self):
        self._strategy.solve(self._problem)
        x_decoded = self._problem.decode_solution(self._strategy.g_best.solution)
        path = x_decoded["placement_var"]
        path_2d = path.reshape((self.size, self.size))
        return path_2d, self.metrics

class BaseACOR(Strategy):
    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Calculate Selection Probabilities
        pop_rank = np.array([idx for idx in range(1, self.pop_size + 1)])
        qn = self.intent_factor * self.pop_size
        matrix_w = 1 / (np.sqrt(2 * np.pi) * qn) * np.exp(-0.5 * ((pop_rank - 1) / qn) ** 2)
        matrix_p = matrix_w / np.sum(matrix_w)  # Normalize to find the probability.
        # Means and Standard Deviations
        matrix_pos = np.array([agent.solution for agent in self.pop])
        matrix_sigma = []
        for idx in range(0, self.pop_size):
            matrix_i = np.repeat(self.pop[idx].solution.reshape((1, -1)), self.pop_size, axis=0)
            D = np.sum(np.abs(matrix_pos - matrix_i), axis=0)
            temp = self.zeta * D / (self.pop_size - 1)
            matrix_sigma.append(temp)
        matrix_sigma = np.array(matrix_sigma)

        # Generate Samples
        pop_new = []
        for idx in range(0, self.sample_count):
            child = np.zeros(self.problem.n_dims)
            for jdx in range(0, self.problem.n_dims):
                rdx = self.get_index_roulette_wheel_selection(matrix_p)
                child[jdx] = self.pop[rdx].solution[jdx] + self.generator.normal() * matrix_sigma[rdx, jdx]  # (1)
            pos_new = self.correct_solution(child)      # (2)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_new, self.pop_size, self.problem.minmax)


'''
Custom implementations go here. BaseACOR is the original Ant Colony Optimization for Continuous Domains (ACOR) algorithm.
In order to implement custom versions of Ant Colony Optimization, modify CustomACOR that inherits from Strategy and
override the evolve() method. The evolve() method is the main operations (equations) of the algorithm.
'''

class CustomACOR(Strategy):
    def evolve(self, epoch):
        pass
