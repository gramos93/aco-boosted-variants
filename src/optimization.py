import numpy as np
from numpy.lib.stride_tricks import as_strided

from gridworld import GridWorld
from abc import ABC, abstractmethod
from typing import List
from view import View


class iACO(ABC):
    def __init__(
        self,
        gridworld,
        alpha,  # controls the importance of pheromone
        beta,  # controls the importance of cost
        gamma,  # controls the importance of optimal path pheromones
        delta,  # controls the importance of exploratory pheromones
        zeta,
        rho,
    ) -> None:
        self._gridworld = gridworld
        self._cost_matrix = gridworld.get_cost_matrix()
        self._pheromone_matrix = np.ones_like(self._cost_matrix).astype(np.float32)
        self._optimal_path = []
        self._optimal_cost = np.inf
        self._agents = gridworld.agents
        self._targets = gridworld.targets

        self.solution_flag = False
        self.view = View()

        self.found = [False for target in self._targets]

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.rho = rho
        self.zeta = zeta

    @abstractmethod
    def normalize_probs(self, probabilities):
        pass

    @abstractmethod
    def path_cost(self, agent):
        pass

    @abstractmethod
    def calculate_pheromone(self, agent, rho):
        pass

    @abstractmethod
    def move_agents(self, alpha, beta):
        pass

    @abstractmethod
    def agent_pheromone_update(self):
        pass

    @abstractmethod
    def solve(self):
        pass


class Search:
    def __init__(self, strategy: iACO) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> iACO:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: iACO) -> None:
        self._strategy = strategy

    def solve(self) -> None:
        result = self._strategy.solve()
        metrics = None
        return result, metrics


class BaseACO(iACO):
    def normalize_probs(self, probabilities):
        # normalize probabilities so they sum to 1
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        return probabilities

    def path_cost(self, agent, delta):
        path_cost = agent.path_cost()
        return (1 / path_cost) * delta if path_cost > 0 else 0

    def calculate_pheromone(self, agent, rho, delta):
        # evaporate pheromone, but to a minimum of 1.
        self._pheromone_matrix *= 1.0 - rho
        self._pheromone_matrix[self._pheromone_matrix < 1] = 1.0
        self._pheromone_matrix[agent.current_location] += self.path_cost(agent, delta)

    def move_agents(self, alpha, beta, zeta, gamma):
        # choose next location for each agent
        for agent in self._agents:
            # get neighbors
            neighbors = agent.get_neighbors()
            # if agent is ever obtaining a path cost worse than the optimal path,
            # send home
            if agent.path_cost() > self._optimal_cost:
                agent.send_home()
                continue
            # get pheromone values
            pheromones = [self._pheromone_matrix[neighbor] for neighbor in neighbors]
            # get cost values
            costs = [
                self._cost_matrix[neighbor] * self._gridworld.gps[neighbor]
                for neighbor in neighbors
            ]
            gps = [self._gridworld.gps[neighbor] for neighbor in neighbors]
            probabilities = [
                pheromone**alpha * (1 / cost) ** beta * signal**zeta
                for pheromone, cost, signal in zip(pheromones, costs, gps)
            ]
            probabilities = self.normalize_probs(probabilities)
            # choose next location
            choice = list(range(len(neighbors)))
            if len(choice) == 0:
                return
            next_location = np.random.choice(choice, p=probabilities)
            next_location = neighbors[next_location]
            # update agent
            agent.update(next_location, self._cost_matrix, self._gridworld.gps)
            # check if target is within vision range
            vision = agent.get_vision(next_location)
            for loc in vision:
                if loc in self._targets:
                    # update found first False to True
                    self.found[self._targets.index(loc)] = True
                    # optimal path check
                    if agent.path_cost() < self._optimal_cost:
                        self.solution_flag = True
                        self._optimal_cost = agent.path_cost()
                        self._optimal_path = agent.get_path()
                        # update pheromone matrix with optimal path
                        for i in range(len(self._optimal_path)):
                            node = self._optimal_path[-i]
                            self._pheromone_matrix[node.location] += (
                                self.path_cost(agent, delta=1.0) * gamma
                            )
                            # diminish gamma slightly
                            gamma *= 0.99
                        # send agent home
                        agent.send_home()

    def agent_pheromone_update(self, rho, delta):
        # update pheromone matrix
        for agent in self._agents:
            self.calculate_pheromone(rho, delta)

    def solve(self):
        count = 0
        ##while not all(self.found):
        while True:
            # pick next best move for each agent
            self.move_agents(self.alpha, self.beta, self.zeta, self.gamma)
            # update pheromone matrix
            self.agent_pheromone_update(self.rho, self.delta)

            if count % 10 == 0 or self.solution_flag:
                self.solution_flag = False
                self.view.display(
                    self._gridworld,
                    self._optimal_path,
                    self._pheromone_matrix,
                    self._cost_matrix,
                )
            count += 1
            if count > 100000:
                break

        print(f"Solution found in {count} iterations.")
        print(f"Optimal path cost: {self._optimal_cost}")
        print(f"Optimal path length: {len(self._optimal_path)}")
        # calculate path from start to finish
        # paths = reconstruct_paths()
        # return paths
        return self._optimal_path, None


class SpittingAnts(BaseACO):
    def calculate_pheromone(self, rho, delta):
        # evaporate pheromone, but to a minimum of 1.
        self._pheromone_matrix *= 1.0 - rho
        self._pheromone_matrix[self._pheromone_matrix < 1] = 1.0
        # self._pheromone_matrix[agent.current_location] += self.path_cost(agent, delta)

    def get_neighbourhood(self, agent, alpha, beta, zeta):
        # get neighbors with visilbility
        neighbourhood = agent.get_neighbors().copy()
        # vision = agent.get_vision(agent.current_location)
        # neighbourhood.extend(vision)
        pheromones, costs, gps, probabilities = [], [], [], []
        for neighbor in neighbourhood:
            pheromones.append(self._pheromone_matrix[neighbor])
            costs.append(self._cost_matrix[neighbor])
            gps.append(self._gridworld.gps[neighbor])
            probabilities.append(
                ((1 / pheromones[-1]) ** alpha) + ((1 / costs[-1]) ** beta) + (gps[-1] ** zeta)
            )
        probabilities = self.normalize_probs(probabilities)
        return neighbourhood, probabilities
    
    def choose_location(self, agent, neighbourhood, probabilities):
        move_vectors = np.array(neighbourhood) - (
            np.array(agent._residual_direction) + np.array(agent.current_location)
        )
        # choose closest point to agent's desired direction.
        move = neighbourhood[np.argmin(np.linalg.norm(move_vectors, axis=1))]
        # Since the world is discretized, we have to calculate a residual direction to
        # keep on the agent's track.
        agent._residual_direction = agent.normalize(
            agent._desired_direction - \
            (np.array(move) - np.array(agent.current_location))
        )
        if (agent._residual_direction == 0).all():
            agent._residual_direction = agent._desired_direction
        # adjust desired direction slightly  at randomly
        # wonder_factor = np.random.uniform(low=-1, high=1, size=2)
        wonder_factor = agent._reflect(agent._desired_direction.copy(), np.random.randint(-20, 20))
        pheromone_factor = np.array(neighbourhood[np.argmax(probabilities)]) - agent.current_location
        adjusted_direction = (
            agent._desired_direction + 
            (pheromone_factor * agent.decisiveness + wonder_factor * (1-agent.decisiveness))
        )
        agent.decisiveness = np.clip(agent.decisiveness / 0.999, 0.1, 0.9)
        agent._desired_direction = agent.normalize(adjusted_direction)
        return move

    def move_agents(self, alpha, beta, zeta, gamma):
        # choose next location for each agent
        for agent in self._agents:
            # if agent is ever obtaining a path cost worse than the optimal path,
            # send home
            if agent.path_cost() > self._optimal_cost * 2:
                agent.send_home()
                continue

            neighbourhood, probabilities = self.get_neighbourhood(
                agent, alpha, beta, zeta
            )
            next_location = self.choose_location(
                agent, neighbourhood, probabilities
            )
            agent.update(next_location, self._cost_matrix, self._gridworld.gps)
            # check if target is within vision range
            vision = agent.get_vision(next_location)
            for loc in vision:
                if loc in self._targets:
                    # update found first False to True
                    self.found[self._targets.index(loc)] = True
                    # optimal path check
                    for node in agent.get_path()[::-1]:
                        self._pheromone_matrix[node.location] += (
                            self.path_cost(agent, delta=1.0) * gamma
                        )
                        # diminish gamma slightly
                        gamma *= 0.99
                    if agent.path_cost() < self._optimal_cost:
                        self.solution_flag = True
                        self._optimal_cost = agent.path_cost()
                        self._optimal_path = agent.get_path()
                        # update pheromone matrix with optimal path
                    # send agent home
                    agent.send_home()

    def spit_pheromone(self, A, kernel_size=3, stride=1, padding=0, pool_mode='avg'):
        '''
        2D Pooling

        Parameters:
            A: input 2D array
            kernel_size: int, the size of the window over which we take pool
            stride: int, the stride of the window
            padding: int, implicit zero paddings on both sides of the input
            pool_mode: string, 'max' or 'avg'
        '''
        h, w = A.shape
        assert h==w, "The height and width of the input should be the same"
        padding = int(((stride - 1) * w - stride + kernel_size)/2)

        # Padding
        A = np.pad(A, padding, mode='constant')

        # Window view of A
        output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                        (A.shape[1] - kernel_size) // stride + 1)

        shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
        strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])

        A_w = as_strided(A, shape_w, strides_w)

        # Return the result of pooling
        if pool_mode == 'max':
            return A_w.max(axis=(2, 3))
        elif pool_mode == 'avg':
            return A_w.mean(axis=(2, 3))

    def solve(self):
        count = 0
        ##while not all(self.found):
        try:
            while True:
                # pick next best move for each agent
                self.move_agents(self.alpha, self.beta, self.zeta, self.gamma)
                # update pheromone matrix
                # self.agent_pheromone_update(self.rho, self.delta)
                self._pheromone_matrix *= 1.0 - self.rho
                self._pheromone_matrix = self.spit_pheromone(self._pheromone_matrix)
                self._pheromone_matrix[self._pheromone_matrix < 1] = 1.0
                if count % 10 == 0 or self.solution_flag:
                    self.solution_flag = False
                self.view.display(
                    self._gridworld,
                    self._optimal_path,
                    self._pheromone_matrix,
                    self._cost_matrix,
                )
                    # self.view.display_ants(self._gridworld)

                count += 1
                if count > 10000:
                    break
        except KeyboardInterrupt:
            pass

        if sum(self.found) == len(self._targets):
            print(f"Solution found in {count} iterations.")
            print(f"Path cost: {self._optimal_cost}")
            print(f"Path length: {len(self._optimal_path)}")
        else:
            print(f"No solution found in {count} iterations.")

        return self._optimal_path, None
