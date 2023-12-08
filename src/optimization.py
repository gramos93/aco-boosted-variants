import numpy as np
from gridworld import GridWorld
from abc import ABC, abstractmethod
from typing import List

class iACO(ABC):

    def __init__(self, gridworld) -> None:
        self._cost_matrix = gridworld.grid
        self._pheromone_matrix = 1/gridworld.probability_matrix
        self._agents = gridworld.agents
        self._targets = gridworld.targets

        self.found = [False for target in self._targets]

    @abstractmethod
    def invert_probabilities(self, probabilities):
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

class Search():

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
    def invert_probabilities(self, probabilities):
        inverted_probabilities = [(1 - p) / sum(1 - q for q in probabilities) for p in probabilities]
        return inverted_probabilities

    def path_cost(self, agent):
        path_cost = agent.path_cost()
        return 1/path_cost if path_cost > 0 else 0
    
    def visibility_case_cost(self, visibility_case):
        case_cost = self._cost_matrix[visibility_case]
        return 1/case_cost if case_cost > 0 else 0
    
    def calculate_pheromone(self, agent, rho):
        # with evaporation
        self._pheromone_matrix *= (1.0-rho)
        self._pheromone_matrix[agent.current_location] += self.path_cost(agent)
        for case in agent.get_visible_cases():
            self._pheromone_matrix[case] += self.visibility_case_cost(case)

    def move_agents(self, alpha, beta):
        # choose next location for each agent
        for agent in self._agents:
            # get neighbors
            neighbors = agent.get_neighbors()
            # get pheromone values
            pheromones = [self._pheromone_matrix[neighbor] for neighbor in neighbors]
            # get cost values
            costs = [self._cost_matrix[neighbor] for neighbor in neighbors]
            denom = np.sum([pheromone**alpha * (1/cost)**beta for pheromone, cost in zip(pheromones, costs)])
            probabilities = [pheromone**alpha * (1/cost)**beta / denom for pheromone, cost in zip(pheromones, costs)]
            # Force ants to explore AWAY from pheromones
            probabilities = self.invert_probabilities(probabilities)
            # choose next location
            choice = list(range(len(neighbors)))
            if len(choice) == 0:
                return
            next_location = np.random.choice(choice, p=probabilities)
            next_location = neighbors[next_location]
            # update agent
            agent.update(next_location, self._cost_matrix[next_location])
            # check if target found
            if next_location in self._targets:
                # update found first False to True
                self.found[self._targets.index(next_location)] = True

        return
    
    def agent_pheromone_update(self):
        # update pheromone matrix
        for agent in self._agents:
            rho = 0.5 # evaporation rate
            self.calculate_pheromone(agent, rho)

    def solve(self):
        count = 0
        while not all(self.found):
            # pick next best move for each agent
            self.move_agents(alpha=1, beta=1)
            # update pheromone matrix
            self.agent_pheromone_update()

            count += 1
            if count > 100000:
                break

        return count # has to return the shortest path, rescue task.