import numpy as np
from gridworld import GridWorld
from abc import ABC, abstractmethod
from typing import List
from view import View

class iACO(ABC):

    def __init__(self, 
                 gridworld,
                 alpha,      # controls the importance of pheromone
                 beta,       # controls the importance of cost
                 gamma,    # controls the importance of optimal path pheromones
                 delta,   # controls the importance of exploratory pheromones
                 zeta,
                 rho) -> None:
        self._gridworld = gridworld
        self._cost_matrix = gridworld.get_cost_matrix()
        self._pheromone_matrix = np.ones_like(self._cost_matrix).astype(np.float32)
        self._optimal_path = []
        self._optimal_path_coords = []
        self._local_points = []
        self._view_local_points = []
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
    def normalize_probs(self, probabilities):
        # normalize probabilities so they sum to 1
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        return probabilities

    def path_cost(self, agent, delta):
        path_cost = agent.path_cost()
        return (1/path_cost) * delta if path_cost > 0 else 0
    
    def calculate_pheromone(self, agent, rho, delta):
        # evaporate pheromone, but to a minimum of 1.
        self._pheromone_matrix *= (1.0-rho)
        self._pheromone_matrix[self._pheromone_matrix < 1] = 1.0
        self._pheromone_matrix[agent.current_location] += self.path_cost(agent, delta)

    def move_agents(self, alpha, beta, zeta, gamma):
        # choose next location for each agent
        for agent in self._agents:
            # get neighbors
            neighbors = agent.get_neighbors()
            # if agent is ever obtaining a path cost worse than the optimal path, send home
            if agent.path_cost() > self._optimal_cost:
                agent.send_home()
                break
            # get pheromone values
            pheromones = [self._pheromone_matrix[neighbor] for neighbor in neighbors]
            # get cost values
            costs = [self._cost_matrix[neighbor] * self._gridworld.gps[neighbor] for neighbor in neighbors]
            gps = [self._gridworld.gps[neighbor] for neighbor in neighbors]
            ##denom = np.sum([pheromone**alpha * (1/cost)**beta * signal**zeta for pheromone, cost, signal in zip(pheromones, costs, gps)]) + 1e-6
            #denom=1
            probabilities = [pheromone**alpha * (1/cost)**beta * signal**zeta for pheromone, cost, signal in zip(pheromones, costs, gps)]
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
                        print(f'***Optimal path found with length {len(agent.get_path())} and cost {agent.path_cost()}.***')
                        self.solution_flag = True
                        self._optimal_cost = agent.path_cost()
                        self._optimal_path = agent.get_path()
                        # update pheromone matrix with optimal path
                        for i in range(len(self._optimal_path)):
                            node = self._optimal_path[-i]
                            self._pheromone_matrix[node.location] += self.path_cost(agent, delta=1.0) * gamma
                            # diminish gamma slightly
                            gamma *= 0.99
                        # send agent home
                        agent.send_home()
                        break
    
    def agent_pheromone_update(self, rho, delta):
        # update pheromone matrix
        for agent in self._agents:
            self.calculate_pheromone(agent, rho, delta)

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
                self.view.display(self._gridworld, self._optimal_path, self._pheromone_matrix, self._cost_matrix)

            count += 1
            if count > 10000:
                break

            if count % 1000 == 0:
                print(f'Iteration {count} complete.')

        print(f'Solution found in {count} iterations.')
        print(f'Optimal path cost: {self._optimal_cost}')
        print(f'Optimal path length: {len(self._optimal_path)}')
        # calculate path from start to finish
        #paths = reconstruct_paths()
        #return paths
        return self._optimal_path, None
    
class CollaborativeAnnealingACO(iACO):
    def normalize_probs(self, probabilities):
        # normalize probabilities so they sum to 1
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        return probabilities

    def path_cost(self, agent, delta):
        path_cost = agent.path_cost()
        return (1/path_cost) * delta if path_cost > 0 else 0

    def move_agents(self, alpha, beta, zeta, gamma, temp, optimal_counter):
        # choose next location for each agent
        for agent in self._agents:
            # get neighbors
            neighbors = agent.get_neighbors()
            # if agent is ever obtaining a path cost worse than the optimal path, send home
            if agent.path_cost() > self._optimal_cost:
                agent.send_home()
                break
            # get pheromone values
            pheromones = [self._pheromone_matrix[neighbor] for neighbor in neighbors]
            # get cost values
            costs = [self._cost_matrix[neighbor] * self._gridworld.gps[neighbor] for neighbor in neighbors]
            gps = [self._gridworld.gps[neighbor] for neighbor in neighbors]
            #alpha = alpha * temp
            probabilities = [pheromone**alpha * (1/cost)**beta * signal**zeta for pheromone, cost, signal in zip(pheromones, costs, gps)]
            # if probabilities contains NaN, set to 0
            probabilities = self.normalize_probs(probabilities)
            # choose next location
            choice = list(range(len(neighbors)))
            if len(choice) == 0:
                return
            try:
                next_location = np.random.choice(choice, p=probabilities)
            except:
                next_location = np.random.choice(choice)
            next_location = neighbors[next_location]
            # update agent
            agent.update(next_location, self._cost_matrix, self._gridworld.gps)
            # check if target is within vision range
            vision = agent.get_vision(next_location)
            for loc in vision:
                # add local vision optimization
                if optimal_counter > 150 and agent.current_location in self._optimal_path_coords:
                    self.optimize_local_path(agent, vision, temp)
                    self._view_local_points = self._local_points.copy()
                #else:
                #    self._local_points = []
                if loc in self._targets:
                    # update found first False to True
                    self.found[self._targets.index(loc)] = True
                    # temperature based pheremone reset
                    #if temp < 0.6:
                    #    for i in range(len(self._optimal_path)):
                    #        node = self._optimal_path[-i]
                    #        self._pheromone_matrix[node.location] += self._optimal_cost * gamma
                    #        # diminish gamma slightly
                    #        gamma *= 0.99
                    # optimal path check
                    if agent.path_cost() < self._optimal_cost:
                        self._local_points = []
                        print(f'***Optimal path found with length {len(agent.get_path())} and cost {agent.path_cost()}.***')
                        self.solution_flag = True
                        self._optimal_cost = agent.path_cost()
                        self._optimal_path = agent.get_path()
                        self._optimal_path_coords = [node.location for node in self._optimal_path]
                        # update pheromone matrix with optimal path
                        for i in range(len(self._optimal_path)):
                            node = self._optimal_path[-i]
                            self._pheromone_matrix[node.location] += self.path_cost(agent, delta=1.0) * gamma
                            # diminish gamma slightly
                            gamma *= 0.99
                        # send agent home
                        agent.send_home()
                        # notify new optimal path found
                        self.solution_flag = True
                        break

    def optimize_local_path(self, agent, vision, temp):
        if agent.current_location in self._local_points or temp == 1.0:
            return
        local_path = []
        # find the cell on the end of the vision that corresponds to the optimal path
        for i in range(len(self._optimal_path)):
            p2 = self._optimal_path[-i]
            if p2.location in vision:
                # end point found
                width = len(range(-agent.visibility, agent.visibility+1))
                num_cells = width**2
                break
        for j in range(num_cells):
            p1 = self._optimal_path[-i-j]
            local_path.append(p1)
            if p1.location not in vision:
                break
        p1 = local_path[-2]
        # start point found
        # heuristic case 1: if local path is not a straight line
        if len(local_path) > 2*width - width//3:
            min_value = 1
            #diminished_values = [original_value - temp * ((original_value - min_value) / min_value) for original_value in numbers]
            for v in vision:
                x, y = v
                # lower pheromone values by temperature adjusted to current path length
                #self._pheromone_matrix[x, y] = self._pheromone_matrix[x, y] * temp
                self._pheromone_matrix[x, y] = self._pheromone_matrix[x, y] - (1-temp) * ((self._pheromone_matrix[x, y] - min_value) / min_value)
            ##p1 = p1.location
            #p2 = p2.location
            #self._pheromone_matrix[p1[0], p1[1]] = self._pheromone_matrix[p1[0], p1[1]] + (1-temp) * self._pheromone_matrix[p1[0], p1[1]]
            #self._pheromone_matrix[p2[0], p2[1]] = self._pheromone_matrix[p2[0], p2[1]] + (1-temp) * self._pheromone_matrix[p2[0], p2[1]]
            print(f'Optimizing local path of length {len(local_path)} at location {agent.current_location}.')
            self._local_points.append(agent.current_location)
            self._pheromone_matrix[self._pheromone_matrix < 1] = 1.0
                
    
    def agent_pheromone_update(self, rho, delta):
        # evaporate pheromone, but to a minimum of 1.
        for agent in self._agents:
            self._pheromone_matrix *= (1.0-rho)
            self._pheromone_matrix[self._pheromone_matrix < 1] = 1.0
            self._pheromone_matrix[agent.current_location] += self.path_cost(agent, delta)

    def temperature_function(self, idx, length, temp):
        return temp - (idx/(length*2))**2

    def anneal_optimal_path(self, temp):
        # update pheromone matrix with optimal path
        length = len(self._optimal_path)
        for idx in range(len(self._optimal_path)):
            node = self._optimal_path[idx]
            energy_scalar = self.temperature_function(idx, length, temp)
            self._pheromone_matrix[node.location] *= energy_scalar

    def solve(self):
        count = 0
        # inverse temperature scaling based on how long optimal solution has been found for
        temp = 1.0
        optimal_counter = 0
        anneal = False
        while True:

            if anneal:
                optimal_counter += 1
            
            # pick next best move for each agent
            self.move_agents(self.alpha, self.beta, self.zeta, self.gamma, temp, optimal_counter)
            # update pheromone matrix
            self.agent_pheromone_update(self.rho, self.delta)

            if optimal_counter > 150:
                #self.anneal_optimal_path(temp)
                temp -= 0.05
                print(f'Optimal path found for {optimal_counter} iterations. Decreasing temperature to {temp}.')
                optimal_counter = 0
                if temp < 0.1:
                    temp = 1.0

            if count % 10 == 0 or self.solution_flag:
                if self.solution_flag:
                    anneal = True
                    optimal_counter = 0
                    temp = 1.0
                self.solution_flag = False
                self.view.display_SA(self._gridworld, self._optimal_path, self._pheromone_matrix, self._cost_matrix, self._view_local_points)

            count += 1
            if count > 10000:
                break
            if count % 1000 == 0:
                print(f'Iteration {count} complete.')

        print(f'Solution found in {count} iterations.')
        print(f'Optimal path cost: {self._optimal_cost}')
        print(f'Optimal path length: {len(self._optimal_path)}')
        # calculate path from start to finish
        #paths = reconstruct_paths()
        #return paths
        return self._optimal_path, None
