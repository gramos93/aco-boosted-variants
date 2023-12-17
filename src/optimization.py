import numpy as np
from gridworld import GridWorld
from abc import ABC, abstractmethod
from typing import List
from view import View
from numpy.lib.stride_tricks import as_strided

class iACO(ABC):

    def __init__(self, 
                 gridworld,
                 alpha,      # controls the importance of pheromone
                 beta,       # controls the importance of cost
                 gamma,    # controls the importance of optimal path pheromones
                 delta,   # controls the importance of exploratory pheromones
                 zeta,
                 rho,
                 max_count=5000,
                 display=True) -> None:
        self._gridworld = gridworld
        self._cost_matrix = gridworld.get_cost_matrix()
        self._pheromone_matrix = np.ones_like(self._cost_matrix).astype(np.float32)
        self._optimal_path = []
        self._optimal_cost = np.inf
        self._agents = gridworld.agents
        self._targets = gridworld.targets
        # For CollaborativeAnnealingACO
        self._optimal_path_coords = []
        self._local_points = []
        self._view_local_points = []
        self.display=display
        self.max_count=max_count

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
                if self.display:
                    self.view.display(self._gridworld, self._optimal_path, self._pheromone_matrix, self._cost_matrix)

            count += 1
            if count > self.max_count:
                break

        print(f'Solution found in {count} iterations.')
        print(f'Optimal path cost: {self._optimal_cost}')
        print(f'Optimal path length: {len(self._optimal_path)}')
        # calculate path from start to finish
        #paths = reconstruct_paths()
        #return paths
        #convert to list of tuples
        optimal_path_coords = [node.location for node in self._optimal_path]
        return optimal_path_coords, None

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
                
    def calculate_pheromone(self, agent, rho):
        pass

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
                if self.display:
                    self.view.display_SA(self._gridworld, self._optimal_path, self._pheromone_matrix, self._cost_matrix, self._view_local_points)

            count += 1
            if count > self.max_count:
                break
            if count % 1000 == 0:
                print(f'Iteration {count} complete.')

        print(f'Solution found in {count} iterations.')
        print(f'Optimal path cost: {self._optimal_cost}')
        print(f'Optimal path length: {len(self._optimal_path)}')
        # calculate path from start to finish
        #paths = reconstruct_paths()
        #return paths
        optimal_path_coords = [node.location for node in self._optimal_path]
        return optimal_path_coords, None
    
class ACOWithMomentumAndVisionUsingDijkstraAlgorithm(iACO):
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
            best_momentum_node = agent.get_best_momentum_node()
            visions = agent.get_vision(agent.get_location())
            for node in visions:
                if self._cost_matrix[node] == -1:
                    visions.remove(node)
            # if agent is ever obtaining a path cost worse than the optimal path, send home
            if agent.path_cost() > self._optimal_cost:
                agent.send_home()
                break

            # get vision pheromone values
            vision_pheromones = [self._pheromone_matrix[vision] for vision in visions]
            # get vision cost values
            vision_costs = [self._cost_matrix[vision] * self._gridworld.gps[vision] for vision in visions]
            # get vision gps
            vision_gps = [self._gridworld.gps[vision] for vision in visions]

            # get probabilities for each nodes in vision
            vision_probabilities = [pheromone**alpha * (1/cost)**beta * signal**zeta for pheromone, cost, signal in zip(vision_pheromones, vision_costs, vision_gps)]
            vision_probabilities = self.normalize_probs(vision_probabilities)
            
            vision_next_location = visions[np.argmax(vision_probabilities)]

            node = agent.find_best_path_to_location(vision_next_location, visions, vision_probabilities, self._cost_matrix)

            vision_node = node[0] if len(node) == 1 else node[1]

            # get pheromoes values
            pheromones = [self._pheromone_matrix[neighbor] for neighbor in neighbors]
            # get cost values
            costs = [self._cost_matrix[neighbor] * self._gridworld.gps[neighbor] for neighbor in neighbors]
            # get gps values
            gps = [self._gridworld.gps[neighbor] for neighbor in neighbors]

            probabilities = [pheromone**alpha * (1/cost)**beta * signal**zeta for pheromone, cost, signal in zip(pheromones, costs, gps)]

            for i in range(len(neighbors)):
                if (neighbors[i] == vision_node):
                    probabilities[i] *= 3
                if (neighbors[i] == best_momentum_node):
                    probabilities[i] *= 1.5

            probabilities = self.normalize_probs(probabilities)

            # choose next location
            choice = list(range(len(probabilities)))
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
                if self.display:
                    self.view.display(self._gridworld, self._optimal_path, self._pheromone_matrix, self._cost_matrix)

            count += 1
            if count > self.max_count:
                break

        print(f'Solution found in {count} iterations.')
        print(f'Optimal path cost: {self._optimal_cost}')
        print(f'Optimal path length: {len(self._optimal_path)}')
        # calculate path from start to finish
        #paths = reconstruct_paths()
        #return paths
        optimal_path_coords = [node.location for node in self._optimal_path]
        return optimal_path_coords, None
    
class RubberBallACO(iACO):
    def __init__(self, gridworld, alpha, beta, gamma, delta, zeta, rho, max_count=5000, display=True) -> None:
        super().__init__(gridworld, alpha, beta, gamma, delta, zeta, rho)
        self.anti_pheromone_matrix= np.zeros_like(self._cost_matrix).astype(np.float32)
        self._directions=[(0,1), (0,-1), (1,0), (-1,0)]
        self.max_count=max_count
        self.display=display
        
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


    def move_agents_rubber_ball(self, alpha, beta, zeta, gamma):
        # choose next location for each agent
        for agent in self._agents:
            # if agent is ever obtaining a path cost worse than the twice optimal path, send home
            if agent.path_cost() > self._optimal_cost:
                agent.send_home()

            if np.random.rand()<=0.1:
                agent.direction=self._directions[np.random.randint(0,4)]

            # choose next location
            next_pos=(agent.get_location()[0]+agent.direction[0],agent.get_location()[1]+agent.direction[1])
            while is_obstacle(self._gridworld,next_pos):
                new_dir=self._directions[np.random.randint(0,4)]
                while new_dir==agent.direction:
                    agent.direction=self._directions[np.random.randint(0,4)]
                next_pos=(agent.get_location()[0]+agent.direction[0],agent.get_location()[1]+agent.direction[1])

            next_location = next_pos
            if next_location in agent.visited:
                #Cuts path, keeps beginning to first encounter of that path
                agent.path=agent.path[0:agent.path.index(next_location)+1]
                agent.visited=set(agent.path)

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
                            self._pheromone_matrix[node.location] += self.path_cost(agent, delta=1.0) * gamma
                            # diminish gamma slightly
                            gamma *= 0.99
                        # send agent home
                        agent.send_home()
                        break
                    
    def move_agents(self, alpha, beta, zeta, gamma):
        # choose next location for each agent
        for agent in self._agents:
            # get neighbors
            neighbors = agent.get_neighbors()
            # if agent is ever obtaining a path cost worse than the optimal path, send home
            if agent.path_cost() > 2*self._optimal_cost:
                agent.send_home()
                break
            # get pheromone values
            pheromones = [self._pheromone_matrix[neighbor] for neighbor in neighbors]
            # get cost values
            costs = [self._cost_matrix[neighbor] * self._gridworld.gps[neighbor] for neighbor in neighbors]
            gps = [self._gridworld.gps[neighbor] for neighbor in neighbors]

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

        solution=None
        while True:
            # pick next best move for each agent
            # if count<0.1*self.max_count:
            self.move_agents_rubber_ball(self.alpha, self.beta, self.zeta, self.gamma)
            # else:
            #     if count%1000<self._gridworld.size:
            #         self.move_agents(self.alpha, self.beta, self.zeta, self.gamma)
            #     else:
            #         self.move_agents_rubber_ball(self.alpha, self.beta, self.zeta, self.gamma)

            # update pheromone matrix
            # self.agent_pheromone_update(self.rho, self.delta)

            if (count % 10 == 0 or self.solution_flag) and self.display:
                self.view.display(self._gridworld, self._optimal_path, self._pheromone_matrix, self._cost_matrix)
            
            if self.solution_flag:
                    self.solution_flag = False
                    solution=(count,self._optimal_cost) 
            
            count += 1
            if count > self.max_count:
                break

        # print(f'Solution found in {count} iterations.')
        # print(f'Optimal path cost: {self._optimal_cost}')
        # print(f'Optimal path length: {len(self._optimal_path)}')
        # return self._optimal_path, None

        return (solution, None)

def in_map(mapsize,pos):
    if pos[0]<0 or pos[1]<0:
        return False
    if pos[0]>=mapsize or pos[1]>=mapsize:
        return False
    return True
def is_obstacle(grid,pos):
    if not in_map(grid.size,pos):
        return True
    elif grid.grid[pos]==7:
        return True
    else:
        return False
    
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
                if self.display:
                    self.view.display(
                        self._gridworld,
                        self._optimal_path,
                        self._pheromone_matrix,
                        self._cost_matrix,
                    )
                    # self.view.display_ants(self._gridworld)

                count += 1
                if count > self.max_count:
                    break
        except KeyboardInterrupt:
            pass

        if sum(self.found) == len(self._targets):
            print(f"Solution found in {count} iterations.")
            print(f"Path cost: {self._optimal_cost}")
            print(f"Path length: {len(self._optimal_path)}")
        else:
            print(f"No solution found in {count} iterations.")

        optimal_path_coords = [node.location for node in self._optimal_path]
        return optimal_path_coords, None