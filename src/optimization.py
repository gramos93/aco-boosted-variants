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
                self.view.display(self._gridworld, self._optimal_path, self._pheromone_matrix, self._cost_matrix)

            count += 1
            if count > 100000:
                break

        print(f'Solution found in {count} iterations.')
        print(f'Optimal path cost: {self._optimal_cost}')
        print(f'Optimal path length: {len(self._optimal_path)}')
        # calculate path from start to finish
        #paths = reconstruct_paths()
        #return paths
        return self._optimal_path, None

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