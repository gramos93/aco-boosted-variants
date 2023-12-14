from copy import deepcopy
import numpy as np


class Agent:
    def __init__(self, idx, visibility, color, size):
        self.idx = idx
        self.current_location = (0, 0)
        self.current_cost = 1
        self.path = [Node((0, 0), 1)]
        self.color = color
        self.neighbors = [(0, 1), (1, 0)]
        self.frontier = []
        self.size = size
        self.visited = set()
        self.visibility = visibility

    def send_home(self):
        self.current_location = (0, 0)
        self.current_cost = 1
        self.path = [Node((0, 0), 1)]
        self.neighbors = [(0, 1), (1, 0)]
        self.visited = set()

    def update(self, new_location, cost_matrix, gps):
        cost = cost_matrix[new_location]
        self.current_location = new_location
        self.current_cost += cost
        self.neighbors = []
        for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if (
                0 <= new_location[0] + i < self.size
                and 0 <= new_location[1] + j < self.size
                and cost_matrix[new_location[0] + i, new_location[1] + j] != -1
            ):
                if (new_location[0] + i, new_location[1] + j) not in self.visited:
                    self.neighbors.append((new_location[0] + i, new_location[1] + j))
                    self.frontier.append((new_location[0] + i, new_location[1] + j))
                elif (new_location[0] + i, new_location[1] + j) in self.frontier:
                    self.frontier.remove((new_location[0] + i, new_location[1] + j))
        self.path.append(Node(new_location, cost))
        self.visited.add(new_location)

        if self.neighbors == []:
            # search through neighbors and find the one with highest gps if agent trapped with no new neighbors
            for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if (
                    0 <= new_location[0] + i < self.size
                    and 0 <= new_location[1] + j < self.size
                    and cost_matrix[new_location[0] + i, new_location[1] + j] != -1
                ):
                    self.neighbors.append((new_location[0] + i, new_location[1] + j))
            gps_neighbors = [gps[n[0], n[1]] for n in self.neighbors]
            max_idx = np.argmax(gps_neighbors)
            self.neighbors = [self.neighbors[max_idx]]

    def get_vision(self, location):
        vision = []
        for i in range(-self.visibility, self.visibility + 1):
            for j in range(-self.visibility, self.visibility + 1):
                if (
                    0 <= location[0] + i < self.size
                    and 0 <= location[1] + j < self.size
                ):
                    vision.append((location[0] + i, location[1] + j))
        return vision

    def get_neighbors(self):
        return self.neighbors

    def get_location(self):
        return self.current_location

    def get_path(self):
        return self.path

    def path_cost(self):
        return sum([node.get_cost() for node in self.path])

    def get_idx(self):
        return self.idx

    def __repr__(self):
        return f"Agent {self.idx} at {self.current_location}"

    def __str__(self):
        return f"Agent {self.idx} at {self.current_location}"

    def __eq__(self, other):
        return self.idx == other.idx


class SpittingAgent(Agent):
    def __init__(self, idx, visibility, color, size):
        super().__init__(idx, visibility, color, size)
        self._desired_direction = self.normalize(
            np.random.uniform(low=0, high=1.0, size=2)
        )
        self._residual_direction = self._desired_direction.copy()
        self.decisiveness = 0.2
        self.current_location = (0, 0)

    def normalize(self, vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _check_for_edge(self, location):
        # Check if agent is at the edge of the grid or obstacle
        return (
            location[0] == 0
            or location[0] == self.size - 1
            or location[1] == 0
            or location[1] == self.size - 1
        )
    
    def _reflect(self, vector, angle=-90):
        vector = vector[::-1]
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        return R.dot(vector)[::-1]

    def update(self, new_location, cost_matrix, gps):
        cost = cost_matrix[new_location]
        self.current_location = new_location
        self.current_cost += cost

        if self._check_for_edge(self.current_location):
            self._desired_direction = self._reflect(self._desired_direction)
            self._residual_direction = self._desired_direction.copy()

        # Defining neighbor positions using array operations
        neighbor_positions = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])
        potential_neighbors = neighbor_positions + np.array(new_location)
        
        # Filtering valid neighbors based on the grid's boundaries and blocked locations
        valid_mask = (
            (potential_neighbors[:, 0] >= 0) &
            (potential_neighbors[:, 0] < self.size) &
            (potential_neighbors[:, 1] >= 0) &
            (potential_neighbors[:, 1] < self.size)
        )
        # Get valid neighbors inside the grid
        valid_neighbors = potential_neighbors[valid_mask]
        # Filter valid neighbors that are not obstacles
        valid_neighbors = valid_neighbors[
            cost_matrix[valid_neighbors[:, 0], valid_neighbors[:, 1]] != -1
        ]
        self.neighbors = list(map(tuple, valid_neighbors))
        # Filtering visited neighbors
        unvisited_neighbors = [
            tuple(neighbor) for neighbor in valid_neighbors
            if tuple(neighbor) not in self.visited[-20:]
        ]
        self.neighbors = unvisited_neighbors

        if not self.neighbors:
            # search through neighbors and find the one with the highest gps 
            # if the agent is trapped with no new neighbors.
            valid_gps_neighbors = [
                gps[neighbor[0], neighbor[1]] for neighbor in valid_neighbors
            ]
            max_idx = np.argmax(valid_gps_neighbors)
            self.neighbors = [tuple(valid_neighbors[max_idx])]
        
        self.path.append(Node(new_location, cost))
        self.visited.add(new_location)



class Node:
    def __init__(self, location, cost):
        self.location = location
        self.i = location[0]
        self.j = location[1]
        self.cost = cost
        self.pheromone = 0

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def get_cost(self):
        return self.cost

    def get_pheromone(self):
        return self.pheromone

    def set_pheromone(self, pheromone):
        self.pheromone = pheromone

    def get_location(self):
        return self.location

    def __repr__(self):
        return f"Node at {self.location}"

    def __str__(self):
        return f"Node at {self.location}"

    def __eq__(self, other):
        return self.location == other.location

    def __hash__(self):
        return hash(self.location)
