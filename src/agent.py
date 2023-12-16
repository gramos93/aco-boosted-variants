import numpy as np
import heapq

class Agent:
    def __init__(self, idx, visibility, color, size):
        self.idx = idx
        self.current_location = (0,0)
        self.current_cost = 1
        self.path = [Node((0,0), 1)]
        self.color = color
        self.neighbors = [(0,1), (1,0)]
        self.frontier = []
        self.size = size
        self.visited = set()
        self.visibility = visibility

    def send_home(self):
        self.current_location = (0,0)
        self.current_cost = 1
        self.path = [Node((0,0), 1)]
        self.neighbors = [(0,1), (1,0)]
        self.visited = set()

    def update(self, new_location, cost_matrix, gps):
        cost = cost_matrix[new_location]
        self.current_location = new_location
        self.current_cost += cost
        self.neighbors = []
        for i, j in [(0,1), (0,-1), (1,0), (-1,0)]:
            if 0 <= new_location[0]+i < self.size and 0 <= new_location[1]+j < self.size and cost_matrix[new_location[0]+i, new_location[1]+j] != -1:
                if (new_location[0]+i, new_location[1]+j) not in self.visited:
                    self.neighbors.append((new_location[0]+i, new_location[1]+j))
                    self.frontier.append((new_location[0]+i, new_location[1]+j))
                elif (new_location[0]+i, new_location[1]+j) in self.frontier:
                    self.frontier.remove((new_location[0]+i, new_location[1]+j))
        self.path.append(Node(new_location, cost))
        self.visited.add(new_location)

        if self.neighbors == []:
            # search through neighbors and find the one with highest gps if agent trapped with no new neighbors
            for i, j in [(0,1), (0,-1), (1,0), (-1,0)]:
                if 0 <= new_location[0]+i < self.size and 0 <= new_location[1]+j < self.size and cost_matrix[new_location[0]+i, new_location[1]+j] != -1:
                    self.neighbors.append((new_location[0]+i, new_location[1]+j))
            gps_neighbors = [gps[n[0], n[1]] for n in self.neighbors]
            max_idx = np.argmax(gps_neighbors)
            self.neighbors = [self.neighbors[max_idx]]


    def get_vision(self, location):
        vision = []
        for i in range(-self.visibility, self.visibility+1):
            for j in range(-self.visibility, self.visibility+1):
                if 0 <= location[0]+i < self.size and 0 <= location[1]+j < self.size:
                    vision.append((location[0]+i, location[1]+j))
        return vision
    
    def get_best_momentum_node(self):
        last_visited = self.path[len(self.path)-2].get_location() if len(self.path) > 1 else None
        momentum_best_node = (self.current_location[0]-last_visited[0]+self.current_location[0], self.current_location[1]-last_visited[1]+self.current_location[1]) if last_visited != None else None

        return momentum_best_node
    
    def find_best_path_to_location(self, destination, all_nodes, probability, cost_matrix):
        probability = np.append(probability, 1)
        all_nodes.append(self.current_location)

        # create graph
        graph = {}
        for vision in all_nodes:
            all_adjacent_nodes = []
            nodes = {}
            for i, j in [(0,1), (0,-1), (1,0), (-1,0)]:
                if 0 <= vision[0]+i < self.size and 0 <= vision[1]+j < self.size and cost_matrix[vision[0]+i, vision[1]+j] != -1:
                    all_adjacent_nodes.append((vision[0]+i, vision[1]+j))
            for adjacent_node in all_nodes:
                if adjacent_node in all_adjacent_nodes:
                    nodes[adjacent_node] = 1/probability[all_nodes.index(adjacent_node)]
            graph[vision] = nodes

        start = self.current_location 

        # Initialize distances with infinity for all nodes except the start node
        distances = {node: float('infinity') for node in graph}
        distances[start] = 0

        # Priority queue to keep track of the next node to visit
        priority_queue = [(0, start)]

        # Dictionary to store the shortest path to each node
        previous_nodes = {node: None for node in graph}

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            # Stop the algorithm if the destination node is reached
            if current_node == destination:
                break

            # Skip if we have already processed this node with a shorter path
            if current_distance > distances[current_node]:
                continue

            # Update distances for neighbors
            for neighbor, weight in graph[current_node].items():
                distance = current_distance + weight

                # If a shorter path is found, update the distance and enqueue the neighbor
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

        # Reconstruct the shortest path
        path = []
        current = destination
        while previous_nodes[current] is not None:
            path.insert(0, current)
            current = previous_nodes[current]

        # Include the start node in the path
        path.insert(0, start)

        return path

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
        return f'Agent {self.idx} at {self.current_location}'
    
    def __str__(self):
        return f'Agent {self.idx} at {self.current_location}'
    
    def __eq__(self, other):
        return self.idx == other.idx
    
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
        return f'Node at {self.location}'
    
    def __str__(self):
        return f'Node at {self.location}'
    
    def __eq__(self, other):
        return self.location == other.location
    
    def __hash__(self):
        return hash(self.location)
