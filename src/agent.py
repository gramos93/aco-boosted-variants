class Agent:
    def __init__(self, idx, visibility, color, size):
        self.idx = idx
        self.visibility = visibility
        self.current_location = (0,0)
        self.current_cost = 1
        self.path = [Node((0,0), 1)]
        self.color = color
        self.neighbors = [(0,1), (1,0)]
        self.size = size
        self.visited = set()

    def update(self, new_location, cost):
        self.current_location = new_location
        self.current_cost = cost
        self.neighbors = []
        for i, j in [(0,1), (0,-1), (1,0), (-1,0)]:
            if 0 <= new_location[0]+i < self.size and 0 <= new_location[1]+j < self.size:
                #if (new_location[0]+i, new_location[1]+j) not in self.visited:
                self.neighbors.append((new_location[0]+i, new_location[1]+j))
        self.path.append(Node(new_location, cost))
        self.visited.add(new_location)

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