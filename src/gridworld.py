from agent import Agent
import numpy as np

class GridWorld:
    def __init__(self, config):
        np.random.seed(config['seed'])
        self.config = config
        self.size = config['dim']
        self.target_locations = config['targets']
        self.num_agents = config['num_agents']
        self.targets = config['targets']
        self.max_obstacle_size=config['max_obstacle_size']
        self.num_obstacle=config['num_obstacle']
        self.id_map = {
            'water': 2,
            'trees': 3,
            'dirt': 4,
            'target': 5,
            'hard_obstacle': 7,
        }
        self.cost_map = {
            1: 1,
            2: 100,
            3: 5,
            4: 2,
            5: 1,
            7: -1
        }
        self.agents = []
        self.grid = self.initialize_world()
        self.gps = self.generate_probabilities()

    def generate_probabilities(self):
        # Generate probabilities for each cell where the target locations are the center of a gaussian
        # and the probability is the value of the gaussian at each cell
        gps = np.zeros((self.size, self.size))
        for t in self.target_locations:
            gps += self.gaussian_probability(t)
        gps = gps / gps.max()
        return gps

    def gaussian_probability(self, target):
        # Generate a gaussian centered at target
        x, y = np.meshgrid(np.linspace(0, self.size-1, self.size), np.linspace(0, self.size-1, self.size))
        d = np.sqrt((y - target[0])**2 + (x - target[1])**2)
        sigma = self.size
        g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )
        return g

    def update(self, solution):
        # Solution is a binary array where 1 indicates a path. Apply solution to grid
        # convert solution to 2d array of ints
        solution = solution.reshape((self.size, self.size)).astype(np.int32)
        self.grid[solution == 1] = self.id_map['path']
        return

    def generate_hard_obstacle(self):
        width=np.random.randint(1,self.max_obstacle_size)
        height=np.random.randint(1,self.max_obstacle_size)
        pos_x=np.random.randint(width,self.size-width)
        pos_y=np.random.randint(height,self.size-height-5)
        return pos_x,pos_y,width,height

    def initialize_world(self):
        grid = np.ones((self.size, self.size), dtype=np.int32)
        noise = self.generate_fractal_noise_2d((self.size, self.size), (1, 1), 6)
        noise = (noise - noise.min()) / (noise.max() - noise.min())

        threshold = 0.3
        grid[noise < threshold] = self.id_map['water']

        potential = ((noise - threshold) / (1 - threshold))**4 * 0.7
        mask = (noise > threshold) * (np.random.rand(self.size, self.size) < potential)
        grid[mask] = self.id_map['trees']

        mask = (grid == 1) * (np.random.rand(self.size, self.size) < 0.05)
        grid[mask] = self.id_map['dirt']

        for _ in range(self.num_obstacle):
            pos_x,pos_y,width,height=self.generate_hard_obstacle()
            grid[pos_x:pos_x+width,pos_y:pos_y+height]=self.id_map['hard_obstacle']

        # Targets
        for t in self.target_locations:
            grid[t[0], t[1]] = self.id_map['target']

        # Agents
        # generate 50 agents with different colors that are shades of red
        colors = np.random.randint(0, 255, (self.num_agents, 3))
        for i in range(self.num_agents):
            start_cost = self.cost_map[grid[0,0]]
            agent = self.config['agent_type'](
                i, self.config['visibility'], colors[i], self.size
            )
            self.agents.append(agent)

        return grid

    def get_cost_matrix(self):
        cost_matrix = np.zeros((self.grid.shape[0], self.grid.shape[1]))
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                cost_matrix[i, j] = self.cost_map[self.grid[i, j]]
        return cost_matrix

    def generate_perlin_noise_2d(self, shape, res):
        def f(t):
            return 6*t**5 - 15*t**4 + 10*t**3

        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
        # Gradients
        angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
        # Ramps
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
        # Interpolation
        t = f(grid)
        n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
        n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
        return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

    def generate_fractal_noise_2d(self, shape, res, octaves=1, persistence=0.5):
        noise = np.zeros(shape)
        frequency = 1
        amplitude = 1
        for _ in range(octaves):
            noise += amplitude * self.generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
            frequency *= 2
            amplitude *= persistence
        return noise
