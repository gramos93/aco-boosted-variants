import numpy as np

class GridWorld:
    def __init__(self, config):
        self.size = config['dim']
        self.target_location = config['target_location']
        self.id_map = {
            'water': 2,
            'trees': 3,
            'dirt': 4,
            'path': 5,
            'target': 6,
        }
        self.cost_map = {
            2: 10,
            3: 5,
            4: 2,
        }

        self.grid = self.initialize_world()

    def update(self, solution):
        # Solution is a binary array where 1 indicates a path. Apply solution to grid
        # convert solution to 2d array of ints
        solution = solution.reshape((self.size, self.size)).astype(np.int32)
        self.grid[solution == 1] = self.id_map['path']
        return

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

        # Agents
        grid[self.target_location[0], self.target_location[1]] = self.id_map['target']

        # Targets

        return grid
        
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