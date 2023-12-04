import numpy as np
import matplotlib.pyplot as plt

class View:
    def __init__(self):
        self.color_map = {
            1: [89, 182, 91],  # grass
            2: [14, 135, 204], # water
            3: [0, 82, 33], # trees
            4: [161, 120, 32],  # dirt
            5: [0, 0, 255], # target
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print('Exiting context: ', self, exc_type, exc_value, traceback)

    def display(self, gridworld):
        grid = gridworld.grid
        agents = gridworld.agents
        # convert grid to image
        image = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                image[i, j] = self.color_map[grid[i, j]]
        for agent in agents:
            path = agent.get_path()
            for node in path:
                image[node.i, node.j] = agent.color
        plt.figure('Search And Rescue: Swarm Optimization')
        plt.imshow(image)
        plt.show()
