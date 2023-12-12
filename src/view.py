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
            6: [0, 0, 255], # solution_path
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print('Exiting context: ', self, exc_type, exc_value, traceback)

    def display(self, gridworld, optimal_path, pheremone_matrix, cost_matrix):
        grid = gridworld.grid
        agents = gridworld.agents
        # convert grid to image
        image = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                image[i, j] = self.color_map[grid[i, j]]

        for node in optimal_path:
            image[node.i, node.j] = self.color_map[6]

        for agent in agents:
            path = agent.get_path()
            for node in path[-2:]:
                image[node.i, node.j] = agent.color

        # rescale pheremone matrix to be between 0 and 1
        pheremone_matrix = pheremone_matrix / pheremone_matrix.max()

        # rescale cost matrix to be between 0 and 1
        cost_matrix = (cost_matrix * gridworld.gps) / cost_matrix.max()

        plt.figure('Search And Rescue: Swarm Optimization')
        # set font size small
        plt.rcParams.update({'font.size': 8})
        plt.ion()
        plt.clf()
        plt.subplot(1,4,1)
        plt.imshow(image)
        plt.title('Gridworld')
        plt.subplot(1,4,2)
        plt.imshow(pheremone_matrix)
        plt.title('Pheromone')
        plt.subplot(1,4,3)
        plt.imshow(gridworld.gps)
        plt.title('GPS')
        plt.subplot(1,4,4)
        plt.title('Adjusted Cost')
        plt.imshow(cost_matrix)
        plt.show()
        plt.pause(0.001)

