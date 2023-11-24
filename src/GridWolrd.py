import numpy as np
import matplotlib.pyplot as plt

def generate_perlin_noise_2d(shape, res):
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
        
def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise

def plot(grid):
    colors = np.array([[255, 164, 6], [138, 181, 73], [95, 126, 48], [186, 140, 93]], dtype=np.uint8)
    image = colors[grid.reshape(-1)].reshape(grid.shape + (3,))
    plt.imshow(image)
    
def get_target_position(grid, n):
    position = np.random.random_integers(n-1, size=(2,))
    while (grid[position[0], position[1]]) != 1 and (grid[position[0], position[1]]) != 3:
        position = np.random.random_integers(n, size=(2,))
    return position

def generate_probability_matrix(position, grid):
    n = grid.shape[0]
    prob_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if grid[i, j] == 1 or grid[i, j] == 3:
                # Calculate the Euclidean distance between the current position and the target position
                distance = np.linalg.norm(np.array([i, j]) - np.array(position))
                # The closer the entry is to the position, the bigger the probability
                prob = 1 / (1 + distance)
                if (grid[i, j] == 3):
                    prob = prob/2
                prob_matrix[i, j] = prob

    # Normalize the probability matrix to ensure the sum of probabilities is 1
    prob_matrix /= np.sum(prob_matrix)

    return prob_matrix    

def getGrid(n):

    grid = np.ones((n, n), dtype=np.int32)

	# Noise
    noise = generate_fractal_noise_2d((n, n), (1, 1), 6)
    noise = (noise - noise.min()) / (noise.max() - noise.min())

	# Fire
    threshold = 0.3
    grid[noise < threshold] = 0

	# Trees
    potential = ((noise - threshold) / (1 - threshold))**4 * 0.7
    mask = (noise > threshold) * (np.random.rand(n, n) < potential)
    grid[mask] = 2

	# Dirt
    mask = (grid == 1) * (np.random.rand(n, n) < 0.05)
    grid[mask] = 3

	# Target's position
    position = get_target_position(grid, n)

    # Probability matrix
    probability_matrix = generate_probability_matrix(position, grid)

    return grid, position, probability_matrix

if __name__ == '__main__':
    (grid, position, probability_matrix) = getGrid(32)
    print(position)
    print(grid)
    print(probability_matrix)
    print(np.sum(probability_matrix))

	# Plot
    plot(grid)
    plt.show()