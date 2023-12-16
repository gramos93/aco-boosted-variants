from view import View
from gridworld import GridWorld
from optimization import Search, BaseACO, CollaborativeAnnealingACO

def run():

    # Initialize GridWorld: edit config to alter problem

    config = {
        'dim': 64,
        'num_agents': 50,
        'visibility': 3,
        'targets': [(60,60)],
        'seed': 89, #73, 81, 89, 71
        'num_obstacle': 10,
        'max_obstacle_size': 10,
    }
    gridworld = GridWorld(config)

    # Search hyperparameters

    alpha = 100.0         # controls the importance of pheromone
    beta = 10.0           # controls the importance of cost
    zeta = 20.0           # controls the importance of gps signal
    gamma = 1000.0        # controls the importance of optimal path pheromones
    delta = 0.0           # controls the importance of exploratory pheromones
    rho = 0.000001       # evaporation rate 0.0000001 

    search = Search(CollaborativeAnnealingACO(gridworld, alpha, beta, gamma, delta, zeta, rho)) # CHANGE ALGO HERE
            
    # Run simulation

    result, metrics = search.solve()

    # Log metrics
    
    # hang until space, then close program
    #input('Press enter to exit')
    #exit()

if __name__ == '__main__':
    run()
