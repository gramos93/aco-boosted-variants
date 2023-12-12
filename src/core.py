from view import View
from gridworld import GridWorld
from optimization import Search, BaseACO

def run():

    # Initialize GridWorld: edit config to alter problem

    '''
    config = {
        'dim': 96,
        'num_agents': 20,
        'visibility': 2,
        'targets': [(90,10)],
        'seed': 63,
    }
    config = {
        'dim': 96,
        'num_agents': 20,
        'visibility': 2,
        'targets': [(90,90)],
        'seed': 65,
    }
    '''
    config = {
        'dim': 96,
        'num_agents': 20,
        'visibility': 2,
        'targets': [(90,90)],
        'seed': 71,
    }
    gridworld = GridWorld(config)

    # Search hyperparameters

    alpha = 100.0         # controls the importance of pheromone
    beta = 10.0           # controls the importance of cost
    zeta = 20.0           # controls the importance of gps signal
    gamma = 1000.0        # controls the importance of optimal path pheromones
    delta = 0.0           # controls the importance of exploratory pheromones
    rho = 0.000001        # evaporation rate

    search = Search(BaseACO(gridworld, alpha, beta, gamma, delta, zeta, rho))
            
    # Run simulation

    result, metrics = search.solve()

    # Log metrics
    
    # hang until space, then close program
    input('Press enter to exit')
    exit()

if __name__ == '__main__':
    run()
