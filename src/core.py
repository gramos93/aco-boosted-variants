from view import View
from agent import Agent, SpittingAgent
from gridworld import GridWorld
from optimization import Search, BaseACO, SpittingAnts

def run():

    # Initialize GridWorld: edit config to alter problem

    config = {
        'dim': 96,
        'num_agents': 20,
        'agent_type': SpittingAgent,
        'visibility': 2,
        'targets': [(90,90)],
        'seed': 89, #73, 81, 89
        'num_obstacle': 10,
        'max_obstacle_size': 20,
    }
    gridworld = GridWorld(config)


    alpha = 20.0         # controls the importance of pheromone
    beta = 1.0         # controls the importance of cost
    zeta = 100.0           # controls the importance of gps signal
    gamma = 1.0        # controls the importance of optimal path pheromones
    delta = 1.0           # controls the importance of exploratory pheromones
    rho = 5E-8        # evaporation rate

    search = Search(SpittingAnts(gridworld, alpha, beta, gamma, delta, zeta, rho))
            
    # Run simulation

    result, metrics = search.solve()

    # Log metrics
    
    # hang until space, then close program
    input('Press enter to exit')
    exit()

if __name__ == '__main__':
    run()
