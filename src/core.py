from view import View
from agent import Agent
from gridworld import GridWorld
from agent import Agent, SpittingAgent
from optimization import Search, BaseACO, CollaborativeAnnealingACO, ACOWithMomentumAndVisionUsingDijkstraAlgorithm, RubberBallACO, SpittingAnts

def run():

    # Initialize GridWorld: edit config to alter problem

    config = {
        'dim': 96,
        'num_agents': 50,
        'agent_type': SpittingAgent, #Agent
        'visibility': 3,
        'targets': [(90,90)],
        'seed': 81, #73, 81, 89
        'num_obstacle': 32,
        'max_obstacle_size': 20,
    }
    gridworld = GridWorld(config)

    # Search hyperparameters

    alpha = 100.0         # controls the importance of pheromone
    beta = 10.0           # controls the importance of cost
    zeta = 20.0           # controls the importance of gps signal
    gamma = 1000.0        # controls the importance of optimal path pheromones
    delta = 0.0           # controls the importance of exploratory pheromones
    rho = 0.0000001       # evaporation rate

    search = Search(BaseACO(gridworld, alpha, beta, gamma, delta, zeta, rho, display=True)) # CHANGE ALGO HERE

    # Run simulation

    result, metrics = search.solve()

    # Log metrics

    # hang until space, then close program
    input('Press enter to exit')
    exit()

if __name__ == '__main__':
    run()
