from view import View
from gridworld import GridWorld
from optimization import SearchAlgorithm, BaseACOR # Add your own algorithm here

def run():

    # Initialize GridWorld: edit config to alter problem

    config = {
        'dim': 32,
        'target_location': (30, 30)
    }

    gridworld = GridWorld(config)

    # Initialize optimization algorithm (Strategy, gridworld)

    search = SearchAlgorithm(BaseACOR(), gridworld)

    # Run simulation

    result, metrics = search.solve()

    gridworld.update(result)

    # Log metrics
    
    # Open UI and display execution log
    with View() as view:
        view.display(gridworld.grid)

if __name__ == '__main__':
    run()
