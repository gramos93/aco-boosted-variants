from view import View
from gridworld import GridWorld
from optimization import Search, BaseACO

def run():

    # Initialize GridWorld: edit config to alter problem

    config = {
        'dim': 96,
        'num_agents': 3,
        'visibility': 3,
        'targets': [(40,50), (50,40)],
    }

    gridworld = GridWorld(config)

    # Initialize optimization algorithm (Strategy, gridworld)

    search = Search(BaseACO(gridworld))

    # Run simulation

    result, metrics = search.solve()

    # Log metrics
    
    # Open UI and display execution log
    with View() as view:
        view.display(gridworld)

if __name__ == '__main__':
    run()
