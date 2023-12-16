from agent import Agent
from gridworld import GridWorld
from optimization import Search, BaseACO,RubberBallACO
import numpy as np
import json
import csv

def run():
    all_solution={}
    #Seeds
    seeds=[61, 74, 81, 89, 100]
    #Map sizes
    dims=[32, 64, 96, 128, 256 ]
    
    #Number of max iteration
    count=[500,1000,3000,3000,3000]

    for id in range(len(dims)):
        dim=dims[id]
        max_count=count[id]
        print(f'Tests for map of size {dim}')
        all_solution[dim]={}
        # Initialize GridWorld: edit config to alter problem
        for seed in seeds:
            print(f'Testing with seed {seed}')
            
            np.random.seed(seed)
            solution=()
            config = {
                'dim': dim,
                'num_agents': 50,
                'agent_type': Agent,
                'visibility': 2,
                'targets': [(dim-np.random.randint(1,(id+2)*2),dim-np.random.randint(1,(id+2)*2))],
                'seed': seed,
                'num_obstacle': 32,
                'max_obstacle_size': int((id+4)*(id+1)),
            }
            gridworld = GridWorld(config)

            # Search hyperparameters

            alpha = 100.0         # controls the importance of pheromone
            beta = 10.0           # controls the importance of cost
            zeta = 20.0           # controls the importance of gps signal
            gamma = 1000.0        # controls the importance of optimal path pheromones
            delta = 0.0           # controls the importance of exploratory pheromones
            rho = 0.0000001       # evaporation rate

            search = Search(RubberBallACO(gridworld, alpha, beta, gamma, delta, zeta, rho,max_count,display=False)) # CHANGE ALGO HERE

            # Run simulation
            solution,_ = search.solve()

            #Save solution
            all_solution[dim][seed]=solution[0]

            with open("results.json", "w") as outfile: 
                json.dump(all_solution, outfile, indent = 4)

def post_processing():
    with open('stats.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["MAP SIZE", "AVG nb step", "Avg path cost", "Min path cost","Max path cost","Min nb step", "Max nb step", "Nb of fails"])
    with open("results.json") as outfile: 
        data=json.load(outfile) 
        stats={}
        for mapsize in data.keys():
            with open('stats.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                counts=[value[0] for value in data[mapsize].values() if value!=None]
                costs=[value[1] for value in data[mapsize].values() if value!=None]
                fails=[1 for value in data[mapsize].values() if value==None] 
                writer.writerow([mapsize, np.mean(counts),np.mean(costs),np.min(costs),np.max(costs),np.min(counts),np.max(counts),int(np.sum(fails))])

if __name__ == '__main__':
    run()
    post_processing()
