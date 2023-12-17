from agent import Agent, SpittingAgent
from gridworld import GridWorld
from optimization import Search, BaseACO, CollaborativeAnnealingACO, ACOWithMomentumAndVisionUsingDijkstraAlgorithm, RubberBallACO, SpittingAnts
import numpy as np
import json
import csv
import os

def run(file_path_json, file_path_csv, agent_type, algorithm):
    all_solution={}
    #Seeds
    seeds=[71, 81]#[61, 74, 81, 89, 100]
    #Map sizes
    dims=[32, 64, 96, 128, 256]

    obs_size=[4,8,12,16,32]
    
    #Number of max iteration
    count=[6000,6000,6000,6000,6000]

    targets=[(30,30),(60,60),(90,90),(120,120),(240,240)]

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
                'agent_type': agent_type,
                'visibility': 3,
                'targets': [targets[id]],##[(dim-np.random.randint(1,(id+2)*2),dim-np.random.randint(1,(id+2)*2))],
                'seed': seed,
                'num_obstacle': 32,
                'max_obstacle_size': obs_size[id]#int((id+4)*(id+1)),
            }
            gridworld = GridWorld(config)

            # Search hyperparameters

            alpha = 100.0         # controls the importance of pheromone
            beta = 10.0           # controls the importance of cost
            zeta = 20.0           # controls the importance of gps signal
            gamma = 1000.0        # controls the importance of optimal path pheromones
            delta = 0.0           # controls the importance of exploratory pheromones
            rho = 0.0000001       # evaporation rate

            search = Search(algorithm(gridworld, alpha, beta, gamma, delta, zeta, rho,max_count,display=False)) # CHANGE ALGO HERE

            # Run simulation
            solution,_ = search.solve()

            #Save solution
            all_solution[dim][seed]=solution[0]

            with open(file_path_json, "w") as outfile: 
                json.dump(all_solution, outfile, indent = 4)

def post_processing(file_path_json, file_path_csv):
    with open(file_path_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["MAP SIZE", "AVG nb step", "Avg path cost", "Min path cost","Max path cost","Min nb step", "Max nb step", "Nb of fails"])
    with open(file_path_json) as outfile: 
        data=json.load(outfile) 
        stats={}
        for mapsize in data.keys():
            with open(file_path_csv, 'a', newline='') as file:
                writer = csv.writer(file)
                counts=[value[0] for value in data[mapsize].values() if value!=None]
                costs=[value[1] for value in data[mapsize].values() if value!=None]
                fails=[1 for value in data[mapsize].values() if value==None] 
                writer.writerow([mapsize, np.mean(counts),np.mean(costs),np.min(costs),np.max(costs),np.min(counts),np.max(counts),int(np.sum(fails))])

if __name__ == '__main__':
    file_path_json = 'C:/Users/laure/Desktop/git/radioactive-goose/src/results_BaseACO.json'
    file_path_csv = 'C:/Users/laure/Desktop/git/radioactive-goose/src/stats_BaseACO.csv'
    run(file_path_json, file_path_csv, Agent, BaseACO)
    post_processing(file_path_json, file_path_csv)
    file_path_json = 'C:/Users/laure/Desktop/git/radioactive-goose/src/results_CollaborativeAnnealing.json'
    file_path_csv = 'C:/Users/laure/Desktop/git/radioactive-goose/src/stats_CollaborativeAnnealing.csv'
    run(file_path_json, file_path_csv, Agent, CollaborativeAnnealingACO)
    post_processing(file_path_json, file_path_csv)
    file_path_json = 'C:/Users/laure/Desktop/git/radioactive-goose/src/results_MomentumD.json'
    file_path_csv = 'C:/Users/laure/Desktop/git/radioactive-goose/src/stats_MomentumD.csv'
    run(file_path_json, file_path_csv, Agent, ACOWithMomentumAndVisionUsingDijkstraAlgorithm)
    post_processing(file_path_json, file_path_csv)
    file_path_json = 'C:/Users/laure/Desktop/git/radioactive-goose/src/results_Ball.json'
    file_path_csv = 'C:/Users/laure/Desktop/git/radioactive-goose/src/stats_Ball.csv'
    run(file_path_json, file_path_csv, Agent, RubberBallACO)
    post_processing(file_path_json, file_path_csv)
    file_path_json = 'C:/Users/laure/Desktop/git/radioactive-goose/src/results_SpittingAnts.json'
    file_path_csv = 'C:/Users/laure/Desktop/git/radioactive-goose/src/stats_SpittingAnts.csv'
    run(file_path_json, file_path_csv, SpittingAgent, SpittingAnts)
    post_processing(file_path_json, file_path_csv)
