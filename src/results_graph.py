import matplotlib.pyplot as plt
import numpy as np

# open all 5 csv files and read them into arrays

path = 'C:/Users/laure/Desktop/git/radioactive-goose/src/'

base_ACO = np.genfromtxt(path+'stats_BaseACO.csv', delimiter=',')
# sort into two arrays, one for each map seed, where the seeds alternatve
base_ACO_71 = base_ACO[1::2,2]
base_ACO_81 = base_ACO[2::2,2]

annealing = np.genfromtxt(path+'stats_CollaborativeAnnealing.csv', delimiter=',')
# sort into two arrays, one for each map seed, where the seeds alternatve
annealing_71 = annealing[1::2,2]
annealing_81 = annealing[2::2,2]

ball = np.genfromtxt(path+'stats_Ball.csv', delimiter=',')
# sort into two arrays, one for each map seed, where the seeds alternatve
ball_71 = ball[1::2,2]
ball_81 = ball[2::2,2]

momentum = np.genfromtxt(path+'stats_MomentumD.csv', delimiter=',')
# sort into two arrays, one for each map seed, where the seeds alternatve
momentum_71 = momentum[1::2,2]
momentum_81 = momentum[2::2,2]

spitting = np.genfromtxt(path+'stats_SpittingAnts.csv', delimiter=',')
# sort into two arrays, one for each map seed, where the seeds alternatve
spitting_71 = spitting[1::2,2]
spitting_81 = spitting[2::2,2]

x_axis = np.array([32, 64, 96, 128, 256])

# plot 2 graphs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Map Seed 71')
plt.xlabel('Map Size')
plt.ylabel('Path Cost')
plt.plot(base_ACO_71, label='Base ACO')
plt.plot(annealing_71, label='Collab. Annealing')
plt.plot(ball_71, label='Rubber Ball')
plt.plot(momentum_71, label='Momentum')
plt.plot(spitting_71, label='Spitting Ants')
plt.xticks(np.arange(5), x_axis)
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.title('Map Seed 81')
plt.xlabel('Map Size')
plt.ylabel('Path Cost')
plt.plot(base_ACO_81, label='Base ACO')
plt.plot(annealing_81, label='Collab. Annealing')
plt.plot(ball_81, label='Rubber Ball')
plt.plot(momentum_81, label='Momentum')
plt.plot(spitting_81, label='Spitting Ants')
plt.xticks(np.arange(5), x_axis)
plt.legend(loc='upper left')

#add slight spacing between the two graphs
plt.subplots_adjust(wspace=0.3)

plt.show()