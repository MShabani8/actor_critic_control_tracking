
from control_env import control_env
from actor_critic import ActorNet, ValueNet
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define simulation parameters
steps = 350
num_agents = 1
alpha = 0.9

# Create an instance of the control environment
env = control_env(steps)

# Create empty lists to store actor and critic networks and their respective optimizers
actor_nets = []
critic_nets = []
opt1s = []
opt2s = []

# Create an initial control input vector of zeros
us = torch.zeros(3)

# Loop through the number of agents and create an actor and critic network for each
for j in range(num_agents):
        actor_net = ActorNet().to(device)
        critic_net = ValueNet().to(device)
        actor_nets.append(actor_net)
        critic_nets.append(critic_net)

        # Create separate AdamW optimizers for the actor and critic networks
        opt1 = torch.optim.AdamW(critic_net.parameters(), lr=0.01)
        opt2 = torch.optim.AdamW(actor_net.parameters(), lr=0.001)
        opt1s.append(opt1)
        opt2s.append(opt2)

# Create empty lists to store reference values and agent states
ref_1 = []
ref_2 = []
agents_1 = []
agents_2 = [] 

# Create a 3D plot for visualizing simulation results
fig = plt.figure()
ax = fig.add_subplot(projection='3d')    
X_0 = torch.zeros((3,2))

# Initialize variables for calculating the index of performance
index_per_old = torch.zeros(num_agents)
value_old = torch.zeros(num_agents)
J_k = 0
logits = torch.zeros(1)

# Loop through simulation time steps
for i in range(steps):
    
    if(i==0):
        # On the first time step, initialize the environment and control input vector
        ei, index_per, u, X_ref, X = env.forward(i, u = None, X_old = X_0)

    else:
        # On subsequent time steps, pass the current state and control input vector to the environment
        ei, index_per, u, X_ref, X = env.forward(i, us, X) 
        ei = ei.data   
            
    # Update the index of performance using the current value and the previous index of performance
    J_k = alpha * J_k + index_per
    
    # Append the reference values and agent states to their respective lists
    ref_1.append(X_ref[0])
    ref_2.append(X_ref[1])

    for j in range(num_agents):
        agents_1.append([])
        agents_2.append([])
        agents_1[j].append(X[j,0])
        agents_2[j].append(X[j,1])

    # Loop through the agents and update their actor and critic networks
    for j in range(num_agents):  # Iterate over all agents
        with torch.no_grad():  # Disable gradient tracking for efficiency
            # Concatenate the current state, control input vector, and other agents' control input vectors for the critic network input
            critic_input = torch.cat((ei[j], u[j:j+1], u[:j], u[j+1:]))
            
            # Use the current state as the input for the actor network
            actor_input = ei[j]


        
        critic_input = critic_input.to(device)  
        actor_input = actor_input.to(device)
        
        # Compute the value function for the current state and input vector
        value = critic_nets[j](critic_input)
        
        # Compute the critic loss using the value function and other variables
        critic_loss = 1/2 * (index_per_old[j] + (alpha * value) - value_old[j]).pow(2)
        
        with torch.no_grad():  
            value_old[j] = value  # Update value_old for next iteration

        critic_loss.backward()
        
        opt1s[j].step()  # Take a step in the optimizer to update the critic network parameters

        # Optimize policy loss (Actor)  
        opt2s[j].zero_grad() 

        # Compute the logits (unnormalized log probabilities) for the current state and input vector
        logits = actor_nets[j](actor_input)  
        
        # Concatenate the current state, logits, and other agents' control input vectors for the critic network input
        critic_input = torch.cat((ei[j], logits , u[:j], u[j+1:]))
        
        # Compute the critic loss using the critic network and other variables
        actor_loss = 1/2 * (critic_nets[j](critic_input)).pow(2)

        actor_loss.backward()
        
        opt2s[j].step()  
        
        index_per_old[j] = index_per[j]  # Update index_per_old for next iteration
        us[j] = logits.detach()  # Detach logits from the computation graph and store them in us for next iteration

with torch.no_grad():      
    xs = ref_1
    ys = ref_2
    zs = list(range(0,steps))
    ax.scatter(xs, ys, zs, marker='o')

    for j in range(num_agents):
        xs = agents_1[j]
        ys = agents_2[j]
        zs = list(range(0,steps))
        ax.scatter(xs, ys, zs, marker='o')

    ax.set_xlabel('Xi1')
    ax.set_ylabel('Xi2')
    ax.set_zlabel('time steps')   
    plt.show()


