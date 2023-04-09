from control_env import control_env
from actor_critic import ActorNet, ValueNet
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

steps = 10
num_agents = 1
alpha = 0.7
env = control_env(steps)

actor_nets = []
critic_nets = []
opt1s = []
opt2s = []
logits = torch.zeros(3)


actor_net = ActorNet().to(device)
critic_net = ValueNet().to(device)
opt1 = torch.optim.Adagrad(critic_net.parameters(), lr=0.001)
opt2 = torch.optim.Adagrad(actor_net.parameters(), lr=0.001)

ref_1 = []
ref_2 = []

agents_1 = []
agents_2 = [] 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')    
X_0 = [0, 10] 

index_per_old = torch.zeros(num_agents)
value_old = torch.zeros(num_agents)


for i in range(steps):
    
    if(i==0):
        ei, index_per, u, X_ref, X = env.forward(i, u = None, X_old = X_0)
        
    else:
        # u_old = logits.detach().numpy()
       
        # u = u_old - (index_per[j].numpy()/100)*np.random.random_sample(num_agents)
        # # print(logits.detach().numpy())

        # for j in range(num_agents):
        #     if(index_per[j] > index_per_old[j]):
        #         u[j] = u_old[j]    

        ei, index_per, u, X_ref, X = env.forward(i, logits.detach().numpy(), X)         

    
   

    ref_1.append(X_ref[0])
    ref_2.append(X_ref[1])

    for j in range(num_agents):
        agents_1.append([])
        agents_2.append([])
        
        agents_1[j].append(X[j,0])
        agents_2[j].append(X[j,1])

    for j in range(num_agents):
        
        critic_input = np.concatenate((ei[j], [u[j]]))


        # Optimize value loss (Critic)
        opt1.zero_grad()
        critic_input = torch.tensor(critic_input, dtype=torch.float).to(device)
        actor_input = torch.tensor(ei[j], dtype=torch.float).to(device)

        index_per = torch.tensor(index_per, dtype=torch.float).to(device)
        
        
        value = critic_net(critic_input)
        critic_loss = (index_per_old[j] + (alpha * value) - value_old[j])
        critic_loss = Variable(critic_loss, requires_grad=True)
       
        value_old[j] = value 

        critic_loss = F.mse_loss(critic_loss, torch.zeros(1), reduction="none")
        critic_loss.sum().backward()

     
        opt1.step() 

        for param in critic_net.parameters():
            print(param)


        # Optimize policy loss (Actor)  
        value = critic_net(critic_input)
        
        opt2.zero_grad()
        value = F.mse_loss(value, torch.zeros(1), reduction="none")
        value.sum().backward()

        opt2.step()

        logits[j] = actor_net(actor_input)

    
    index_per_old = index_per
    
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


