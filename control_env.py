import numpy as np
import matplotlib.pyplot as plt
import torch


class control_env():
    def __init__(self, steps):    
        self.num_agents = 3
        self.Qii = torch.tensor([[1.0, 0.0],[0.0, 1.0]])
        self.Rij = torch.tensor([[1.0, 0.0, 0.0],[0.0, 1.0 ,0.0],[0.0, 1.0, 1.0]])
        self.A = torch.tensor([[0.495, 0.09983], [0.09983, 0.495]])
        self.B = torch.tensor([[0, 0.1],[0, 0.9],[0, 0.8]])
        self.aij = torch.tensor([[0.0, 0.0, 0.0],[0.0, 0.0 ,0.0],[0.0, 1.0, 0.0]])
        self.bi = torch.tensor([[1],[1],[0]])
        self.n_steps = steps
        
    def forward(self, i, u = None, X_old = None):
        # Initialize the control environment with the given number of steps
        num_agents = self.num_agents 
        Qii = self.Qii 
        Rij = self.Rij 
        A = self.A 
        B = self.B 
        aij = self.aij 
        bi = self.bi
        n_steps = self.n_steps 

        pi = torch.tensor(np.pi)
        
        def Ji(k, ei, ein, U, alpha, Qii, Rij):
            # Calculate the cost given the current inputs, state, and control errors
            def reward(e):
                reward = torch.dot(torch.matmul(e, Qii), e) + U[k] * Rij[k][k] * U[k]
                for i in range(num_agents):
                    reward = reward + U[i] * Rij[k][i] * U[i]
                return reward

            cost = reward(ei) 
            # print(cost)
            return cost

        t = torch.linspace(0, 15, n_steps)
        X_ref = torch.zeros((2, 2, n_steps))
        X = torch.zeros((num_agents, 2, 2, n_steps))

        if X_old == None:
            X_old = torch.zeros((3,2))
  

        if u is None:
            u = ((torch.randn(num_agents)) * 0) - 2.5
            # print(u)
            
        X[:,0,:,i] = X_old

        track_error = torch.zeros((num_agents, 2))
        index_per = torch.zeros((num_agents))

      
        x0 = torch.tensor([[3*torch.sin(1/2*pi*t[i]), 2*torch.sin(1/2*pi*t[i] + pi/2)]]).T
        # Compute state at each time step
        X_ref[0,:,i] = x0.flatten()
        X_ref[1,:,i] = torch.matmul(A, X_ref[0,:,i])
        ei = torch.zeros((num_agents, 2))
        ein = torch.zeros((num_agents, 2))

        for j in range(num_agents):
             # Compute the new state vector based on the current input and state vector
            # X[j,0,:,i] = x0.flatten()
            X[j,1,:,i] = torch.matmul(A, X[j,0,:,i]) + B[j] * u[j]
                
        for k in range(num_agents):
            
            for j in range(num_agents):
                ei[k,:] = ei[k,:] + aij[k][j] * (X[k,0,:,i] - X[j,0,:,i])
                ein[k,:] = ein[k,:] + aij[k][j] * (X[k,1,:,i] - X[j,1,:,i])
            ei[k,:] = ei[k,:] + bi[k] * (X[k,0,:,i] - X_ref[0,:,i])         
            ein[k,:] = ein[k,:] + bi[k] * (X[k,1,:,i] - X_ref[1,:,i])
            index_per[k] = Ji(k, ei[k,:], ein[k,:], u, 0.5, Qii, Rij)

            # print(track_error)

        return ei, index_per, u, X_ref[1,:,i], X[:,1,:,i]
        
if __name__ == "__main__":
    env = control_env(300)
    ei, index_per, u, X_ref, X = env.forward(1)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # Initialize lists to store data for visualization
    ref_1 = []
    ref_2 = []

    agents_1 = []
    agents_2 = [] 

    X_old = torch.zeros((3,2))

    # Loop over each time step
    for i in range(300):
        # Get outputs from environment
        u = torch.randn(3) 
        ei, index_per, u, X_ref, X = env.forward(i, u, X_old)
        
        # Store relevant data for visualization
        ref_1.append(X_ref[0])
        ref_2.append(X_ref[1])
        for j in range(3):
            agents_1[j].append(X[j,0])
            agents_2[j].append(X[j,1])
        
        # Update X_old for next iteration
        X_old = X
        
    # Visualize data in the plot
    xs = ref_1
    ys = ref_2
    zs = list(range(0,300))
    ax.scatter(xs, ys, zs, marker='o')

    for j in range(3):
        xs = agents_1[j]
        ys = agents_2[j]
        zs = list(range(0,300))
        ax.scatter(xs, ys, zs, marker='o')

    # Add labels to the plot
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')   
    plt.show()
    
    # print(ei)
    # print(ei[1])
    itorchuts = torch.concatenate((ei.flatten(), u))
    # print(itorchuts)
    itorchuts = torch.tensor(itorchuts, dtype=torch.float).to("cpu")
    

   
    
    


