import numpy as np
import matplotlib.pyplot as plt
import torch


class control_env():
    def __init__(self, steps):    
        self.num_agents = 3
        self.Qii = np.identity(2)
        self.Rij = np.array([[1, 0, 0],[0, 1 ,0],[0, 1, 1]])
        self.A = np.array([[0.495, 0.09983], [0.09983, 0.495]])
        self.B = np.array([[0, 0.1],[0, 0.9],[0, 0.8]])
        self.aij = np.array([[0, 0, 0],[0, 0 ,0],[0, 1, 0]])
        self.bi = np.array([[1],[1],[0]])
        self.n_steps = steps
        
    def forward(self, i, u = None, X_old = None):
        num_agents = self.num_agents 
        Qii = self.Qii 
        Rij = self.Rij 
        A = self.A 
        B = self.B 
        aij = self.aij 
        bi = self.bi
        n_steps = self.n_steps 
        
        def Ji(k, ei, ein, U, alpha, Qii, Rij):
            def reward(e):
                reward = np.dot(np.dot(e, Qii), np.transpose(e)) + np.dot(np.dot(U[k], Rij[k][k]), np.transpose(U[k]))
                for i in range(num_agents):
                    reward = reward + np.dot(np.dot(U[i], Rij[k][i]), np.transpose(U[i]))
                return reward

            cost = reward(ei) 
            # print(cost)
            return cost

        t = np.linspace(0, 15, n_steps)
        X_ref = np.zeros((2, 2, n_steps))
        X = np.zeros((num_agents, 2, 2, n_steps))

        print(X_old)
        

        if u is None:
            u = ((np.random.random_sample(num_agents)) * 0) - 2.5
            # print(u)
            


        X[:,0,:,i] = X_old

        print(X[:,0,:,i])

        track_error = np.zeros((num_agents, 2))
        index_per = np.zeros((num_agents))

      
        x0 = np.array([[3*np.sin(1/2*np.pi*t[i]), 2*np.sin(1/2*np.pi*t[i] + np.pi/2)]]).T
        # Compute state at each time step
        X_ref[0,:,i] = x0.flatten()
        X_ref[1,:,i] = np.dot(A, X_ref[0,:,i])
        ei = np.zeros((num_agents, 2))
        ein = np.zeros((num_agents, 2))

        for j in range(num_agents):
            # X[j,0,:,i] = x0.flatten()
            X[j,1,:,i] = np.dot(A, X[j,0,:,i]) + np.dot(B[j], u[j])
                
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

    ref_1 = []
    ref_2 = []

    agents_1 = []
    agents_2 = [] 

    X_old = np.zeros((3, 2))

    for i in range(300):
        u = (np.random.random_sample(3) * 10) - 5
        ei, index_per, u, X_ref, X = env.forward(i, u, X_old)
        ref_1.append(X_ref[0])
        ref_2.append(X_ref[1])
        for j in range(3):
            agents_1.append([])
            agents_2.append([])
            agents_1[j].append(X[j,0])
            agents_2[j].append(X[j,1])

        X_old = X

    xs = ref_1
    ys = ref_2
    zs = list(range(0,300))
    ax.scatter(xs, ys, zs, marker='o')

    for j in range(3):
        xs = agents_1[j]
        ys = agents_2[j]
        zs = list(range(0,300))
        ax.scatter(xs, ys, zs, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')   
    plt.show()
    
    # print(ei)
    # print(ei[1])
    inputs = np.concatenate((ei.flatten(), u))
    # print(inputs)
    inputs = torch.tensor(inputs, dtype=torch.float).to("cpu")
    

   
    
    


