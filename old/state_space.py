import numpy as np
import matplotlib.pyplot as plt

def Ji(k, ei, ein, U, alpha, Qii, Rij):
    def reward(e):
        reward = np.dot(np.dot(e, Qii), np.transpose(e)) + np.dot(np.dot(U[k], Rij[k,k]), np.transpose(U[k]))
        for i in range(k):
            reward = reward + np.dot(np.dot(U[i], Rij[k,i]), np.transpose(U[i]))
        return reward

    cost = reward(ei) 
    print(cost)
 
    return cost



# Define system parameters
N = 2  # number of state variables
num_agents = 3
Qii = np.identity(2)
Rij = np.array([[1, 0, 0],[0, 1 ,0],[0, 1, 1]])


A = np.array([[0.495, 0.09983], [0.09983, 0.495]])  # state transition matrix
B = np.array([[0, 1],[0, 0.9],[0, 0.8]])
aij = np.array([[0, 0, 0],[0, 0 ,0],[0, 1, 0]])
bi = np.array([[1],[1],[0]])



# Define initial state as sine functions with phase difference
n_steps = 400

t = np.linspace(0, 15, n_steps)
X_ref = np.zeros((2, 2, n_steps))
X = np.zeros((num_agents, 2, 2, n_steps))
U = np.zeros((num_agents, n_steps))
track_error = np.zeros((num_agents, 2))
index_per = np.zeros((num_agents))


for i in range(n_steps):
    x0 = np.array([[3*np.sin(1/2*np.pi*t[i]), 2*np.sin(1/2*np.pi*t[i] + np.pi/2)]]).T
    # Compute state at each time step
    X_ref[0,:,i] = x0.flatten()
    X_ref[1,:,i] = np.dot(A, X_ref[0,:,i])

    for j in range(num_agents):
        X[j,0,:,i] = 0
        # X[j,0,:,i] = x0.flatten()
        X[j,1,:,i] = np.dot(A, X[j,0,:,i]) + np.dot(B[j], U[j,i])
        
    for k in range(num_agents):
        ei = np.zeros((num_agents, 2))
        ein = np.zeros((num_agents, 2))
        for j in range(num_agents):
            ei[k,:] = ei[k,:] + aij[k][j] * (X[k,0,:,i] - X[j,0,:,i])
            ein[k,:] = ein[k,:] + aij[k][j] * (X[k,1,:,i] - X[j,1,:,i])
        ei[k,:] = ei[k,:] + bi[k] * (X[k,0,:,i] - X_ref[0,:,i])
        ein[k,:] = ein[k,:] + bi[k] * (X[k,1,:,i] - X_ref[1,:,i])
        index_per[k] = Ji(k, ei[k,:], ein[k,:], U[...,i], 0.5, Qii, Rij)

    
    # print(track_error)
    
    



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X_ref[0,0,:], X_ref[0,1,:], t, label='X0')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time')
ax.legend()

plt.show()
    
