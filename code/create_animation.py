import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


dataset_name = 'P1C1_alldata'
n_offx = 61 #Number of offx data points in the dataset
n_offy = 61
n_equi = 76

#%% Load and Process Training Data
with open('training_data/' + dataset_name + '.npy', 'rb') as f:
    X, Y = np.load(f,allow_pickle=True)   #Principal stretches in X[lambda1, lambda2] and the 

# Create a figure and a 3D Axes
fig = plt.figure()
ax = Axes3D(fig)

def init():
    ax.scatter(X[:61,0], X[:61,1], Y[:61,1], marker='o', c='r')#,label='Off-x')
    ax.scatter(X[61:122,0], X[61:122,1], Y[61:122,1], marker='o', c='y')#,label='Off-y')
    ax.scatter(X[122:198,0], X[122:198,1], Y[122:198,1], marker='o', c='g')#, label='Equibi')
    ax.legend(['Off-x','Off-y','Equibi'])
    ax.set_xlabel('$\lambda_1$')
    ax.set_ylabel('$\lambda_2$')
    ax.set_zlabel('$\sigma_2$')
    return fig,

def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,

# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)
# Save
anim.save('P1C1_alldata_animation.gif', fps=30)#, extra_args=['-vcodec', 'libx264'])