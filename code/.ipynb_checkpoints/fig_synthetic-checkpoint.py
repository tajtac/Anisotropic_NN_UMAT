# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:03:26 2021

@author: vtac
"""

#Figure 2
#Hint: run this code by selecting the whole code and pressing F9 AFTER NNMAT_v3.py is executed

import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
from misc import preprocessing, normalization

#Read data
dataset_name = 'P1C1_s'
ndata, I1, I2, I4a, I4s, Psi_gt, X, Y, sigma_gt, F, C, C_inv  = preprocessing(dataset_name)

meanPsi, meanI1, meanI2, meanI4a, meanI4s, stdPsi, stdI1, stdI2, stdI4a, stdI4s = normalization(True, 'P1C1_s', Psi_gt, I1, I2, I4a, I4s)

Psinorm = (Psi_gt - meanPsi)/stdPsi
I1norm  = (I1     - meanI1) /stdI1
I2norm  = (I2     - meanI2) /stdI2
I4anorm = (I4a    - meanI4a)/stdI4a
I4snorm = (I4s    - meanI4s)/stdI4s

#### Combine the NN inputs
inputs = np.zeros([ndata,4])
inputs[:,0] = I1norm
inputs[:,1] = I2norm
inputs[:,2] = I4anorm
inputs[:,3] = I4snorm
inpten = tf.Variable(inputs)

#Load the model
model_fname   = 'savednet/P1C1_s.json'
weights_fname = 'savednet/P1C1_s_weights.h5'
model = tf.keras.models.model_from_json(open(model_fname).read())
model.load_weights(weights_fname)

#Make predictions
with tf.GradientTape() as t:
    y_pred = model(inpten)
grad = t.jacobian(y_pred, inpten)
gradPsi = np.zeros([grad.shape[0],grad.shape[3]])
for i in range(grad.shape[0]):
    gradPsi[i,:] = grad[i,0,i,:]      #0 stands for derivatives of the first output, Psi
dPsidI1, dPsidI2, dPsidI4a, dPsidI4s = gradPsi[:,0], gradPsi[:,1], gradPsi[:,2], gradPsi[:,3]
Psi_pred = y_pred[:,0]*stdPsi + meanPsi
d1 = y_pred[:,1]*stdPsi/stdI1
d2 = y_pred[:,2]*stdPsi/stdI2
d3 = y_pred[:,3]*stdPsi/stdI4a
d4 = y_pred[:,4]*stdPsi/stdI4s

sigma_pred = np.zeros_like(C)
a0a0 = np.zeros_like(C) #a0 dyadic a0
s0s0 = np.zeros_like(C) #s0 dyadic s0
# p = -(2*d1[:]*C[:,2,2] + 2*d2[:]*(I1-C[:,2,2]))*C[:,2,2] #from sigma_3 = 0
p = -(2*d1[:] + 2*d2[:]*(I1-C[:,2,2]))*C[:,2,2] #from sigma_3 = 0
# p = d1*0
I1_2 = np.zeros_like(C)
for i in range(ndata):
    a0a0[i,0,0] = 1
    s0s0[i,1,1] = 1
    I1_2[i,:,:] = I1[i]*np.eye(3)
    sigma_pred[i,:,:] = 2*d1[i]*np.eye(3) + 2*d2[i]*(I1_2[i,:,:]-C[i,:,:]) 
    sigma_pred[i,:,:]+= 2*d3[i]*a0a0[i,:,:] + 2*d4[i]*s0s0[i,:,:] + p[i]*C_inv[i,:,:]
    sigma_pred[i,:,:] = F[i,:,:]*sigma_pred[i,:,:]*F[i,:,:]

#Read fitting history:
with open('savednet/P1C1_s_history.pkl', 'rb') as f:
    [total_loss, sigma_loss, symmetry_loss, convexity_loss, Psi_loss] = pickle.load(f)

res1 = 11
res2 = 4
res3 = 11
n_hf = 0 #number of high fidelity data points (experimental)

#%% VERSION 4
fsize=10
pltparams = {'legend.fontsize': 'large',
          'figure.figsize': (fsize*1.75,fsize),
          'font.size'     : 1.4*fsize,
          'axes.labelsize': 1.6*fsize,
          'axes.titlesize': 1.4*fsize,
          'xtick.labelsize': 1.4*fsize,
          'ytick.labelsize': 1.4*fsize,
          'lines.linewidth': 2,
          'lines.markersize': 7,
          'axes.titlepad': 25,
          "mathtext.fontset": 'dejavuserif',
          'axes.labelpad': 5}
plt.rcParams.update(pltparams)
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.inferno(np.linspace(0, 0.8,res1)))

fig = plt.figure()
gs = fig.add_gridspec(2,3, wspace=0.2, hspace=0.25) #nrows, ncols

#top left: lambda1-lambda2
ax1 = fig.add_subplot(gs[0, 0])
for i in range(res1):
    start = n_hf + i*(res2+res3)
    end = n_hf + (i+1)*(res2+res3)
    ax1.plot(X[start:end,1], X[start:end,0],'.-', color='k')
ax1.set(xlabel = '$\\lambda_x$', ylabel = '$\\lambda_y$')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

#top center: 3d I1-I2-I4s
plt.rcParams.update({'axes.labelpad': 10})
ax2 = fig.add_subplot(gs[0, 1], projection = '3d')
for i in range(res1):
    start = n_hf + i*(res2+res3)
    end = n_hf + (i+1)*(res2+res3)
    ax2.plot(I1[start:end], I2[start:end], I4s[start:end])
ax2.set(xlabel = '$I_1$', ylabel = '$I_2$', zlabel = '$I_{4s}$')
ax2.view_init(azim=10)
ax2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.xaxis.set_rotate_label(False)
ax2.yaxis.set_rotate_label(False)
ax2.zaxis.set_rotate_label(False)

#top right: losses
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.inferno(np.linspace(0.2, 0.8, 4)))


plt.rcParams.update({'axes.labelpad': 5})
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(Psi_loss,       label = '$\\mathcal{L}_1$')
ax3.plot(sigma_loss,     label = '$\\mathcal{L}_2$')
ax3.plot(symmetry_loss,  label = '$\\mathcal{L}_3$')
ax3.plot(convexity_loss,  label = '$\\mathcal{L}_4$')
ax3.set(yscale = 'log', xscale='linear')
ax3.legend()

ax3.set(xlabel = 'Iteration', ylabel = 'Training Loss')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

#bottom left
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.inferno(np.linspace(0, 0.8,res1)))
plt.rcParams.update({'axes.labelpad': 5})
ax4 = fig.add_subplot(gs[1, 0])
for i in range(res1):
    start = n_hf + i*(res2+res3)
    end = n_hf + (i+1)*(res2+res3)
    if i == 0:
        ax4.plot(X[start:end,1], Psi_gt[start:end]*1000,'.', label = 'Ground Truth')
    else:
        ax4.plot(X[start:end,1], Psi_gt[start:end]*1000,'.')
for i in range(res1):
    start = n_hf + i*(res2+res3)
    end = n_hf + (i+1)*(res2+res3)
    if i == 0:
        ax4.plot(X[start:end,1], Psi_pred[start:end]*1000, label = 'Predicted')
    else: 
        ax4.plot(X[start:end,1], Psi_pred[start:end]*1000)
ax4.set(xlabel = '$\\lambda_y$', ylabel = '$\\Psi$ (KJ)')
ax4.legend()
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

#bottom center
ax5 = fig.add_subplot(gs[1, 1])
for i in range(res1):
    start = n_hf + i*(res2+res3)
    end   = n_hf + (i+1)*(res2+res3)
    ax5.plot(X[start:end,1], sigma_pred[start:end,0,0])
for i in range(res1):
    start = n_hf + i*(res2+res3)
    end   = n_hf + (i+1)*(res2+res3)
    ax5.plot(X[start:end,1], sigma_gt[start:end,0,0],'.')
ax5.set(xlabel = '$\\lambda_y$', ylabel = '$\\sigma_x$ (MPa)')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

#bottom right
ax6 = fig.add_subplot(gs[1, 2])
for i in range(res1):
    start = n_hf + i*(res2+res3)
    end   = n_hf + (i+1)*(res2+res3)
    ax6.plot(X[start:end,1], sigma_pred[start:end,1,1])
for i in range(res1):
    start = n_hf + i*(res2+res3)
    end   = n_hf + (i+1)*(res2+res3)
    ax6.plot(X[start:end,1], sigma_gt[start:end,1,1],'.')
ax6.set(xlabel = '$\\lambda_y$', ylabel = '$\\sigma_y$ (MPa)')
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)

fig.savefig('fig_synthetic.png', dpi = 500)











