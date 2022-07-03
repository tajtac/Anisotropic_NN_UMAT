# -*- coding: utf-8 -*-
"""
Created on Wed May 19 17:07:58 2021

@author: vtac
"""

#Figure 4
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from material_models import GOH_fullyinc
GOH_params = [2.69804100e-03, 2.13677988e-02, 1.14909551e+01, 3.33333333e-01, 5.29620414e-03]
GOH = GOH_fullyinc(X, GOH_params[0], GOH_params[1], GOH_params[2], GOH_params[3], GOH_params[4])
sigma_GOH = GOH.s(X)

error = np.zeros(sigma_gt.shape[0])
for i in range(sigma_gt.shape[0]):
    for j in range(3):
        for k in range(3):
            error[i]+= (sigma_gt[i,k,j] - sigma_pred[i,k,j])**2
    error[i] = np.sqrt(error[i])
error = error*1000 #MPa -> kPa
offx_error = np.sum(error[:183])/183
offy_error = np.sum(error[183:2*183])/183
equi_error = np.sum(error[2*183:])/183
print('L2 norm of error (NN): \n Off-x: %f \n Off-y: %f \n Equibiaxial: %f' %(offx_error, offy_error, equi_error))
error_GOH = np.zeros(sigma_gt.shape[0])
for i in range(sigma_gt.shape[0]):
    for j in range(3):
        for k in range(3):
            error_GOH[i]+= (sigma_gt[i,k,j] - sigma_GOH[i,k,j])**2
    error_GOH[i] = np.sqrt(error_GOH[i])
error_GOH = error_GOH*1000 #MPa -> kPa
offx_error_GOH = np.sum(error_GOH[:183])/183
offy_error_GOH = np.sum(error_GOH[183:2*183])/183
equi_error_GOH = np.sum(error_GOH[2*183:])/183
print('L2 norm of error (GOH): \n Off-x: %f \n Off-y: %f \n Equibiaxial: %f' %(offx_error, offy_error, equi_error))

fsize=5
pltparams = {'legend.fontsize': 'large',
          'figure.figsize': (fsize*5,fsize),
          'font.size'     : 4*fsize,
          'axes.labelsize': 4*fsize,
          'axes.titlesize': 4*fsize,
          'xtick.labelsize': 3.5*fsize,
          'ytick.labelsize': 3.5*fsize,
          'lines.linewidth': 5,
          'lines.markersize': 10,
          'lines.markeredgewidth': 2,
          'axes.titlepad': 25,
          "mathtext.fontset": 'dejavuserif',
          'axes.labelpad': 5}
plt.rcParams.update(pltparams)
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.inferno(np.linspace(0.1, 0.7, 2)))

fig = plt.figure()
gs = fig.add_gridspec(1,2, wspace=0.1, hspace=0.2, left = 0.05, right = 0.4) #nrows, ncols

#left
ax1 = fig.add_subplot(gs[0,0])
start = 0
end = 183
ax1.plot(X[start:end,0], sigma_gt[start:end,0,0]*1000,'.', label = '$\sigma_x$ (Exp.)', alpha = 0.2)
ax1.plot(X[start:end,0], sigma_gt[start:end,1,1]*1000,'.', label = '$\sigma_y$ (Exp.)', alpha = 0.2)
# ax1.plot(X[start:end,0], sigma_GOH[start:end,0,0]*1000,'--', label = '$\sigma_x$ (GOH)', linewidth = 2)
# ax1.plot(X[start:end,0], sigma_GOH[start:end,1,1]*1000,'--', label = '$\sigma_y$ (GOH)', linewidth = 2)
ax1.plot(X[start:end,0], sigma_pred[start:end,0,0]*1000, label = '$\sigma_x$ (NN)')
ax1.plot(X[start:end,0], sigma_pred[start:end,1,1]*1000, label = '$\sigma_y$ (NN)')
# title = 'Avg Error ($NN_E$)= \n$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ $%.2f' %offx_error #1st row
# ax1.set_title(title, y=0.85, x = 0.35, pad = -50)
title = '$NN_E$= %.2f [kPa]' %offx_error #1st row
ax1.set_title(title, y = 0.85, x = 0.35, pad = -25)

ax1.set(xlabel = '$\lambda_x$', ylabel = '$\sigma$ [kPa]', ylim=[0,50])
# ax1.legend(loc = 'upper left')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2 = fig.add_subplot(gs[0,1])
start = 183
end = 183*2
ax2.plot(X[start:end,0], sigma_gt[start:end,0,0]*1000,'.', alpha = 0.2)
ax2.plot(X[start:end,0], sigma_gt[start:end,1,1]*1000,'.', alpha = 0.2)
# ax2.plot(X[start:end,0], sigma_GOH[start:end,0,0]*1000,'--', label = '$\sigma_x$ (GOH)', linewidth = 2)
# ax2.plot(X[start:end,0], sigma_GOH[start:end,1,1]*1000,'--', label = '$\sigma_y$ (GOH)', linewidth = 2)
ax2.plot(X[start:end,0], sigma_pred[start:end,0,0]*1000)
ax2.plot(X[start:end,0], sigma_pred[start:end,1,1]*1000)
title = '$NN_E$= %.2f' %offy_error #1st row
ax2.set_title(title, y = 0.85, x = 0.35, pad = -25)
ax2.set(xlabel = '$\lambda_x$', yticklabels = [], ylim=[0,50])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

gs = fig.add_gridspec(1,2, wspace=0.3, hspace=0.2, left = 0.48, right = 0.84) #nrows, ncols

ax3 = fig.add_subplot(gs[0,0])
start = 183*2
end = 183*3
ax3.plot(X[start:end,0], sigma_gt[start:end,0,0]*1000,'.', alpha = 0.2)
ax3.plot(X[start:end,0], sigma_gt[start:end,1,1]*1000,'.', alpha = 0.2)
# ax3.plot(X[start:end,0], sigma_GOH[start:end,0,0]*1000,'--', linewidth = 2)
# ax3.plot(X[start:end,0], sigma_GOH[start:end,1,1]*1000,'--', linewidth = 2)
ax3.plot(X[start:end,0], sigma_pred[start:end,0,0]*1000)
ax3.plot(X[start:end,0], sigma_pred[start:end,1,1]*1000)
title = '$NN_E$= %.2f' %equi_error #1st row
ax3.set_title(title, y = 0.85, x = 0.35, pad = -25)
ax3.set(xlabel = '$\lambda_x$', ylabel = '$\sigma$ [kPa]')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

pltparams = {'lines.linewidth': 1,}
plt.rcParams.update(pltparams)

with open('S111S1_dense_convex_contour.npy', 'rb') as f:
    lm1, lm2, convexity_loss = np.load(f)
ax4 = fig.add_subplot(gs[0, 1])
levels = np.linspace(0, 10, 11)
ax4.contour(lm1, lm2, convexity_loss, colors='k', levels = levels)
cntr1 = ax4.contourf(lm1, lm2, convexity_loss, cmap="inferno_r", levels = levels, vmin = 0, vmax = 10, extend='max')
ax4.set(xlabel = '$\lambda_x$', ylabel = '$\lambda_y$')
ax4.set(facecolor='black')
fig.colorbar(cntr1, ax=ax4)


fig.savefig('murine_part_4.jpg', bbox_inches='tight')
plt.show()





