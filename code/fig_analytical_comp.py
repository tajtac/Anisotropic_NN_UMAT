# -*- coding: utf-8 -*-
"""
Created on Sun May  2 17:37:29 2021

@author: vtac
"""
#Figure 4

import matplotlib.pyplot as plt
import numpy as np
from material_models import GOH_fullyinc
from material_models_fig_analytical_comp import HGO, Fung, MR, neoHook
from misc import preprocessing, normalization, predict

n_offx = 61 #Number of offx data points in the dataset
n_offy = 61

model_name = 'P1C1_xy'
dataset_name = 'P1C1_xy'
ndata, I1, I2, I4a, I4s, Psi_gt, X, Y, sigma_gt, F, C, C_inv  = preprocessing(dataset_name)
meanPsi, meanI1, meanI2, meanI4a, meanI4s, stdPsi, stdI1, stdI2, stdI4a, stdI4s = normalization(True, model_name, Psi_gt, I1, I2, I4a, I4s)

sigma_pred,_,_,_,_,Psi_pred = predict(model_name, dataset_name)
GOH = GOH_fullyinc(X, 9.86876414e-04, 5.64353050e-01, 7.95242698e+01, 2.94747207e-01, 1.57079633e+00)
MR_params =  [0.,         0.,         0.14424528]
HGO_params = [0.012902496702913772,0.01724173170395558,14.00442692847235,2.110210658359853]
Fung_params = [0.0024147281291801714,-1.74859889140465,-21.453946421295953,49.84357587843394]
neoHook_params =  [0.04902344]
MR = MR(MR_params)
HGO = HGO(HGO_params)
Fung = Fung(Fung_params)
neoHook = neoHook(neoHook_params)

sigma_GOH = GOH.s(X)
Psi_GOH = GOH.U(X)
sigma_MR = MR.sigma(X)
Psi_MR = MR.Psi(X)
sigma_HGO = HGO.sigma(X)
Psi_HGO = HGO.Psi(X)
sigma_Fung = Fung.sigma(X)
Psi_Fung = Fung.Psi(X)
sigma_neoHook = neoHook.sigma(X)
Psi_neoHook = neoHook.Psi(X)

fsize=8
pltparams = {'legend.fontsize': 'large',
          'figure.figsize': (2*fsize,fsize),
          'axes.labelsize': 1.75*fsize,
          'axes.titlesize': 1.75*fsize,
          'xtick.labelsize': 1.25*fsize,
          'ytick.labelsize': 1.25*fsize,
          'legend.fontsize': 1.25*fsize,
          'legend.title_fontsize': 1.25*fsize,
          'axes.titlepad': 10,
          'lines.linewidth': 1,
          'lines.markersize': 10,
          "mathtext.fontset": 'dejavuserif',
          'axes.labelpad': 5}
plt.rcParams.update(pltparams)
# plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.inferno(np.linspace(0.2, 0.8, 2)))

fig, ax = plt.subplots(2,3)
fig.subplots_adjust(wspace=0.3)

linewd  = ['1',      '2',        '1',       '1',       '1',      '1',        '1'          ]
markers = ['k.',     'k-',       '-x',      '-_',      '-+',     '-1',       '-2'         ]
Psis    = [Psi_gt,   Psi_pred,   Psi_GOH,   Psi_HGO,   Psi_MR,   Psi_Fung,   Psi_neoHook  ]
sigmas  = [sigma_gt, sigma_pred, sigma_GOH, sigma_HGO, sigma_MR, sigma_Fung, sigma_neoHook]
labels  = ['Exp.',   'DNN',      'GOH',     'HGO',     'MR',     'Fung',     'nH'         ]

for Psi, sigma, marker, label, lw in zip(Psis, sigmas, markers, labels, linewd):

    i1 = n_offx
    i2 = n_offx+n_offy
    ax[0,0].plot(X[ 0:i1,1], Psi[ 0:i1]*1000, marker, label=label, linewidth=lw, markevery=3)
    ax[1,0].plot(X[i1:i2,1], Psi[i1:i2]*1000, marker, label=label, linewidth=lw, markevery=3)
    
    ax[0,1].plot(X[ 0:i1,1], sigma[ 0:i1,0,0], marker, label=label, linewidth=lw, markevery=3)
    ax[1,1].plot(X[i1:i2,1], sigma[i1:i2,0,0], marker, label=label, linewidth=lw, markevery=3)
    
    ax[0,2].plot(X[ 0:i1,1], sigma[ 0:i1,1,1], marker, label=label, linewidth=lw, markevery=3)
    ax[1,2].plot(X[i1:i2,1], sigma[i1:i2,1,1], marker, label=label, linewidth=lw, markevery=3)

for i in range(2):
    for j in range(3):
        ax[i,j].spines['top'].set_visible(False)
        ax[i,j].spines['right'].set_visible(False)

ylabels = ['$\Psi$ [kJ]', '$\sigma_x$ [MPa]','$\sigma_y$ [MPa]']
for j in range(3):
    ax[0,j].set(ylabel = ylabels[j])
    ax[1,j].set(ylabel = ylabels[j])
    ax[0,j].set(xlabel='$\lambda_y$')

ax[0,0].legend()

#Calculate the error to report
error = np.zeros([6,sigma_gt.shape[0]])
for i in range(sigma_gt.shape[0]):
    for j in range(3):
        for k in range(3):
            for l, sigma in enumerate(sigmas[1:]):
                error[l,i]+= (sigma_gt[i,k,j] - sigma[i,k,j])**2
    for l, sigma in enumerate(sigmas[1:]):
        error[l,i] = np.sqrt(error[l,i])
print('Average errors')
averr = np.average(error*1000, axis=1)
for l,label in enumerate(labels[1:]):
    print(label, averr[l])
text = 'Average errors [kPa]: \n DNN: {:.3f}\n GOH: {:.3f}\n HGO: {:.3f}\n MR: {:.3f}\n Fung: {:.3f}\n nH: {:.3f}'.format(
    averr[0], averr[1], averr[2], averr[3], averr[4], averr[5]
)
ax[0,1].text(0.05,0.95, text, transform=ax[0,1].transAxes, va='top')

ax[0,0].text(-0.2,1.0, 'A', transform=ax[0,0].transAxes, va='top', fontsize=20)
ax[1,0].text(-0.2,1.0, 'B', transform=ax[1,0].transAxes, va='top', fontsize=20)

fig.savefig('figs/fig_analytical_comp.jpg', dpi=300, bbox_inches='tight')