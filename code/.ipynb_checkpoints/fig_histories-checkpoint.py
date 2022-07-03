#History figure for Appendix

# Panels:

# Fig. 4
# S111S1_xy_nonconv
# S111S1_xy_conv
# S111S1_xys_nonconv
# S111S1_xys_conv

# Fig. 5
# P1C1_xy
# P1C1_xys

# Fig. 6
# S111S1_xy
# S111S1_sxsy
# S111S1_xyb
# S111S1_bsxsy
# S111S1_xysxsy

import matplotlib.pyplot as plt 
import numpy as np
import pickle

#Top row
files = ['S111S1_xy_nonconv', 'S111S1_xy', 'S111S1_xys_nonconv', 'S111S1_xys']
data = []
for file in files:
	with open('savednet/'+file+'_history.pkl', 'rb') as f:
	    losses = pickle.load(f)
	    data.append(losses)
print(len(data))

fig, ax = plt.subplots(1,4,figsize=[24,6])

for axi, losses in zip(ax, data):
	axi.plot(losses["loss"],label='$\\mathcal{L}_1$')
	axi.plot(losses["sigma_loss"],label='$\\mathcal{L}_2$')
	axi.plot(losses["symmetry"],label='$\\mathcal{L}_3$')
	axi.plot(losses["convexity"],label='$\\mathcal{L}_4$')