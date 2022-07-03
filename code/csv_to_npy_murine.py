import numpy as np
from material_models import GOH_fullyinc

param_sets = [[2.69804100e-03, 2.13677988e-02, 1.14909551e+01, 3.33333333e-01, 5.29620414e-03],
			  [0.00000000e+00, 3.67463536e-02, 1.14910263e+01, 3.31002082e-01, 5.29863821e-03],
			  [0.00000000e+00, 3.15918839e-02, 1.14909877e+01, 3.33333333e-01, 5.30096719e-03],
			  [3.44146561e-04, 1.11816314e-02, 1.14885011e+01, 3.33333333e-01, 5.28486316e-03],
			  [0.,             0.06119954,     16.95850061,    0.33333333,     0.02521206    ],
			  [0.00000000e+00, 1.97567494e-02, 1.14904873e+01, 3.33333333e-01, 5.29524367e-03],
			  [2.19404642e-03, 4.24951393e-03, 1.14855650e+01, 3.33333333e-01, 5.36256541e-03],
			  [4.28020227e-03, 1.59229630e-05, 1.14220508e+01, 2.74081721e-01, 5.45201343e-03],
			  [3.83973446e-03, 3.75989596e-04, 1.14773922e+01, 3.08257022e-01, 5.46525554e-03],
			  [8.45394781e-04, 6.14428149e-03, 1.14900602e+01, 3.08286186e-01, 5.28501177e-03],
			  [0.00000000e+00, 5.54908164e-02, 1.14910353e+01, 3.27687526e-01, 5.30391414e-03]]
sample_set = ['S111S1', 'S111S4', 'S112S2', 'S112S3', 'S112S4', 'S113S3', 'S113S4', 'S143S1', 'S143S2', 'S144S3', 'S144S4']

for sample, GOH_params in zip(sample_set, param_sets):
	offx_path = 'training_data/All_Murine_InvitroUnloaded_YoungDorsal/'+sample+'_OffbiaxialX.csv'
	offy_path = 'training_data/All_Murine_InvitroUnloaded_YoungDorsal/'+sample+'_OffbiaxialY.csv'
	equi_path = 'training_data/All_Murine_InvitroUnloaded_YoungDorsal/'+sample+'_Equibiaxial.csv'

	data_offx = np.genfromtxt(offx_path,delimiter=',')[1:]
	data_offy = np.genfromtxt(offy_path,delimiter=',')[1:]
	data_equi = np.genfromtxt(equi_path,delimiter=',')[1:]

	data_xy  = np.vstack((data_offx, data_offy))
	data_b   = np.vstack((data_equi))
	data_xyb = np.vstack((data_offx, data_offy, data_equi))

	for data, load in zip([data_xy, data_b, data_xyb], ['xy', 'b', 'xyb']):
		save_path = 'training_data/'+sample+'_'+load+'.npy'
		X = np.vstack((data[:,0], data[:,1]))
		X = np.transpose(X,[1,0])
		Y = np.vstack((data[:,2], data[:,3]))
		Y = np.transpose(Y,[1,0])
		with open(save_path, 'wb') as f:
		    np.save(f,[X,Y])


	# Synthetic data
	X = np.vstack((data_xy[:,0], data_xy[:,1]))
	X = np.transpose(X,[1,0])
	Y = np.vstack((data_xy[:,2], data_xy[:,3]))
	Y = np.transpose(Y,[1,0])
	save_path_aug = 'training_data/'+sample+'_xys.npy'
	lambda_max = 1.45
	res1 = 11 #Resolution of data points (lambda_1)
	res2 = 4 #Resolution of lambda_2 when lambda_1 = 1
	res3 = res1 #Resolution of lambda_2 when lambda_1 varies
	lambda1 = 1.0
	X_synth = np.ones([res1*(res2+res3),2])
	Y_synth = np.zeros([res1*(res2+res3),2])
	for i in range(res1):
		X_synth[i*(res2 + res3)       : i*(res2 + res3) +          res2,0] = np.linspace(1.0,lambda1,res2) 
		X_synth[i*(res2 + res3) + res2: i*(res2 + res3) + (res2 + res3),0] = [lambda1]*res3
		X_synth[i*(res2 + res3) + res2: i*(res2 + res3) + (res2 + res3),1] = np.linspace(1.0, lambda_max, res3)
		lambda1+= (lambda_max-1.0)/res3

	GOH = GOH_fullyinc(X_synth, GOH_params[0], GOH_params[1], GOH_params[2], GOH_params[3], GOH_params[4])
	Y_synth[:,0] = GOH.s(X_synth)[:,0,0]
	Y_synth[:,1] = GOH.s(X_synth)[:,1,1]

	X_aug = np.vstack((X,X_synth))
	Y_aug = np.vstack((Y,Y_synth))

	with open(save_path_aug,'wb') as f:
		np.save(f,[X_aug, Y_aug])


