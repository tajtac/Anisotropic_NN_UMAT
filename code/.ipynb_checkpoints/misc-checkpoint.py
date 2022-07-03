import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

def preprocessing(dataset_name):
	with open('training_data/' + dataset_name + '.npy', 'rb') as f:
	    X, Y = np.load(f,allow_pickle=True)   #Principal stretches in X[lambda1, lambda2] and the 
	                        #corresponding stresses in Y[sigma1, sigma2] lambda3 
	                        #can be calculated from the incompressibility of the 
	                        #material and sigma3 = 0.
	#### Preprocessing: s1,s2,lm1,lm2 -> I1,I2,I4a,I4s,Psi
	ndata = X.shape[0]
	F = np.zeros([ndata,3,3])
	F[:,0,0] = X[:,0]
	F[:,1,1] = X[:,1]
	F[:,2,2] = 1/(X[:,0]*X[:,1])
	sigma_gt = np.zeros_like(F) #This is the same as Y but in 3x3 format
	sigma_gt = sigma_gt.astype('float32')
	sigma_gt[:,0,0] = Y[:,0]
	sigma_gt[:,1,1] = Y[:,1]

	#Use this ONLY when generating figure 3 for the paper
	#with open('val_dsets.npy', 'rb') as f:
	#    F, sigma_gt = np.load(f, allow_pickle=True)
	#ndata = F.shape[0]
	    
	F_inv = np.linalg.inv(F)
	F_inv_T = np.linalg.inv(np.transpose(F,[0,2,1]))
	S = np.einsum('...ik,...kl,...lj->...ij',F_inv, sigma_gt, F_inv_T)
	b = C = F*F #Since F^T = F
	C_inv = np.linalg.inv(C)
	I1 = C.trace(axis1=1, axis2=2)
	I2 = 1/2*(I1**2 - np.trace(np.einsum("...ij,...jk->...ik", C, C),axis1=1, axis2=2))
	a0 = [1,0,0]
	s0 = [0,1,0]
	I4a = np.einsum('i,pij,j->p',a0,C,a0)
	I4s = np.einsum('i,pij,j->p',s0,C,s0)
	Psi_gt = np.zeros(ndata)
	for i in range(ndata):
	    if C[i,0,0] != 1 or C[i,1,1] != 1:
	        # delPsi = 1/2*S*delC
	        Psi_gt[i] = Psi_gt[i-1]
	        Psi_gt[i]+= 0.5*0.5*(S[i,0,0] + S[i-1,0,0])*(C[i,0,0] - C[i-1,0,0])
	        Psi_gt[i]+= 0.5*0.5*(S[i,1,1] + S[i-1,1,1])*(C[i,1,1] - C[i-1,1,1])
	return ndata, I1, I2, I4a, I4s, Psi_gt, X, Y, sigma_gt, F, C, C_inv


def normalization(load_existing_model, model_name, Psi_gt, I1, I2, I4a, I4s):
	if load_existing_model == True:
	    with open('savednet/' + model_name + '_factors.pkl','rb') as f:
	        norm_factors = pickle.load(f)
	    meanPsi, stdPsi = norm_factors['meanPsi'], norm_factors['stdPsi']
	    meanI1, stdI1   = norm_factors['meanI1'],  norm_factors['stdI1']
	    meanI2, stdI2   = norm_factors['meanI2'],  norm_factors['stdI2']
	    meanI4a, stdI4a = norm_factors['meanI4a'], norm_factors['stdI4a']
	    meanI4s, stdI4s = norm_factors['meanI4s'], norm_factors['stdI4s']
	else:
	    meanPsi, stdPsi = np.mean(Psi_gt), np.std(Psi_gt)
	    meanI1, stdI1 = np.mean(I1), np.std(I1)
	    meanI2, stdI2 = np.mean(I2), np.std(I2)
	    meanI4a, stdI4a = np.mean(I4a), np.std(I4a)
	    meanI4s, stdI4s = np.mean(I4s), np.std(I4s)
	    
	    norm_factors = {"meanPsi":meanPsi, "stdPsi":stdPsi, 
	                    "meanI1" :meanI1 , "stdI1" :stdI1 ,
	                    "meanI2" :meanI2 , "stdI2" :stdI2 ,
	                    "meanI4a":meanI4a, "stdI4a":stdI4a,
	                    "meanI4s":meanI4s, "stdI4s":stdI4s}
	    with open('savednet/' + model_name + '_factors.pkl','wb') as f:
	        pickle.dump(norm_factors,f)


	return meanPsi, meanI1, meanI2, meanI4a, meanI4s, stdPsi, stdI1, stdI2, stdI4a, stdI4s


def predict(model_name, dataset_name, impose_convexity=True):
	ndata, I1, I2, I4a, I4s, Psi_gt, X, Y, sigma_gt, F, C, C_inv  = preprocessing(dataset_name)
	meanPsi, meanI1, meanI2, meanI4a, meanI4s, stdPsi, stdI1, stdI2, stdI4a, stdI4s = normalization(True, model_name, Psi_gt, I1, I2, I4a, I4s)

    if impose_convexity:
        model_fname   = 'savednet/'+model_name+'.json'
        weights_fname = 'savednet/'+model_name+'_weights.h5'
    else:
        model_fname   = 'savednet/'+model_name+'_nonconv.json'
        weights_fname = 'savednet/'+model_name+'_nonconv_weights.h5'
	model = tf.keras.models.model_from_json(open(model_fname).read())
	model.load_weights(weights_fname)

	Psinorm = (Psi_gt - meanPsi)/stdPsi
	I1norm  = (I1     - meanI1) /stdI1
	I2norm  = (I2     - meanI2) /stdI2
	I4anorm = (I4a    - meanI4a)/stdI4a
	I4snorm = (I4s    - meanI4s)/stdI4s

	inputs = np.zeros([ndata,4])
	inputs[:,0] = I1norm
	inputs[:,1] = I2norm
	inputs[:,2] = I4anorm
	inputs[:,3] = I4snorm

	y_pred = model(inputs)
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

	return sigma_pred, d1, d2, d3, d4


