#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 20:18:00 2021

@author: vahidullahtac
Version Description: Non-eager execution
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle
plt.rcParams.update(plt.rcParamsDefault)


a0vals = [25, 50, 75]
a1vals = [5, 10, 15]
a2vals = [50, 100, 150]
avals = np.meshgrid(a0vals, a1vals, a2vals)
avals = np.array(avals).transpose().reshape(-1,3)
for ai in avals:
    model_name = 'P1C1_s_a0_{a0}_a1_{a1}_a2_{a2}'.format(a0=ai[0], a1=ai[1], a2=ai[2])
    a0, a1, a2 = ai
    a3 = 50
    print('Starting training of {m}'.format(m=model_name))
    print(a0,a1,a2,a3)
    #%% User Inputs
    load_existing_model = False #True: Inference mode, False: Training mode
    impose_convexity = True #In inference mode this determines which saved model to load
    dataset_name = 'P1C1_s'
    learning_rate = 0.0001
    n_epoch = 100000
    #P1C1: n_offx = n_offy = 61, n_equi = n_strx = n_stry = 0
    #S111S1: n_offx = n_offy = n_equi = 183, n_strx = n_stry = 0
    #P12AC1: n_offx = 72, n_offy = 76, n_equi = 81, n_strx = 101, n_stry = 72
    #P12BC2: n_offx = 76, n_offy = 76, n_equi = 95, n_strx = 101, n_stry = 84
    n_offx = 0#183#72#183#61 #Number of offx data points in the dataset. 
    n_offy = 0#183#76#183#61
    n_equi = 0#183#0
    n_strx = 0#101
    n_stry = 0#84
    plot_derivatives = False
    n_hf = n_offx + n_offy + n_equi + n_strx + n_stry #number of high fidelity data points

    #%% Load and Process Training Data
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
    v0 = [1,0,0]
    w0 = [0,1,0]
    I4a = np.einsum('i,pij,j->p',v0,C,v0)
    I4s = np.einsum('i,pij,j->p',w0,C,w0)
    Psi_gt = np.zeros(ndata)
    for i in range(ndata):
        if C[i,0,0] != 1 or C[i,1,1] != 1:
            # delPsi = 1/2*S*delC
            Psi_gt[i] = Psi_gt[i-1]
            Psi_gt[i]+= 0.5*0.5*(S[i,0,0] + S[i-1,0,0])*(C[i,0,0] - C[i-1,0,0])
            Psi_gt[i]+= 0.5*0.5*(S[i,1,1] + S[i-1,1,1])*(C[i,1,1] - C[i-1,1,1])

    #### Normalize the data
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

    #%% Custom Loss Function
    #Define custom loss function
    inputstensor = tf.Variable(inputs)
    def custom_loss(y_true, y_pred):
        global a0, a1, a2, a3
        #Going from dPsi_norm/dI1_norm to dPsi/dI1 etc
        d1 = y_pred[:,1]*stdPsi/stdI1
        d2 = y_pred[:,2]*stdPsi/stdI2
        d3 = y_pred[:,3]*stdPsi/stdI4a
        d4 = y_pred[:,4]*stdPsi/stdI4s
        
        # p = -(2*d1[:]*C[:,2,2] + 2*d2[:]*(I1-C[:,2,2]))*C[:,2,2]
        p = -(2*d1[:] + 2*d2[:]*(I1-C[:,2,2]))*C[:,2,2]
        # p = d1*0
        I1_2 = np.zeros_like(C)
        a0a0 = np.zeros_like(C) #a0 dyadic a0
        s0s0 = np.zeros_like(C) #s0 dyadic s0
        a0a0[:,0,0] = 1
        s0s0[:,1,1] = 1
        sigma_list = []
        for i in range(ndata):
            I1_2[i,:,:] = I1[i]*np.eye(3)
            dummy = F[i,:,:]*(2*d1[i]*np.eye(3) + 2*d2[i]*(I1_2[i,:,:]-C[i,:,:]) + 
                              2*d3[i]*a0a0[i,:,:] + 2*d4[i]*s0s0[i,:,:] + 
                              p[i]*C_inv[i,:,:])*F[i,:,:]
            sigma_list.append(dummy)
        sigma_pred = tf.stack(sigma_list)
        out = model(inputstensor)
        gradPsi = K.gradients(out[:,0],inputstensor)[0]
        dPsidI1, dPsidI2, dPsidI4a, dPsidI4s = gradPsi[:,0], gradPsi[:,1], gradPsi[:,2], gradPsi[:,3]
        dPsidI1 = tf.cast(dPsidI1,tf.float32)
        dPsidI2 = tf.cast(dPsidI2,tf.float32)
        dPsidI4a = tf.cast(dPsidI4a,tf.float32)
        dPsidI4s = tf.cast(dPsidI4s,tf.float32)
        
        #Second derivatives
        gradPsi1  = K.gradients(out[:,1],inputstensor)[0] #gradient of dPsidI1 (=Psi1) wrt all input nodes
        dI1dI1    = gradPsi1[:,0] #d^2Psi/d^I1
        dI1dI2    = gradPsi1[:,1] #d^2Psi/dI1dI2
        gradPsi2  = K.gradients(out[:,2],inputstensor)[0] #gradient of dPsidI2 (=Psi2) wrt all input nodes
        dI2dI2    = gradPsi2[:,1]
        gradPsi4a = K.gradients(out[:,3],inputstensor)[0] #gradient of dPsidI4a (=Psi4a) wrt all input nodes
        gradPsi4s = K.gradients(out[:,4],inputstensor)[0] #gradient of dPsidI4s (=Psi4s) wrt all input nodes
        
        #Symmetry Loss
        loss_symm = K.sum(K.abs(gradPsi1[:,1]  - gradPsi2[:,0] )) #dPsidI1dI2 - dPsidI2dI1
        loss_symm+= K.sum(K.abs(gradPsi1[:,2]  - gradPsi4a[:,0]))
        loss_symm+= K.sum(K.abs(gradPsi1[:,3]  - gradPsi4s[:,0]))
        loss_symm+= K.sum(K.abs(gradPsi2[:,2]  - gradPsi4a[:,1]))
        loss_symm+= K.sum(K.abs(gradPsi2[:,3]  - gradPsi4s[:,1]))
        loss_symm+= K.sum(K.abs(gradPsi4a[:,3] - gradPsi4s[:,2]))
        loss_symm = tf.cast(loss_symm, tf.float32)
        
        #Convexity Loss
        #For the time being apply positive definiteness (instead of positive semidefiniteness)
        #Source: Prussing 1985
        #Leading Principal Minors, LPM
        matrix1 = [gradPsi1[:,:3], gradPsi2[:,:3], gradPsi4a[:,:3]] #Hessian matrix minus the last row and column
        matrix1 = tf.stack(matrix1)
        matrix1 = tf.transpose(matrix1, perm=[1,0,2])
        
        matrix2 = [gradPsi1, gradPsi2, gradPsi4a, gradPsi4s] #Hessian matrix
        matrix2 = tf.stack(matrix2)
        matrix2 = tf.transpose(matrix2, perm=[1,0,2])
        
        LPM1 = dI1dI1
        LPM2 = dI1dI1*dI2dI2 - dI1dI2*dI1dI2
        LPM3 = tf.linalg.det(matrix1)
        LPM4 = tf.linalg.det(matrix2)
        
        zeros = tf.zeros_like(LPM1)
        loss_conv = K.sum(K.max((-LPM1, zeros),axis=0))
        loss_conv+= K.sum(K.max((-LPM2, zeros),axis=0))
        loss_conv+= K.sum(K.max((-LPM3, zeros),axis=0))
        loss_conv+= K.sum(K.max((-LPM4, zeros),axis=0))
        loss_conv = tf.cast(loss_conv,tf.float32)
        
        #original values: a0 = 10, a2 = 10
        # a0 = 25         #Stress
        # a1 = 10         #Symmetry
        # a2 = 100        #Convexity
        # a3 = 50         #High fidelity
        
        #High Fidelity and Low Fidelity Loss --- MSE
        loss_sigma_hf = K.sum((sigma_pred[:n_hf,0,0] - sigma_gt[:n_hf,0,0])**2)
        loss_sigma_hf+= K.sum((sigma_pred[:n_hf,1,1] - sigma_gt[:n_hf,1,1])**2)
        loss_sigma_hf+= K.sum((sigma_pred[:n_hf,2,2] - sigma_gt[:n_hf,2,2])**2)
        loss_Psi_hf   = K.sum((y_pred[:n_hf,0]       -   y_true[:n_hf,0]  )**2)
        loss_Psi_hf  += K.sum((y_pred[:n_hf,1]       -  dPsidI1[:n_hf]    )**2)
        loss_Psi_hf  += K.sum((y_pred[:n_hf,2]       -  dPsidI2[:n_hf]    )**2)
        loss_Psi_hf  += K.sum((y_pred[:n_hf,3]       - dPsidI4a[:n_hf]    )**2)
        loss_Psi_hf  += K.sum((y_pred[:n_hf,4]       - dPsidI4s[:n_hf]    )**2)
        
        loss_sigma_lf = K.sum((sigma_pred[n_hf:,0,0] - sigma_gt[n_hf:,0,0])**2)
        loss_sigma_lf+= K.sum((sigma_pred[n_hf:,1,1] - sigma_gt[n_hf:,1,1])**2)
        loss_sigma_lf+= K.sum((sigma_pred[n_hf:,2,2] - sigma_gt[n_hf:,2,2])**2)
        loss_Psi_lf   = K.sum((y_pred[n_hf:,0]       -   y_true[n_hf:,0]  )**2)
        loss_Psi_lf  += K.sum((y_pred[n_hf:,1]       -  dPsidI1[n_hf:]    )**2)
        loss_Psi_lf  += K.sum((y_pred[n_hf:,2]       -  dPsidI2[n_hf:]    )**2)
        loss_Psi_lf  += K.sum((y_pred[n_hf:,3]       - dPsidI4a[n_hf:]    )**2)
        loss_Psi_lf  += K.sum((y_pred[n_hf:,4]       - dPsidI4s[n_hf:]    )**2)
        
        loss_hf = a0*loss_sigma_hf + loss_Psi_hf
        loss_lf = a0*loss_sigma_lf + loss_Psi_lf
        total_loss = a3*loss_hf + loss_lf
        if impose_convexity == True:
            total_loss+= a1*loss_symm + a2*loss_conv
        return total_loss

    #%% Custom Metrics
    def symmetry(y_true, y_pred):
        out = model(inputstensor)
        gradPsi1  = K.gradients(out[:,1],inputstensor)[0]
        gradPsi2  = K.gradients(out[:,2],inputstensor)[0]
        gradPsi4a = K.gradients(out[:,3],inputstensor)[0]
        gradPsi4s = K.gradients(out[:,4],inputstensor)[0]
        loss_symm = K.sum(K.abs(gradPsi1[:,1]  - gradPsi2[:,0] )) #dPsidI1dI2 - dPsidI2dI1
        loss_symm+= K.sum(K.abs(gradPsi1[:,2]  - gradPsi4a[:,0]))
        loss_symm+= K.sum(K.abs(gradPsi1[:,3]  - gradPsi4s[:,0]))
        loss_symm+= K.sum(K.abs(gradPsi2[:,2]  - gradPsi4a[:,1]))
        loss_symm+= K.sum(K.abs(gradPsi2[:,3]  - gradPsi4s[:,1]))
        loss_symm+= K.sum(K.abs(gradPsi4a[:,3] - gradPsi4s[:,2]))
        return loss_symm
    def convexity(y_true, y_pred):
        out = model(inputstensor)
        gradPsi1  = K.gradients(out[:,1],inputstensor)[0]
        gradPsi2  = K.gradients(out[:,2],inputstensor)[0]
        gradPsi4a = K.gradients(out[:,3],inputstensor)[0]
        gradPsi4s = K.gradients(out[:,4],inputstensor)[0]
        dI1dI1    = gradPsi1[:,0] #d^2Psi/d^I1
        dI1dI2    = gradPsi1[:,1] #d^2Psi/dI1dI2
        dI2dI2    = gradPsi2[:,1]
        matrix1 = [gradPsi1[:,:3], gradPsi2[:,:3], gradPsi4a[:,:3]] #Hessian matrix minus the last row and column
        matrix1 = tf.stack(matrix1)
        matrix1 = tf.transpose(matrix1, perm=[1,0,2])
        matrix2 = [gradPsi1, gradPsi2, gradPsi4a, gradPsi4s] #Hessian matrix
        matrix2 = tf.stack(matrix2)
        matrix2 = tf.transpose(matrix2, perm=[1,0,2])
        LPM1 = dI1dI1
        LPM2 = dI1dI1*dI2dI2 - dI1dI2*dI1dI2
        LPM3 = tf.linalg.det(matrix1)
        LPM4 = tf.linalg.det(matrix2)
        zeros = tf.zeros_like(LPM1)
        loss_conv = K.sum(K.max((-LPM1-0.01, zeros),axis=0)) #This is to ensure that LPM1 > 0.01 (=a positive number)
        loss_conv+= K.sum(K.max((-LPM2-0.01, zeros),axis=0))
        loss_conv+= K.sum(K.max((-LPM3-0.01, zeros),axis=0))
        loss_conv+= K.sum(K.max((-LPM4-0.01, zeros),axis=0))
        return loss_conv
    def loss_sigma(y_true, y_pred):
        global a2
        #Going from dPsi_norm/dI1_norm to dPsi/dI1 etc
        d1 = y_pred[:,1]*stdPsi/stdI1
        d2 = y_pred[:,2]*stdPsi/stdI2
        d3 = y_pred[:,3]*stdPsi/stdI4a
        d4 = y_pred[:,4]*stdPsi/stdI4s
        
        p = -(2*d1[:]*C[:,2,2] + 2*d2[:]*(I1-C[:,2,2]))*C[:,2,2]
        I1_2 = np.zeros_like(C)
        a0a0 = np.zeros_like(C) #a0 dyadic a0
        s0s0 = np.zeros_like(C) #s0 dyadic s0
        a0a0[:,0,0] = 1
        s0s0[:,1,1] = 1
        sigma_list = []
        for i in range(ndata):
            I1_2[i,:,:] = I1[i]*np.eye(3)
            dummy = F[i,:,:]*(2*d1[i]*np.eye(3) + 2*d2[i]*(I1_2[i,:,:]-C[i,:,:]) + 
                              2*d3[i]*a0a0[i,:,:] + 2*d4[i]*s0s0[i,:,:] + 
                              p[i]*C_inv[i,:,:])*F[i,:,:]
            sigma_list.append(dummy)
        sigma_pred = tf.stack(sigma_list)
        
        loss_sigma_hf = K.sum((sigma_pred[:n_hf,0,0] - sigma_gt[:n_hf,0,0])**2)
        loss_sigma_hf+= K.sum((sigma_pred[:n_hf,1,1] - sigma_gt[:n_hf,1,1])**2)
        loss_sigma_lf = K.sum((sigma_pred[n_hf:,0,0] - sigma_gt[n_hf:,0,0])**2)
        loss_sigma_lf+= K.sum((sigma_pred[n_hf:,1,1] - sigma_gt[n_hf:,1,1])**2)
        return a2*loss_sigma_hf + loss_sigma_lf

    #%% Define and Fit the Model
    if load_existing_model == True:
        if impose_convexity == True:
            model_fname   = 'savednet/' + model_name + '.json'
            weights_fname = 'savednet/' + model_name + '_weights.h5'
        else:
            model_fname   = 'savednet/' + model_name + '_nonconv.json'
            weights_fname = 'savednet/' + model_name + '_nonconv_weights.h5'
            
        model = tf.keras.models.model_from_json(open(model_fname).read())
        model.load_weights(weights_fname)
    else:
        model = Sequential()
        model.add(Dense(4, input_dim=4, activation='sigmoid')) #Inputs: I1, I2, I4a, I4s
        model.add(Dense(8, activation='sigmoid'))
        model.add(Dense(5, activation='linear')) #Outputs: Psi, Psi1, Psi2, Psi4a, Psi4s

        model.compile(loss = custom_loss, optimizer = Adam(learning_rate = learning_rate), 
                      metrics = [symmetry, convexity, loss_sigma])

        fit = model.fit(inputs, Psinorm, epochs = n_epoch, batch_size = ndata, verbose = 0, shuffle=False, workers=4)
        
        if impose_convexity == True:
            model_fname   = 'savednet/' + model_name + '.json'
            weights_fname = 'savednet/' + model_name + '_weights.h5'
            history_fname = 'savednet/' + model_name + '_history.pkl'
        else:
            model_fname   = 'savednet/' + model_name + '_nonconv.json'
            weights_fname = 'savednet/' + model_name + '_nonconv_weights.h5'
            history_fname = 'savednet/' + model_name + '_nonconv_history.pkl'
            
        with open(model_fname, 'w') as f:
           f.write(model.to_json())
        model.save_weights(weights_fname, overwrite=True)
        
        total_loss     = np.array(fit.history['loss'])
        sigma_loss     = np.array(fit.history['loss_sigma'])
        symmetry_loss  = np.array(fit.history['symmetry'])
        convexity_loss = np.array(fit.history['convexity'])
        Psi_loss       = total_loss - a0*sigma_loss - a1*symmetry_loss - a2*convexity_loss
        with open(history_fname, 'wb') as f:
            pickle.dump([total_loss, sigma_loss, symmetry_loss, convexity_loss, Psi_loss], f)
    
