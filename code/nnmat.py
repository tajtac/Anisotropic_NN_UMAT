import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import keras
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle
plt.rcParams.update(plt.rcParamsDefault)

# samples = ['S111S1', 'S111S4', 'S112S2', 'S112S3', 'S112S4', 'S113S3', 'S113S4', 'S143S1', 'S143S2', 'S144S3', 'S144S4']
# nn_arch = [[4,8,5], [4,4,8,5], [4,4,8,8,5], [4,4,4,5], [4,8,16,5]]
# samples = ['P1C1_s_noisy001', 'P1C1_s_noisy005', 'P1C1_s_noisy010']
weights = [[0,0],
                [0.050, 0.002], [0.050, 0.004], [0.050, 0.008], 
                [0.075, 0.002], [0.075, 0.004], [0.075, 0.008],
                [0.100, 0.002], [0.100, 0.004], [0.100, 0.008],
                                                            [0.2,0.016],
                                                                    [0.4, 0.032],
                                                                                [0.8, 0.064]]
# loadings = ['xy', 'sxsy', 'xyb', 'bsxsy', 'xysxsy']

for dataset_name in ['P1C1_s']:
    theta = 0
    NN = [4,4,8,8,5]
    #%% User Inputs
    # dataset_name = 'P1C1_s'
    # model_name = 'P1C1_s_a1_{p1}_a3_{p3}'.format(p1=a1,p3=a3)
    model_name = dataset_name
    # dataset_name = model_name
    theta = theta/180*np.pi
    print('Starting the training of '+ model_name)
    impose_convexity = True
    learning_rate = 0.00004
    n_epoch = 200000
    #P1C1: n_offx = n_offy = 61, n_equi = n_strx = n_stry = 0
    #S111S1: n_offx = n_offy = n_equi = 183, n_strx = n_stry = 0
    #P12AC1: n_offx = 72, n_offy = 76, n_equi = 81, n_strx = 101, n_stry = 72
    #P12BC2: n_offx = 76, n_offy = 76, n_equi = 95, n_strx = 101, n_stry = 84
    n_offx = 0#72#183#61 #Number of offx data points in the dataset. 
    n_offy = 0#76#183#61
    n_equi = 0#183#0
    n_strx = 0#101
    n_stry = 0#84
    n_hf = n_offx + n_offy + n_equi + n_strx + n_stry

    a1 = 0.1#0.075       #Psi (Internal consistency)
    a2 = 1.0         #Sigma (Data)
    a3 = 0.008#0.004       #Symmetry + convexity
    a_hf = 50


    #%% Load and Process Training Data
    with open('training_data/' + dataset_name + '.npy', 'rb') as f:
        X, Y = np.load(f,allow_pickle=True)
    #### Preprocessing: s1,s2,lm1,lm2 -> I1,I2,I4a,I4s,Psi
    ndata = X.shape[0]
    F = np.zeros([ndata,3,3])
    F[:,0,0] = X[:,0]
    F[:,1,1] = X[:,1]
    F[:,2,2] = 1/(X[:,0]*X[:,1])
    FT = np.transpose(F, axes=[0,2,1])
    sigma_gt = np.zeros_like(F)
    sigma_gt[:,0,0] = Y[:,0]
    sigma_gt[:,1,1] = Y[:,1]
    
    F_inv = np.linalg.inv(F)
    F_inv_T = np.linalg.inv(np.transpose(F,[0,2,1]))
    S = np.einsum('...ik,...kl,...lj->...ij',F_inv, sigma_gt, F_inv_T)
    b = C = F*F #Since F^T = F
    C_inv = np.linalg.inv(C)
    I1 = C.trace(axis1=1, axis2=2)
    I2 = 1/2*(I1**2 - np.trace(np.einsum("...ij,...jk->...ik", C, C),axis1=1, axis2=2))
    a0 = np.array([ np.cos(theta), np.sin(theta), 0])
    s0 = np.array([-np.sin(theta), np.cos(theta), 0])
    a0a0 = np.outer(a0,a0)
    s0s0 = np.outer(s0,s0)
    I4a = np.einsum('i,pij,j->p',a0,C,a0)
    I4s = np.einsum('i,pij,j->p',s0,C,s0)
    Psi_gt = np.zeros(ndata)
    for i in range(ndata):
        if C[i,0,0] != 1 or C[i,1,1] != 1:
            Psi_gt[i] = Psi_gt[i-1]
            Psi_gt[i]+= 0.5*0.5*(S[i,0,0] + S[i-1,0,0])*(C[i,0,0] - C[i-1,0,0])
            Psi_gt[i]+= 0.5*0.5*(S[i,1,1] + S[i-1,1,1])*(C[i,1,1] - C[i-1,1,1])

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

    I4a = np.clip(I4a, a_min=1.0, a_max=None)
    I4s = np.clip(I4s, a_min=1.0, a_max=None)

    Psinorm = (Psi_gt - meanPsi)/stdPsi
    I1norm  = (I1     - meanI1) /stdI1
    I2norm  = (I2     - meanI2) /stdI2
    I4anorm = (I4a    - meanI4a)/stdI4a
    I4snorm = (I4s    - meanI4s)/stdI4s

    #### Combine the NN inputs
    inputs = np.array([I1norm, I2norm, I4anorm, I4snorm]).transpose()
    inputstensor = tf.Variable(inputs)
    def custom_loss(y_true, y_pred):
        global a1, a2, a3, a_hf
        if impose_convexity == True:
            d1 = y_pred[:,1]**2*stdPsi/stdI1
            d2 = y_pred[:,2]**2*stdPsi/stdI2
            d3 = y_pred[:,3]**2*stdPsi/stdI4a
            d4 = y_pred[:,4]**2*stdPsi/stdI4s
        else:
            d1 = y_pred[:,1]*stdPsi/stdI1
            d2 = y_pred[:,2]*stdPsi/stdI2
            d3 = y_pred[:,3]*stdPsi/stdI4a
            d4 = y_pred[:,4]*stdPsi/stdI4s

        
        p = -(2*d1[:] + 2*d2[:]*(I1-C[:,2,2]))*C[:,2,2]
        I1_2 = np.zeros_like(C)
        sigma_list = []
        for i in range(ndata):
            I1_2[i,:,:] = I1[i]*np.eye(3)
            dummy = (2*d1[i]*np.eye(3) + 2*d2[i]*(I1_2[i,:,:]-C[i,:,:]) + 2*d3[i]*a0a0[:,:] + 2*d4[i]*s0s0[:,:] + 
                            p[i]*C_inv[i,:,:])
            dummy = tf.tensordot(F[i,:,:], tf.tensordot(dummy, FT[i,:,:], axes=1), axes=1)
            sigma_list.append(dummy)
        sigma_pred = tf.stack(sigma_list)
        out = model(inputstensor)
        gradPsi = K.gradients(out[:,0],inputstensor)[0]
        dPsidI1, dPsidI2, dPsidI4a, dPsidI4s = gradPsi[:,0], gradPsi[:,1], gradPsi[:,2], gradPsi[:,3]
        
        #Second derivatives
        if impose_convexity == True:
            gradPsi1  = K.gradients(out[:,1]**2/stdI1,inputstensor)[0] #gradient of dPsidI1 (=Psi1) wrt all input nodes
            gradPsi2  = K.gradients(out[:,2]**2/stdI2,inputstensor)[0] #gradient of dPsidI2 (=Psi2) wrt all input nodes
            gradPsi4a = K.gradients(out[:,3]**2/stdI4a,inputstensor)[0] #gradient of dPsidI4a (=Psi4a) wrt all input nodes
            gradPsi4s = K.gradients(out[:,4]**2/stdI4s,inputstensor)[0] #gradient of dPsidI4s (=Psi4s) wrt all input nodes
        else:
            gradPsi1  = K.gradients(out[:,1]/stdI1,inputstensor)[0]
            gradPsi2  = K.gradients(out[:,2]/stdI2,inputstensor)[0]
            gradPsi4a = K.gradients(out[:,3]/stdI4a,inputstensor)[0]
            gradPsi4s = K.gradients(out[:,4]/stdI4s,inputstensor)[0]

        scalingfactors = [stdI1, stdI2, stdI4a, stdI4s]
        gradPsi1list = []
        gradPsi2list = []
        gradPsi4alist = []
        gradPsi4slist = []
        for i in range(4):
            dummy1 = gradPsi1[:,i]/scalingfactors[i]
            dummy2 = gradPsi2[:,i]/scalingfactors[i]
            dummy3 = gradPsi4a[:,i]/scalingfactors[i]
            dummy4 = gradPsi4s[:,i]/scalingfactors[i]
            gradPsi1list.append(dummy1)
            gradPsi2list.append(dummy2)
            gradPsi4alist.append(dummy3)
            gradPsi4slist.append(dummy4)
        gradPsi1  = tf.transpose(tf.stack(gradPsi1list))
        gradPsi2  = tf.transpose(tf.stack(gradPsi2list))
        gradPsi4a = tf.transpose(tf.stack(gradPsi4alist))
        gradPsi4s = tf.transpose(tf.stack(gradPsi4slist))
        dI1dI1    = gradPsi1[:,0] #d^2Psi/d^I1
        dI1dI2    = gradPsi1[:,1] #d^2Psi/dI1dI2
        dI2dI2    = gradPsi2[:,1]
        
        #Symmetry Loss
        loss_symm = K.sum(K.abs(gradPsi1[:,1]  - gradPsi2[:,0] )) #dPsidI1dI2 - dPsidI2dI1
        loss_symm+= K.sum(K.abs(gradPsi1[:,2]  - gradPsi4a[:,0]))
        loss_symm+= K.sum(K.abs(gradPsi1[:,3]  - gradPsi4s[:,0]))
        loss_symm+= K.sum(K.abs(gradPsi2[:,2]  - gradPsi4a[:,1]))
        loss_symm+= K.sum(K.abs(gradPsi2[:,3]  - gradPsi4s[:,1]))
        loss_symm+= K.sum(K.abs(gradPsi4a[:,3] - gradPsi4s[:,2]))
        
        #Convexity Loss
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
        
        #High Fidelity and Low Fidelity Loss --- MSE
        loss_sigma_hf = K.sum((sigma_pred[:n_hf,0,0] - sigma_gt[:n_hf,0,0])**2)
        loss_sigma_hf+= K.sum((sigma_pred[:n_hf,1,1] - sigma_gt[:n_hf,1,1])**2)
        loss_sigma_hf+= K.sum((sigma_pred[:n_hf,2,2] - sigma_gt[:n_hf,2,2])**2)
        loss_Psi_hf   = K.sum((y_pred[:n_hf,0]       -   y_true[:n_hf,0]  )**2)
        
        loss_sigma_lf = K.sum((sigma_pred[n_hf:,0,0] - sigma_gt[n_hf:,0,0])**2)
        loss_sigma_lf+= K.sum((sigma_pred[n_hf:,1,1] - sigma_gt[n_hf:,1,1])**2)
        loss_sigma_lf+= K.sum((sigma_pred[n_hf:,2,2] - sigma_gt[n_hf:,2,2])**2)
        loss_Psi_lf   = K.sum((y_pred[n_hf:,0]       -   y_true[n_hf:,0]  )**2)

        if impose_convexity == True:
            loss_Psi_hf  += K.sum((y_pred[:n_hf,1]**2    -  dPsidI1[:n_hf])**2)
            loss_Psi_hf  += K.sum((y_pred[:n_hf,2]**2    -  dPsidI2[:n_hf])**2)
            loss_Psi_hf  += K.sum((y_pred[:n_hf,3]**2    - dPsidI4a[:n_hf])**2)
            loss_Psi_hf  += K.sum((y_pred[:n_hf,4]**2    - dPsidI4s[:n_hf])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,1]**2    -  dPsidI1[n_hf:])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,2]**2    -  dPsidI2[n_hf:])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,3]**2    - dPsidI4a[n_hf:])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,4]**2    - dPsidI4s[n_hf:])**2)
        else:
            loss_Psi_hf  += K.sum((y_pred[:n_hf,1]    -  dPsidI1[:n_hf])**2)
            loss_Psi_hf  += K.sum((y_pred[:n_hf,2]    -  dPsidI2[:n_hf])**2)
            loss_Psi_hf  += K.sum((y_pred[:n_hf,3]    - dPsidI4a[:n_hf])**2)
            loss_Psi_hf  += K.sum((y_pred[:n_hf,4]    - dPsidI4s[:n_hf])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,1]    -  dPsidI1[n_hf:])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,2]    -  dPsidI2[n_hf:])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,3]    - dPsidI4a[n_hf:])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,4]    - dPsidI4s[n_hf:])**2)
        
        loss_hf = a2*loss_sigma_hf + a1*loss_Psi_hf
        loss_lf = a2*loss_sigma_lf + a1*loss_Psi_lf
        total_loss = a_hf*loss_hf + loss_lf
        if impose_convexity == True:
            total_loss+= a3*(loss_symm + loss_conv)
        return total_loss
    def symmetry(y_true, y_pred):
        out = model(inputstensor)
        if impose_convexity == True:
            gradPsi1  = K.gradients(out[:,1]**2/stdI1,inputstensor)[0] #gradient of dPsidI1 (=Psi1) wrt all input nodes
            gradPsi2  = K.gradients(out[:,2]**2/stdI2,inputstensor)[0] #gradient of dPsidI2 (=Psi2) wrt all input nodes
            gradPsi4a = K.gradients(out[:,3]**2/stdI4a,inputstensor)[0] #gradient of dPsidI4a (=Psi4a) wrt all input nodes
            gradPsi4s = K.gradients(out[:,4]**2/stdI4s,inputstensor)[0] #gradient of dPsidI4s (=Psi4s) wrt all input nodes
        else:
            gradPsi1  = K.gradients(out[:,1]/stdI1,inputstensor)[0]
            gradPsi2  = K.gradients(out[:,2]/stdI2,inputstensor)[0]
            gradPsi4a = K.gradients(out[:,3]/stdI4a,inputstensor)[0]
            gradPsi4s = K.gradients(out[:,4]/stdI4s,inputstensor)[0]

        scalingfactors = [stdI1, stdI2, stdI4a, stdI4s]
        gradPsi1list = []
        gradPsi2list = []
        gradPsi4alist = []
        gradPsi4slist = []
        for i in range(4):
            dummy1 = gradPsi1[:,i]/scalingfactors[i]
            dummy2 = gradPsi2[:,i]/scalingfactors[i]
            dummy3 = gradPsi4a[:,i]/scalingfactors[i]
            dummy4 = gradPsi4s[:,i]/scalingfactors[i]
            gradPsi1list.append(dummy1)
            gradPsi2list.append(dummy2)
            gradPsi4alist.append(dummy3)
            gradPsi4slist.append(dummy4)
        gradPsi1  = tf.transpose(tf.stack(gradPsi1list))
        gradPsi2  = tf.transpose(tf.stack(gradPsi2list))
        gradPsi4a = tf.transpose(tf.stack(gradPsi4alist))
        gradPsi4s = tf.transpose(tf.stack(gradPsi4slist))

        loss_symm = K.sum(K.abs(gradPsi1[:,1]  - gradPsi2[:,0] )) #dPsidI1dI2 - dPsidI2dI1
        loss_symm+= K.sum(K.abs(gradPsi1[:,2]  - gradPsi4a[:,0]))
        loss_symm+= K.sum(K.abs(gradPsi1[:,3]  - gradPsi4s[:,0]))
        loss_symm+= K.sum(K.abs(gradPsi2[:,2]  - gradPsi4a[:,1]))
        loss_symm+= K.sum(K.abs(gradPsi2[:,3]  - gradPsi4s[:,1]))
        loss_symm+= K.sum(K.abs(gradPsi4a[:,3] - gradPsi4s[:,2]))
        return loss_symm
    def convexity(y_true, y_pred):
        out = model(inputstensor)
        if impose_convexity == True:
            gradPsi1  = K.gradients(out[:,1]**2/stdI1,inputstensor)[0] #gradient of dPsidI1 (=Psi1) wrt all input nodes
            gradPsi2  = K.gradients(out[:,2]**2/stdI2,inputstensor)[0] #gradient of dPsidI2 (=Psi2) wrt all input nodes
            gradPsi4a = K.gradients(out[:,3]**2/stdI4a,inputstensor)[0] #gradient of dPsidI4a (=Psi4a) wrt all input nodes
            gradPsi4s = K.gradients(out[:,4]**2/stdI4s,inputstensor)[0] #gradient of dPsidI4s (=Psi4s) wrt all input nodes
        else:
            gradPsi1  = K.gradients(out[:,1]/stdI1,inputstensor)[0]
            gradPsi2  = K.gradients(out[:,2]/stdI2,inputstensor)[0]
            gradPsi4a = K.gradients(out[:,3]/stdI4a,inputstensor)[0]
            gradPsi4s = K.gradients(out[:,4]/stdI4s,inputstensor)[0]

        scalingfactors = [stdI1, stdI2, stdI4a, stdI4s]
        gradPsi1list = []
        gradPsi2list = []
        gradPsi4alist = []
        gradPsi4slist = []
        for i in range(4):
            dummy1 = gradPsi1[:,i]/scalingfactors[i]
            dummy2 = gradPsi2[:,i]/scalingfactors[i]
            dummy3 = gradPsi4a[:,i]/scalingfactors[i]
            dummy4 = gradPsi4s[:,i]/scalingfactors[i]
            gradPsi1list.append(dummy1)
            gradPsi2list.append(dummy2)
            gradPsi4alist.append(dummy3)
            gradPsi4slist.append(dummy4)
        gradPsi1  = tf.transpose(tf.stack(gradPsi1list))
        gradPsi2  = tf.transpose(tf.stack(gradPsi2list))
        gradPsi4a = tf.transpose(tf.stack(gradPsi4alist))
        gradPsi4s = tf.transpose(tf.stack(gradPsi4slist))

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
        global a_hf
        if impose_convexity == True:
            d1 = y_pred[:,1]**2*stdPsi/stdI1
            d2 = y_pred[:,2]**2*stdPsi/stdI2
            d3 = y_pred[:,3]**2*stdPsi/stdI4a
            d4 = y_pred[:,4]**2*stdPsi/stdI4s
        else:
            d1 = y_pred[:,1]*stdPsi/stdI1
            d2 = y_pred[:,2]*stdPsi/stdI2
            d3 = y_pred[:,3]*stdPsi/stdI4a
            d4 = y_pred[:,4]*stdPsi/stdI4s

        
        p = -(2*d1[:] + 2*d2[:]*(I1-C[:,2,2]))*C[:,2,2]
        I1_2 = np.zeros_like(C)
        sigma_list = []
        for i in range(ndata):
            I1_2[i,:,:] = I1[i]*np.eye(3)
            dummy = (2*d1[i]*np.eye(3) + 2*d2[i]*(I1_2[i,:,:]-C[i,:,:]) + 2*d3[i]*a0a0[:,:] + 2*d4[i]*s0s0[:,:] + 
                            p[i]*C_inv[i,:,:])
            dummy = tf.tensordot(F[i,:,:], tf.tensordot(dummy, FT[i,:,:], axes=1), axes=1)
            sigma_list.append(dummy)
        sigma_pred = tf.stack(sigma_list)
        
        loss_sigma_hf = K.sum((sigma_pred[:n_hf,0,0] - sigma_gt[:n_hf,0,0])**2)
        loss_sigma_hf+= K.sum((sigma_pred[:n_hf,1,1] - sigma_gt[:n_hf,1,1])**2)
        loss_sigma_hf+= K.sum((sigma_pred[:n_hf,2,2] - sigma_gt[:n_hf,2,2])**2)
        loss_sigma_lf = K.sum((sigma_pred[n_hf:,0,0] - sigma_gt[n_hf:,0,0])**2)
        loss_sigma_lf+= K.sum((sigma_pred[n_hf:,1,1] - sigma_gt[n_hf:,1,1])**2)
        loss_sigma_lf+= K.sum((sigma_pred[n_hf:,2,2] - sigma_gt[n_hf:,2,2])**2)
        return a_hf*loss_sigma_hf + loss_sigma_lf
    def loss_Psi(y_true, y_pred):
        out = model(inputstensor)
        gradPsi = K.gradients(out[:,0],inputstensor)[0]
        dPsidI1, dPsidI2, dPsidI4a, dPsidI4s = gradPsi[:,0], gradPsi[:,1], gradPsi[:,2], gradPsi[:,3]

        loss_Psi_hf   = K.sum((y_pred[:n_hf,0]       -   y_true[:n_hf,0]  )**2)
        loss_Psi_lf   = K.sum((y_pred[n_hf:,0]       -   y_true[n_hf:,0]  )**2)
        if impose_convexity == True:
            loss_Psi_hf  += K.sum((y_pred[:n_hf,1]**2    -  dPsidI1[:n_hf])**2)
            loss_Psi_hf  += K.sum((y_pred[:n_hf,2]**2    -  dPsidI2[:n_hf])**2)
            loss_Psi_hf  += K.sum((y_pred[:n_hf,3]**2    - dPsidI4a[:n_hf])**2)
            loss_Psi_hf  += K.sum((y_pred[:n_hf,4]**2    - dPsidI4s[:n_hf])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,1]**2    -  dPsidI1[n_hf:])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,2]**2    -  dPsidI2[n_hf:])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,3]**2    - dPsidI4a[n_hf:])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,4]**2    - dPsidI4s[n_hf:])**2)
        else:
            loss_Psi_hf  += K.sum((y_pred[:n_hf,1]    -  dPsidI1[:n_hf])**2)
            loss_Psi_hf  += K.sum((y_pred[:n_hf,2]    -  dPsidI2[:n_hf])**2)
            loss_Psi_hf  += K.sum((y_pred[:n_hf,3]    - dPsidI4a[:n_hf])**2)
            loss_Psi_hf  += K.sum((y_pred[:n_hf,4]    - dPsidI4s[:n_hf])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,1]    -  dPsidI1[n_hf:])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,2]    -  dPsidI2[n_hf:])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,3]    - dPsidI4a[n_hf:])**2)
            loss_Psi_lf  += K.sum((y_pred[n_hf:,4]    - dPsidI4s[n_hf:])**2)
        loss_Psi = a_hf*loss_Psi_hf + loss_Psi_lf
        return loss_Psi
    class custom_out_layer(keras.layers.Layer):
        def __init__(self, units=32, input_dim=32):
            super(custom_out_layer, self).__init__()
            self.w1 = self.add_weight(shape=(input_dim, units), initializer="random_normal", trainable=True)
            self.w2 = self.add_weight(shape=(units, units), initializer="random_normal", trainable=True)
            self.b  = self.add_weight(shape=(1,units), initializer="random_normal", trainable=True)

        def call(self, inputs):
            x = tf.matmul(inputs, self.w1) + self.b
            x = tf.math.sigmoid(x)
            x = tf.matmul(x+1, self.w2)
            return x
    #%% Define and Fit the Model
    model = Sequential()
    model.add(Dense(NN[1], input_dim=NN[0], activation='sigmoid')) #Inputs: I1, I2, I4a, I4s
    for i in range(2,len(NN)-1):
        model.add(Dense(NN[i], activation='sigmoid'))
    model.add(Dense(NN[-1], activation='linear')) #Outputs: Psi, Psi1, Psi2, Psi4a, Psi4s
    # model.add(Dense(NN[-1], activation='relu'))
    # model.add(custom_out_layer(NN[-1], NN[-2]))
    model.compile(loss = custom_loss, optimizer = Adam(learning_rate = learning_rate), 
                metrics = [symmetry, convexity, loss_sigma, loss_Psi])

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
    Psi_loss       = np.array(fit.history['loss_Psi'])
    print('-----------------------------')
    print('Total Loss = ', total_loss[-1])
    print('sigma loss = ', sigma_loss[-1])
    print('smtry loss = ', symmetry_loss[-1])
    print('cnvex_loss = ', convexity_loss[-1])
    print('Psi_loss = ', Psi_loss[-1])
    print('-----------------------------')
    with open(history_fname, 'wb') as f:
        pickle.dump([total_loss, sigma_loss, symmetry_loss, convexity_loss, Psi_loss], f)
