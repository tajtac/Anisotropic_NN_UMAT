import numpy as np

#Only used for generating synthetic data for the synthetic validation figure because this one accepts F instead of lambdas.
class GOH_fullyinc():
    def __init__(self, F, mu = 0.04498, k1 = 4.9092, k2 = 76.64134, kappa = 1/3, theta = 0): #lm.shape = (n,2)
        self.mu = mu
        self.k1 = k1
        self.k2 = k2
        self.kappa = kappa
        n = F.shape[0]
        # self.C = F*F #Since F^T = F.
        self.C = np.einsum('...ji,...jk->...ik', F, F)
        self.F = F
        self.I1 = np.zeros(n)
        self.I1 = self.C.trace(axis1=1, axis2=2) #In this case C_isoch = C because J = 1.
        e_0 = [np.cos(theta), np.sin(theta), 0] #Fiber direction.
        self.I4 = np.zeros(n)
        for i in range(0,3):
            for j in range(0,3):
                self.I4[:] = self.I4[:] + e_0[i]*self.C[:,i,j]*e_0[j]
        self.E = kappa*(self.I1-3) + (1-3*kappa)*(self.I4-1)
        self.e_0 = e_0
    def s(self): #Cauchy stress
        mu = self.mu
        k1 = self.k1
        k2 = self.k2
        kappa = self.kappa
        I1 = self.I1
        I4 = self.I4
        E = self.E
        F = self.F
        C = self.C
        C_inv = np.linalg.inv(self.C)
        n = F.shape[0]
        I = np.identity(3)
        S_iso = np.zeros([n,3,3])
        S_vol = np.zeros([n,3,3])
        for i in range(0,3):
            for j in range(0,3):
                S_iso[:,i,j] = mu*(I[i,j] - 1/3*I1[:]*C_inv[:,i,j])
        #There is a mistake in the paper in the equation above.
        #The J**(-2/3) is supposed to be factored out. See Prof. Tepole's notes.
        #S_vol = 0.0 because we are assuming fully incompressible material
        eiej = np.outer(self.e_0,self.e_0) #e_i dyadic e_j
        dI1dC = np.zeros(n)
        dI4dC = np.zeros(n)
        S_aniso = np.zeros([n,3,3])
        for i in range(0,3):
            for j in range(0,3):
                dI1dC[:] = I[i,j] - 1/3*I1[:]*C_inv[:,i,j]
                # dI1dC[:] = I[i,j] #Both this and the line above are true when C_isoch = C
                dI4dC[:] = eiej[i,j] - 1/3*I4[:]*C_inv[:,i,j]
                # dI4dC[:] = eiej[i,j] #Both this and the line above are true when C_isoch = C

                S_aniso[:,i,j] = 2*k1*np.exp(k2*E[:]**2)*E[:]*(kappa*dI1dC[:] + (1-3*kappa)*dI4dC[:])
        p = -(S_iso[:,2,2] + S_aniso[:,2,2])*C[:,2,2]
        for i in range(0,3):
            for j in range(0,3):
                S_vol[:,i,j] = p[:]*C_inv[:,i,j]
        S = S_iso + S_aniso + S_vol
        # s = F*S*F #Since F^T = F
        s = np.einsum('...ij,...jk,...lk->...il', F, S, F)
        return s

