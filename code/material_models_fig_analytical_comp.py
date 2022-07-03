import numpy as np

# This is a later version of the material_models module which I only use for fig_analytical_comp because it has new material models.
class neoHook(): #Incompressible neo-Hookean
    n_params = 1
    def __init__(self, param):
        self.C1 = param[0]
    def kinematics(self, lm):
        n = np.size(lm,0)
        F = np.zeros([n,3,3])
        F[:,0,0] = lm[:,0]
        F[:,1,1] = lm[:,1]
        F[:,2,2] = 1/(lm[:,0]*lm[:,1])
        C = np.einsum('...ji,...jk->...ik', F, F)
        I1 = C.trace(axis1=1, axis2=2)
        return F, C, I1
    def Psi(self, lm):
        C1 = self.C1
        _,_,I1 = self.kinematics(lm)
        return C1*(I1-3)
    def S(self, lm):
        C1 = self.C1
        _, C, _ = self.kinematics(lm)
        C_inv = np.linalg.inv(C)
        n = C.shape[0]
        I = np.array([np.identity(3) for i in range(n)])
        p = 2*C1*C[:,2,2]
        S = 2*C1*I - p[:, None, None]*C_inv
        return S
    def sigma(self, lm):
        F, _, _, = self.kinematics(lm)
        S = self.S(lm)
        sigma = np.einsum('...ij,...jk,...lk->...il', F, S, F)
        return sigma
        

class MR(): #Source: Continuummechanics
    #Assumptions: Fully incompressible material. Thus paramter D is irrelevant.
    #Strain Energy
    n_params = 3
    def __init__(self, params):
        self.C10, self.C01, self.C20 = params
        
    def Psi(self, lm): #lm.shape = (n,2)
        C10, C01, C20 = self.C10, self.C01, self.C20
        lm3 = np.zeros(lm.shape[0])
        lm3[:] = 1/(lm[:,0] * lm[:,1])
        I1 = lm[:,0]**2 + lm[:,1]**2 + lm3**2
        I2 = 1/lm[:,0]**2 + 1/lm[:,1]**2 + 1/lm3**2
        return C10*(I1-3) + C01*(I2-3) + C20*(I1-3)**2
    
    def partials(self, lm):
        C10, C01, C20 = self.C10, self.C01, self.C20
        lm3 = np.zeros(lm.shape[0])
        lm3[:] = 1/(lm[:,0] * lm[:,1])
        I1 = lm[:,0]**2 + lm[:,1]**2 + lm3**2
        I2 = 1/lm[:,0]**2 + 1/lm[:,1]**2 + 1/lm3**2
        return C10 + 2*C20*(I1-3), C01
        
    #stress tensor given lm1 and lm2, assuming sigma3=0 and J=1
    def sigma(self, lm): #lm.shape = (n,2)
        C10, C01, C20 = self.C10, self.C01, self.C20
        lm1 = lm[:,0]
        lm2 = lm[:,1]
        lm3 = 1/(lm1*lm2)
        sigma1 = 2*(C10*(lm1**2 - lm3**2) - C01*(1/lm1**2 - 1/lm3**2) +
                  2*C20*(lm1**2 - lm3**2)*(lm1**2 + lm2**2 + lm3**2 - 3))
        sigma2 = 2*(C10*(lm2**2 - lm3**2) - C01*(1/lm2**2 - 1/lm3**2) +
                  2*C20*(lm2**2 - lm3**2)*(lm1**2 + lm2**2 + lm3**2 - 3))
        sigma = np.zeros((lm.shape[0],3,3))
        sigma[:,0,0] = sigma1
        sigma[:,1,1] = sigma2
        return sigma

class MR_compressible():
    #source: http://www.oofem.org/resources/doc/matlibmanual/html/node9.html
    n_params = 3
    def __init__(self, params):
        self.params = params

    def kinematics(self, lm): #lm: [n,3]
        n = np.size(lm,0)
        F = np.zeros([n,3,3])
        F[:,0,0] = lm[:,0]
        F[:,1,1] = lm[:,1]
        F[:,2,2] = lm[:,2]
        Finv = np.linalg.inv(F)
        FinvT = np.transpose(Finv,axes=[0,2,1])
        C = np.einsum('...ji,...jk->...ik', F, F)
        C2 = np.einsum('pij,pjk->pik', C, C)
        trC2 = C2.trace(axis1=1, axis2=2)
        I1 = C.trace(axis1=1, axis2=2)
        I2 = 1/2*(I1**2-trC2)
        J = np.linalg.det(F)
        I1bar = J**(-2/3)*I1
        I2bar = J**(-4/3)*I2
        return F,FinvT, C, I1bar, I2bar, J

    def P(self, lm):
        C1, C2, K = self.params
        F,FinvT,C,I1bar,I2bar,J = self.kinematics(lm)
        dI1bardF = 2*J[:, None, None]**(-2/3)*F - 2/3*I1bar[:, None, None]*FinvT
        dI2bardF = 2*I1bar[:, None, None]*F - 4/3*I2bar[:, None, None]*FinvT - 2/3*J[:, None, None]**(-4/3)*np.einsum('pij,pjk->pik',F,C)
        P = C1*dI1bardF + C2*dI2bardF + K*np.log(J)[:, None, None]*FinvT
        return P

    def sigma(self, lm):
        F,FinvT,C,I1bar,I2bar,J = self.kinematics(lm)
        P = self.P(lm)
        sigma = J[:, None, None]**(-1)*np.einsum('pij,pjk->pik', P, np.transpose(F,axes=[0,2,1]))
        return sigma

        

class GOH():
    #Paper: Propagation of material behavior uncertainty in a nonlinear finite
    #element model of reconstructive surgery
    #This assumes fully incompressible material.
    n_params = 5
    def __init__(self, params): #lm.shape = (n,2)
        self.mu, self.k1, self.k2, self.kappa, self.theta = params
        
    def kill_I4(self, E, I4):
        for i in range(E.shape[0]):
            E[i] = np.max([0,E[i]])
#             if I4[i]<0:
#                 E[i] = 0
        return E

    def lm2F(self, lm):
        theta = self.theta
        n = np.size(lm,0)
        F = np.zeros([n,3,3])
        F[:,0,0] = lm[:,0]
        F[:,1,1] = lm[:,1]
        F[:,2,2] = 1/(lm[:,0]*lm[:,1])
        return F
        
    def kinematics(self, F):
        theta = self.theta
        C = np.einsum('...ji,...jk->...ik', F, F)
        I1 = C.trace(axis1=1, axis2=2)
        e_0 = [np.cos(theta), np.sin(theta), 0]
        I4 = np.einsum('i,pij,j->p', e_0, C, e_0)
        return C, I1, I4, e_0
    
    def Psi_from_inv(self, I1, I4): #lm.shape = (n,2)
        mu, k1, k2, kappa, theta = self.mu, self.k1, self.k2, self.kappa, self.theta
        E = kappa*(I1-3) + (1-3*kappa)*(I4-1)
        E = self.kill_I4(E, I4)
        Psi_iso = mu/2*(I1-3)
        Psi_aniso = k1/2/k2*(np.exp(k2*E**2) - 1)
        Psi = Psi_iso + Psi_aniso
        return Psi
    
    def Psi(self, lm):
        F = self.lm2F(lm)
        _, I1, I4, _ = self.kinematics(F)
        return self.Psi_from_inv(I1, I4)
    
    def partials_from_inv(self, I1, I4):
        mu, k1, k2, kappa, theta = self.mu, self.k1, self.k2, self.kappa, self.theta
        E = kappa*(I1-3) + (1-3*kappa)*(I4-1)
        E = self.kill_I4(E, I4)
        Psi1 = mu/2 + k1*np.exp(k2*E**2)*E*kappa
        Psi4 = k1*np.exp(k2*E**2)*E*(1-3*kappa)
        return Psi1, Psi4
    
    def partials(self, lm):
        F = self.lm2F(lm)
        _, I1, I4, _ = self.kinematics(F)
        return self.partial_from_inv(I1, I4)
    
    def S_from_F(self, F):
        mu, k1, k2, kappa, theta = self.mu, self.k1, self.k2, self.kappa, self.theta
        C, I1, I4, e_0 = self.kinematics(F)
        eiej = np.outer(e_0,e_0)
        E = kappa*(I1-3) + (1-3*kappa)*(I4-1)
        E = self.kill_I4(E, I4)
        C_inv = np.linalg.inv(C)
        n = F.shape[0]
        I = np.identity(3)
        S_iso = mu*(I - 1/3*np.einsum('...,...ij->...ij', I1, C_inv))
        eiej = np.outer(e_0,e_0)
        dI1dC = I    - 1/3*np.einsum('...,...ij->...ij', I1, C_inv)
        dI4dC = eiej - 1/3*np.einsum('...,...ij->...ij', I4, C_inv)
        aux = 2*k1*np.exp(k2*E**2)*E
        S_aniso = np.einsum('...,...ij->...ij', aux, kappa*dI1dC + (1-3*kappa)*dI4dC)
        p = -(S_iso[:,2,2] + S_aniso[:,2,2])/C_inv[:,2,2]
        S_vol = np.einsum('...,...ij->...ij', p, C_inv)
        S = S_iso + S_aniso + S_vol
        return S
    
    def sigma_from_F(self, F):
        S = self.S_from_F(F)
        sigma = np.einsum('...ij,...jk,...lk->...il', F, S, F)
        return sigma
    
    def S(self, lm): 
        F = self.lm2F(lm)
        return self.S_from_F(F)
    
    def sigma(self, lm):
        F = self.lm2F(lm)
        return self.sigma_from_F(F)

class HGO():
    n_params = 4
    #Ref: M. Liu et al 2020
    def __init__(self, params):
        self.C10, self.k1, self.k2, self.theta = params
        
    def kill_I4(self, E, I4):
        for i in range(E.shape[0]):
            E[i] = np.max([0,E[i]])
    #             if I4[i]<0:
    #                 E[i] = 0
        return E

    def kinematics(self, lm):
        theta = self.theta
        v0 = np.array([np.cos(theta), np.sin(theta), 0])
        w0 = np.array([np.cos(theta),-np.sin(theta), 0])
        V0 = np.outer(v0,v0)
        W0 = np.outer(w0,w0)
        n = lm.shape[0]
        F = np.zeros([n,3,3])
        F[:,0,0] = lm[:,0]
        F[:,1,1] = lm[:,1]
        F[:,2,2] = 1/(lm[:,0]*lm[:,1])
        C = np.einsum('...ji,...jk->...ik', F, F)
        I1 = C[:,0,0] + C[:,1,1] + C[:,2,2]
        I4 = np.tensordot(C,V0)
        I6 = np.tensordot(C,W0)
        return F, C, I1, I4, I6, V0, W0
    
    def S(self, lm):
        C10, k1, k2, theta = self.C10, self.k1, self.k2, self.theta
        F, C, I1, I4, I6, V0, W0 = self.kinematics(lm)
        invC = np.linalg.inv(C)
        I = np.eye(3)
        Psi1 = C10
        E = I4-1
        E = self.kill_I4(E, I4)
        Psi4 = k1*(I4-1)*np.exp(k2*E**2)
        E = I6-1
        E = self.kill_I4(E, I6)
        Psi6 = k1*(I6-1)*np.exp(k2*(I6-1)**2)
        S2 = 2*Psi1*I + 2*Psi4[:, None, None]*V0 + 2*Psi6[:, None, None]*W0
        p = S2[:,2,2]/invC[:,2,2]
        S = S2 - p[:, None, None]*invC
        return S
    
    def sigma(self, lm):
        F, _, _, _, _, _, _ = self.kinematics(lm)
        S = self.S(lm)
        sigma = np.einsum('...ij,...jk,...lk->...il', F, S, F)
        return sigma
    
    def Psi(self, lm):
        C10, k1, k2, theta = self.C10, self.k1, self.k2, self.theta
        _, C, I1, I4, I6, _, _ = self.kinematics(lm)
        invC = np.linalg.inv(C)
        I = np.eye(3)
        Psi = C10*(I1-3) + k1/2/k2*(np.exp(k2*(I4-1)**2)-1 + np.exp(k2*(I6-1)**2)-1)
        return Psi
    
class Fung():
    n_params = 4
    #Source: Fung et al. 1979. Replace F with Q and C with c1 to avoid confusion with deformation tensors.
    #Also, set * values to zero.
    def __init__(self, params):
        self.c1, self.a1, self.a2, self.a4 = params
        
    def kinematics(self, lm):
        n = lm.shape[0]
        F = np.zeros([n,3,3])
        F[:,0,0] = lm[:,0]
        F[:,1,1] = lm[:,1]
        F[:,2,2] = 1/(lm[:,0]*lm[:,1])
        C = F*F
        E_11 = 0.5*(C[:,0,0]-1)
        E_22 = 0.5*(C[:,1,1]-1)
        E_Z = E_11
        E_theta = E_22
        return F, E_Z, E_theta

    def S(self, lm):
        c1, a1, a2, a4 = self.c1, self.a1, self.a2, self.a4
        F, E_Z, E_theta = self.kinematics(lm)
        Q = a1*E_theta**2 + a2*E_Z**2 + 2*a4*E_theta*E_Z
        S_theta = c1*(a1*E_theta + a4*E_Z)*np.exp(Q) #Eq. (4)
        S_Z     = c1*(a4*E_theta + a2*E_Z)*np.exp(Q) #Eq. (4)
        S = np.zeros((lm.shape[0],3,3))
        S[:,0,0] = S_theta
        S[:,1,1] = S_Z
        return S
    
    def sigma(self, lm):
        F, _, _ = self.kinematics(lm)
        S = self.S(lm)
        sigma = np.einsum('...ij,...jk,...lk->...il', F, S, F)
        return sigma
        
    def Psi(self, lm):
        c1, a1, a2, a4 = self.c1, self.a1, self.a2, self.a4
        _, E_Z, E_theta = self.kinematics(lm)
        Q = a1*E_theta**2 + a2*E_Z**2 + 2*a4*E_theta*E_Z
        Psi = c1/2*(np.exp(Q)-1)
        return Psi

class Fung3D():
    n_params = 10
    #Source: Initially proposed by Chuong & Fung 1986 But they only suggested only the first 6 out of 9 terms.
    #Then E_ZTheta term was added by Deng et al. 1994
    #Explained in J. Humphrey 1995 Mechanics of Arterial Walls: Review and Directions
    def __init__(self, params):
        self.c = np.abs(params[0])
        self.b = params[1:]
    
    def kinematics(self, lm):
        c, b = self.c, self.b
        n = lm.shape[0]
        F = np.zeros([n,3,3])
        F[:,0,0] = lm[:,0]
        F[:,1,1] = lm[:,1]
        F[:,2,2] = 1/(lm[:,0]*lm[:,1])
        C = np.einsum('pji,pjk->pik',F,F)
        Cinv = np.linalg.inv(C)
        E = 0.5*(C-np.eye(3))
        
        Q = np.zeros(n)
        itojk = [[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]]
        for i in range(6):
            j,k = itojk[i]
            Q+= b[i]*E[:,j,k]**2
        Q+= b[6]*E[:,0,0]*E[:,1,1]
        Q+= b[7]*E[:,0,0]*E[:,2,2]
        Q+= b[8]*E[:,1,1]*E[:,2,2]
        return F, E, Cinv, Q
    
    def Psi(self, lm):
        c, b = self.c, self.b
        _, _, _, Q = self.kinematics(lm)
        
        Psi = c/2*(np.exp(Q)-1)
        return Psi
    
    def S(self, lm):
        c, b = self.c, self.b
        _, E, Cinv, Q = self.kinematics(lm)
        
        S = np.zeros([lm.shape[0],3,3])
        S[:,0,0] = c*np.exp(Q)*(b[0]*E[:,0,0] + b[6]*E[:,1,1] + b[7]*E[:,2,2])
        S[:,1,1] = c*np.exp(Q)*(b[1]*E[:,1,1] + b[6]*E[:,0,0] + b[8]*E[:,2,2])
        S[:,2,2] = c*np.exp(Q)*(b[2]*E[:,2,2] + b[7]*E[:,0,0] + b[8]*E[:,1,1])
        S[:,0,1] = c*np.exp(Q)* b[3]*E[:,0,1]
        S[:,0,2] = c*np.exp(Q)* b[4]*E[:,0,2]
        S[:,1,2] = c*np.exp(Q)* b[5]*E[:,1,2]
        S[:,1,0] = S[:,0,1]
        S[:,2,0] = S[:,0,2]
        S[:,2,1] = S[:,1,2]
        p = S[:,2,2]/Cinv[:,2,2]
        S = S - p[:, None, None]*Cinv
        return S
    
    def sigma(self, lm):
        F, _, _, _ = self.kinematics(lm)
        S = self.S(lm)
        sigma = np.einsum('...ij,...jk,...lk->...il', F, S, F)
        return sigma

class Ogden(): #probably incomplete...
    def __init__(self, N, params):
        self.N = N #number of branches
        self.mu = np.zeros(N)
        self.alpha = np.zeros(N)
        for i in range(N):
            self.mu[i] = params[i]
        for i in range(N):
            self.alpha[i] = params[N+i]
            
    def kinematics(self, lm):
        n = lm.shape[0]
        F = np.zeros([n,3,3])
        F[:,0,0] = lm[:,0]
        F[:,1,1] = lm[:,1]
        F[:,2,2] = 1/(lm[:,0]*lm[:,1])
        C = np.einsum('pji,pjk->pik',F,F)
        Cinv = np.linalg.inv(C)
        return F, Cinv
    
    def Psi(self, lm):
        N, mu, alpha = self.N, self.mu, self.alpha
        Psi = np.zeros(lm.shape[0])
        for i in range(N):
            Psi+= mu[i]/alpha[i]*(lm[0]**alpha[i] + lm[1]**alpha[i] + (lm[0]*lm[1])**(-alpha[i]) - 3)
        return Psi
    
    def S(self, lm):
        N, mu, alpha = self.N, self.mu, self.alpha
        #since we only have stretches, the eigenvalues of C will be N1=[1,0,0], N2=[0,1,0] ...
        N1, N2, N3 = [1,0,0], [0,1,0], [0,0,1]
        n = lm.shape[0]
        S = np.zeros([n,3,3])
        
        lm1, lm2 = lm
        lm3 = 1/(lm1*lm2)
        for i in range(N):
            S+= mu[i]*lm1**(alpha[i]-2)*np.outer(N1,N1) 
            S+= mu[i]*lm2**(alpha[i]-2)*np.outer(N2,N2) 
            S+= mu[i]*lm3**(alpha[i]-2)*np.outer(N3,N3)
        _, Cinv = self.kinematics(lm)
        
        p = S[:,2,2]/Cinv[:,2,2]
        S = S - p[:, None, None]*Cinv
        return S
    def sigma(self, lm):
        F, _, = self.kinematics(lm)
        S = self.S(lm)
        sigma = np.einsum('...ij,...jk,...lk->...il', F, S, F)
        return sigma
        
        
        
def lm2F(lm):
    n = np.size(lm,0)
    F = np.zeros([n,3,3])
    F[:,0,0] = lm[:,0]
    F[:,1,1] = lm[:,1]
    F[:,2,2] = 1/(lm[:,0]*lm[:,1])
    return F

# Work in progress:
def fit(model, lm, stress, initial_guess=[], stresstype='sigma'):
    from scipy import optimize
    #from autograd import jacobian
    class obj_fun():
        def __init__(self, model, lm, stress_gt, stresstype):
            self.lm = lm
            self.stress_gt = stress_gt
            self.model = model
            self.stresstype = stresstype
            
        def loss(self, params):
            lm  = self.lm
            n = lm.shape[0]
            stress_gt   = self.stress_gt
            model = self.model(params)

            if stresstype == 'sigma':
                stress_pr = model.sigma(lm)
            elif stresstype == 'P':
                stress_pr = model.P(lm)
            elif stresstype == 'S':
                stress_pr = model.S(lm)
            stress_pr = np.array([stress_pr[:,0,0], stress_pr[:,1,1]]).transpose()
            err = np.sum((stress_pr[:,0]-stress_gt[:,0])**2)
            err+= np.sum((stress_pr[:,1]-stress_gt[:,1])**2)
            err = err/n
            return err
        
    if len(initial_guess)==0:
        initial_guess = np.random.randn(model.n_params)
    fun = obj_fun(model, lm, stress, stresstype).loss
    #jac = jacobian(fun)
    opt = optimize.minimize(fun,initial_guess)
    return model(opt.x), opt.x


def Govindjee(F, C_i_inv, dt):
    #Material parameters for Vulcanized rubber (Ogden material for neq):
    mu_m = np.array([51.4, -18, 3.86])
    alpha_m = np.array([1.8, -2, 7])
    K_m = 10000
    tau = 17.5
    shear_mod = 1/2*(mu_m[0]*alpha_m[0] + mu_m[1]*alpha_m[1] + mu_m[2]*alpha_m[2])
    eta_D = tau*shear_mod
    eta_V = tau*K_m
    #Neo Hookean for eq parts:
    mu = shear_mod #same as mu for neq parts for simplicity.
    K = K_m

    be_trial = np.einsum('ij,jk,kl->il', F, C_i_inv, F.transpose())
    lamb_e_trial, n_A = np.linalg.eig(be_trial)
    lamb_e_trial = np.sqrt(np.real(lamb_e_trial))
    eps_e_trial = np.log(lamb_e_trial)
    
    #Initial guess for eps_e
    eps_e = eps_e_trial
    normres = 1
    iter = 0
    itermax = 20
    while normres>1.e-6 and iter < itermax:
        lamb_e = np.exp(eps_e)
        Je = lamb_e[0]*lamb_e[1]*lamb_e[2]
        bbar_e = Je**(-2/3)*lamb_e**2 #(54)
        #Calculate K_AB
        ddevtau_AdepsBe = np.zeros([3,3])
        for A in range(3):
            for B in range(3):
                if A==B:
                    for r in range(3):
                        oi = np.array([0,1,2]) #Other indices. Indices other than A=B
                        oi = np.delete(oi, A)
                        ddevtau_AdepsBe[A,B]+= mu_m[r]*alpha_m[r]*(4/9*bbar_e[A]**(alpha_m[r]/2) + 1/9*bbar_e[oi[0]]**(alpha_m[r]/2)
                                                                   + 1/9*bbar_e[oi[1]]**(alpha_m[r]/2)) #(B12)
                else:
                    for r in range(3):
                        oi = np.array([0,1,2]) #Other index. Index other than A or B.
                        oi = np.delete(oi, [A,B])
                        ddevtau_AdepsBe[A,B]+= mu_m[r]*alpha_m[r]*(-2/9*bbar_e[A]**(alpha_m[r]/2) - 2/9*bbar_e[B]**(alpha_m[r]/2)
                                                                   + 1/9*bbar_e[oi[0]]**(alpha_m[r]/2)) #(B13)
        
        
        K_AB = np.eye(3) + dt/2/eta_D*ddevtau_AdepsBe - dt/3/eta_V*K_m*Je**2*np.ones([3,3]) #(B15)
        K_AB_inv = np.linalg.inv(K_AB)

        devtau = np.zeros(3)
        for A in range(3):
            for r in range(3):
                oi = np.array([0,1,2])
                oi = np.delete(oi, A)
                devtau[A]+= mu_m[r]*(2/3*bbar_e[A]**(alpha_m[r]/2) - 1/3*bbar_e[oi[0]]**(alpha_m[r]/2) - 1/3*bbar_e[oi[1]]**(alpha_m[r]/2)) #(B8)
        tau_NEQdyadicI = 3/2*K_m*(Je**2-1) #(B8)

        res = eps_e + dt*(1/2/eta_D*devtau + 1/9/eta_V*tau_NEQdyadicI*np.ones(3))-eps_e_trial #(60), res=r in the paper
        deps_e = np.einsum('ij,j->i', K_AB_inv, -res)
        eps_e = eps_e + deps_e
        normres = np.linalg.norm(res)
        iter+= 1
    if iter==itermax:
        print('no local convergence')
        print(res, normres)
    #Now we have the value of eps_e
    #Calculate tau_NEQ
    lamb_e = np.exp(eps_e)
    Je = lamb_e[0]*lamb_e[1]*lamb_e[2]
    bbar_e = Je**(-2/3)*lamb_e**2
    devtau = np.zeros(3)
    for A in range(3):
        for r in range(3):
            oi = np.array([0,1,2])
            oi = np.delete(oi, A)
            devtau[A]+= mu_m[r]*(2/3*bbar_e[A]**(alpha_m[r]/2) - 1/3*bbar_e[oi[0]]**(alpha_m[r]/2) - 1/3*bbar_e[oi[1]]**(alpha_m[r]/2)) #(B8)
    tau_NEQdyadicI = 3*K_m/2*(Je**2-1) #(B8)
    tau_A = devtau + 1/3*tau_NEQdyadicI #(B8)
    tau_NEQ = np.einsum('i,ji,ki->jk', tau_A, n_A, n_A) #(58)
    tr_tau = np.trace(tau_NEQ)
    
    be = np.einsum('i,ji,ki->jk', lamb_e**2, n_A, n_A)
    F_inv = np.linalg.inv(F)
    C_i_inv_new = np.einsum('ij,jk,kl->il', F_inv, be, F_inv.transpose())
    
    J = np.linalg.det(F)
    b = np.einsum('ij,kj->ik', F, F)
    sigma_EQ = mu/J*(b-np.eye(3)) + 2*K*(J-1)*np.eye(3) #neo Hookean material
    sigma = 1/Je*tau_NEQ + sigma_EQ #(7)
    return sigma, C_i_inv_new, lamb_e
