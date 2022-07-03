c...  ------------------------------------------------------------------
      subroutine sdvini(statev,coords,nstatv,ncrds,noel,npt,layer,kspt)
c...  ------------------------------------------------------------------
      include 'aba_param.inc'


      dimension statev(nstatv)

      statev(1)=1.0d0    

      return
      end



c...  ------------------------------------------------------------------
      subroutine umat(stress,statev,ddsdde,sse,spd,scd,
     #rpl,ddsddt,drplde,drpldt,
     #stran,dstran,time,dtime,temp,dtemp,predef,dpred,cmname,
     #ndi,nshr,ntens,nstatv,props,nprops,coords,drot,pnewdt,
     #celent,dfgrd0,dfgrd1,noel,npt,layer,kspt,kstep,kinc)
c...  ------------------------------------------------------------------
      include 'aba_param.inc'

      character*80 cmname
      dimension stress(ntens),statev(nstatv),
     #ddsdde(ntens,ntens),ddsddt(ntens),drplde(ntens),
     #stran(ntens),dstran(ntens),time(2),predef(1),dpred(1),
     #props(nprops),coords(3),drot(3,3),dfgrd0(3,3),dfgrd1(3,3)

      call umat_NN(stress,statev,ddsdde,sse,
     #                       time,dtime,coords,props,dfgrd1,
     #                       ntens,ndi,nshr,nstatv,nprops,
     #                       noel,npt,kstep,kinc)

      return
      end

c...  ------------------------------------------------------------------
      subroutine umat_NN(stress,statev,ddsdde,sse,
     #                             time,dtime,coords,props,dfgrd1,
     #                             ntens,ndi,nshr,nstatv,nprops,
     #                             noel,npt,kstep,kinc)
c...  ------------------------------------------------------------------

c...  ------------------------------------------------------------------

      implicit none

c...  variables to be defined
      real*8  stress(ntens), ddsdde(ntens,ntens), statev(nstatv), sse

c...  variables passed in for information
      real*8  time(2), dtime, coords(3), props(nprops), dfgrd1(3,3)
      integer ntens, ndi, nshr, nstatv, nprops, noel, npt, kstep, kinc

c...  local variables (mostly mechanics part)
      real*8  finv(3,3), detf, b(6),lnJ
      real*8 Fiso(3,3), biso(6), biso2(6), sigmabar(6), sigmaiso(6)
      real*8 bisomat(3,3), kronmat(3,3), sigmamat(3,3), bb
      real*8 sigma(6), p, Psivol, dpdJ
      real*8 I1bar, I2bar, trb2
      real*8 Psi1, Psi2, Psi11, Psi12, Psi21, Psi22, gamma1, gamma2
      real*8 delta1, delta2, delta3, delta4, delta5, delta6, delta7
      real*8 tr_sigmabar, Jccbar(6,6), kron(6), IIII 
      real*8 cciso(6,6), ccvol(6,6)

c...  some auxiliar variables, tensors 
      integer i, j, k, l, nitl, II, JJ, Itoi(6), Itoj(6)

c...  material properties, read in the weights and biases 
      integer weight_count, ind, bias_count, n_input, nlayers 
c...  values needed for normalization of I1, I2, in our approach
      real*8 I1mean,I1var,I2mean,I2var
c...  Bulk modulus
      real*8 Kvol
c...  nlayers = props(1)
c...  n_input = props(2)
      integer n_neuronsperlayer(props(1)) 
      real*8 input_param(props(2))
c...  weight_count = props(3)
c...  bias_count = props(4)
      real*8 ALLweights(props(3)), ALLbiases(props(4)) 
      integer activtypes(props(1)-1)
      real*8 activout, gradactivout
c...      real*8 output_vector(bias_count+n_input+2)
c...      real*8 output_gradient((n_input+2)*(n_input+2+bias_count))
      real*8 output_vector(props(4)+props(2)+2)
      real*8 output_grad((props(2)+2)*(props(2)+2+props(4)) )
      integer io1, iw1, ig1, io2, iw2, ig2, ib1


c...  initialize material parameters
      nlayers   = props(1)    ! number of layers of the NN, including input
c...      print *, 'neurons per layer'
      do i = 1,nlayers 
        n_neuronsperlayer(i)    = props(i+4)
c...        print *, n_neuronsperlayer(i)
      end do
c...  read the normalizing constants for I1, I2
      I1mean = props(4+nlayers+1)
      I1var = props(4+nlayers+2)
      I2mean = props(4+nlayers+3)
      I2var = props(4+nlayers+4)
c...  Bulk modulus 
      Kvol = props(4+nlayers+4+1)
c...  read the value of the other inputs that are not I1bar, I2bar
      n_input = props(2)
c...      print *,'inputs: ',n_input
      do i = 1,n_input
        input_param(i) = props(i+4+nlayers+4+1)
c...       print *, input_param(i)
      end do 

c...  read in weights and biases
      weight_count = props(3)
      bias_count = props(4)
      ind = 1
c...      print *,'weights'
      do i=1,nlayers-1
c...       print *, 'layer',i
        do j=1,n_neuronsperlayer(i)
          do k=1,n_neuronsperlayer(i+1)
            ALLweights(ind) = props(ind+nlayers+n_input+4+4+1)
c...            print *, ALLweights(ind)
            ind = ind+1
          end do
        end do
      end do
      ind = 1
c...      print *, 'biases' 
      do i=2,nlayers
c...        print *, 'layer', i
        do j=1,n_neuronsperlayer(i)
          ALLbiases(ind) = props(ind+weight_count+nlayers+n_input+4+4+1)
c...          print *,ALLbiases(ind)
          ind = ind + 1
        end do
      end do
      ind = 1
c...      print *,'activation types'
      do i=2,nlayers
c...        print *, 'layer', i
        activtypes(ind)=int(props(ind+bias_count+weight_count+nlayers+n_input+4+4+1))
c...        print *, activtypes(ind)
        ind = ind+1
      end do

c...      print *,'finished reading'

c...      print *, 'deformation gradient'
c...      print *, dfgrd1(1,1),dfgrd1(1,2),dfgrd1(1,3)
c...      print *, dfgrd1(2,1),dfgrd1(2,2),dfgrd1(2,3)
c...      print *, dfgrd1(3,1),dfgrd1(3,2),dfgrd1(3,3)
c...  calculate determinant of deformation gradient
      detf = +dfgrd1(1,1)*(dfgrd1(2,2)*dfgrd1(3,3)-dfgrd1(2,3)*dfgrd1(3,2))
     #       -dfgrd1(1,2)*(dfgrd1(2,1)*dfgrd1(3,3)-dfgrd1(2,3)*dfgrd1(3,1))
     #       +dfgrd1(1,3)*(dfgrd1(2,1)*dfgrd1(3,2)-dfgrd1(2,2)*dfgrd1(3,1))
c...      print *, 'detF'
c...      print *, detf

c...  calculate inverse of F
      finv(1,1) = (+dfgrd1(2,2)*dfgrd1(3,3) - dfgrd1(2,3)*dfgrd1(3,2))/detf
      finv(1,2) = (-dfgrd1(1,2)*dfgrd1(3,3) + dfgrd1(1,3)*dfgrd1(3,2))/detf
      finv(1,3) = (+dfgrd1(1,2)*dfgrd1(2,3) - dfgrd1(1,3)*dfgrd1(2,2))/detf
      finv(2,1) = (-dfgrd1(2,1)*dfgrd1(3,3) + dfgrd1(2,3)*dfgrd1(3,1))/detf
      finv(2,2) = (+dfgrd1(1,1)*dfgrd1(3,3) - dfgrd1(1,3)*dfgrd1(3,1))/detf
      finv(2,3) = (-dfgrd1(1,1)*dfgrd1(2,3) + dfgrd1(1,3)*dfgrd1(2,1))/detf
      finv(3,1) = (+dfgrd1(2,1)*dfgrd1(3,2) - dfgrd1(2,2)*dfgrd1(3,1))/detf
      finv(3,2) = (-dfgrd1(1,1)*dfgrd1(3,2) + dfgrd1(1,2)*dfgrd1(3,1))/detf
      finv(3,3) = (+dfgrd1(1,1)*dfgrd1(2,2) - dfgrd1(1,2)*dfgrd1(2,1))/detf


c...      C = F^T*F
c...      b = F*F^T -> full notation [[b11, b12, b13],[b12,b22,b23],[b13,b23,b33]]
c...      b = [b11,b22,b33,b12,b13,b23] -> voigt notation

c...  calculate left cauchy-green deformation tensor b = f * f^t
      b(1) = dfgrd1(1,1)*dfgrd1(1,1) + dfgrd1(1,2)*dfgrd1(1,2) + dfgrd1(1,3)*dfgrd1(1,3)
      b(2) = dfgrd1(2,1)*dfgrd1(2,1) + dfgrd1(2,2)*dfgrd1(2,2) + dfgrd1(2,3)*dfgrd1(2,3)
      b(3) = dfgrd1(3,1)*dfgrd1(3,1) + dfgrd1(3,2)*dfgrd1(3,2) + dfgrd1(3,3)*dfgrd1(3,3)
      b(4) = dfgrd1(1,1)*dfgrd1(2,1) + dfgrd1(1,2)*dfgrd1(2,2) + dfgrd1(1,3)*dfgrd1(2,3)
      b(5) = dfgrd1(1,1)*dfgrd1(3,1) + dfgrd1(1,2)*dfgrd1(3,2) + dfgrd1(1,3)*dfgrd1(3,3)
      b(6) = dfgrd1(2,1)*dfgrd1(3,1) + dfgrd1(2,2)*dfgrd1(3,2) + dfgrd1(2,3)*dfgrd1(3,3)
c...      print *, 'b'
c...      print *, b(1), b(2), b(3), b(4), b(5), b(6)

c...  get the isochoric split 
      Fiso(1,1) = detf**(-1./3.)*dfgrd1(1,1) 
      Fiso(1,2) = detf**(-1./3.)*dfgrd1(1,2) 
      Fiso(1,3) = detf**(-1./3.)*dfgrd1(1,3) 
      Fiso(2,1) = detf**(-1./3.)*dfgrd1(2,1) 
      Fiso(2,2) = detf**(-1./3.)*dfgrd1(2,2) 
      Fiso(2,3) = detf**(-1./3.)*dfgrd1(2,3) 
      Fiso(3,1) = detf**(-1./3.)*dfgrd1(3,1) 
      Fiso(3,2) = detf**(-1./3.)*dfgrd1(3,2) 
      Fiso(3,3) = detf**(-1./3.)*dfgrd1(3,3)

c... get the isochoric b or b_bar, you can also get Cbar
      biso(1) = Fiso(1,1)*Fiso(1,1) + Fiso(1,2)*Fiso(1,2) + Fiso(1,3)*Fiso(1,3)
      biso(2) = Fiso(2,1)*Fiso(2,1) + Fiso(2,2)*Fiso(2,2) + Fiso(2,3)*Fiso(2,3)
      biso(3) = Fiso(3,1)*Fiso(3,1) + Fiso(3,2)*Fiso(3,2) + Fiso(3,3)*Fiso(3,3)
      biso(4) = Fiso(1,1)*Fiso(2,1) + Fiso(1,2)*Fiso(2,2) + Fiso(1,3)*Fiso(2,3)
      biso(5) = Fiso(1,1)*Fiso(3,1) + Fiso(1,2)*Fiso(3,2) + Fiso(1,3)*Fiso(3,3)
      biso(6) = Fiso(2,1)*Fiso(3,1) + Fiso(2,2)*Fiso(3,2) + Fiso(2,3)*Fiso(3,3)
      bisomat(1,1) = biso(1)
      bisomat(2,2) = biso(2)
      bisomat(3,3) = biso(3)
      bisomat(1,2) = biso(4)
      bisomat(2,1) = biso(4)
      bisomat(1,3) = biso(5)
      bisomat(3,1) = biso(5)
      bisomat(2,3) = biso(6)
      bisomat(3,2) = biso(6)
c...      print *, 'biso' 
c...      print *, biso(1), biso(2), biso(3), biso(4), biso(5), biso(6)

c... get biso^2 (you could also get C^2)
      biso2(1) = biso(1)**2 + biso(4)**2 + biso(5)**2
      biso2(2) = biso(4)**2 + biso(2)**2 + biso(6)**2
      biso2(3) = biso(5)**2 + biso(6)**2 + biso(3)**2
      biso2(4) = biso(1)*biso(4) + biso(4)*biso(2) + biso(5)*biso(6)
      biso2(5) = biso(1)*biso(5) + biso(4)*biso(6) + biso(5)*biso(3)
      biso2(6) = biso(4)*biso(5) + biso(2)*biso(6) + biso(6)*biso(3)

c... get the invariants of biso or b_bar (are the same as invariants of Cbar)
      I1bar = biso(1)+biso(2)+biso(3)
      trb2 = biso2(1)+biso2(2)+biso2(3)
      bb = biso(1)*biso(1)+biso(2)*biso(2)+biso(3)*biso(3)+
     #     2*(biso(4)*biso(4)+biso(5)*biso(5)+biso(6)*biso(6))
      I2bar = 0.5*(I1bar*I1bar-trb2)
c      print *, 'I1bar', I1bar
c      print *, 'I2bar', I2bar

c...  evaluate the NN and derivatives 
c...  fill out the input vector
      output_vector = 0
      output_grad = 0 
c      print *, 'output grad'
      do i= 1,n_input
        output_vector(i) = input_param(i)
c...  fill out the first jacobian is just the identity 
        output_grad((i-1)*(n_input+2)+i) = 1
      end do
c...  CAREFUL!!!!!!! 
c...  Normalizing the input 
c...  If your NN was trained with I1, I2 directly then you can pass 0 for the mean and 1 for the variance 
      output_vector(n_input+1) = (I1bar-I1mean)/I1var 
      output_vector(n_input+2) = (I2bar-I2mean)/I2var
      output_grad(n_input*(n_input+2)+n_input+1) = 1
      output_grad((n_input+1)*(n_input+2)+n_input+2) = 1
c...  print output grad to debug 
      io1 = 0
      iw1 = 0
      ig1 = 0
      ib1 = 0
      do i =1,nlayers-1
c        print *, 'Jacobian for layer ',i
c...    Beginning and end of the chunk in output vector to be used as input
        io2 = io1 + n_neuronsperlayer(i) 
c...    Beginning and end of the chunk in weight array defining matrix 
        iw2 = iw1 + n_neuronsperlayer(i)*n_neuronsperlayer(i+1) 
c...    Beginning and end of the chunk for grad outputs 
        ig2 = ig1 + n_neuronsperlayer(i)*(n_input+2)
c...    do the matrix vector product and store in output chunk 
        do k = 1,n_neuronsperlayer(i+1)
          do j = 1,n_neuronsperlayer(i) 
c...        Matrix*vector + bias 
            output_vector(io2+k) = output_vector(io2+k) + ALLweights(iw1+(j-1)* 
     #                n_neuronsperlayer(i+1)+k)*output_vector(io1+j)
c...        Matrix*Matrix for jacobian 
            do l = 1,n_input+2
              output_grad(ig2+(k-1)*(n_input+2)+l) = output_grad(ig2+(k-1)*(n_input+2)+l ) + 
     #                ALLweights(iw1+(j-1)*n_neuronsperlayer(i+1)+k)
     #                *output_grad(ig1+(j-1)*(n_input+2)+l)
            end do
          end do
          activout = output_vector(io2+k) + ALLbiases(ib1+k)
          gradactivout = activout
          call activation(activout,activtypes(i))  
          output_vector(io2+k) = activout
          call grad_activation(gradactivout,activtypes(i))
          do l = 1,n_input+2
            output_grad(ig2+(k-1)*(n_input+2)+l) = gradactivout*output_grad(ig2+(k-1)*(n_input+2)+l)
c            print *, output_grad(ig2+(k-1)*(n_input+2)+l)
          end do
        end do
        io1 = io2 
        iw1 = iw2
        ig1 = ig2
        ib1 = ib1 + n_neuronsperlayer(i+1)
      end do

c... here we should have first derivatives Psi1, Psi2
      Psi1 = output_vector(bias_count+n_input+1)
      Psi2 = output_vector(bias_count+n_input+2)
c... and second derivatives Psi11,Psi12,Psi21,Psi22
      Psi11 = output_grad((n_input+0+bias_count)*(n_input+2)+n_input+1)
      Psi12 = output_grad((n_input+0+bias_count)*(n_input+2)+n_input+2)
      Psi21 = output_grad((n_input+1+bias_count)*(n_input+2)+n_input+1)
      Psi22 = output_grad((n_input+1+bias_count)*(n_input+2)+n_input+2)
c      print *, 'Psi1', Psi1
c      print *, 'Psi2', Psi2
c      print *, 'Psi11', Psi11
c      print *, 'Psi12', Psi12
c      print *, 'Psi21', Psi21
c      print *, 'Psi22', Psi22
 

c...  Ficticious stress sigma_bar, holzapfel page 235
      gamma1 = 2.0*(Psi1+I1bar*Psi2)
      gamma2 = -2.0*Psi2
      sigmabar(1) = (1.0/detf)*(gamma1*biso(1) +gamma2*biso2(1))
      sigmabar(2) = (1.0/detf)*(gamma1*biso(2) +gamma2*biso2(2))
      sigmabar(3) = (1.0/detf)*(gamma1*biso(3) +gamma2*biso2(3))
      sigmabar(4) = (1.0/detf)*(gamma1*biso(4) +gamma2*biso2(4))
      sigmabar(5) = (1.0/detf)*(gamma1*biso(5) +gamma2*biso2(5))
      sigmabar(6) = (1.0/detf)*(gamma1*biso(6) +gamma2*biso2(6))

c...  sigmaiso = Projection::sigmabar, super simple in eulerian 
      tr_sigmabar = sigmabar(1)+sigmabar(2)+sigmabar(3)
      sigmaiso(1) = sigmabar(1) -(1./3.)*tr_sigmabar
      sigmaiso(2) = sigmabar(2) -(1./3.)*tr_sigmabar
      sigmaiso(3) = sigmabar(3) -(1./3.)*tr_sigmabar
      sigmaiso(4) = sigmabar(4)
      sigmaiso(5) = sigmabar(5)
      sigmaiso(6) = sigmabar(6)
c...      print *, 'sigma_iso'
c...      print *, sigmaiso(1), sigmaiso(2), sigmaiso(3), sigmaiso(4), sigmaiso(5), sigmaiso(6)

c...  volumetric part (we can change this later)
c...  sigmavol = J*p*Identity, with p = dPsivol/dJ
      Psivol = Kvol*(detf-1)**2
      p = 2*Kvol*(detf-1)
      dpdJ = 2*Kvol
      sigma(1) = sigmaiso(1) + p
      sigma(2) = sigmaiso(2) + p
      sigma(3) = sigmaiso(3) + p
      sigma(4) = sigmaiso(4)
      sigma(5) = sigmaiso(5)
      sigma(6) = sigmaiso(6) 
      sigmamat(1,1) = sigma(1)
      sigmamat(2,2) = sigma(2)
      sigmamat(3,3) = sigma(3)
      sigmamat(1,2) = sigma(4)
      sigmamat(2,1) = sigma(4)
      sigmamat(1,3) = sigma(5)
      sigmamat(3,1) = sigma(5)
      sigmamat(2,3) = sigma(6)
      sigmamat(3,2) = sigma(6)
c...     print *, 'sigma vol, i.e. p' 
c...     print *, p

c...  tangent in voigt notation (see holland abaqus documentation)
c...  The NN computes the two derivatives Psi12 and Psi21 independently, there
c...  is no guarantee that they are the same... this should be enforced during NN training
c...  Psi12 with the average of the two 
      Psi12 = 0.5*(Psi12+Psi21)
      delta1 = 4*(Psi2+Psi11+2*(I1bar*Psi12)+I1bar*I1bar*Psi22)
      delta2 = -4*(Psi12+I1bar*Psi22)
      delta3 = 4*Psi22
      delta4 = -4*Psi2
      delta5 = -1.0/3.0*I1bar*delta1-1.0/3.0*trb2*delta2
      delta6 = -1.0/3.0*I1bar*delta2-1.0/3.0*trb2*delta3-1.0/3.0*delta4
      delta7 = (1.0/9.0)*(I1bar*I1bar*delta1+2*I1bar*trb2*delta2+trb2*trb2*delta3+bb*delta4)
c...  Need the kron delta in voigt 
      kron(1) = 1.0
      kron(2) = 1.0
      kron(3) = 1.0
      kron(4) = 0.0
      kron(5) = 0.0
      kron(6) = 0.0
      kronmat(1,1) = 1.0
      kronmat(2,2) = 1.0
      kronmat(3,3) = 1.0
      kronmat(1,2) = 0.0
      kronmat(2,1) = 0.0
      kronmat(1,3) = 0.0
      kronmat(3,1) = 0.0
      kronmat(2,3) = 0.0
      kronmat(3,2) = 0.0
      Itoi(1) = 1
      Itoi(2) = 2
      Itoi(3) = 3
      Itoi(4) = 1
      Itoi(5) = 1
      Itoi(6) = 2
      Itoj(1) = 1
      Itoj(2) = 2
      Itoj(3) = 3
      Itoj(4) = 2
      Itoj(5) = 3
      Itoj(6) = 3

c...  Fill part of the tangent in voigt notation
c...  1->11, 2->22, 3->33, 4->12, 5->13, 6->23
c...  Itoi = [1,2,3,1,1,2] (definition above)
c...  Itoj = [1,2,3,2,3,3] (definition above)
      do II=1,6
        stress(II) = sigma(II)
        do JJ=II,6
          Jccbar(II,JJ) = delta1*(biso(II)*biso(JJ)-(I1bar/3.0)*biso(II)*kron(JJ) 
     #      -(I1bar/3.0)*kron(II)*biso(JJ)+(I1bar*I1bar/9.0)*kron(II)*kron(JJ))
     #      +delta2*(biso(II)*biso2(JJ)+biso2(II)*biso(JJ)-1.0/3.0*(trb2*biso(II)*kron(JJ)
     #      +I1bar*biso2(II)*kron(JJ))-1.0/3.0*(I1bar*kron(II)*biso2(JJ)+trb2*kron(II)*biso2(JJ))
     #      +2.0/9.0*I1bar*trb2*kron(II)*kron(JJ))
     #      +delta3*(biso2(II)*biso2(JJ)-1.0/3.0*trb2*(biso2(II)*kron(JJ)+kron(II)*biso2(JJ))
     #      +1.0/9.0*trb2*trb2*kron(II)*kron(JJ))
     #      +delta4*(-1.0/3.0*(biso2(II)*kron(JJ)+kron(II)*biso2(JJ))+1.0/9.0*bb*kron(II)*kron(JJ))
c...   There is ont term in the tangent which requires special tensor product
          i = Itoi(II)
          j = Itoj(II)
          k = Itoi(JJ)
          l = Itoj(JJ)
          IIII = 0.5*(kronmat(i,k)*kronmat(j,l)+kronmat(i,l)*kronmat(j,k))
          Jccbar(II,JJ) = Jccbar(II,JJ) + delta4*(bisomat(i,k)*bisomat(j,l))
          cciso(II,JJ) = (1.0/detf)*Jccbar(II,JJ) 
     #       +(2.0/3.0)*tr_sigmabar*(IIII-(1.0/3.0)*kron(II)*kron(JJ))
     #       -(2.0/3.0)*(kron(II)*sigmaiso(JJ)+sigmaiso(II)*kron(JJ))
          ccvol(II,JJ) = (p +detf*dpdJ)*kron(II)*kron(JJ) -2.0*p*IIII
c...  Abaqus corrections 
          ddsdde(II,JJ) = cciso(II,JJ)+ccvol(II,JJ)+0.5*(kronmat(i,k)*sigmamat(j,l)
     #                    +kronmat(i,l)*sigmamat(j,k)+kronmat(j,k)*sigmamat(i,l)
     #                    +kronmat(j,l)*sigmamat(i,k))
          if (JJ>II) then
            cciso(JJ,II) = cciso(II,JJ)
            ccvol(JJ,II) = ccvol(II,JJ)
            ddsdde(JJ,II) = ddsdde(II,JJ)
          end if
        end do
      end do
c...      print *, 'STRESS'
c...      print *, stress(1), stress(2), stress(3), stress(4), stress(5), stress(6)
c...      print *, 'DDSDDE' 
c...      print *, ddsdde(1,1), ddsdde(1,2), ddsdde(1,3), ddsdde(1,4), ddsdde(1,5), ddsdde(1,6)
c...      print *, ddsdde(2,1), ddsdde(2,2), ddsdde(2,3), ddsdde(2,4), ddsdde(2,5), ddsdde(2,6)
c...      print *, ddsdde(3,1), ddsdde(3,2), ddsdde(3,3), ddsdde(3,4), ddsdde(3,5), ddsdde(3,6)
c...      print *, ddsdde(4,1), ddsdde(4,2), ddsdde(4,3), ddsdde(4,4), ddsdde(4,5), ddsdde(4,6)
c...      print *, ddsdde(5,1), ddsdde(5,2), ddsdde(5,3), ddsdde(5,4), ddsdde(5,5), ddsdde(5,6)
c...      print *, ddsdde(6,1), ddsdde(6,2), ddsdde(6,3), ddsdde(6,4), ddsdde(6,5), ddsdde(6,6)
c...  calculate strain energy
      sse = Psivol

      return
      end

c...  ------------------------------------------------------------------

      subroutine cross(aa, bb,cc)
      implicit none

      real*8 :: cc(3)
      real*8 :: aa(3), bb(3)

      cc(1) = aa(2) * bb(3) - aa(3) * bb(2)
      cc(2) = aa(3) * bb(1) - aa(1) * bb(3)
      cc(3) = aa(1) * bb(2) - aa(2) * bb(1)

      return
      end
c...  ------------------------------------------------------------------

      subroutine activation(value, typea)
      implicit none

      real*8 value
      integer typea

      if (typea==0) then
        if (value<0) then
          value = 0
        end if
      end if

      return
      end
c...  ------------------------------------------------------------------

      subroutine grad_activation(value, typea)
      implicit none

      real*8 value
      integer typea

      if (typea==0) then
        if (value<1e-9) then
          value = 0
        else
          value = 1
        end if
      else
        value = 1
      end if

      return
      end

c...  ------------------------------------------------------------------


c...  ------------------------------------------------------------------
      end
c...  ------------------------------------------------------------------
