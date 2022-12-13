from backward_step_mod import physical_grid, pressure_grid, horizontal_vel, vert_vel, F_x, G_y, H_x, H_y, inflow, outflow, Get_A, visualize_grid
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
from scipy import sparse
from scipy.sparse import linalg
import sys
np.set_printoptions(threshold=sys.maxsize)

H = 1; rho = 1; Q = 2/3
f = 8; Ny = 32; Nx = f*Ny
Ly = 2*H; Lx=f*Ly
Re = 200; nu = (2*Q)/Re


X,Y,h=physical_grid(Lx, Ly, Nx, Ny)
uData=horizontal_vel(h, Lx, Ly, Nx, Ny)
vData= vert_vel(h, Lx, Ly, Nx, Ny)

NiterMax = int(1e6)

#visualize_grid(X,Y, uData, vData, pData, hxData)

#Initializing arrays of zeros for pressure, flux, velocity
U = np.zeros((Nx+3,Ny+2))
V = np.zeros((Nx+2,Ny+3))


A = Get_A(Nx, Ny, h)
L1 = np.zeros(NiterMax)
#======================================#
#Step 1: Initialize u v, get para. prof#
#======================================#

p1,y1 = inflow(uData, Ny, h, H) #Inflow Horizontal Velocity Points
p2,y2=outflow(uData, Ny, h, H)  #Outflow Horizontal Velocity Points
h2=h**2
beta=1.5
Umax = np.max(p1)                #Stability Factor
dt = beta*np.min([0.25*h2/nu, 4*nu/Umax**2])  #Determining time step

n=0                                             #Setting n = 0
while n<NiterMax:
# ===========================================#
#  Step 2: Boundary & Ghost Cell Allocation  #
# ===========================================#
    # Inflow Initial Horizontal Velocity
    U[1, int(Ny/2)+1:Ny+1] = p1[:]                      #Inflow points
    U[1, 1:int(Ny/2)+1] = 0                             #Wall points. u = 0
    U[0, int(Ny/2)+1:Ny+1] = p1[:]                      #Ghost Cell @ inflow
    U[0, 1:int(Ny/2)+1]= U[2, 1:int(Ny/2)+1]            #Ghost Cell @ Wall

    # Outflow Initial Horizontal Velocity
    U[-2, 1:Ny+1] = p2[:]                               #Outflow points
    U[-1, :] = U[-2, :]                                 #Ghost Cell Value. Enforced by fully-dev. laminar flow
    # Bottom Wall
    U[:, 0] = -U[:, 1]                                  #Bottom Wall BC.
    # Top Wall
    U[:, -1] = -U[:, -2]                                #Top Wall BC.

    # Inflow Initial Vert. Velocity
    V[0, int(Ny/2) + 1:Ny+2] = 0                        #Zero at fully-dev inflow
    V[0, 1:int(Ny/2) + 1] = -V[1, 1:int(Ny / 2) + 1]    #Ghost Cell Value: Vout = -Vin

    # Outflow Initial Vert Velocity
    V[-1, 1:Ny + 2] = 0                                 #Ghost cells = zero on RHS, fully-developed

    # Top Wall Vert. Velocity
    V[:, -2] = 0                                        #Zero at wall, no through flow
    V[:, -1] = V[:, -3]                                 #Ghost points: Vout = Vin

    # Bottom Wall Vert. Velocity
    V[:, 1] = 0                                         #Zero at Wall
    V[:, 0] = V[:, 2]                                   #Ghost point: Vout = Vin

#======================================#
#       Step 3: Computing Flux         #
#======================================#

    #Transport of U in x-direction
    F = np.zeros((Nx, Ny))
    q = np.zeros((Nx,Ny))
    phi = np.zeros((Nx,Ny))
    for i in range(Nx):
      for j in range(Ny):
            q[i,j]=(U[i+1,j+1]+U[i+2,j+1])/2
            if q[i,j] > 0:
                phi[i,j]=(3*U[i+2,j+1]+6*U[i+1,j+1]-U[i,j+1])/8
            else:
                phi[i, j]= (3*U[i+1,j+1]+6*U[i+2,j+1]-U[i+3,j+1])/8
    for i in range(Nx):
        for j in range(Ny):
            F[i,j]=q[i,j]*phi[i,j]-(nu/h)*(U[i+2,j+1]-U[i+1,j+1])

    #Transport of v in y-direction
    G = np.zeros((Nx, Ny))
    q = np.zeros((Nx,Ny))
    phi = np.zeros((Nx,Ny))
    for i in range(Nx):
      for j in range(Ny):
            q[i,j]=(V[i+1,j+2]+V[i+1,j+1])/2
            if q[i,j] > 0:
                phi[i,j]=(3*V[i+1,j+2]+6*V[i+1,j+1]-V[i+1,j])/8
            else:
                phi[i,j]= (3*V[i+1,j+1]+6*V[i+1,j+2]-V[i+1,j+3])/8
    for i in range(Nx):
        for j in range(Ny):
            G[i,j]=q[i,j]*phi[i,j]-(nu/h)*(V[i+1,j+2]-V[i+1,j+1])

    #Transport of u in y-direction
    Hx=np.zeros((Nx+1, Ny+1))
    q = np.zeros((Nx+1,Ny+1))
    phi = np.zeros((Nx+1,Ny+1))
    for i in range(1,Nx):               #Not evaluating on boundary
      for j in range(1,Ny):
            q[i,j]=(V[i,j+1]+V[i+1,j+1])/2
            if q[i,j] > 0:
                phi[i,j]=(3*U[i+1,j+1]+6*U[i+1,j]-U[i+1,j-1])/8
            else:
                phi[i, j]= (3*U[i+1,j]+6*U[i+1,j+1]-U[i+1,j+2])/8
    for i in range(Nx+1):
        for j in range(Ny+1):
            Hx[i,j]=q[i,j]*phi[i,j]-(nu/h)*(U[i+1,j+1]-U[i+1,j])

    #Transport of v in x-direction
    Hy=np.zeros((Nx+1, Ny+1))
    q = np.zeros((Nx+1,Ny+1))
    phi = np.zeros((Nx+1,Ny+1))
    for i in range(1,Nx):               #Not evaluating on boundary
      for j in range(1,Ny):
            q[i,j]=(U[i+1,j]+U[i+1,j+1])/2
            if q[i,j] > 0:
               phi[i,j]=(3*V[i+1,j+1]+6*V[i,j+1]-V[i-1,j+1])/8
            else:
                phi[i, j]= (3*V[i,j+1]+6*V[i+1,j+1]-V[i+2,j+1])/8
    for i in range(Nx+1):
        for j in range(Ny+1):
            Hy[i,j]=q[i,j]*phi[i,j]-(nu/h)*(V[i+1,j+1]-V[i,j+1])
#======================================#
#   Step 4: Fractional Step Update     #
#======================================#

    #Intermediate Horizontal Velocity Update
    U_int = U.copy(); V_int = V.copy()
    for i in range(Nx-1):
        for j in range(Ny):
            U_int[i+2,j+1] = U[i+2,j+1]-(dt/h)*((F[i+1,j]-F[i,j])+(Hx[i+1,j+1]-Hx[i+1,j]))

    #Intermediate Vertical Velocity Update
    for i in range(Nx):
        for j in range(Ny-1):
            V_int[i+1,j+2]= V[i+1,j+2]-(dt/h)*((Hy[i+1,j+1]-Hy[i,j+1])+(G[i,j+1]-G[i,j]))

#======================================#
#   Step 5: Solve Pressure Poisson     #
#======================================#

    #PPE Formulation
    P=np.zeros((Nx,Ny))
    del_v = np.zeros((Nx,Ny))
    for i in range(Nx):
        for j in range(Ny):
            del_v[i,j]=(1/dt)*((U_int[i+2,j+1]-U_int[i+1,j+1])/h+(V_int[i+1,j+2]-V_int[i+1,j+1])/h)

    b = np.reshape(del_v,((Nx*Ny),1), order ='F')
    b[-1]=1

    Pv = linalg.spsolve(A, b)  # solution at all points
    P = np.reshape(Pv, (Nx, Ny), order='F')  # reshape into matrix
#======================================#
#   Step 6: Correct velocity field     #
#======================================#
    #U_new = np.zeros_like(U); V_new = np.zeros_like(V)
    U_new=U_int.copy(); V_new=V_int.copy()
    #Horizontal Velocity Update
    for i in range(Nx-1):
        for j in range(Ny):
            U_new[i+2,j+1]=U_int[i+2,j+1]-(dt/h)*(P[i+1,j]-P[i,j])

    for i in range(Nx):
        for j in range(Ny-1):
            V_new[i+1,j+2] = V_int[i+1,j+2] - (dt/h)*(P[i,j+1]-P[i,j])

    #Divergence Test:
    test=np.zeros((Nx,Ny))
    for i in range(Nx):
        for j in range(Ny):
            test[i,j]=(U_new[i+2,j+1]-U_new[i+1,j+1])/h+(V_new[i+1,j+2]-V_new[i+1,j+1])/h

#======================================#
#   Step 7: Compute Residuals          #
#======================================#
    R_y = np.zeros((Nx-1,Ny))
    for i in range(Nx-1):
        for j in range(Ny):
            R_y[i,j]=h*(F[i+1,j]+P[i+1,j]-F[i,j]-P[i,j])+h*(Hx[i+1,j+1]-Hx[i+1,j])

    R_x = np.zeros((Nx,Ny-1))
    for i in range(Nx):
        for j in range(Ny-1):
            R_x[i,j]=h*(G[i,j+1]+P[i,j+1]-G[i,j]-P[i,j])+h*(Hy[i+1,j+1]-Hy[i,j+1])

    rx=np.sum(np.abs(R_x)); ry = np.sum(np.abs(R_y))
    L1[n]=rx+ry
    if L1[n]<10**-5:
        break
    else:
        print(n, L1[n])
        n+=1
        U = U_new;  V = V_new

tf = n

#Plotting L1 Norm
plt.figure()
fname=r'acc_ConvNy%dRe%d'%(Ny,Re)
plt.title(r'$Convergence\ History\ (Ny= {:.0f},\ Re={:.0f}, \beta={:.2f})$'.format(Ny,Re,beta))
plt.xlabel('$Number\ of\ Interations$')
plt.ylabel('$|R|_{L1}$')
plt.semilogy(np.arange(tf),L1[:tf], color = 'b')
plt.grid()
plt.tight_layout()
plt.show(block=False)
plt.savefig(fname)
np.savez('accelerated_Ny%dRe%d.npz'%(Ny,Re), U=U, V=V)
