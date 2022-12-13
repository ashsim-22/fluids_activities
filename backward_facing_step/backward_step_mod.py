import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy import sparse

#Physical Grid - x and y-co-ordinates
def physical_grid(Lx, Ly, Nx,Ny):
    x=np.linspace(0,Lx,Nx+1);y=np.linspace(0,Ly,Ny+1)
    h = np.subtract(x[2],x[1])

    yc,xc = np.meshgrid(y,x)
    return xc,yc,h

#Pressure Grid
def pressure_grid(h,Lx, Ly, Nx,Ny):
    P = np.zeros((Nx, Ny))
    nElem = P.size
    xp=np.linspace(h/2,Lx-h/2,Nx);yp=np.linspace(h/2,Ly-h/2,Ny)
    Yp,Xp = np.meshgrid(yp,xp)
    pData = np.zeros((nElem,2))
    i=0
    for j in range(Nx):
        for k in range(Ny):
            pData[i,0]=Xp[j,k]
            pData[i,1] = Yp[j,k]
            i+=1
    return pData

#U-velocity
def horizontal_vel(h,Lx, Ly, Nx,Ny):
    U = np.zeros((Nx+3, Ny+2))
    nElem = U.size
    xu=np.linspace(-h,Lx+h,Nx+3);yu=np.linspace(-h/2,Ly+h/2,Ny+2)
    Yu,Xu = np.meshgrid(yu,xu)
    uData = np.zeros((nElem,2))
    i=0

    for j in range(Nx+3):
        for k in range(Ny+2):
            uData[i,0]=Xu[j,k]
            uData[i,1] = Yu[j,k]
            i+=1
    return uData

#V-velocity
def vert_vel(h,Lx, Ly, Nx,Ny):
    V = np.zeros((Nx + 2, Ny + 3))
    nElem = V.size
    xv=np.linspace(-h/2,Lx+h/2,Nx+2);yv=np.linspace(-h,Ly+h,Ny+3)
    Yv,Xv = np.meshgrid(yv,xv)
    vData = np.zeros((nElem,2))
    i=0
    for j in range(Nx+2):
        for k in range(Ny+3):
            vData[i,0]=Xv[j,k]
            vData[i,1] = Yv[j,k]
            i+=1
    return vData

def F_x(h,Lx, Ly, Nx,Ny):
    F = np.zeros((Nx,Ny))
    nElem = F.size
    Fx=np.linspace(h/2,Lx-h/2,Nx);Fy=np.linspace(h/2,Ly-h/2,Ny)
    fy,fx = np.meshgrid(Fy,Fx)
    fData = np.zeros((nElem,2))
    i=0
    for j in range(Nx):
        for k in range(Ny):
            fData[i,0]=fx[j,k]
            fData[i,1] = fy[j,k]
            i+=1
    return fData

def G_y(h,Lx, Ly, Nx,Ny):
    G = np.zeros((Nx, Ny))
    nElem = G.size
    Gx=np.linspace(h/2,Lx-h/2,Nx);Gy=np.linspace(h/2,Ly-h/2,Ny)
    gy,gx = np.meshgrid(Gy,Gx)
    gData = np.zeros((nElem, 2))
    i = 0
    for j in range(Nx):
        for k in range(Ny):
            gData[i, 0] = gx[j, k]
            gData[i, 1] = gy[j, k]
            i += 1
    return gData

def H_x(Lx, Ly, Nx,Ny):
    H=np.zeros((Nx+1, Ny+1))
    nElem=H.size
    Hx=np.linspace(0,Lx,Nx+1);Hy=np.linspace(0,Ly,Ny+1)
    hxy,hxx = np.meshgrid(Hy,Hx)
    hxData = np.zeros((nElem, 2))
    i = 0
    for j in range(Nx+1):
        for k in range(Ny+1):
            hxData[i, 0] = hxx[j, k]
            hxData[i, 1] = hxy[j, k]
            i += 1
    return hxData


def H_y(Lx, Ly, Nx,Ny):
    H=np.zeros((Nx+1, Ny+1))
    nElem=H.size
    Hx=np.linspace(0,Lx,Nx+1);Hy=np.linspace(0,Ly,Ny+1)
    hyy,hyx = np.meshgrid(Hy,Hx)
    hyData = np.zeros((nElem, 2))
    i = 0
    for j in range(Nx+1):
        for k in range(Ny+1):
            hyData[i, 0] = hyx[j, k]
            hyData[i, 1] = hyy[j, k]
            i += 1
    return hyData

def inflow(uData, Ny,h,H):
    y = uData[int(Ny / 2) + 1:Ny + 1, 1]

    prof = -(y - H) * (y - 2 * H)
    C = (2 / 3) / (np.sum(h * prof))
    profile = C * prof
    sum = np.sum(profile)
    return profile, y



def outflow(uData, Ny,h,H):
    y = uData[1:Ny + 1, 1]
    prof = -((y-H)*(y-H))+H
    C = (2/3)/(np.sum(h*prof))
    profile=C*prof
    return profile, y

def Get_A(Nx, Ny, h):
    n = (Nx) * (Ny)  # number of unknowns
    d = np.ones(n)  # diagonals
    d0 = d.copy()*-4      #main diagional
    #Corners
    d0[0]=-2            #Bottem Left Corner [0]
    d0[Nx-1]=-2         #Bottom Right Corner [Nx,0]
    d0[n-1]=-2          #Top Right Corner [N-1]
    d0[n-Nx]=-2         #Top Left Corner    [N-Nx]
    #Edges
    d0[1:Nx-1]=-3       #Bottom Edge
    d0[n-(Nx-1):n-1]=-3 #Top Edge

    for i in range(1,Ny-1):
        d0[i*Nx]=-3
    for i in range(1,Ny-1):
        d0[(i+1)*Nx-1]=-3
    d1_lower = d.copy()[0:-1]
    d1_upper = d1_lower.copy()
    d1_upper[Nx-1] = 0
    d1_lower[-Nx] = 0
    d1_upper[2*Nx-1::Nx]=0
    d1_lower[Nx-1::Nx] = 0
    dnx_lower = d.copy()[0:-Nx]
    dnx_upper = dnx_lower.copy()
    d0[-1]=1                        #Setting Final Diagonal to 1. Pinning pressure
    d1_lower[-1]=0                  #Setting Final Lower Diag to 0. Pinning pressure
    dnx_lower[-1] = 0               #Setting Final off diag to 0. Pinning pressure  (Set b[-1]=1)
    A = scipy.sparse.diags([d0, d1_upper, d1_lower, dnx_upper, dnx_lower], [0, 1, -1, Nx, -Nx], format='csc')
    return A/(h**2)

def visualize_grid(X,Y, uData, vData, pData, hxData):
    plt.figure(figsize=(8, 8), dpi=100)
    plt.axis("equal")
    plt.plot(X, Y, color="k")
    #plt.scatter(X, Y, color="k", marker='o')
    plt.plot(X.transpose(), Y.transpose(), color="k")
    # Staggered grids
    plt.title('$Staggered\ Grid\ Arrangement$')
    plt.scatter(uData[:,0], uData[:,1], color="b", marker="o", s=14, label ='$u$')      #U-grid
    plt.scatter(vData[:,0], vData[:,1], color="r", marker="o", s=14, label = '$v$')     #V-grid
    plt.scatter(pData[:,0], pData[:,1], color="k", marker="o", s=14, label = '$P/G/F$')     #p-grid
    plt.scatter(hxData[:,0], hxData[:,1], color="purple", marker="s", s=14 , label ='$H^{x}/H^{y}$')    #Hx-grid
    plt.legend(loc='best')
    plt.show()

def stream(U,V,Nx,Ny,h):
    psi = np.zeros((Nx+1, Ny+1))
    for i in range(1,Nx+1):
        psi[i,0]=psi[i-1,0]+(-V[i,1])*h #Bottom Edge Streamline
    for i in range(Nx+1):
        for j in range(1,Ny+1):
            psi[i, j] = psi[i,j-1]+(U[i+1,j])*h #Right Edge Streamline

    return psi

def wall_shear(U,h,x):
    tau_upper=(U[1:-1,-1]-U[1:-1,-2])/h
    tau_lower=(U[1:-1,1]-U[1:-1, 0])/h

    #Lower Wall Z.C.
    xl_idxs = np.where(np.diff(np.sign(tau_lower)))[0]  #Indices of element before zero crossing
    xl_zero = []
    for i in xl_idxs:  #Interpolate Each Zero Crossing
        x1 = x[i]; x2 = x[i + 1]
        tl1 = tau_lower[i]; tl2 = tau_lower[i + 1]
        xl_zero.append(x1 + (0 - tl1) * ((x2 - x1) / (tl2 - tl1)))
    print('Lower Separation Point:', xl_zero[0], 'Lower Reattachment Point:', xl_zero[-1], 'Reattachment Length:',xl_zero[-1]-xl_zero[0])

    #Upper Wall Z.C.
    xu_idxs = np.where(np.diff(np.sign(tau_upper)))[0]  # indices of element before zero crossing
    xu_zero = []
    for i in xu_idxs:  # interpolate each zero crossing
        x1 = x[i]; x2 = x[i + 1]
        tu1 = tau_upper[i]; tu2 = tau_upper[i+1]
        xu_zero.append(x1 + (0-tu1)*((x2-x1)/(tu2-tu1)))

    if len(xu_zero)<2:
        print('No bubble :(')
    else:
        print('Upper Separation Point:', xu_zero[0], 'Upper Reattachment Point:', xu_zero[-1], 'Separation-Bubble Length:',
          xu_zero[-1] - xu_zero[0])

    return tau_upper,tau_lower

