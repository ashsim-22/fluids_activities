import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from backward_step_mod import physical_grid, stream, wall_shear
matplotlib.rcParams["contour.negative_linestyle"] = 'dashed'

import matplotlib.ticker as tick



with np.load('accelerated_Ny16Re400.npz') as data:
    U = data['U']
    V = data['V']

Ny=len(U[0,:])-2; Nx=len(U[:,0])-3; H=1
f=Nx/Ny; Ly=2*H; Lx=int(f*Ly)

if f==8:
    Re=200
elif f==12:
    Re=400
elif f==16:
    Re=600

x = np.linspace(0,Lx,Nx+1)
X,Y,h=physical_grid(Lx,Ly,Nx,Ny)
psi=stream(U, V, Nx, Ny, h)
levels=np.linspace(np.min(psi),np.max(psi),Ny+1)
tu,tl= wall_shear(U,h,x)


figname=[r'jcontour_Ny%dRe%d'%(Ny,Re),r'jshear_Ny%dRe%d'%(Ny,Re)]
plt.figure(figsize=(10, 4), dpi=100)
plt.title('$Flow\ Field\ (Re=%d,\ Ny=%d)$'%(Re,Ny))
plt.contour(X,Y,psi,levels=levels, linewidths=0.5, colors='k')
cont = plt.contourf(X,Y,psi,levels=levels)
cbar =plt.colorbar(cont)
cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
cbar.set_label('$\psi$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig(figname[0])
plt.show()


plt.figure(figsize=(10, 4), dpi=100)
plt.plot(x,tl,linewidth=2, color='b', label='Lower Wall')
plt.plot(x,tu,linewidth=2, color='r', label ='Upper Wall')
plt.plot(x, np.zeros_like(x), ':', color = 'k')
plt.title('$Wall\ Shear\ Stress\ (Re=%d,\ Ny=%d)$'%(Re,Ny))
plt.legend(loc='lower right')
plt.xlabel('$x$')
plt.ylabel(r'$\tau_{w}$')
plt.grid()
plt.savefig(figname[1])
plt.show()