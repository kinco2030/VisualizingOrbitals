import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

L = 10
a0 = 0.529 # in A
X, Y, Z = np.mgrid[-L:L:40j, -L:L:40j, -L:L:40j]

r = np.sqrt(X**2 + Y**2 + Z**2)
theta = np.arccos(Z/r)
phi = np.arctan2(Y, X)

sig = r/a0

# Radial part
row_1 = 2 * sig
R1s = a0 **(-3/2) * np.exp(-row_1/2) * 2

row_2 = sig
R2s = a0 **(-3/2) * np.exp(-row_2/2) * 8 ** (-1/2) * (2-row_2)
R2p = a0 **(-3/2) * np.exp(-row_2/2) * 24**(-1/2) * row_2

row_3 = (2/3)*sig
R3s = a0 **(-3/2) * np.exp(-row_3/2) * (81*3)**(-1/2) * (6-6*row_3+row_3**2)
R3p = a0 **(-3/2) * np.exp(-row_2/2) * (81*6)**(-1/2) * (4-row_3)*row_3
R3d = a0**(-3/2) * np.exp(-row_3/2) * (81*30)**(-1/2) * row_3**2

row_4 = (1/2)*sig
R4s = a0**(-3/2) * np.exp(-row_4/2) * (1/96) * (24-(36*row_4)+12*row_4**2-row_4**3)
R4p = a0**(-3/2) * np.exp(-row_4/2) * row_4 * (15/9)**(1/2) * (1/8) * (1 - row_4/2 + (3/64)*(row_4**2))
R4d = (8/15)*(5**(1/2))*(4*row_4)**(3/2) * np.exp(-(1/2)*row_4) * (2*row_4 - ((1/4)*row_4)**2 + (8/3)*((1/4)*row_4)**3)

# Angular part 
As = (4*np.pi)**(-1/2)
Apz = (3/4*np.pi)**(1/2) * np.cos(theta)
Apy = (3/4*np.pi)**(1/2) * np.sin(theta) * np.sin(phi)
Apx = (3/4*np.pi)**(1/2) * np.sin(theta) * np.cos(phi)
Adz2 = (5/4*np.pi)**(1/2) * (3*np.cos(theta)**2-1)
Adxz = (15/4*np.pi)**(1/2) * np.cos(theta) * np.sin(theta) * np.cos(phi)
Adyz = (15/4*np.pi)**(1/2) * np.cos(theta) * np.sin(theta) * np.sin(phi)
Adxxyy = (5/4*np.pi)**(1/2) * np.sin(theta)**2 * np.cos(2*phi)
Adxy = (5/4*np.pi)**(1/2) * np.sin(theta)**2 * np.sin(2*phi)

# Define the orbital to plot
values = R2s*As + R2p*Apz + R2p*Apy + R2p*Apx

# Plot the orbital
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Normalize values for colormap
norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))

# Define contour levels
levels = np.linspace(np.min(norm_values), np.max(norm_values), 10)

# Plot the isosurface
isosurface = ax.contourf(X[:,:,0], Y[:,:,0], Z[:,:,0], norm_values[:,:,int(len(Z)/2)], levels=levels, cmap='plasma')

# Add color bar
cbar = fig.colorbar(isosurface, ax=ax, orientation='vertical')
cbar.set_label('Normalized Value')

plt.show()
