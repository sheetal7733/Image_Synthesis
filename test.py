import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = pd.read_csv('base_layout_gmrt.enu.30x4.txt',delimiter=" ",header=None)
e = data[1]
n = data[2]
U = data[3]
ref_lat = (19.0931)*(np.pi/180)
ref_long = 74.0506*(np.pi/180)
ref_alt = 650
a = 6378137.0
f = 1/298.257223563


# Calculate N
N = a / np.sqrt(1 - (2*f - f*f) * np.sin(ref_lat)**2)

# Convert ENU to XYZ
X = (N + ref_alt) * np.cos(ref_lat) * np.cos(ref_long) + e
Y = (N + ref_alt) * np.cos(ref_lat) * np.sin(ref_long) + n
Z = ((1 - f)**2 * N + ref_alt) * np.sin(ref_lat) + U

# Print the ITRF coordinates  1.657672e+06

#N = a/np.sqrt(1-(2*f-f*f)*np.sin(ref_lat)**(2))
#x_ref = (N+ref_alt)*np.cos(ref_lat)*np.cos(ref_long)
#y_ref = (N+ref_alt)*np.cos(ref_lat)*np.sin(ref_long)
#z_ref = (((1-f)**(2))*N+ref_alt)*np.sin(ref_lat)
#X = -e*np.sin(ref_long)-n*np.sin(ref_lat)*np.cos(ref_long)+U*np.cos(ref_lat)*np.cos(ref_long)+x_ref
#Y = e*np.cos(ref_long)-n*np.sin(ref_lat)*np.sin(ref_long)+U*np.cos(ref_lat)*np.sin(ref_long)+y_ref
#Z = n*np.cos(ref_lat)+U*np.sin(ref_lat)+z_ref

#Another data
data_1 = pd.read_csv('test1.txt',delimiter=" ",header=None)
x = data_1[0]
y = data_1[1]
z = data_1[2]
print(x-X,y-Y,z-Z)
