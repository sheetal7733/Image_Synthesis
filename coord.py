import numpy as np
import pandas as pd

# Read ENU data from the file
data = pd.read_csv('base_layout_gmrt.enu.30x4.txt', delimiter=" ", header=None)
e = data[1]
n = data[2]
U = data[3]

# GMRT-specific coordinates
gmrt_lat = 19.0931 * (np.pi / 180)  # GMRT latitude in radians
gmrt_long = 74.0506 * (np.pi / 180)  # GMRT longitude in radians
gmrt_alt = 588  # GMRT altitude in meters

# Constants
a = 6378137.0
f = 1/298.257223563
N = a / np.sqrt(1 - (2 * f - f * f) * np.sin(gmrt_lat) ** 2)

# Transformation from ENU to ITRF
X = -e * np.sin(gmrt_long) - n * np.sin(gmrt_lat) * np.cos(gmrt_long) + U * np.cos(gmrt_lat) * np.cos(gmrt_long) + (N + gmrt_alt) * np.cos(gmrt_lat) * np.cos(gmrt_long)
Y = e * np.cos(gmrt_long) - n * np.sin(gmrt_lat) * np.sin(gmrt_long) + U * np.cos(gmrt_lat) * np.sin(gmrt_long) + (N + gmrt_alt) * np.cos(gmrt_lat) * np.sin(gmrt_long)
Z = n * np.cos(gmrt_lat) + U * np.sin(gmrt_lat) + ((1 - f) ** 2) * N + gmrt_alt
print(X,Y,Z)

