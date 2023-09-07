import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from pyslalib import slalib


data = pd.read_csv('untitled4.txt',delimiter=" ",header=None)
e = data[0]
n = data[1]
U = data[2]
# #calculating the ITRF coordinates from the ENU coordinates
ref_lat = (19.06)*(np.pi/180)
ref_long = (74.03)*(np.pi/180)
ref_alt = 588
a = 6378137.0
f = 1/298.257223563
N = a/np.sqrt(1-(2*f-f*f)*np.sin(ref_lat)**(2))
x_ref = (N+ref_alt)*np.cos(ref_lat)*np.cos(ref_long)
y_ref = (N+ref_alt)*np.cos(ref_lat)*np.sin(ref_long)
z_ref = (((1-f)**(2))*N+ref_alt)*np.sin(ref_lat)
X = -e*np.sin(ref_long)-n*np.sin(ref_lat)*np.cos(ref_long)+U*np.cos(ref_lat)*np.cos(ref_long)+x_ref
Y = e*np.cos(ref_long)-n*np.sin(ref_lat)*np.sin(ref_long)+U*np.cos(ref_lat)*np.sin(ref_long)+y_ref
Z = n*np.cos(ref_lat)+U*np.sin(ref_lat)+z_ref
mat = np.array([X,Y,Z])
print(mat)
#Calculating the Baseline vector
bx=[]
by=[]
bz=[]
for i in range (len(X)):
  for j in range(len(X)):
    if i==j:
      continue
    b_x = X[j]-X[i]
    b_y = Y[j]-Y[i]
    b_z = Z[j]-Z[i]
    bx.append(b_x)
    by.append(b_y)
    bz.append(b_z)

base = np.array([bx,by,bz])
#print(np.shape(base))
base_t = np.transpose(base)
print(np.shape(base_t))

from astropy.time import Time
#start time of the obsservation
date1 = '2020-12-17T18:34:59.9'
date2 = '2020-12-17T23:51:30.4'
time_1 = Time(date1 ,format='isot')
time_2 = Time(date2,format='isot')
start_mjd = time_1.mjd
end_mjd = time_2.mjd
# print(mjd_1)
# print(mjd_2)
mjd = (start_mjd+end_mjd)/2
# print(mjd)
epoch1 = 2000
print(start_mjd)


class UvParType:
    def __init__(self):
        self.u = 0.0
        self.v = 0.0
        self.w = 0.0
        self.ra_app = 1.5*(np.pi/12)
        self.dec_app = 0.0*(np.pi/180)

uv_list = [UvParType() for _ in range(len(bx))]
for i in range(len(bx)):
    uv_list[i].u = base_t[i][0]
    uv_list[i].v = base_t[i][1]
    uv_list[i].w = base_t[i][2]

def prenut(uv, mjd, ra_app, dec_app, epoch1):
    TINY = 1.0e-9
    p = np.zeros((3, 3))
    p1 = np.zeros((3, 3))
    p2 = np.zeros((3, 3))
    p3 = np.zeros((3, 3))
    rm = np.zeros((3, 3))
    nm = np.zeros((3, 3))
    pm = np.zeros((3, 3))
    a = np.zeros((3, 3))
    v = np.zeros(3)
    v1 = np.zeros(3)

    r = d = -4 * np.pi
    m = e = 0.0
    new_epoch = new_source = 0

    if abs(m - mjd) > TINY or abs(e - epoch1) > TINY:
        new_epoch = 1
        new_source = 1
        r = ra_app
        d = dec_app
    else:
        if abs(r - ra_app) > TINY or abs(d - dec_app) > TINY:
            new_source = 1
            r = ra_app
            d = dec_app

    if new_epoch:
        a = slalib.sla_nut(mjd)
        nm = np.array(a).reshape(3, 3)

        epoch = slalib.sla_epj(mjd)
        a1 = slalib.sla_prec(epoch, epoch1)
        pm = np.array(a1).reshape(3, 3)

        rm = np.dot(pm, nm)

    if new_source:
        t = np.pi / 2 - d  # Use 'd' instead of 'dec_app'
        p = np.array([[1.0, 0.0, 0.0],
                      [0.0, np.cos(t), -np.sin(t)],
                      [0.0, np.sin(t), np.cos(t)]])

        t = np.pi / 2 + r  # Use 'r' instead of 'ra_app'
        p1 = np.array([[np.cos(t), -np.sin(t), 0.0],
                       [np.sin(t), np.cos(t), 0.0],
                       [0.0, 0.0, 1.0]])

        ra_mean, dec_mean = slalib.sla_amp(r, d, mjd, epoch1)  # Use 'r' and 'd' here
        t = np.pi / 2 - dec_mean
        p2 = np.array([[1.0, 0.0, 0.0],
                       [1.0, np.cos(t), -np.sin(t)],
                       [0.0, np.sin(t), np.cos(t)]])

        t = np.pi / 2 + ra_mean
        p3 = np.array([[np.cos(t), -np.sin(t), 0],
                       [np.sin(t), np.cos(t), 0.0],
                       [0.0, 0.0, 1.0]])

    v = np.array([uv.u, uv.v, uv.w])

    v1 = np.dot(p, v)
    v = np.dot(p1, v1)
    v1 = np.dot(rm, v)

    v = np.dot(p3, v1)
    v1 = np.dot(p2, v)

    uv.u = v1[0]
    uv.v = v1[1]
    uv.w = v1[2]

    return uv      


def lmst(mjd):
    # Calculate Local Mean Sidereal Time
    ut = mjd
    tu = (ut - 51544.5) / 36525.
    res = ut + 74.05 / 360.
    res = res - np.floor(res)
    lmstime = res + (((0.093104 - tu * 6.2e-6) * tu + 8640184.812866) * tu + 24110.54841) / 86400.0
    lmstime = (lmstime - np.floor(lmstime)) * 2.0 * np.pi
    return lmstime

sets = len(base_t)
# ha_values =np.linspace(-4,4,100)*(15)*(np.pi/180)
# ha = 4*(np.pi/12)
# dec = -30*(np.pi/180)
base = base_t


for k in range(len(base)):
   uv = uv_list[k]
   updated_uv = prenut(uv, mjd, uv.ra_app, uv.dec_app, epoch1)


lmst_start = lmst(start_mjd)
lmst_end = lmst(end_mjd)
start_ha = lmst_start-updated_uv.ra_app
end_ha = lmst_end-updated_uv.ra_app
#hour_angles = np.linspace(start_ha,end_ha,100)
hour_angles = np.arange(start_ha,end_ha,(8/3600)*(np.pi/12))


#def calculate_uvw_coordinates(mjd, epoch1, base,ha_rad):
    # Calculate Local Mean Sidereal Time
    #lmst_value = lmst(start_mjd)

    # Calculate UVW coordinates
    #uvw_coordinates = np.zeros((len(base), 3))
    #for k in range(len(base)):
        #uv = uv_list[k]
        #updated_uv = prenut(uv, start_mjd, uv.ra_app, uv.dec_app, epoch1)

        #Apply transformation matrices to baseline coordinates
        # ch = np.cos(lmst_value - updated_uv.ra_app)
        # sh = np.sin(lmst_value - updated_uv.ra_app)
        # cd = np.cos(updated_uv.dec_app)
        # sd = np.sin(updated_uv.dec_app)
        #ch = np.cos(ha_rad)
        #sh = np.sin(ha_rad)
        #cd = np.cos(updated_uv.dec_app)
        #sd = np.sin(updated_uv.dec_app)

        #u = base[k][0] * sh + base[k][1] * ch
        #v = -sd * ch * base[k][0] + base[k][1] * sd * sh + cd * base[k][2]
        #w = cd * ch * base[k][0] - cd * sh * base[k][1] + sd * base[k][2]

        #uvw_coordinates[k] = np.array([u, v, w])

    #return uvw_coordinates


# Initialize lists to store U and V coordinates
#U_list = []
#V_list = []

# Loop through all hour angles
#for ha in hour_angles:
    #uvw_coords = calculate_uvw_coordinates(start_mjd, epoch1, base_t, ha)
    #U = uvw_coords[:, 0]
    #V = uvw_coords[:, 1]

    #U_list.append(U)
    #V_list.append(V)

# Convert the lists to NumPy arrays
#U_array = np.array(U_list)
#V_array = np.array(V_list)

# Flatten the arrays for plotting
#U_flat = U_array.flatten()
#V_flat = V_array.flatten()

#print(U_flat)
#print(V_flat)
# Plot the scatter plot
#plt.figure(figsize=(8,8),dpi=240)
#plt.scatter(U_flat, V_flat,s =0.2,color='r')
#plt.xlabel('U Coordinate(k\u03BB)')
#plt.ylabel('V Coordinate(k\u03BB)')
# plt.title('UV Coverage for 0.5 hour angle and 45 degree declination')
# plt.savefig('Uv coverage at 45 dec.pdf')
def calculate_uvw_coordinates(mjd, epoch1, base, ha_rad):
    # Calculate Local Mean Sidereal Time
    lmst_value = lmst(mjd)

    # Calculate UVW coordinates
    uvw_coordinates = np.zeros((len(base), 3))
    for k in range(len(base)):
        uv = uv_list[k]
        updated_uv = prenut(uv, mjd, uv.ra_app, uv.dec_app, epoch1)

        # Apply transformation matrices to baseline coordinates
        ch = np.cos(lmst_value - updated_uv.ra_app)
        sh = np.sin(lmst_value - updated_uv.ra_app)
        cd = np.cos(updated_uv.dec_app)
        sd = np.sin(updated_uv.dec_app)

        u = base[k][0] * sh + base[k][1] * ch
        v = -sd * ch * base[k][0] + base[k][1] * sd * sh + cd * base[k][2]
        w = cd * ch * base[k][0] - cd * sh * base[k][1] + sd * base[k][2]

        uvw_coordinates[k] = np.array([u, v, w])

    return uvw_coordinates

def plot_uv_coverage(U, V):
    # Flatten the arrays for plotting
    U_flat = U.flatten()
    V_flat = V.flatten()

    # Plot the scatter plot
    plt.scatter(U_flat, V_flat, s=0.05, color='r')
    plt.xlabel('U Coordinate')
    plt.ylabel('V Coordinate')
    plt.title('UV Coverage for 0.5 hour angle and 45-degree declination')
    plt.savefig('Uv_coverage_at_45_dec.pdf')
    plt.show()

# Initialize lists to store U and V coordinates
U_list = []
V_list = []

# Loop through all hour angles
for ha in hour_angles:
    uvw_coords = calculate_uvw_coordinates(start_mjd, epoch1, base_t, ha)
    U = uvw_coords[:, 0]
    V = uvw_coords[:, 1]

    U_list.append(U)  # Append U coordinates for this angle
    V_list.append(V)  # Append V coordinates for this angle

# Convert the lists to NumPy arrays
U_array = np.array(U_list)
V_array = np.array(V_list)

# Call the function to plot UV coverage
plot_uv_coverage(U_array, V_array)

plt.show()
