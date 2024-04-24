import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
import pyslalib
from pyslalib import slalib
from scipy.optimize import curve_fit
import pyproj

#---------Fumction for calculating the ITRF coordinates---------

def llh2itrf(lon, lat, height):
   wgs84 = pyproj.CRS("EPSG:4326")
   itrf2005 = pyproj.CRS("EPSG:4896")
   transformer = pyproj.Transformer.from_crs(wgs84, itrf2005, always_xy=True)
   x, y,z = transformer.transform(lon, lat, height)
   return x, y, z

#--------Writing the ITRF into a txt file------
coordinates = [
    (75.926703600893, 22.5233500765833, 619),
    (75.9271759274595, 22.5234119637904, 619),
    (75.9270470010471, 22.5238770319285, 619),
    (75.9266371267733, 22.5238597555395, 619)
   ]

itrf_coordinates = [llh2itrf(lon, lat, height) for lon, lat, height in coordinates]


output_file_path = "four_antennas_itrf.txt"

with open(output_file_path, "w") as file:
    for (lon, lat, height), (x, y, z) in zip(coordinates, itrf_coordinates):
        file.write(f"\t{x} {y} {z}\n")

print(f"Results written to {output_file_path}")

#---------Load the antennas coordinates file-------
coord = pd.read_csv("four_antennas_itrf.txt",delimiter = ' ', header = None )
x = coord[0]
y = coord[1]
z = coord[2]
b_x = []
b_y = []
b_z = []

for _ in range(155):
  for i in range(len(x)):
    bxx = x[i]-x[2]
    byy = y[i]-y[2]
    bzz = z[i]-z[2]
    b_x.append(bxx)
    b_y.append(byy)
    b_z.append(bzz)

sets = len(x)
print(sets)
# print(b_x)


#------Function for calculating (u,v,w) coordinates-----
def gvgetuvw(sets, ha, dec):
    ch = np.cos(ha)
    sh = np.sin(ha)
    cd = np.cos(dec)
    sd = np.sin(dec)

    c = 2.99792458e8
    freq = 1.4e+09
    lam = c/freq

    u_values = []
    v_values = []
    w_values = []

    for k in range(sets):
        bxch = b_x[k] * ch
        bysh = b_y[k] * sh

        u = (b_x[k] * sh + b_y[k] * ch)/lam
        v = (b_z[k] * cd - sd * (bxch - bysh))/lam
        w = (cd * (bxch - bysh) + sd * b_z[k])/lam

        u_values.append(u)
        v_values.append(v)
        w_values.append(w)

    return u_values, v_values, w_values


#------Function for calcualating Local Mean Sidereal Time------

def lmst(mjd):
  ut = mjd
  tu = (ut - 51544.5) / 36525.
  res = ut + 74.05 / 360.
  res = res - np.floor(res)
  lmstime = res + (((0.093104 - tu * 6.2e-6) * tu + 8640184.812866) * tu + 24110.54841) / 86400.0
  lmstime = (lmstime - np.floor(lmstime)) * 2.0 * np.pi
  return lmstime

#-------Converting string to seconds--------

def str2sec(time_str):
    hours, minutes, seconds = map(int, time_str.split(':'))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds

#--------Function for calculating the Precession and Nutation Effect on (u,v) points--------
def prenut(uu, vv, ww, mjd, ra_app, dec_app, epoch1):
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
        m = mjd
        e = epoch1
    else:
        if abs(r - ra_app) > TINY or abs(d - dec_app) > TINY:
            new_source = 1
            r = ra_app
            d = dec_app


    if new_epoch:

        a = slalib.sla_nut(mjd)
        a=np.transpose(a)
        a = a.flatten()
        #print(a)
        for i in range(3):
          for j in range(3):
            nm[j][i] = a[3*i+j]

        epoch = slalib.sla_epj(mjd)
        a = slalib.sla_prec(epoch, epoch1)
        a=np.transpose(a)
        a = a.flatten()
        #print(a)
        for i in range(3):
          for j in range(3):
            pm[j][i] = a[3*i+j]


        for i in range(3):
          for j in range(3):
            rm[i][j] = 0.0
            for k in range(3):
              rm[i][j] += pm[i][k] * nm[j][k]


    if new_source:
        t = np.pi / 2 - dec_app
        p = np.array([[1.0, 0.0, 0.0],
                      [0.0, np.cos(t), -np.sin(t)],
                      [0.0, np.sin(t), np.cos(t)]])

        t = np.pi / 2 + ra_app
        p1 = np.array([[np.cos(t), -np.sin(t), 0.0],
                       [np.sin(t), np.cos(t), 0.0],
                       [0.0, 0.0, 1.0]])

        ra_mean, dec_mean = slalib.sla_amp(ra_app, dec_app, mjd, epoch1)
        # print(ra_mean)
        # print(dec_mean)
        t = np.pi / 2 - dec_mean
        p2 = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(t), -np.sin(t)],
                       [0.0, np.sin(t), np.cos(t)]])

        t = np.pi / 2 + ra_mean
        p3 = np.array([[np.cos(t), -np.sin(t), 0],
                       [np.sin(t), np.cos(t), 0.0],
                       [0.0, 0.0, 1.0]])

    v = np.array([uu, vv, ww])

    for i in range(3):
      v1[i] = 0.0
      for j in range(3):
        v1[i] += p[i][j]*v[j]


    for i in range(3):
      v[i] = 0.0
      for j in range(3):
        v[i] += p1[i][j]*v1[j]

    for i in range(3):
      v1[i] = 0.0
      for j in range(3):
        v1[i] += rm[i][j]*v[j]

    for i in range(3):
      v[i] = 0.0
      for j in range(3):
        v[i] += p3[j][i]*v1[j]

    for i in range(3):
      v1[i] = 0.0
      for j in range(3):
        v1[i] += p2[j][i]*v[j]

    updated_u = v1[0]
    updated_v = v1[1]
    updated_w = v1[2]

    return updated_u, updated_v, updated_w

#2D gaussian
def Gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    g = offset + amplitude * np.exp( -(((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))
    return g.ravel()



# observation_paramters = pd.read_csv('observation_parameter.txt')

with open('random_Parameters.txt', 'w') as file:
    file.write("RA\tDEC\tStart_time\tStop_time\tintegration_time\tRef_Date\n")
    data_rows = ['13:31:08.27999', '30:00:00', "02:00:00", "8:00:00", 4.0233, "2024-02-14T18:30:00"] 
    row_str = '\t'.join(map(str, data_rows))
    file.write(row_str + '\n')

observation_parameter = pd.read_csv('random_Parameters.txt',delimiter="\t")
print("Columns in DataFrame:", observation_parameter.columns)
ra_value = observation_parameter['RA']
ra_value = ra_value.iloc[0]
#print(ra_value)
ra_split = ra_value.split(':')
#print(ra_split)
ra_split_float = [float(i) for i in ra_split]
ra = (ra_split_float[0] + (ra_split_float[1] / 60) + (ra_split_float[2] / 3600)) * (np.pi / 12)
#print(ra_app)
dec = observation_parameter['DEC']
dec_value = dec.iloc[0]
#print(dec_value)
dec_split = dec_value.split(':')
#print(dec_split)
dec_split_float = [float(i) for i in dec_split]
dec = (dec_split_float[0] + (dec_split_float[1] / 60) + (dec_split_float[2] / 3600)) * (np.pi / 180)
#print(dec_app)
date = observation_parameter['Ref_Date']
date_obs = Time(observation_parameter['Ref_Date'].iloc[0],format='isot')
mjd_ref = date_obs.mjd
start_time = observation_parameter['Start_time'].iloc[0]
stop_time = observation_parameter['Stop_time'].iloc[0]
sr_tm = str2sec(start_time)
st_tm = str2sec(stop_time)
integration_time = observation_parameter['integration_time'].iloc[0]
tm = np.arange(sr_tm,st_tm,integration_time)
# print(tm)
mjd0 = round(mjd_ref)
print(mjd0)
epoch = 2000 + (mjd_ref - 51544.5)/365.25
print(epoch)
epoch1 = 2000
if (epoch1 <  0.0):
  epoch1 = epoch
ra_ap,dec_ap = slalib.sla_preces("FK5",epoch,epoch1,ra,dec)
# print(ra_app)
lst= lmst(mjd_ref + tm/86400)
jd = mjd_ref + 2400000.5 + tm/86400
ha = lst - ra


uvw = gvgetuvw(sets,ha,dec)
u = uvw[0]
v = uvw[1]
w = uvw[2]
# print(u)
uuu = []
vvv = []
www = []
print(len(u))
# for _ in range(155):
for i in range(len(u)):
  for j in range(i+1,len(u)):
    u_new = u[j]-u[i]
    v_new = v[j]-v[i]
    w_new = w[j]-w[i]
    uuu.append(u_new)
    vvv.append(v_new)
    www.append(w_new)
# print(uuu)
uuu = np.array(uuu).flatten(order="F")
vvv = np.array(vvv).flatten(order="F")
www = np.array(www).flatten(order="F")
# print(uuu)
# print(vvv)

uuu_new = np.concatenate((uuu,uuu*(-1)))
vvv_new = np.concatenate((vvv,vvv*(-1)))
www_new = np.concatenate((vvv,vvv*(-1)))
# print(uuu_new)
# print(vvv_new)
# plt.figure(figsize=(7,7),dpi=240)
# labels = ['A4','A1','A2','A3']
# # plt.figure(dpi=240)
# plt.rcParams.update({'font.size':15})
# plt.scatter(x,y,marker="^", s=75)
# for i, label in enumerate(labels):
#    plt.text(x[i],y[i],label,fontsize=9,ha='right',va='bottom')
# plt.xlabel('x(m)',fontsize=15)
# plt.ylabel('y(m)',fontsize=15)
# # plt.title("Configuration of IIRI")
# plt.savefig("Config_IIRI.png")
# plt.show()
# plt.scatter(uuu_new,vvv_new,s = 0.3)
# plt.xlabel("u(m)")
# plt.ylabel('v(m)')
# plt.title("Observed (u,v) distribution of IIRI")
# plt.show()
epoch = 2000 + (mjd_ref - 51544.5)/365.25
print(epoch)
epoch1 = 2000
if (epoch1 <  0.0):
  epoch1 = epoch
# for i in range(len(u_org)):
uu_mast=[]
vv_mast=[]
ww_mast=[]

for i in range(len(uuu)):
  updated_uu, updated_vv, updated_ww = prenut(uuu[i], vvv[i], www[i], mjd_ref, ra, dec, epoch1)
  uu_mast.append(updated_uu)
  vv_mast.append(updated_vv)
  ww_mast.append(updated_ww)

uu_mast=np.array(uu_mast)
vv_mast = np.array(vv_mast)
# print(uu_mast)

uuu_mast = np.concatenate((uu_mast,uu_mast*(-1)))
vvv_mast = np.concatenate((vv_mast,vv_mast*(-1)))
# print(uuu_mast)
# print(len(uuu_mast))
# print(len(vvv_mast))
# plt.rcParams.update({'font.size':15})
# plt.scatter(uuu_mast,vvv_mast,s=0.003,color='r',label="\u03b4 = 0\N{DEGREE SIGN}")
# plt.xlabel('u(m)',fontsize=15)
# plt.ylabel('v(m)',fontsize=15)
# plt.legend(loc='upper right', fontsize=11)
# plt.savefig("u-v coverage_4antennas_0dec.png")

def gaussian_2d(xy, mu_x, mu_y, sigma_x, sigma_y, amplitude):
    x, y = xy
    return amplitude * np.exp(-(((x - mu_x) ** 2) / (2 * sigma_x ** 2) + ((y - mu_y) ** 2) / (2 * sigma_y ** 2)))

# Assuming you have defined uuu_mast, vvv_mast
# Step 1: Normalize u-v coordinates
def normalize_uv(u, v):
    u_max = np.max(np.abs(u))
    v_max = np.max(np.abs(v))
    u_norm = u / u_max
    v_norm = v / v_max
    return u_norm, v_norm

# Step 2: Grid the u-v points
def grid_uv(u, v, grid_size=64):
    u_grid = np.linspace(-1, 1, grid_size)
    v_grid = np.linspace(-1, 1, grid_size)
    counts, _, _ = np.histogram2d(u, v, bins=[u_grid, v_grid])
    return counts

# Step 3: Compute the Fourier transform
def compute_synthesized_beam(uv_counts):
    image_plane = np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(uv_counts)))
    return image_plane

# Step 4: Normalize the synthesized beam
def normalize_beam(image_plane):
    normalized_beam = np.abs(image_plane) / np.max(np.abs(image_plane))
    return normalized_beam

# Assuming you have defined uuu_mast, vvv_mast
#Step 1: Normalize u-v coordinates
u_norm, v_norm = normalize_uv(uuu_mast, vvv_mast)

# Step 2: Grid the u-v points
uv_counts = grid_uv(u_norm, v_norm)

# Step 3: Compute the Fourier transform
synthesized_beam = compute_synthesized_beam(uv_counts)

#Step 4: Normalize the synthesized beam
normalized_beam = normalize_beam(synthesized_beam)

# Plot the real part of the normalized beam
plt.rcParams.update({'font.size':15})
plt.imshow(np.real(normalized_beam), cmap='viridis')
plt.xlabel("RA",fontsize=15)
plt.ylabel("DEC",fontsize=15)
plt.colorbar()
plt.savefig('beam_iiri_30.png')

# Generate X, Y coordinates
# x = np.linspace(-1, 1, normalized_beam.shape[0])
# y = np.linspace(-1, 1, normalized_beam.shape[1])
# X, Y = np.meshgrid(x, y)

# initial_guess = (0, 0, 0.1, 0.1, 1)

# # Fit the Gaussian
# popt, pcov = curve_fit(gaussian_2d, (X.flatten(), Y.flatten()), normalized_beam.ravel(), p0=initial_guess)

# Print optimized parameters
# print("Optimized Parameters:")
# print("mu_x:", popt[0])
# print("mu_y:", popt[1])
# print("sigma_x:", popt[2])
# print("sigma_y:", popt[3])
# print("A:", popt[4])

# Calculate the fitted Gaussian
# fitted_gaussian = gaussian_2d((X, Y), *popt)

# # Calculate RMS value
# rms_value = np.std(normalized_beam - fitted_gaussian)
# print("RMS value:", rms_value)
# histogram, xedges, yedges = np.histogram2d(uuu_new, vvv_new, bins=(40, 40))
# plt.plot(histogram)
# plt.show()
# beam = np.fft.fftshift(np.fft.fft2(histogram))
# beam = np.absolute(beam)
# plt.imshow(beam, cmap='magma')
# plt.colorbar()
# plt.show()



# x_1 = np.linspace(0, beam.shape[1] - 1, beam.shape[1])
# y_1 = np.linspace(0, beam.shape[0] - 1, beam.shape[0])
# x_1, y_1 = np.meshgrid(x_1, y_1)



# xy = np.vstack([x_1.ravel(), y_1.ravel()])
# data = beam.ravel()

# initial_guess = (data.max(), beam.shape[1]//2, beam.shape[0]//2, 10, 10, data.min())

# popt, pcov = curve_fit(Gaussian_2d, xy, data, p0 = initial_guess)
# print("Fitted parameters (amplitude, xo, yo, sigma_x, sigma_y, offset):", popt)
# fitted_data = Gaussian_2d((x_1, y_1), *popt).reshape(beam.shape)
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.imshow(beam, cmap='magma', origin='lower')
# plt.title('Original Beam Data')
# plt.colorbar()

# plt.subplot(1, 2, 2)
# plt.imshow(fitted_data, cmap='viridis', origin='lower') 
# plt.title('Fitted Gaussian')
# plt.colorbar()
# # contour_levels = np.linspace(fitted_data.min(), fitted_data.max(), 5)  
# # contour = plt.contour(x_1, y_1, fitted_data, levels=contour_levels, colors='red', linewidths=2, alpha=0.8)
# levels = np.logspace(np.log10(fitted_data.min()+1), np.log10(fitted_data.max()), 10)
# contour = plt.contour(x_1, y_1, fitted_data, levels=levels, colors='red', linewidths=2, alpha=0.8)
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.imshow(beam, cmap='magma', origin='lower', extent=(0, beam.shape[1], 0, beam.shape[0]))
# plt.colorbar()
# plt.title('Original Beam Data with Fitted Gaussian Contours')
# contour_levels = np.logspace(np.log10(fitted_data.min()+1), np.log10(fitted_data.max()), 10) 
# plt.contour(x_1, y_1, fitted_data, levels=contour_levels, colors='white', linewidths=1.5, alpha=0.7)

# plt.show()
