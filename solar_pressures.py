import numpy as np
import matplotlib.pyplot as plt
import time
import os

''' different coordinate systems definitions ***
(all coordinate systems are right handed)

Earth coordinate system [xe, ye, ze] 
		- origin in Earth's center 
		- xe always towards the sun center 
		- ze towards north pole

Orbit coordinate system [xo, yo, zo] 
		- origin in Earth's center, 
		- xo always towards equatorial ascending node
		- zo so that satellite orbits in a positive way around it

Body carried velocity coordinate system [xv, yv, zv] 
		- origin in CubeSat's c.g.
		- xv always coincides velocity vector
		- zv towards Earth's center

Body coordinate system [xb, yb, zb] 
		- origin in CubeSat's c.g.
		- xb opposite of thrust vector
		- zv coincides with nominal camera pointing vector

x axis = 0
y axis = 1
z axis = 2
'''
# GENERAL COORDINATE SYSTEM TRANSFORMATIONS
def tf_matrix(angle, axis=0, degrees=True):
	if degrees:
		angle = angle*np.pi/180.
	if axis==0:
		a=np.array([[1, 	0., 			0.], \
			        [0., 	np.cos(angle), 	np.sin(angle)], \
					[0., 	-np.sin(angle), np.cos(angle)]])
		return a
	elif axis==1:
		a=np.array([[np.cos(angle), 	0., 	-np.sin(angle)], \
			        [0., 				1., 	0. 			  ], \
					[np.sin(angle), 	0., 	np.cos(angle) ]])
		return a
	elif axis==2:
		a=np.array([[np.cos(angle), 	np.sin(angle),	0.], \
			        [-np.sin(angle), 	np.cos(angle), 	0.], \
					[0., 				0., 			1.]])
		return a
def axis_tf(*args, degrees=True):
	tf = np.identity(3)
	for arg in args:
		mat = tf_matrix(arg[0], axis=arg[1], degrees=degrees)
		tf = np.matmul(mat,tf)
		tf[np.abs(tf)<1e-10]=0
	return tf

# SPECIFIC ORBIT TRANSFORMATIONS
def EtoO(arg_asc_node=0, inclination=0):
	e2o = arg_asc_node
	e0o= inclination
	return axis_tf((e2o,2),(e0o,0))
def OtoV(arg_lat=0):
	o2v = 90
	o0v = -90
	o1v = -arg_lat
	return axis_tf((o2v,2), (o0v,0), (o1v,1))
def VtoB(roll=0, pitch=0, yaw=0):
	v0b = roll
	v1b = pitch
	v2b = yaw
	return axis_tf((v0b,0), (v1b,1), (v2b,2))



# INPUTS
battery_packs = 2
side_units = 6
alt = 340000	# m
mission_length= 5 #years
n = 7500		# steps per orbit
orbits = 15	# simulated orbits
case = 'hot'

print(case)
comm_out = 2	 #W   2-4W
thrust_out = 34*2 	 #W

# ENVIRONMENT CONSTANTS
CtoK = 273.15 #celsius to Kelvin
stefan_boltz = 5.67e-8 		# W/m^2/K^4
earth_r = 6378000			# m
myu = 3.986e14 				# m^3/s^2

if case=='hot':
	e_albedo = 0.4 				# reflected ratio 0.24-0.4 for SSO
	earth_T = 256 				# 246K-256K
	solar_flux = 1371			# W/m^2
else:
	e_albedo = 0.24 				# reflected ratio 0.24-0.4 for SSO
	earth_T = 246 				# 246K-256K
	solar_flux = 1322			# W/m^2
space_T = 2.7 				# K
rad_conv = 180/np.pi 		#57.3..

# SOME ORBIT AND SHADE SHIT
inclination = 96.75			# 96.66-96.84   96.75	
arg_asc_node = -45 	 	 	# -45 and 45
tot_orbit_time = 2*np.pi*((earth_r+alt)**3/myu)**0.5  	 	# s
angle_of_sun_behind_pole = np.arccos(earth_r/(earth_r+alt)) # rads
shade_angle = 30/tot_orbit_time*2*np.pi  			  	 	# 30s of eclipse
EtoS_r = (4*np.pi*earth_r**2) / (4*np.pi*(alt+earth_r)**2)  # earth to sat-elite ratio for scaling 
rho = np.arcsin(EtoS_r**0.5)  		 	 		 	 	 	# sum altitude factor
Ka = 0.664 + 0.521*rho + 0.203*rho**2  	 	 	 	  	 	# sum albedo scaling factor



# AREAS OF FACES FOR SAT-ELITE
# BODY = 229.4mm W x 222.8mm H x 366mm L 	SOLAR PANELS = 212.8mm W x 2.1mm H x 366mm L
angl=26.5
body_w = 229.4/1000.  		# m 
body_h = 222.8/1000.  		# m 
body_l = 366/1000.  		# m 
sp_w = 212.8/1000.  		# m 
sp_h = 2.1/1000.  		# m 
sp_l = 366/1000.  		# m 
sat_posX = 4*sp_h*sp_w + body_w*body_h  #m^2
sat_negX = sat_posX  	 	#m^2
sat_posY = body_l*body_h + 4*sp_w*sp_l*np.cos(angl/rad_conv) 	#m^2
sat_negY = sat_posY
sat_posZ = body_l*body_h + 4*sp_w*sp_l*np.sin(angl/rad_conv) 	#m^2
sat_negZ = sat_posZ 	 	#m^2
areas = np.array([sat_posX, sat_negX,   sat_posY, sat_negY,   sat_posZ, sat_negZ]).reshape((1,6)) # [+X, -X, +Y, -Y, +Z, -Z]


# ABSORTIVITY AND REFLECTIVITY ARRAYS FOR SURFACES [+X, -X, +Y, -Y, +Z, -Z]
absortivity = np.zeros((1,6))
emisivity = np.zeros((1,6))
for i in range(np.shape(absortivity)[1]):
	absortivity[0,i]=0.15
	emisivity[0,i]=0.05

absortivity[0,2]= 179/779*0.15 + (1-179/779)*0.9
emisivity[0,2]  = 179/779*0.05 + (1-179/779)*0.85


orbit_time = np.linspace(0, tot_orbit_time, num=n).reshape((n,1)) 	# time linspace for one orbit

panels_power = np.zeros((n,6)) 	 	# power created by diff. sat. sides [+X, -X, +Y, -Y, +Z, -Z]
EtoB_transform = np.zeros((n,3,3))  # EtoB transforations with varying orbit angle
sun_incidence = np.zeros((n,6))  	# 0-1(cosine of sun incidence angle) [+X, -X, +Y, -Y, +Z, -Z]
earth_incidence = np.zeros((n,6))   # 0-1(cosine of earth incidence angle) [+X, -X, +Y, -Y, +Z, -Z]
albedo_incidence = np.zeros((n,6))  # 0-1(cosine of albedo incidence angle) [+X, -X, +Y, -Y, +Z, -Z]
sun_vec_B = np.zeros((n,3)) 		# Sun vector transformed to body system along the orbit
earth_vec_B = np.zeros((n,3)) 		# Earth vector transformed to body system along the orbit
sun_visibility = np.ones(np.shape(orbit_time)) 			# 0 or 1 for orbit linspace if the sun is visible
albedo_visibility = np.ones(np.shape(orbit_time)) 			# 0 or 1 for orbit linspace if the albedo is visible
sun_vec_E = np.array([1,0,0]) 		# Earth-Sun vector in Earth coordinate system
earth_vec_V = np.array([0,0,1]) 	# C.G.- Earth vector in Velocity coordinate system
body_vec_V = [0,0,-(earth_r+alt)] 		# C.G.- Earth vector in Velocity coordinate system with magnitude


t_step = (orbit_time[1]-orbit_time[0])					# step in s
orbit_angle = orbit_time/tot_orbit_time*2*np.pi 		# angle linspace for one orbit (0-2pi) 0 starts in the middle of shadow cone
orbit_ang = orbit_angle * rad_conv 						# same shit but in degrees, just for plotting


# GETTING SOLAR PANEL SUN EXPOSURE + SHADE CONDITION ALONG ORBIT
arg_lat = orbit_ang
sun_visibility = np.ones(np.shape(orbit_time)) 
albedo_visibility= np.ones(np.shape(orbit_time)) 
for i in range(n):
	vtob = VtoB(roll=0, pitch=0, yaw=0)
	EtoB_transform[i,:,:] = np.matmul( vtob, np.matmul( OtoV(arg_lat=arg_lat[i,0]) ,\
									   EtoO(arg_asc_node=arg_asc_node, inclination=inclination) ))
	#shade and albedo
	body_vec_E = np.matmul(np.linalg.inv(EtoB_transform[i,:,:]), body_vec_V)
	if body_vec_E[0]<0:
		albedo_visibility[i]=0
		if (body_vec_E[1]**2 + body_vec_E[2]**2)**0.5 < earth_r:
			sun_visibility[i]=0
		
	sun_vec_B[i,:] = np.matmul(EtoB_transform[i,:,:], sun_vec_E)	
	sun_incidence[i,:] = sun_vec_B[i,0], -sun_vec_B[i,0], \
						   sun_vec_B[i,1], -sun_vec_B[i,1], \
						   sun_vec_B[i,2], -sun_vec_B[i,2]
						   
	earth_vec_B[i,:] = np.matmul(vtob, earth_vec_V)
	earth_incidence[i,:] = earth_vec_B[i,0], -earth_vec_B[i,0], \
						   earth_vec_B[i,1], -earth_vec_B[i,1], \
						   earth_vec_B[i,2], -earth_vec_B[i,2]

# INCIDENCES AND INSTANTANEOUS POWER PRODUCED BY SOLAR PANELS
albedo_incidence = earth_incidence
sun_incidence[sun_incidence < 0] = 0 	 
earth_incidence[earth_incidence < 0] = 0 
sun_incidence = sun_incidence*sun_visibility 
albedo_incidence = albedo_incidence*albedo_visibility




















