import numpy as np
import matplotlib.pyplot as plt
import time
import os

#http://www.iosrjournals.org/iosr-jeee/Papers/Vol12%20Issue%203/Version-5/G1203053745.pdf

def less_dec(a,decs=2):
	b = 10**decs
	return int((a) * b) / float(b)

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

# THERMAL FUNCTIONS
def solar_input(areas, sun_incidence, absortivity, solar_flux = 1368 ):
	heatflow = areas*sun_incidence*absortivity*solar_flux
	return heatflow
#a = solar_input(areas, sun_incidence, absortivity, solar_flux = solar_flux )	
def E_IR_input(areas, earth_incidence, emisivity, altitude = 340, earth_T = 256, stefan_boltz = 5.67e-8, EtoS_r = 0.9 ):
	earth_flux = stefan_boltz*earth_T**4 
	heatflow = areas*earth_incidence*emisivity*EtoS_r*earth_flux
	return heatflow 		 #J/s
#a = E_IR_input(areas, earth_incidence, emisivity, altitude=altitude, earth_T=earth_T, stefan_boltz=stefan_boltz, EtoS_r=EtoS_r  )	
def albedo_input(areas, earth_incidence, absortivity, altitude = 340, e_albedo = 0.3, solar_flux = 1368, Ka = 0 ):
	albedo_flux = solar_flux*e_albedo
	heatflow = areas*earth_incidence*absortivity*Ka*albedo_flux
	return heatflow 		 #J/s
#a = albedo_input(areas, earth_incidence, absortivity, altitude = altitude, e_albedo = e_albedo, solar_flux = solar_flux, Ka=Ka)
def from_batteries(P_to_storage):
	return -P_to_storage 		 #J/s
#a = to_batteries(P_to_storage)	
def sys_output(thruster_on = False, comms_on = False, thrust_out = 30, comm_out = 10):
	heatflow = 0
	if thruster_on:
		heatflow+=thrust_out
	if comms_on:
		heatflow+=comm_out	
	return -heatflow 		 #J/s
#a = sys_output(thruster_on = False, comms_on = False, thrust_out = thrust_out, comm_out = comm_out)
def IR_output(areas, emisivity, sat_T, stefan_boltz = 5.67e-8, space_T = 3 ):
	heatflow = areas*emisivity*stefan_boltz*(sat_T-space_T)**4
	return -heatflow 		 #J/s
#a = IR_output(areas, emisivity, sat_T, stefan_boltz = stefan_boltz, space_T = space_T )



# INPUTS
units = 12	 	# CubeSat units
batts = 24	 	# battery units
panels = 5		# solar panel cubesat squares
alt = 345000	# km
mission_length= 5 #years
n = 10			# steps per orbit
orbits = 3 		# simulated orbits

comm_out = 2	 #W   2-4W
thrust_out = 6 	 #W

# ENVIRONMENT CONSTANTS
stefan_boltz = 5.67e-8 		# W/m^2/K^4
earth_r = 6378000			# km
myu = 3.986e14 				# m^3/s^2
solar_flux = 1368			# W/m^2
e_albedo = 0.3 				# reflected ratio 0.24-0.4 for SSO
earth_T = 256 				# 246K-256K
space_T = 2.7 				# K
rad_conv = 180/np.pi 		#57.3..
# SOME ORBIT AND SHADE SHIT
inclination = 96.75			 	# 96.66-96.84
arg_asc_node = -45 	 	 	 	# -45 and 45
tot_orbit_time = 2*np.pi*((earth_r+alt)**3/myu)**0.5  	 	# s
angle_of_sun_behind_pole = np.arccos(earth_r/(earth_r+alt)) # rads
shade_angle = 30/tot_orbit_time*2*np.pi  			  	 	# 30s of eclipse
EtoS_r = (4*np.pi*earth_r**2) / (4*np.pi*(alt+earth_r)**2)  # earth to sat-elite ratio for scaling 
rho = np.arcsin(EtoS_r**0.5)  		 	 		 	 	 	# sum altitude factor
Ka = 0.664 + 0.521*rho + 0.203*rho**2  	 	 	 	  	 	# sum albedo scaling factor



#SOLAR CELL AREA
# single cell dims 4.1 cm height, 8 cm  -1.96cm^2 (total 30cm^2)
# 3Ux2U solar panels have 20 solar cells (20*30cm^2 = 600cm^2  (90.5% eff. placement))
area_sc = 30*0.0001  #m^2
# GET SOLAR PANEL AREAS
solar_area_negX = 0  	#m^2
solar_area_posX = 0 	#1*10*area_sc
solar_area_posY = 20*area_sc + 4*20*area_sc*np.cos(45/rad_conv)
solar_area_negY = 0
solar_area_posZ = 0
solar_area_negZ = 20*area_sc + 4*20*area_sc*np.cos(45/rad_conv)
SP_areas = np.array([solar_area_posX, solar_area_negX,   solar_area_posY, solar_area_negY,\
					 solar_area_posZ, solar_area_negZ]).reshape((1,6)) 	 	 	# [+X, -X, +Y, -Y, +Z, -Z]

# AREAS OF FACES FOR SAT-ELITE
# BODY = 229.4mm W x 222.8mm H x 366mm L 	SOLAR PANELS = 212.8mm W x 2.1mm H x 366mm L
body_w = 229.4/1000.  		# m 
body_h = 222.8/1000.  		# m 
body_l = 366/1000.  		# m 
sp_w = 212.8/1000.  		# m 
sp_h = 2.1/1000.  		# m 
sp_l = 366/1000.  		# m 
sat_posX = 4*sp_h*sp_w + body_w*body_h  #m^2
sat_negX = sat_posX  	 	#m^2
angl = 28.5
sat_posY = body_l*body_h + 4*sp_w*sp_l*np.cos(angl/rad_conv) 	#m^2
sat_negY = sat_posY
sat_posZ = body_l*body_h + 4*sp_w*sp_l*np.sin(angl/rad_conv) 	#m^2
sat_negZ = sat_posZ 	 	#m^2
areas = np.array([sat_posX, sat_negX,   sat_posY, sat_negY,   sat_posZ, sat_negZ]).reshape((1,6)) # [+X, -X, +Y, -Y, +Z, -Z]




# HARDWARE CONSTANTS
unit_V = 0.1**3 			# m^3
unit_max_mass = 2 			# kg

panel_V_ps = (2.3*98*83)/1000**3 	# m^3
panel_mass_ps = 45/1000 	# kg

batt_max_Volt = 5
# m(x) = a + b*x 	V(x) = a + b*x 
batt_V_a = (96*92*16)/1000**3200 	# m^3
batt_V_b = (96*92*5.5)/1000**3		# m^3
batt_mass_a = 70/1000		# kg
batt_mass_b = 65/1000 		# kg
batt_mAh_pb = 3200 			# mAh
batt_temp_range = [-20,60] 	# °C
batt_capacity_pb = 154 		# Wh/kg

yearly_degradation = 1-0.0275
EOL_sp_eff = yearly_degradation**mission_length
sp_eff = 0.3 # GaAs solar panel efficiency

step_up_conv_eff = 0.95
battery_charger_eff = 0.90
harness_eff = 0.9
#http://spaceflight.com/wp-content/uploads/2015/05/201410-6U-20W-Solar-Panel-Datasheet.pdf
#for thermal decrease in power
Mbatts = batt_mass_a + batt_mass_b*batts 	# kg
batt_capacity = batt_capacity_pb * Mbatts * 3600 	# J

#MASSES AND SPECIFIC HEATS
masses = np.array([10,10]) # kg
spec_heats = np.array([1000,1000]) # J/kg/K
print('total mass =',np.sum(masses))
print('average specific heat =',np.sum(masses*spec_heats)/np.sum(masses))

# ABSORTIVITY AND REFLECTIVITY ARRAYS FOR SURFACES [+X, -X, +Y, -Y, +Z, -Z]
absortivity = np.zeros((1,6))
emisivity = np.zeros((1,6))
for i in range(np.shape(absortivity)[1]):
	absortivity[0,i]=0.5
	emisivity[0,i]=0.5




orbit_time = np.linspace(0, tot_orbit_time, num=n).reshape((n,1)) 	# time linspace for one orbit

panels_power = np.zeros((n,6)) 	 	# power created by diff. sat. sides [+X, -X, +Y, -Y, +Z, -Z]
EtoB_transform = np.zeros((n,3,3))  # EtoB transforations with varying orbit angle
sun_incidence = np.zeros((n,6))  	# 0-1(cosine of sun incidence angle) [+X, -X, +Y, -Y, +Z, -Z]
earth_incidence = np.zeros((n,6))   # 0-1(cosine of earth incidence angle) [+X, -X, +Y, -Y, +Z, -Z]
albedo_incidence = np.zeros((n,6))   # 0-1(cosine of albedo incidence angle) [+X, -X, +Y, -Y, +Z, -Z]
sun_vec_B = np.zeros((n,3)) 		# Sun vector transformed to body system along the orbit
earth_vec_B = np.zeros((n,3)) 		# Earth vector transformed to body system along the orbit
sun_visibility = np.zeros(np.shape(orbit_time)) 			# 0 or 1 for orbit linspace if the sun is visible
albedo_visibility = np.zeros(np.shape(orbit_time)) 			# 0 or 1 for orbit linspace if the albedo is visible
sun_vec_E = np.array([1,0,0]) 		# Earth-Sun vector in Earth coordinate system
earth_vec_V = np.array([0,0,1]) 	# C.G.- Earth vector in Velocity coordinate system

t_step = (orbit_time[1]-orbit_time[0])					# step in s
orbit_angle = orbit_time/tot_orbit_time*2*np.pi 		# angle linspace for one orbit (0-2pi) 0 starts in the middle of shadow cone
orbit_ang = orbit_angle * rad_conv 						# same shit but in degrees, just for plotting

# GETTING SHADE CONDITION ALONG ORBIT
for i in range(n):
	if (orbit_angle[i] - angle_of_sun_behind_pole + shade_angle) < 90/rad_conv or \
	   (orbit_angle[i] + angle_of_sun_behind_pole - shade_angle) > 270/rad_conv : 
		   sun_visibility[i] = 1
	if (orbit_angle[i]) < 90/rad_conv or \
	   (orbit_angle[i]) > 270/rad_conv : 
		   albedo_visibility[i] = 1

# GETTING SOLAR PANEL SUN EXPOSURE
arg_lat = orbit_ang
for i in range(n):
	vtob = VtoB(roll=0, pitch=0, yaw=0)
	EtoB_transform[i,:,:] = np.matmul( vtob, np.matmul( OtoV(arg_lat=arg_lat[i,0]) ,\
									   EtoO(arg_asc_node=arg_asc_node, inclination=inclination) ))
	sun_vec_B[i,:] = np.matmul(EtoB_transform[i,:,:], sun_vec_E )	
	sun_incidence[i,:] = sun_vec_B[i,0], -sun_vec_B[i,0], \
						   sun_vec_B[i,1], -sun_vec_B[i,1], \
						   sun_vec_B[i,2], -sun_vec_B[i,2]
						   
	earth_vec_B[i,:] = np.matmul(vtob, earth_vec_V)
	earth_incidence[i,:] = earth_vec_B[i,0], -earth_vec_B[i,0], \
						   earth_vec_B[i,1], -earth_vec_B[i,1], \
						   earth_vec_B[i,2], -earth_vec_B[i,2]
albedo_incidence = earth_incidence
sun_incidence[sun_incidence < 0] = 0 	 
earth_incidence[earth_incidence < 0] = 0 	


# INCIDENCES AND INSTANTANEOUS POWER PRODUCED BY SOLAR PANELS
sun_incidence = sun_incidence*sun_visibility 
albedo_incidence = albedo_incidence*albedo_visibility
panels_power = sun_incidence * SP_areas* solar_flux* EOL_sp_eff* sp_eff

# SWITCHING STARTING POINT SUCH THAT IT STARTS IN MIDNIGHT FOR SAT-ELITE
Placeholder = np.sum(panels_power,axis=1)
P_produced = np.zeros(np.shape(Placeholder))
P_produced[0:int(n/2)] = Placeholder[int(n/2):]
P_produced[int(n/2):] = Placeholder[0:int(n/2)]



# TIMES ALONG THE ORBIT
print('\n',less_dec( orbit_time[0]/60),'min -midnight for satellite\n',
	  less_dec( orbit_time[(np.abs((orbit_angle + angle_of_sun_behind_pole - shade_angle) - 90/rad_conv)).argmin()]/60),'min -sunrise for satellite\n',
	  less_dec( orbit_time[(np.abs(orbit_ang - 90)).argmin()]/60),'min -sunrise on Earth \n',	  
	  less_dec( orbit_time[(np.abs(orbit_ang - 180)).argmin()]/60),'min -noon for satellite\n',
	  less_dec( orbit_time[(np.abs(orbit_ang - 270)).argmin()]/60),'min -sunset on Earth \n',
	  less_dec( orbit_time[(np.abs((orbit_angle - angle_of_sun_behind_pole + shade_angle) - 270/rad_conv)).argmin()]/60),'min -sunset for satellite\n',
	  less_dec( orbit_time[-1]/60),'min -midnight for satellite\n')

# POWER CONSUMPTION ARRAY THROUGHOUT ORBIT (W)
P_usage = [5,10,15,70] # W
P_usage_times = [[0,93], [23,67], [19,24], [67,80]]# min
P_consumed = np.zeros(np.shape(orbit_time)) # W
def put_in_power(P_usage,P_usage_time, P_consumed, orbit_time):
	a = np.where( np.logical_and( orbit_time>=P_usage_time[0]*60, orbit_time<=P_usage_time[1]*60 ) )[0]
	P_consumed[a] = P_consumed[a] + P_usage
	return P_consumed
for i in range(len(P_usage)):
	put_in_power(P_usage[i], P_usage_times[i], P_consumed, orbit_time)



#plt.plot(orbit_time,P_consumed,orbit_time,P_produced)

# P_produced , P_consumed

solar_inputs = solar_input(areas, sun_incidence, absortivity, solar_flux = solar_flux )
E_IR_inputs = E_IR_input(areas, earth_incidence, emisivity, altitude=alt, earth_T=earth_T, stefan_boltz=stefan_boltz, EtoS_r=EtoS_r  )
albedo_inputs = albedo_input(areas, albedo_incidence, absortivity, altitude=alt, e_albedo = e_albedo, solar_flux = solar_flux, Ka=Ka)



T_start = 271 #K
charge_start = batt_capacity*0.8 #J
heatflows = np.zeros((n,6))
heatflows[:,0] = np.sum(solar_inputs, axis=1)
heatflows[:,1] = np.sum(E_IR_inputs, axis=1)
heatflows[:,2] = np.sum(albedo_inputs, axis=1)


for n_orbit in range(orbits):
	
	E_stored = np.zeros(np.shape(orbit_time)) # J
	sat_temp = np.zeros(np.shape(orbit_time)) # K
	sat_temp[0] = T_start
	E_stored[0] = charge_start
	heatflows[:,3:] = 0
	for i in range(1, n):
		eff_due_charging = (P_produced[i]*1+P_consumed[i]*battery_charger_eff)/(P_produced[i]+P_consumed[i])
		E_stored[i] = P_produced[i]*t_step*step_up_conv_eff*harness_eff - P_consumed[i]*t_step/eff_due_charging/harness_eff + E_stored[i-1]
		if E_stored[i] > batt_capacity: E_stored[i] = batt_capacity
		
		heatflows[i,3] = from_batteries((E_stored[i] - E_stored[i-1])/t_step)
		heatflows[i,4] = sys_output(thruster_on=False, comms_on=False, thrust_out=thrust_out, comm_out=comm_out)
		heatflows[i,5] = (np.sum(IR_output(areas, emisivity, sat_temp[i], stefan_boltz=stefan_boltz, space_T=space_T )))
		print(heatflows[i,:])
		delta_T = np.sum(heatflows[i,:]) * t_step / (np.sum(masses*spec_heats))
		sat_temp[i] = sat_temp[i-1] + delta_T

	
	if n_orbit==0:
		full_P_produced = P_produced
		charge = E_stored
		temp = sat_temp
		all_time = orbit_time
	else:		
		full_P_produced = np.concatenate((full_P_produced,P_produced), axis=0)
		charge = np.concatenate((charge,E_stored), axis=0)
		temp = np.concatenate((temp,sat_temp), axis=0)
		all_time = np.concatenate((all_time,orbit_time+all_time[-1]), axis=0)
	T_start = temp[-1]
	charge_start = E_stored[-1]

	

#plt.plot(all_time,full_P_produced)#,all_time,P_produced)
'''

def get_E(panels_power, P_consumed, batt_capacity, t_step):
	#a= np.zeros(np.shape(orbit_time))
	E_stored = np.zeros(np.shape(orbit_time)) # J
	batt_E_stored = np.zeros(np.shape(orbit_time)) # J
	
	for i in range(1, n):
		eff_due_charging = (P_produced[i]*1+P_consumed[i]*battery_charger_eff)/(P_produced[i]+P_consumed[i])
		E_stored[i] = P_produced[i]*t_step*step_up_conv_eff*harness_eff - P_consumed[i]*t_step/eff_due_charging/harness_eff + E_stored[i-1]
		

	return E_stored, batt_E_stored

E_produced, batt_E_stored = get_E(panels_power, P_consumed, batt_capacity, t_step)



print(E_produced[-1]/tot_orbit_time)
#plt.plot(orbit_ang,panels_power)
#plt.plot(orbit_ang,sun_visibility)
#plt.plot(orbit_ang, E_produced, orbit_ang, batt_E_stored)
'''
'''

plt.figure(1)
rows=2
columns=1
#plt.subplot(int(str(rows)+str(columns)+str(1)))
plt.plot(orbit_ang, P_consumed, orbit_ang, panels_power)
plt.title('Power usage')
plt.xlabel(' Theta [°] ')
plt.ylabel(' Power [W]')
plt.legend(['Power used','Power produced'], loc='best',fontsize = 'small',borderpad=0.2,labelspacing=0.05) #x-small
plt.grid(True)

plt.subplot(int(str(rows)+str(columns)+str(2)))
plt.plot(orbit_ang, batt_E_stored, orbit_ang, batt_E_stored*0+batt_capacity )
plt.title('Energy storage')
plt.xlabel(' Theta [°] ')
plt.ylabel(' Energy [J]')
plt.legend(['Energy stored','Max. energy stored'], loc='best',fontsize = 'small',borderpad=0.2,labelspacing=0.05)
plt.grid(True)
plt.figure(1).tight_layout()

'''