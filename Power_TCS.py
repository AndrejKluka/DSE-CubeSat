import numpy as np
import matplotlib.pyplot as plt
import time
import os

def less_dec(a,decs=2):
	b = 10**decs
	return int((a) * b) / float(b)
rad_conv = 180/np.pi
''' different coordinate systems definitions ***
(all coordinate systems are right handed)

Earth coordinate system [xe, ye, ze] 
		- origin in Earth's center 
		- xe always towards the sun center 
		- ze towards north pole

Orbit coordinate system [xo, yo, zo] 
		- origin in Earth's center, 
		- xo always towards equatorial ascending node
		- zo so that satellite orbits  in a positive way around it

Body carried north coordinate system [xn, yn, zn] 
		- origin in CubeSat's c.g. 
		- xn always towards the north pole
		- zn towards Earth's center

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
'''
arg_asc_node = 45   # Sun-Earths_center- equatorial ascending node (positive towards east)
inclination = 96  #angle between orbital and equatorial plane
e2o = arg_asc_node
e0o= inclination
print('Earth to Orbit\n',axis_tf((e2o,2),(e0o,0)))


arg_lat = 10 # orbital angle ( asc_node - earth's center - body) positive is going north from asc node

o2v = 90
o0v = -90
o1v = -arg_lat
print('Orbit to Velocity\n',axis_tf((o2v,2), (o0v,0), (o1v,1)))

roll = 0
pitch = 0
yaw = 0
v0b = roll
v1b = pitch
v2b = yaw
print('Velocity to Body\n',axis_tf((v0b,0), (v1b,1), (v2b,2)))
'''
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

'''
x=np.array([1,0,0])
first=EtoO(arg_asc_node=0, inclination=0)
sec = OtoV(arg_lat=0)
thir= VtoB(roll=0, pitch=0, yaw=0)
x1= np.matmul(first, x)
x2= np.matmul(sec, x1)
x3= np.matmul(thir, x2)

print('\n','\n',first,'\n\n',sec,'\n\n',thir,'\n\n x=',x,'\n x1=',x1,'\n x2=',x2,'\n x3=',x3,'\n','\n')
a=np.array([1,0,0])
#print(np.cross(b,a))
print(np.matmul(a,x3))
'''

# INPUTS
units = 1	 	# CubeSat units
batts = 1	 	# battery units
panels = 5		# solar panel cubesat squares
alt = 380	 	# km
mission_length= 5 #years

#SOLAR CELL AREA
# single cell dims 4.1 cm height, 8 cm  -1.96cm^2 (total 30cm^2)
# 3Ux2U solar panels have 20 solar cells (20*30cm^2 = 600cm^2  (90.5% eff. placement))

area_sc = 30*0.0001  #m^2

solar_area_top = 4*20*area_sc + 2*20*area_sc*np.cos(45/rad_conv)

solar_area_negX = 0  #m^2
solar_area_posX = 0#1*10*area_sc

solar_area_posY = 20*area_sc + 4*20*area_sc*np.cos(45/rad_conv)#5*20*area_sc # np.cos(45/rad_conv) * solar_area_top
solar_area_negY = 0

solar_area_posZ = 0
solar_area_negZ = 20*area_sc + 4*20*area_sc*np.cos(45/rad_conv)#20*area_sc#np.cos(45/rad_conv) * solar_area_top


# ENVIRONMENT CONSTANTS
stefan_boltz = 5.67e-8 		# W/m^2/K^4
earth_r = 6378 				# km

solar_flux = 1368			# W/m^2
e_albedo = 0.3 				# reflected ratio



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



# PRELIM HARDWARE CALCULATIONS
Vtot = units * unit_V 	    # m^3
Vpanels = panel_V_ps*panels # m^3
Vbatts = batt_V_a + batt_V_b*batts
print('Solar panels: ',int(Vpanels*1e7)/1e7,'m^3   ', int(Vpanels/Vtot*10000)/100,'% total maximum volume ',\
	  '\nBatteries: ',int(Vbatts*1e7)/1e7,'m^3   ', int(Vbatts/Vtot*10000)/100,'% total maximum volume \n')

Mtot = units * unit_max_mass 				# kg
Mpanels = panels * panel_mass_ps 			# kg
Mbatts = batt_mass_a + batt_mass_b*batts 	# kg
print('Solar panels: ',int(Mpanels*1e4)/1e4,'kg   ', int(Mpanels/Mtot*10000)/100,'% total maximum mass ',\
	  '\nBatteries: ', int(Mbatts*1e5)/1e5,'kg   ', int(Mbatts/Mtot*10000)/100,'% total maximum mass ')
batt_capacity = batt_capacity_pb * Mbatts * 3600 	# J



# SOME ORBIT AND SHADE SHIT
inclination = 96.75 # 96.66-96.84
arg_asc_node = -65 # -45 and 45
sun_rad_size = np.arctan(695510/149597871) # rad size of sun from Earth
penumbra_effect = 8			 # s
earth_flattening = 22		 # s
atmosphere_earth = 8		 # s
tot_orbit_time = 90.52*60	 # s
angle_of_sun_behind_pole = np.arccos(earth_r/(earth_r+alt)) #rads
shade_angle = 30/tot_orbit_time*2*np.pi  # 30s of eclipse
omega = 0 / rad_conv 		 # rads between sun-earth-equatorial ascension node


def wow(angl):
	solar_area_negX = 0  #m^2
	solar_area_posX = 0#1*10*area_sc
	
	solar_area_posY = 20*area_sc + 4*20*area_sc*np.cos(angl/rad_conv)#5*20*area_sc # np.cos(45/rad_conv) * solar_area_top
	solar_area_negY = 0
	
	solar_area_posZ = 0
	solar_area_negZ = 20*area_sc + 4*20*area_sc*np.sin(angl/rad_conv)#20*area_sc#np.cos(45/rad_conv) * solar_area_top
	
	
	
	n = 200
	orbit_time = np.linspace(0, tot_orbit_time, num=n) 	# time linspace for one orbit
	
	panels_power = np.zeros((np.shape(orbit_time)[0],6)) # power created by diff. sat. sides [+X, -X, +Y, -Y, +Z, -Z]
	EtoB_transform = np.zeros((np.shape(orbit_time)[0],3,3)) # EtoB transforations with varying orbit angle
	panels_exposure = np.zeros((np.shape(orbit_time)[0],6))  # 0-1(cosine of incidence angle) [+X, -X, +Y, -Y, +Z, -Z]
	B_sun_vec = np.zeros((np.shape(orbit_time)[0],3)) # Sun vector transformed to body system along the orbit
	sun_visibility = np.zeros(np.shape(orbit_time)) 			# 0 or 1 for orbit linspace if the sun is visible
	SP_areas = [solar_area_posX, solar_area_negX,   solar_area_posY, solar_area_negY,   solar_area_posZ, solar_area_negZ]
	E_sun_vec = np.array([1,0,0])
	
	t_step = (orbit_time[1]-orbit_time[0])					# step in s
	orbit_angle = orbit_time/tot_orbit_time*2*np.pi 		# angle linspace for one orbit (0-2pi) 0 starts in the middle of shadow cone
	orbit_ang = orbit_angle * rad_conv 						# same shit but in degrees, just for plotting
	
	# getting shade
	for i in range(n):
		if (orbit_angle[i] - angle_of_sun_behind_pole + shade_angle) < 90/rad_conv or \
		   (orbit_angle[i] + angle_of_sun_behind_pole - shade_angle) > 270/rad_conv : 
			   sun_visibility[i]=1
	
	# getting panel exposure
	arg_lat = orbit_ang
	for i in range(n):
		EtoB_transform[i,:,:] = np.matmul( VtoB(roll=0, pitch=0, yaw=0) ,np.matmul( OtoV(arg_lat=arg_lat[i]) ,\
										   EtoO(arg_asc_node=arg_asc_node, inclination=inclination) ))
		B_sun_vec[i,:] = np.matmul(EtoB_transform[i,:,:], E_sun_vec )	
		panels_exposure[i,:] = B_sun_vec[i,0], -B_sun_vec[i,0], \
							   B_sun_vec[i,1], -B_sun_vec[i,1], \
							   B_sun_vec[i,2], -B_sun_vec[i,2],
	panels_exposure[panels_exposure < 0] = 0 	 
	
	# getting instantaneous power
	for i in range(np.shape(panels_exposure)[1]):
		panels_exposure[:,i] = panels_exposure[:,i]*sun_visibility 
		panels_power[:,i] = panels_exposure[:,i] * SP_areas[i]	
	panels_power = panels_power* solar_flux* EOL_sp_eff* sp_eff
	
	
	'''
	for i in range(6):
			plt.plot(orbit_ang, panels_power[:,i])
	plt.plot(orbit_ang, np.sum(panels_power,axis=1))
	plt.legend(['+X', '-X', '+Y', '-Y', '+Z', '-Z', 'sum'])
	
	# 53.33
	print('sum= ',np.sum(panels_power)/n)
	'''
	return np.sum(panels_power)/n


nn=15
powaa = np.zeros((nn))
angl = np.linspace(0, 90, num=nn)
for i in range(nn):
	powaa[i] = wow(angl[i])

plt.plot(angl, powaa)
#plt.plot(orbit_ang, np.sum(panels_power,axis=1))
#plt.legend(['+X', '-X', '+Y', '-Y', '+Z', '-Z', 'sum'])

'''
ech = np.ones((1,6))
for i in range(np.shape(panels_exposure)[1]):
	ech[0,i] = np.sum(panels_power[:,i])
	#plt.plot(orbit_ang,panels_power[:,i])
#plt.legend(['+X', '-X', '+Y', '-Y', '+Z', '-Z'])
ech = ech/np.sum(ech)	
nn=150
asc_node = np.linspace(0, -90, num=nn)
effectiveness = np.zeros((nn,6))
for i in range(nn):
	effectiveness[i,:] = wow(asc_node[i])
for i in range(6):
		plt.plot(asc_node,effectiveness[:,i])
plt.legend(['+X', '-X', '+Y', '-Y', '+Z', '-Z'])
#the sun angle of the ascending node
plt.title('Effectiveness of various satellite faces')
plt.xlabel('Sun angle of ascending node [°]')
plt.ylabel('Effectiveness [-]')
plt.show()
print(effectiveness)
'''

'''
#plt.plot(orbit_ang, P_consumed, orbit_ang, panels_power)

print('\n',less_dec( orbit_time[0]/60),'min -midnight for satellite\n',
	  less_dec( orbit_time[(np.abs((orbit_angle + angle_of_sun_behind_pole - shade_angle) - 90/rad_conv)).argmin()]/60),'min -sunrise for satellite\n',
	  less_dec( orbit_time[(np.abs(orbit_ang - 90)).argmin()]/60),'min -sunrise on Earth \n',	  
	  less_dec( orbit_time[(np.abs(orbit_ang - 180)).argmin()]/60),'min -noon for satellite\n',
	  less_dec( orbit_time[(np.abs(orbit_ang - 270)).argmin()]/60),'min -sunset on Earth \n',
	  less_dec( orbit_time[(np.abs((orbit_angle - angle_of_sun_behind_pole + shade_angle) - 270/rad_conv)).argmin()]/60),'min -sunset for satellite\n',
	  less_dec( orbit_time[-1]/60),'min -midnight for satellite\n')



P_usage = [1,3,4,4] # W
P_usage_times = [[0,93], [23,67], [19,24], [67,73]]# min
P_consumed = np.zeros(np.shape(orbit_time)) # W
def put_in_power(P_usage,P_usage_time, P_consumed, orbit_time):
	a = np.where( np.logical_and( orbit_time>=P_usage_time[0]*60, orbit_time<=P_usage_time[1]*60 ) )[0]
	P_consumed[a] = P_consumed[a] + P_usage
	return P_consumed


for i in range(len(P_usage)):
	put_in_power(P_usage[i], P_usage_times[i], P_consumed, orbit_time)
P_consumed=P_consumed*0

def get_E(panels_power, P_consumed, batt_capacity, t_step):
	E_produced = np.zeros(np.shape(orbit_time)) # J
	batt_E_stored = np.zeros(np.shape(orbit_time)) # J
	for i in range(1, len(orbit_angle)):
		E_produced[i] = panels_power[i]*t_step - P_consumed[i]*t_step + E_produced[i-1]
		
	batt_E_stored = E_produced + batt_capacity
	batt_E_stored[batt_E_stored > batt_capacity] = batt_capacity
	if batt_E_stored[-1] < batt_capacity: print('\nConsuming more power overall than producing!')
	return E_produced, batt_E_stored

E_produced, batt_E_stored = get_E(panels_power, P_consumed, batt_capacity, t_step)

print(E_produced[-1]/tot_orbit_time)
#plt.plot(orbit_ang,panels_power)
#plt.plot(orbit_ang,sun_visibility)
#plt.plot(orbit_ang, E_produced, orbit_ang, batt_E_stored)



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