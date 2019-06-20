import numpy as np
import matplotlib.pyplot as plt
import time
import os
from TCS import IR_output, sys_output, from_batteries, albedo_input, E_IR_input, solar_input,\
VtoB, OtoV, EtoO, tf_matrix, axis_tf, less_dec, time_print
start = time_print(None, None, None)
stop = time_print(start, None, 'Modules imported')

#http://www.iosrjournals.org/iosr-jeee/Papers/Vol12%20Issue%203/Version-5/G1203053745.pdf

'''

-proper modes
-proper masses and specific heats
-proper absorbtivity
-proper emisivity
-albedo heating coefficient?
-battery heating?
-separate panels and body?

-find EPS
-get efficiencies
-find possible TCS surfaces

'''





# INPUTS
battery_packs = 2
side_units = 6
alt = 340000	# m
mission_length= 5 #years
n = 7500		# steps per orbit
orbits = 15	# simulated orbits
case = 'cold'

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



#SOLAR CELL AREA
# single cell dims 4.1 cm height, 8 cm  -1.96cm^2 (total 30cm^2)
# 3Ux2U solar panels have 20 solar cells (20*30cm^2 = 600cm^2  (90.5% eff. placement))
area_sc = 30*0.0001  #m^2
# GET SOLAR PANEL AREAS
angl=26.5
solar_area_negX = 0  	#m^2
solar_area_posX = 0 	#1*10*area_sc
solar_area_posY = 20*area_sc + (4*20)*area_sc*np.cos(angl/rad_conv)
solar_area_negY = 0
solar_area_posZ = 0
solar_area_negZ = 20*area_sc + (4*20)*area_sc*np.sin(angl/rad_conv)
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
sat_posY = body_l*body_h + 4*sp_w*sp_l*np.cos(angl/rad_conv) 	#m^2
sat_negY = sat_posY
sat_posZ = body_l*body_h + 4*sp_w*sp_l*np.sin(angl/rad_conv) 	#m^2
sat_negZ = sat_posZ 	 	#m^2
areas = np.array([sat_posX, sat_negX,   sat_posY, sat_negY,   sat_posZ, sat_negZ]).reshape((1,6)) # [+X, -X, +Y, -Y, +Z, -Z]




# HARDWARE CONSTANTS
unit_V = 0.1**3 			# m^3
unit_max_mass = 2 			# kg

sp_mass = 350*side_units/1000 # kg
fr_sp_mass = 21.3*1.8*36.6*1.85*side_units/1000 # kg

batt_mass = 0.5*battery_packs 	 	# kg
batt_capacity = 77*3600*battery_packs 	# J
lion_mass = 48*8*battery_packs/1000 # kg

yearly_degradation = 1-0.0275
EOL_sp_eff = yearly_degradation**mission_length
sp_eff = 0.295 # GaAs solar panel efficiency


eff_mppt = 0.90
eff_converter = 0.98
eff_charger = 0.96
eff_battery = 0.91
eff_discharge = 0.97
eff_harness = 0.90
eff_straight = eff_mppt* eff_converter* eff_harness
eff_from_batts = eff_battery* eff_discharge* eff_harness

#http://spaceflight.com/wp-content/uploads/2015/05/201410-6U-20W-Solar-Panel-Datasheet.pdf
#for thermal decrease in power


#MASSES AND SPECIFIC HEATS
#                    kg        J/kg/K 	
masses = np.array([[lion_mass,  830], #EPS
				   [batt_mass-lion_mass, 930], #EPS
				   [sp_mass, 1600], #EPS
				   [0.4, 	930], #EPS
				   [0.6,	800], #EPS
				   [1.6,	960], #STRUCTURE
				   [1.74, 	822], #PROP
				   [10, 	750], #payload
				   [0.55,	900], #TT
				   [0.094,	1000],#CDH
				   [1.440,	614], #ADCS
				   [0.2, 	1000]]) # thermal??

spec_heats = masses[:,1] # J/kg/K
masses = masses[:,0]
print('total mass =',less_dec(np.sum(masses)))
print('average specific heat =',less_dec(np.sum(masses*spec_heats)/np.sum(masses)))

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
panels_power = sun_incidence * SP_areas* solar_flux* EOL_sp_eff* sp_eff
P_produced = np.sum(panels_power,axis=1)



	
# TIMES ALONG THE ORBIT
print('\n',less_dec( orbit_time[0]/60),'min -noon for satellite\n',
	  less_dec( orbit_time[albedo_visibility.argmin()]/60),'min -sunset on Earth \n',	
	  less_dec( orbit_time[sun_visibility.argmin()]/60),'min -sunset for satellite\n',	    
	  less_dec( orbit_time[(np.abs(orbit_ang - 180)).argmin()]/60),'min -midnight for satellite\n',
	  less_dec( orbit_time[::-1][sun_visibility[::-1].argmin()]/60),'min -sunrise for satellite\n',
	  less_dec( orbit_time[::-1][albedo_visibility[::-1].argmin()]/60),'min -sunrise on Earth \n',
	  less_dec( orbit_time[-1]/60),'min -noon for satellite\n')



# POWER CONSUMPTION ARRAY THROUGHOUT ORBIT (W)
# Hot= hot standby, 	Delq= deliquifying propellant, 	Des= desaturation
comps_dict = dict(TC  =dict( On=12 , Off=0.5, 					minT=-40,	maxT=70),
				  PROP=dict( On=80 , Hot=7, Delq=16, Off=0, 	minT=-20,	maxT=50),
				  ADCS=dict( On=7 , Coms=6, Off=0 , Des=7, 		minT=-10,	maxT=60),
				  CDH =dict( On=0.4 , Off=0 , 			 		minT=-25,	maxT=60),
				  PAYLOAD=dict( On=30 , Off=0,	 				minT=15,	maxT=25))

modes = dict(imaging = 		[['TC',''],		['PROP','Hot'],	['ADCS','On'],	['CDH','On'],	['PAYLOAD','On']],
			 desaturation = [['TC',''],		['PROP','Hot'],	['ADCS','Des'],	['CDH','On'],	['PAYLOAD','']],
			 communication =[['TC','On'],	['PROP','Hot'],	['ADCS','Coms'],['CDH','On'],	['PAYLOAD','']],
			 thrusting=		[['TC',''],		['PROP','On'],	['ADCS','On'],	['CDH','On'],	['PAYLOAD','']],
			 switching=		[['TC',''],		['PROP','Hot'],	['ADCS','On'],	['CDH','On'],	['PAYLOAD','']])

# imaging, desaturation, communication, thrusting, switching
mode_l = dict(imaging =17.404, desaturation =31.052, communication=4.95, thrusting=3.7565, switching= 0.5)
mode_schedule = [['thrusting', 0.5],
			  ['switching', 1],
			  ['imaging', 	1],
			  ['switching', 1],
			  ['communication', 1],
			  ['switching', 1],
			  ['thrusting', 1],
			  ['switching',	1],
			  ['desaturation',1],
			  ['switching',	1],
			  ['thrusting',	1],
			  ['switching',	1],
			  ['communication',	1],
			  ['switching',	1],
			  ['imaging',	1],
			  ['switching',	1],
			  ['thrusting', 0.5]]

mode_times =[[mode_schedule[0][0],0,mode_schedule[0][1]*mode_l[mode_schedule[0][0]]]]
for i in range(1,len(mode_schedule)):
	mode_times.append([mode_schedule[i][0],  mode_times[i-1][2], mode_times[i-1][2] + mode_schedule[i][1]*mode_l[mode_schedule[i][0]]])
	if i == len(mode_schedule)-1 : mode_times[i][2] = tot_orbit_time/60
def put_in_power(comps_dict, modes, mode_times, orbit_time):
	P_usage = np.zeros((n,len(comps_dict)))	
	state_mask = np.zeros((n,len(comps_dict)))	
	for mode in mode_times:
		for i in range(len(modes[mode[0]])):
			comp = modes[mode[0]][i]
			if not comp[1]=='':
				if mode[2]>mode[1]:
					a = np.where( np.logical_and( orbit_time>=mode[1]*60, orbit_time<=mode[2]*60 ) )[0]					
					P_usage[a,i] = comps_dict[comp[0]][comp[1]]
					if not comp[1]=='Off': state_mask[a,i] = 1
				else:
					a = np.where(orbit_time>=mode[1]*60)[0]		
					b = np.where(orbit_time<=mode[2]*60)[0]		
					P_usage[a,i] = comps_dict[comp[0]][comp[1]]		
					P_usage[b,i] = comps_dict[comp[0]][comp[1]]	
					if not comp[1]=='Off':
						state_mask[a,i] = 1
						state_mask[b,i] = 1
	return P_usage, state_mask


P_usage, state_mask = put_in_power(comps_dict, modes, mode_times, orbit_time)
P_consumed = np.sum(P_usage, axis=1)*1.1
#comp_temps = np.zeros((np.shape(state_mask)))	


solar_inputs = solar_input(areas, sun_incidence, absortivity, solar_flux = solar_flux )
E_IR_inputs = E_IR_input(areas, earth_incidence, emisivity, altitude=alt, earth_T=earth_T, stefan_boltz=stefan_boltz, EtoS_r=EtoS_r  )
albedo_inputs = albedo_input(areas, albedo_incidence, absortivity, altitude=alt, e_albedo = e_albedo, solar_flux = solar_flux, Ka=1)


T_start = 281 #K
charge_start = batt_capacity*0.95 #J
heatflows = np.zeros((n,6))
heatflows[:,0] = np.sum(solar_inputs, axis=1)
heatflows[:,1] = np.sum(E_IR_inputs, axis=1)
heatflows[:,2] = np.sum(albedo_inputs, axis=1)


stop = time_print(start, stop, 'Starting loops')
# MAIN LOOP FOR ORBITAL SIMULATION
for n_orbit in range(orbits):	
	E_stored = np.zeros(np.shape(orbit_time)) # J
	sat_temp = np.zeros(np.shape(orbit_time)) # K
	sat_temp[0] = T_start
	E_stored[0] = charge_start
	heatflows[:,3:] = 0
	for i in range(1, n):
		
		if P_produced[i] < 1e-5 :
			E_stored[i] = -P_consumed[i]*t_step / eff_from_batts + E_stored[i-1]	
		
		elif P_produced[i]*eff_straight > P_consumed[i]:
			E_stored[i] = (P_produced[i]*eff_straight - P_consumed[i]) * t_step * eff_charger + E_stored[i-1]
		
		else:
			E_stored[i] = (P_produced[i]*eff_straight - P_consumed[i]) * t_step / eff_from_batts + E_stored[i-1]

		if E_stored[i] > batt_capacity: E_stored[i] = batt_capacity
		
		
		
		heatflows[i,3] = from_batteries((E_stored[i] - E_stored[i-1])/t_step)
		heatflows[i,4] = sys_output(state_mask[i-1,:], thrust_out=thrust_out, comm_out=comm_out)
		heatflows[i,5] = np.sum(IR_output(areas, emisivity, sat_temp[i-1], stefan_boltz=stefan_boltz, space_T=space_T ))
		delta_T = np.sum(heatflows[i,:]) * t_step / (np.sum(masses*spec_heats))
		sat_temp[i] = sat_temp[i-1] + delta_T

	
	if n_orbit==0:
		full_P_p_c = np.concatenate((P_produced.reshape((n,1)),P_consumed.reshape((n,1))), axis=1)
		charge = E_stored
		temp = sat_temp
		all_time = orbit_time
	else:		
		full_P_p_c = np.concatenate((full_P_p_c, np.concatenate((P_produced.reshape((n,1)),P_consumed.reshape((n,1))), axis=1)), axis=0)
		charge = np.concatenate((charge,E_stored), axis=0)
		temp = np.concatenate((temp,sat_temp), axis=0)
		all_time = np.concatenate((all_time,orbit_time+all_time[-1]), axis=0)
	T_start = sat_temp[-1]
	charge_start = E_stored[-1]

	
DoD = (1 - (1 - np.min(charge[int(np.shape(charge)[0]*0.6):,0])/batt_capacity)*1.1)
print('\nDoD =',less_dec(DoD,decs=3))
mintemp = np.min(temp[int(np.shape(charge)[0]*0.6):,0])-CtoK
maxtemp = np.max(temp[int(np.shape(charge)[0]*0.6):,0])-CtoK
print('\n Total temp range ',less_dec(mintemp),'°C :',less_dec(maxtemp),'°C\n')
print('Average power production ',less_dec(np.sum(P_produced)/np.shape(P_produced)[0]))
print('Average power consumption ',less_dec(np.sum(P_consumed)/np.shape(P_consumed)[0]))
for i in range(len(comps_dict)):
	surv = ''
	temps = (state_mask*sat_temp)[:,i]
	comp = modes['imaging'][i][0]
	try: 
		minT = np.min(temps[temps!=0])-CtoK
		maxT = np.max(temps[temps!=0])-CtoK
		if minT<comps_dict[comp]['minT']	or maxT>comps_dict[comp]['maxT']: 
			surv='subsystem did not operate all the time'
		print(comp,	' experienced',less_dec(minT),'°C :',less_dec(maxT),'°C  ',surv)
	except: 
		print(comp,' was not on')

'''
plt.plot(all_time,full_P_p_c)
plt.plot(all_time,temp-CtoK)
plt.plot(orbit_time/60,(-charge[:n]+charge[1:n+1])/t_step)
plt.plot(all_time,charge)
plt.plot(orbit_time,panels_power,  orbit_time,P_produced)
plt.plot(orbit_time,P_produced,  orbit_time,P_consumed)
plt.plot(orbit_time,P_produced-P_consumed)
plt.plot(orbit_time,heatflows[:,:3])
plt.plot(orbit_time,solar_inputs)
plt.plot(orbit_time,E_IR_inputs)
plt.plot(orbit_time,albedo_inputs)



plt.plot(orbit_time/60,panels_power)
plt.title('Power production at i=96.75°, b=-45° orbit')
plt.xlabel(' Orbit time [min] ')
plt.ylabel(' Power [W]')
plt.legend(['+X', '-X', '+Y', '-Y', '+Z', '-Z'], loc='best',fontsize = 'medium',borderpad=0.2,labelspacing=0.05) #x-small


#, orbit_time/60,(-charge[:n]+charge[1:n+1])/t_step
plt.plot(orbit_time/60,P_produced,  orbit_time/60,P_consumed)
plt.title('Power production and consumption at i=96.75°, b=-45° orbit')
plt.xlabel('Orbit time [min]')
plt.ylabel('Power [W]')
plt.legend(['Production', 'Consumption'], loc='best',fontsize = 'medium',borderpad=0.2,labelspacing=0.05) #x-small

plt.plot(orbit_time/60,P_consumed)
plt.title('Power consumption')
plt.xlabel('Orbit time [min]')
plt.ylabel('Power [W]')
plt.legend(['Consumption'], loc='best',fontsize = 'medium', borderpad=0.2, labelspacing=0.05) #x-small



a= solar_inputs + E_IR_inputs + albedo_inputs
plt.plot(orbit_time/60,a)
#plt.plot(orbit_time/60,solar_inputs,  orbit_time/60,E_IR_inputs,  orbit_time/60,albedo_inputs)
plt.title('Thermal external inputs')
plt.xlabel(' Orbit time [min] ')
plt.ylabel(' Thermal external input [W]')
plt.legend(['+X', '-X', '+Y', '-Y', '+Z', '-Z'], loc='best',fontsize = 'medium',borderpad=0.2,labelspacing=0.05) #x-small

plt.plot(all_time/60,temp-CtoK)
plt.title('Satellite temperature')
plt.xlabel('Orbit time [min]')
plt.ylabel('T [°C]')






plt.plot(orbit_time/60,sun_incidence)
plt.title('i=96.75°, b=-45°')
plt.xlabel('Orbit time [min]')
plt.ylabel('cos(i) [-]')
plt.legend(['+X = 15.85%', '- X = 15.85%', '+Y = 44.02%', '- Y = 0%', '+Z = 2.26%', '- Z = 22.01%'], loc='best',fontsize = 'medium',borderpad=0.2,labelspacing=0.05) #x-small


np.sum(sun_incidence,axis=0)/n/np.sum(np.sum(sun_incidence,axis=0)/n)

sun_incidence









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


stop = time_print(start, stop, 'Finished')