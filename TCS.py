import numpy as np
import matplotlib.pyplot as plt
import time
import os


altitude = 340000#m
comm_out = 10   #W
thrust_out = 50 #W




# ENVIRONMENT CONSTANTS
stefan_boltz = 5.67e-8 		# W/m^2/K^4
earth_r = 6378000 				# m
solar_flux = 1368			# W/m^2
e_albedo = 0.3 				# reflected ratio 0.24-0.4 for SSO
earth_T = 256 				# 246K-256K
space_T = 2.7 				# K
rad_conv = 180/np.pi 		#57.3..


print((4*np.pi*earth_r**2)/(4*np.pi*(altitude+earth_r)))

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
areas = np.array([sat_posX, sat_negX,   sat_posY, sat_negY,   sat_posZ, sat_negZ]) # [+X, -X, +Y, -Y, +Z, -Z]


# ABSORTIVITY AND REFLECTIVITY ARRAYS
absortivity = np.zeros((6,1))
emisivity = np.zeros((6,1))
for i in range(np.shape(absortivity)[0]):
	absortivity[i]=0.5
	emisivity[i]=0.5

# HEAT CAPACITANCE
h_cap = 10 # J/K     kg*m^2 / (s^2*K)

def solar_input(areas, sun_incidence, absortivity, solar_flux = 1368 ):
	heatflow = areas*sun_incidence*absortivity*solar_flux
	return heatflow
#a = solar_input(areas, sun_incidence, absortivity, solar_flux = solar_flux )
	
def E_IR_input(areas, earth_incidence, emisivity, altitude = 340, earth_T = 256, stefan_boltz = 5.67e-8  ):
	earth_flux = earth_T*stefan_boltz #FINISH
	heatflow = areas*earth_incidence*emisivity*earth_flux
	return heatflow 		 #J/s
#a = E_IR_input(areas, earth_incidence, emisivity, altitude = altitude, earth_T = earth_T, stefan_boltz = stefan_boltz  )
	
def albedo_input(areas, earth_incidence, absortivity, altitude = 340, e_albedo = 0.3, solar_flux = 1368):
	albedo_flux = solar_flux*e_albedo #FINISH
	heatflow = areas*earth_incidence*emisivity*albedo_flux
	return heatflow 		 #J/s
#a = albedo_input(areas, earth_incidence, absortivity, altitude = altitude, e_albedo = e_albedo, solar_flux = solar_flux)

def power_gen(areas, sun_incidence, absortivity, solar_flux):
	heatflow = areas*sun_incidence*absortivity*solar_flux
	return heatflow

def internal_input(P_consumption, P_production, P_to_storage):
	heatflow = P_production - P_to_storage
	return heatflow 		 #J/s
#a = internal_input(P_consumption, P_production, P_to_storage)
	
def sys_output(thruster_on = False, comms_on = False, thrust_out = 30, comm_out = 10):
	heatflow = 0
	if thruster_on:
		heatflow+=thrust_out
	if comms_on:
		heatflow+=comm_out	
	return heatflow 		 #J/s
#a = sys_output(thruster_on = False, comms_on = False, thrust_out = thrust_out, comm_out = comm_out)

def IR_output(areas, emisivity, sat_T, stefan_boltz = 5.67e-8, space_T = 3 ):
	heatflow = emisivity*areas*stefan_boltz*(sat_T-space_T)**4
	return heatflow 		 #J/s
#a = IR_output(areas, emisivity, sat_T, stefan_boltz = stefan_boltz, space_T = space_T )













































