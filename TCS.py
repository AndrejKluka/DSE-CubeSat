import numpy as np
import matplotlib.pyplot as plt
import time
import os






def time_print(start, stop, mtext):
	now_time = time.process_time()
	if start==None and stop==None:
		return now_time
	elif stop==None:
		t = int((now_time - start) * 100) / 100.
		print('T+'+str(int(t/3600))+':'+str(int(t%3600/60))+':'+str(t%60),' process took:',\
		      str(int(t/3600))+':'+str(int(t%3600/60))+':'+str(t%60), 'sec -- {}'.format(mtext))
		return now_time
	else:
		t = int((now_time - start) * 100) / 100.
		pt = int((now_time - stop) * 100) / 100.
		print('T+'+str(int(t/3600))+':'+str(int(t%3600/60))+':'+str(t%60),' process took:',\
		      str(int(pt/3600))+':'+str(int(pt%3600/60))+':'+str(pt%60), 'sec -- {}'.format(mtext))
		return now_time



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
def sys_output(state_mask, thrust_out = 30, comm_out = 10):
	heatflow = 0
	if state_mask[0]: heatflow+=comm_out
	if state_mask[1]: heatflow+=thrust_out	
	return -heatflow 		 #J/s
#a = sys_output(thruster_on = False, comms_on = False, thrust_out = thrust_out, comm_out = comm_out)
def IR_output(areas, emisivity, sat_T, stefan_boltz = 5.67e-8, space_T = 2.7 ):
	heatflow = areas*emisivity*stefan_boltz*(sat_T**4 - space_T**4)
	return -heatflow 		 #J/s
#a = IR_output(areas, emisivity, sat_T, stefan_boltz = stefan_boltz, space_T = space_T )













































