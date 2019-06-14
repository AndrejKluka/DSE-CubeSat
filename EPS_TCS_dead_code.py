
# COORDINATE SYSTEMS DEFINITIONS
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


# TRANSFORMATION DEFINITIONS
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


# TRANSFORMATION TESTING
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


# GETTING THE EFFECTIVENESS FOR DIFF FACES
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


# GETTING THE OPTIMAL ANGLE FO SOLAR PANELS
'''
return np.sum(panels_power)/n
nn=500
powaa = np.zeros((nn))
angl = np.linspace(25, 30, num=nn)
for i in range(nn):
	powaa[i] = wow(angl[i])

plt.plot(angl, powaa)
#plt.plot(orbit_ang, np.sum(panels_power,axis=1))
#plt.legend(['+X', '-X', '+Y', '-Y', '+Z', '-Z', 'sum'])
'''


# OLD VARIABLES
'''
sun_rad_size = np.arctan(695510/149597871) # rad size of sun from Earth
penumbra_effect = 8			 # s
earth_flattening = 22		 # s
atmosphere_earth = 8		 # s

'''


# SUM VALUES FOR DENSITY AND HEAT CAPACITY
'''
3U
v=1.6*326*82.6/1000
mFR4 = 79 #g
tot = 127
FR4 = 1.850 #g/cm3
alu  950 J/Kg*K
Copper shc=589 d=2223

'''
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



# GET SOLAR PANEL AREAS
angl = 28.5
solar_area_negX = 0  #m^2
solar_area_posX = 0#1*10*area_sc
solar_area_posY = 20*area_sc + 4*20*area_sc*np.cos(angl/rad_conv)#5*20*area_sc # np.cos(45/rad_conv) * solar_area_top
solar_area_negY = 0
solar_area_posZ = 0
solar_area_negZ = 20*area_sc + 4*20*area_sc*np.sin(angl/rad_conv)#20*area_sc#np.cos(45/rad_conv) * solar_area_top

solar_area_top = 4*20*area_sc + 2*20*area_sc*np.cos(45/rad_conv)
#5*20*area_sc # np.cos(45/rad_conv) * solar_area_top
#20*area_sc#np.cos(45/rad_conv) * solar_area_top



# getting effictiveness of different faces
'''
for i in range(6):
		plt.plot(orbit_ang, panels_power[:,i])
plt.plot(orbit_ang, np.sum(panels_power,axis=1))
plt.legend(['+X', '-X', '+Y', '-Y', '+Z', '-Z', 'sum'])

# 53.33
print('sum= ',np.sum(panels_power)/n)
'''

# USELESS THERMAL FUNCTION
'''
def sp_power_gen(areas, sun_incidence, absortivity, solar_flux, sp_eff):
	heatflow = areas*sun_incidence*absortivity*solar_flux*sp_eff
	return heatflow
#a = sp_power_gen(areas, sun_incidence, absortivity, solar_flux = solar_flux, sp_eff )
'''


'''
# INSTANTANEOUS POWER PRODUCED BY SOLAR PANELS
for i in range(np.shape(sun_incidence)[1]):
	sun_incidence[:,i] = sun_incidence[:,i]*sun_visibility 
	albedo_incidence[:,i] = albedo_incidence[:,i]*albedo_visibility 
	panels_power[:,i] = sun_incidence[:,i] * SP_areas[i]	
panels_power = panels_power* solar_flux* EOL_sp_eff* sp_eff
'''


