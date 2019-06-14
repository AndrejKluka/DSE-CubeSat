import numpy as np
import matplotlib.pyplot as plt
import time
import os







# ENVIRONMENT CONSTANTS
stefan_boltz = 5.67e-8 		# W/m^2/K^4
earth_r = 6378000 				# m
solar_flux = 1368			# W/m^2
e_albedo = 0.3 				# reflected ratio 0.24-0.4 for SSO
earth_T = 256 				# 246K-256K
space_T = 2.7 				# K
rad_conv = 180/np.pi 		#57.3..
sp_eff = 0.3 # GaAs solar panel efficiency


EtoS_r = (4*np.pi*earth_r**2) / (4*np.pi*(altitude+earth_r)**2)
rho = np.arcsin(EtoS_r**0.5)
Ka = 0.664 + 0.521*rho + 0.203*rho**2



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


















































