import numpy as np
from matplotlib import pyplot as plt
#Mass moment of Inertia calculations

#Constants
#All constansts are in SI units unless mentioned elsewise

#AL-7075

rho_al = 2810
E_al = 71700000000

#Sat-ELITE

x_sat = 0.3
y_sat = 0.2
z_sat = 0.1
x_cube = 0.1

x_spacing = 0.005

#Solar

m_sol = 0.053
t_sol = 0.002
w_sol = 0.1

#Structures

M_struc = 1.8
n_struc = 92.-4.
l_struc = 0.1

m_struc = M_struc/n_struc
v_struc = m_struc/rho_al
a_struc = v_struc/l_struc
t_struc = np.sqrt(a_struc)



#Model the parts

class Part(object):
    def __init__(self, name, m, x, y, z):
        self.name = name
        self.m = m
        self.x = x
        self.y = y
        self.z = z
        self.Ixx = (1./12.)*m*(y**2+z**2)
        self.Iyy = (1./12.)*m*(x**2+z**2)
        self.Izz = (1./12.)*m*(x**2+y**2)
        self.Ixy = 0.
        self.Ixz = 0.
        self.Iyz = 0.

    def __repr__(self):
        return str(self.name)

    #If part is not modelled as a cuboid redifine Ixx,Iyy,Izz and define Ixy,Ixz,Iyz if not 0
cam = Part("SEEING 7 Camera",10.,0.16,0.16,0.14)
cam.Ixx = (1./2.)*cam.m*cam.x**2
cam.Iyy = (1./12.)*cam.m*(3*cam.x**2+cam.z**2)
cam.Izz = (1./12.)*cam.m*(3*cam.x**2+cam.z**2)
adcs = Part("ADCS",1.4,0.09,0.09,0.09)
power = Part("Batteries",0.4,0.09,0.09,0.05)
tele = Part("Telecommunications",0.4,0.04,0.09,0.09)
cdh = Part("C&DH",0.2,0.04,0.09,0.09)
prop = Part("Propulsion unit",0.86,0.078,0.094,0.09)
horsol = Part("Horizontal 2x3 solar panel",6*m_sol,3*w_sol,2*w_sol,t_sol)
versol = Part("Vertical 2x3 solar panel",m_sol,3*w_sol,t_sol,2*w_sol)
diasol = Part("Diagonal 2x3 solar panel",m_sol,3*w_sol,2*w_sol,2*w_sol)
diasol.Ixx = (1./12.)*diasol.m*((2*w_sol)**2+t_sol**2)
diasol.Iyy = (1./12.)*diasol.m*((t_sol**2)*(np.cos(np.pi/4)**2)+(((2*w_sol)**2)*(np.sin(np.pi/4)**2))+(3*w_sol)**2)
diasol.Izz = (1./12.)*diasol.m*(((2*w_sol)**2)*(np.cos(np.pi/4)**2)+((t_sol**2)*(np.sin(np.pi/4)**2))+(3*w_sol)**2)
slantsol = Part("28.5 deg 2x3 solar panel",m_sol,3*w_sol,2*w_sol,2*w_sol)
slantsol.Ixx = (1./12.)*diasol.m*((2*w_sol)**2+t_sol**2)
slantsol.Iyy = (1./12.)*diasol.m*((t_sol**2)*(np.cos((61.5/180)*np.pi)**2)+(((2*w_sol)**2)*(np.sin((61.5/180)*np.pi)**2))+(3*w_sol)**2)
slantsol.Izz = (1./12.)*diasol.m*(((2*w_sol)**2)*(np.cos((61.5/180)*np.pi)**2)+((t_sol**2)*(np.sin((61.5/180)*np.pi)**2))+(3*w_sol)**2)
xstruc3 = Part("Longitudinal beam length 3U",3*m_struc,3*l_struc,t_struc,t_struc)
xstruc1 = Part("Longitudinal beam length 1U",m_struc,l_struc,t_struc,t_struc)
ystruc2 = Part("Horizontal beam length 2U",2*m_struc,t_struc,2*l_struc,t_struc)
zstruc2 = Part("Vertical beam length 2U",2*m_struc,t_struc,t_struc,2*l_struc)

testm_tip = 0.4
testr_tip = 0.15
testh_tip = 0.1
testm_boom = 0.010
testl_boom = 0.34775
testm_body =  3.18244-2*testm_boom-2*testm_tip
testx_body = 0.1
testy_body = 0.1
testz_body = 0.3405

testtip = Part("Tip mass",testm_tip,testh_tip,testr_tip,testr_tip)
testtip.Ixx = (1./12.)*testm_tip*(3*testr_tip**2+testh_tip**2)
testtip.Iyy = (1./12.)*testm_tip*testr_tip**2
testtip.Izz = (1./12.)*testm_tip*(3*testr_tip**2+testh_tip**2)
testboom = Part("Boom",testm_boom,0.,0.,testl_boom)
testboom.Ixx = (1./12.)*testm_boom*testl_boom**2
testboom.Iyy = (1./12.)*testm_boom*testl_boom**2
testboom.Izz = 0.
testbody = Part("Body",testm_body,testx_body,testy_body,testz_body)

class Component(object):
    def __init__(self,part,xpos,ypos,zpos):
        self.part = part
        self.xpos = xpos
        self.ypos = ypos
        self.zpos = zpos

    def __repr__(self):
        return str(self.part)

#Model the system

##system = [Component(cam,0.2,0.1,0.13),
##          Component(adcs,0.05,0.15,0.05),
##          Component(power,0.15,0.05,0.03),
##          Component(power,0.15,0.15,0.03),
##          Component(power,0.25,0.05,0.03),
##          Component(power,0.25,0.15,0.03),
##          Component(tele,0.025,0.05,0.15),
##          Component(cdh,0.075,0.05,0.15),
##          Component(prop,0.039,0.05,0.05),
##          Component(prop,0.039,0.15,0.15),
##          Component(horsol,0.15,0.1,0.0),
##          Component(versol,0.15,0.2,0.1),
##          Component(slantsol,0.15,-np.cos((61.5/180)*np.pi)*0.1,-np.sin((61.5/180)*np.pi)*0.1),
##          Component(slantsol,0.15,-np.cos((61.5/180)*np.pi)*0.3,-np.sin((61.5/180)*np.pi)*0.3),
##          Component(slantsol,0.15,0.2+np.cos((61.5/180)*np.pi)*0.1,0.2+np.sin((61.5/180)*np.pi)*0.1),
##          Component(slantsol,0.15,0.2+np.cos((61.5/180)*np.pi)*0.3,0.2+np.sin((61.5/180)*np.pi)*0.3),
##          Component(xstruc3,0.15,0.0,0.0),
##          Component(xstruc3,0.15,0.1,0.0),
##          Component(xstruc3,0.15,0.2,0.0),
##          Component(xstruc3,0.15,0.2,0.1),
##          Component(xstruc3,0.15,0.2,0.2),
##          Component(xstruc1,0.025,0.1,0.2),
##          Component(xstruc3,0.15,0.0,0.2),
##          Component(xstruc3,0.15,0.0,0.1),
##          Component(ystruc2,0.0,0.1,0.0),
##          Component(ystruc2,0.1,0.1,0.0),
##          Component(ystruc2,0.2,0.1,0.0),
##          Component(ystruc2,0.3,0.1,0.0),
##          Component(ystruc2,0.3,0.1,0.1),
##          Component(ystruc2,0.3,0.1,0.2),
##          Component(ystruc2,0.2,0.1,0.2),
##          Component(ystruc2,0.1,0.1,0.2),
##          Component(ystruc2,0.0,0.1,0.1),
##          Component(ystruc2,0.0,0.1,0.0),
##          Component(zstruc2,0.0,0.0,0.1),
##          Component(zstruc2,0.1,0.0,0.1),
##          Component(zstruc2,0.2,0.0,0.1),
##          Component(zstruc2,0.3,0.0,0.1),
##          Component(zstruc2,0.3,0.1,0.1),
##          Component(zstruc2,0.3,0.2,0.1),
##          Component(zstruc2,0.2,0.2,0.1),
##          Component(zstruc2,0.1,0.2,0.1),
##          Component(zstruc2,0.0,0.2,0.1),
##          Component(zstruc2,0.0,0.1,0.1)]

system = [Component(testtip,0,0,-0.5),
          Component(testtip,0,-0.05,0.5),
          Component(testboom,0,0,-0.335125),
          Component(testboom,0,-0.05,0.335125),
          Component(testbody,0,0,0)]

#Centroid calculations

summdx = 0.
summdy = 0.
summdz = 0.
summ = 0.

for component in system:
    component.mdx = component.part.m*component.xpos
    component.mdy = component.part.m*component.ypos
    component.mdz = component.part.m*component.zpos
    summdx = summdx + component.mdx
    summdy = summdy + component.mdy
    summdz = summdz + component.mdz
    summ = summ + component.part.m

centroidx = summdx/summ
centroidy = summdy/summ
centroidz = summdz/summ

deltax = centroidx - 0.15
deltay = centroidy - 0.1
deltaz = centroidz - 0.1

#Area moment of intertia calculations

sumIxx = 0.
sumIyy = 0.
sumIzz = 0.

for component in system:
    component.Ixxconf = component.part.Ixx + component.part.m*((component.ypos-centroidy)**2 + (component.zpos-centroidz)**2)
    component.Iyyconf = component.part.Iyy + component.part.m*((component.xpos-centroidx)**2 + (component.zpos-centroidz)**2)
    component.Izzconf = component.part.Izz + component.part.m*((component.xpos-centroidx)**2 + (component.ypos-centroidy)**2)
    sumIxx = sumIxx + component.Ixxconf
    sumIyy = sumIyy + component.Iyyconf
    sumIzz = sumIzz + component.Izzconf

systemIxx = sumIxx
systemIyy = sumIyy
systemIzz = sumIzz

#Moment tensor calculations

sumIxy = 0.
sumIxz = 0.
sumIyz = 0.

for component in system:
    component.Ixyconf = component.part.Ixy - component.part.m*(component.xpos-centroidx)*(component.ypos-centroidy)
    component.Ixzconf = component.part.Ixz - component.part.m*(component.xpos-centroidx)*(component.zpos-centroidz)
    component.Iyzconf = component.part.Iyz - component.part.m*(component.ypos-centroidy)*(component.zpos-centroidz)
    sumIxy = sumIxy + component.Ixyconf
    sumIxz = sumIxz + component.Ixzconf
    sumIyz = sumIyz + component.Iyzconf

systemIxy = sumIxy
systemIxz = sumIxz
systemIyz = sumIyz

Isystem = np.array([[systemIxx,systemIxy,systemIxz],[systemIxy,systemIyy,systemIyz],[systemIxz,systemIyz,systemIzz]])

#Final print

print "Delta centroid: "
print deltax,deltay,deltaz
print "Moments of Inertia: "
print Isystem


#diagonalizing the matrix

eigval,eigvec = np.linalg.eig(Isystem)
dia = np.diag(eigval)

print "Diagonal Inertia matrix: "
print dia
