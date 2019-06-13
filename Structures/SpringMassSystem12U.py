import numpy as np
from matplotlib import pyplot as plt
#12U Sat-ELITE frequency calculations

#Constants
#All constansts are in SI units unless mentioned elsewise

#AL-7075

rho_al = 2810
E_al = 71700000000

#Sat-ELITE

x_sat = 0.3
y_sat = 0.2
z_sat = 0.2
x_cube = 0.1

#Camera

m_cam = 10
x_cam = 0.2
y_cam = 0.1
z_cam = 0.13

#ADCS

m_adcs = 1.4
x_adcs = 0.05
y_adcs = 0.15
z_adcs = 0.05

#Power

m_pow = 1.6
x_pow = 0.2
y_pow = 0.1
z_pow = 0.03

#Telecommunications

m_tele = 0.4
x_tele = 0.025
y_tele = 0.05
z_tele = 0.15

#C&DH

m_cdh = 0.2
x_cdh = 0.075
y_cdh = 0.05
z_cdh = 0.15

#Propulsion1

m_prop1 = 0.86
x_prop1 = 0.039
y_prop1 = 0.05
z_prop1 = 0.05

#Propulsion2

m_prop2 = 0.86
x_prop2 = 0.039
y_prop2 = 0.15
z_prop2 = 0.15

#Structures

M_struc = 1.8
n_struc = 92.-4.
l_struc = 0.1

m_struc = M_struc/n_struc
v_struc = m_struc/rho_al
a_struc = v_struc/l_struc
t_struc = np.sqrt(a_struc)

#beam distances

l1 = x_prop1
l2 = x_adcs
l3 = x_tele
l4 = x_prop2
l5 = z_tele-z_prop1
l6 = z_cdh-z_prop1
l7 = x_cdh-x_tele
l8 = y_prop2-y_tele
l9 = y_adcs-y_prop1
l10 = x_cam-x_cdh
l11 = y_prop2-y_cdh
l12 = z_prop2-z_adcs
l13 = x_pow-x_prop1
l14 = x_pow-x_adcs
l15 = x_cam-x_adcs
l16 = x_cam-x_prop2
l17 = x_cam-x_prop1
l18 = z_cam-z_pow
l19 = x_sat-x_pow
l20 = x_sat-x_cam
l21 = z_prop1
l22 = z_adcs
l23 = z_pow
l24 = z_sat-z_tele
l25 = z_sat-z_cdh
l26 = z_sat-z_prop2
l27 = z_sat-z_cam
l28 = y_prop1
l29 = y_tele
l30 = y_cdh
l31 = y_pow
l32 = y_cam
l33 = y_sat-y_adcs
l34 = y_sat-y_prop2
l35 = y_sat-y_pow
l36 = y_sat-y_cam

################

#longitudinal frequencies

k1lat = 3*((3*E_al*((t_struc**4)/12))/(l1**3))
k2lat = 3*((3*E_al*((t_struc**4)/12))/(l2**3))
k3lat = 3*((3*E_al*((t_struc**4)/12))/(l3**3))
k4lat = 3*((3*E_al*((t_struc**4)/12))/(l4**3))
k5lat = 2*((3*E_al*((t_struc**4)/12))/(l5**3))
k6lat = 1*((3*E_al*((t_struc**4)/12))/(l6**3))
k7lat = 2*((3*E_al*((t_struc**4)/12))/(l7**3))
k8lat = 1*((3*E_al*((t_struc**4)/12))/(l8**3))
k9lat = 3*((3*E_al*((t_struc**4)/12))/(l9**3))
k10lat = 1/((1/((2*E_al*((t_struc**4)/12))/(x_cube**3)))+(1/((3*E_al*((t_struc**4)/12))/((l10-x_cube)**3))))
k11lat = 1*((3*E_al*((t_struc**4)/12))/(l11**3))
k12lat = 3*((3*E_al*((t_struc**4)/12))/(l12**3))
k13lat = 2*((3*E_al*((t_struc**4)/12))/(l13**3))
k14lat = 2*((3*E_al*((t_struc**4)/12))/(l14**3))
k15lat = 1*((3*E_al*((t_struc**4)/12))/(l15**3))
k16lat = 1/((1/((2*E_al*((t_struc**4)/12))/(x_cube**3)))+(1/((3*E_al*((t_struc**4)/12))/((l6-x_cube)**3))))
k17lat = 1*((3*E_al*((t_struc**4)/12))/(l17**3))
k18lat = 7*((3*E_al*((t_struc**4)/12))/(l18**3))
k19lat = 3*((3*E_al*((t_struc**4)/12))/(l19**3))
k20lat = 4*((3*E_al*((t_struc**4)/12))/(l20**3))
k21lat = 3*((3*E_al*((t_struc**4)/12))/(l21**3))
k22lat = 3*((3*E_al*((t_struc**4)/12))/(l22**3))
k23lat = 7*((3*E_al*((t_struc**4)/12))/(l23**3))
k24lat = 2*((3*E_al*((t_struc**4)/12))/(l24**3))
k25lat = 1*((3*E_al*((t_struc**4)/12))/(l25**3))
k26lat = 3*((3*E_al*((t_struc**4)/12))/(l26**3))
k27lat = 7*((3*E_al*((t_struc**4)/12))/(l27**3))
k28lat = 3*((3*E_al*((t_struc**4)/12))/(l28**3))
k29lat = 2*((3*E_al*((t_struc**4)/12))/(l29**3))
k30lat = 1*((3*E_al*((t_struc**4)/12))/(l30**3))
k31lat = 3*((3*E_al*((t_struc**4)/12))/(l31**3))
k32lat = 4*((3*E_al*((t_struc**4)/12))/(l32**3))
k33lat = 3*((3*E_al*((t_struc**4)/12))/(l33**3))
k34lat = 3*((3*E_al*((t_struc**4)/12))/(l34**3))
k35lat = 3*((3*E_al*((t_struc**4)/12))/(l35**3))
k36lat = 4*((3*E_al*((t_struc**4)/12))/(l36**3))

k1long = 3*((E_al*t_struc**2)/l1)
k2long = 3*((E_al*t_struc**2)/l2)
k3long = 3*((E_al*t_struc**2)/l3)
k4long = 3*((E_al*t_struc**2)/l4)
k5long = 2*((E_al*t_struc**2)/l5)
k6long = 1*((E_al*t_struc**2)/l6)
k7long = 2*((E_al*t_struc**2)/l7)
k8long = 1*((E_al*t_struc**2)/l8)
k9long = 3*((E_al*t_struc**2)/l9)
k10long = 1/((1/(2*((E_al*t_struc**2)/x_cube)))+(1/(3*((E_al*t_struc**2)/(l10-x_cube)))))
k11long = 1*((E_al*t_struc**2)/l11)
k12long = 3*((E_al*t_struc**2)/l12)
k13long = 2*((E_al*t_struc**2)/l13)
k14long = 2*((E_al*t_struc**2)/l14)
k15long = 1*((E_al*t_struc**2)/l15)
k16long = 1/((1/(2*((E_al*t_struc**2)/x_cube)))+(1/(3*((E_al*t_struc**2)/(l16-x_cube)))))
k17long = 1*((E_al*t_struc**2)/l17)
k18long = 7*((E_al*t_struc**2)/l18)
k19long = 3*((E_al*t_struc**2)/l19)
k20long = 4*((E_al*t_struc**2)/l20)
k21long = 3*((E_al*t_struc**2)/l21)
k22long = 3*((E_al*t_struc**2)/l22)
k23long = 7*((E_al*t_struc**2)/l23)
k24long = 2*((E_al*t_struc**2)/l24)
k25long = 1*((E_al*t_struc**2)/l25)
k26long = 3*((E_al*t_struc**2)/l26)
k27long = 7*((E_al*t_struc**2)/l27)
k28long = 3*((E_al*t_struc**2)/l28)
k29long = 2*((E_al*t_struc**2)/l29)
k30long = 1*((E_al*t_struc**2)/l30)
k31long = 3*((E_al*t_struc**2)/l31)
k32long = 4*((E_al*t_struc**2)/l32)
k33long = 3*((E_al*t_struc**2)/l33)
k34long = 3*((E_al*t_struc**2)/l34)
k35long = 3*((E_al*t_struc**2)/l35)
k36long = 4*((E_al*t_struc**2)/l36)

Mmat = np.array([[m_prop1,0,0,0,0,0,0],
                 [0,m_adcs,0,0,0,0,0],
                 [0,0,m_tele,0,0,0,0],
                 [0,0,0,m_cdh,0,0,0],
                 [0,0,0,0,m_prop2,0,0],
                 [0,0,0,0,0,m_pow,0],
                 [0,0,0,0,0,0,m_cam]])
Kmatxdir = np.array([[(k1long+k5lat+k6lat+k9lat+k13long+k17long+k21lat+k28lat),-k9lat,-k5lat,-k6lat,0,-k13long,-k17long],
                     [-k9lat,(k2long+k9lat+k12lat+k14long+k15long+k22lat+k33lat),0,0,-k12lat,-k14long,-k15long],
                     [-k5lat,0,(k3long+k5lat+k7long+k8lat+k24lat+k29lat),-k7long,-k8lat,0,0],
                     [-k6lat,0,-k7long,(k6lat+k7long+k10long+k11lat+k25lat+k30lat),-k11lat,0,-k10long],
                     [0,-k12lat,-k8lat,-k11lat,(k4long+k8lat+k11lat+k12lat+k16long+k26lat+k34lat),0,-k16long],
                     [-k13long,-k14long,0,0,0,(k13long+k14long+k18lat+k19long+k23lat+k31lat+k35lat),-k18lat],
                     [-k17long,-k15long,0,-k10long,-k16long,-k18lat,(k10long+k15long+k16long+k17long+k18lat+k20long+k27lat+k32lat+k36lat)]])
Kmatydir = np.array([[(k1lat+k5lat+k6lat+k9long+k13lat+k17lat+k21lat+k28long),-k9long,-k5lat,-k6lat,0,-k13lat,-k17lat],
                     [-k9long,(k2lat+k9long+k12lat+k14lat+k15lat+k22lat+k33long),0,0,-k12lat,-k14lat,-k15lat],
                     [-k5lat,0,(k3lat+k5lat+k7lat+k8long+k24lat+k29long),-k7lat,-k8long,0,0],
                     [-k6lat,0,-k7lat,(k6lat+k7lat+k10lat+k11long+k25lat+k30long),-k11long,0,-k10lat],
                     [0,-k12lat,-k8long,-k11long,(k4lat+k8long+k11long+k12lat+k16lat+k26lat+k34long),0,-k16lat],
                     [-k13lat,-k14lat,0,0,0,(k13lat+k14lat+k18lat+k19lat+k23lat+k31long+k35long),-k18lat],
                     [-k17lat,-k15lat,0,-k10lat,-k16lat,-k18lat,(k10lat+k15lat+k16lat+k17lat+k18lat+k20lat+k27lat+k32long+k36long)]])
Kmatzdir = np.array([[(k1lat+k5long+k6long+k9lat+k13lat+k17lat+k21long+k28lat),-k9lat,-k5long,-k6long,0,-k13lat,-k17lat],
                     [-k9lat,(k2lat+k9lat+k12long+k14lat+k15lat+k22long+k33lat),0,0,-k12long,-k14lat,-k15lat],
                     [-k5long,0,(k3lat+k5long+k7lat+k8lat+k24long+k29lat),-k7lat,-k8lat,0,0],
                     [-k6long,0,-k7lat,(k6long+k7lat+k10lat+k11lat+k25long+k30lat),-k11lat,0,-k10lat],
                     [0,-k12long,-k8lat,-k11lat,(k4lat+k8lat+k11lat+k12long+k16lat+k26long+k34lat),0,-k16lat],
                     [-k13lat,-k14lat,0,0,0,(k13lat+k14lat+k18long+k19lat+k23long+k31lat+k35lat),-k18long],
                     [-k17lat,-k15lat,0,-k10lat,-k16lat,-k18long,(k10lat+k15lat+k16lat+k17lat+k18long+k20lat+k27long+k32lat+k36lat)]])


Minvsqrt = np.linalg.inv(Mmat**0.5)
Ktildexdir = Minvsqrt*Kmatxdir*Minvsqrt
Ktildeydir = Minvsqrt*Kmatydir*Minvsqrt
Ktildezdir = Minvsqrt*Kmatzdir*Minvsqrt

eigenvaluesx = np.linalg.eigvals(Ktildexdir)
eigenvaluesy = np.linalg.eigvals(Ktildeydir)
eigenvaluesz = np.linalg.eigvals(Ktildezdir)

xfreqs = (eigenvaluesx**0.5)/(2*np.pi)
yfreqs = (eigenvaluesy**0.5)/(2*np.pi)
zfreqs = (eigenvaluesz**0.5)/(2*np.pi)

print xfreqs,yfreqs,zfreqs

print "Min x freq: ",min(xfreqs)
print "Min y freq: ",min(yfreqs)
print "Min z freq: ",min(zfreqs)
