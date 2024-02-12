# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 00:59:33 2024

@author: nachi
"""
#%% Librerias a utilizar

import numpy as np
from scipy.spatial.transform import Rotation
import functions    
import pandas as pd


#%% importar archivo con SGP4 y mediciones ECI

# Nombre del archivo CSV
archivo_csv = "vectores_40k.csv"

# Leer el archivo CSV en un DataFrame de pandas
df = pd.read_csv(archivo_csv)

# Convertir el DataFrame a un array de NumPy
array_datos = df.to_numpy()

t_aux = array_datos[:,0]
Bx_IGRF = array_datos[:,10]
By_IGRF = array_datos[:,11]
Bz_IGRF = array_datos[:,12]
vsun_x = array_datos[:,13]
vsun_y = array_datos[:,14]
vsun_z = array_datos[:,15]

position = np.transpose(np.vstack((array_datos[:,1], array_datos[:,2], array_datos[:,3])))
velocity = np.transpose(np.vstack((array_datos[:,4], array_datos[:,5], array_datos[:,6])))

Z_orbits = position[:,:] / np.linalg.norm(position[:,:])
X_orbits = np.cross(velocity[:,:],Z_orbits) / np.linalg.norm(np.cross(velocity[:,:],Z_orbits))
Y_orbits = np.cross(Z_orbits,X_orbits)

q0_e2o = []
q1_e2o = []
q2_e2o = []
q3_e2o = []

Bx_orbit = []
By_orbit = []
Bz_orbit = []
vx_sun_orbit = []
vy_sun_orbit = []
vz_sun_orbit = []
for i in range(len(t_aux)):
    Rs_ECI_orbit = np.vstack((X_orbits[i,:],Y_orbits[i,:],Z_orbits[i,:]))
    q_ECI_orbit= Rotation.from_matrix(Rs_ECI_orbit).as_quat()
    
    q0_e2o.append(q_ECI_orbit[0])
    q1_e2o.append(q_ECI_orbit[1])
    q2_e2o.append(q_ECI_orbit[2])
    q3_e2o.append(q_ECI_orbit[3])
    
    B_ECI_quat = [Bx_IGRF[i],By_IGRF[i],Bz_IGRF[i],0]
    inv_q_e2o = functions.inv_q(q_ECI_orbit)
    B_orbit = functions.quat_mult(functions.quat_mult(q_ECI_orbit,B_ECI_quat),inv_q_e2o)
    B_orbit_n = np.array([B_orbit[0],B_orbit[1],B_orbit[2]])
    B_orbit = B_orbit_n / np.linalg.norm(B_orbit_n)
    
    vsun_ECI_quat = [vsun_x[0],vsun_y[0],vsun_z[0],0]
    inv_qi_s = functions.inv_q(q_ECI_orbit)
    vsun_orbit = functions.quat_mult(functions.quat_mult(q_ECI_orbit,vsun_ECI_quat),inv_qi_s)
    vsun_orbit_n = np.array([vsun_orbit[0],vsun_orbit[1],vsun_orbit[2]]) 
    vsun_orbit =  vsun_orbit_n / np.linalg.norm(vsun_orbit_n)

    Bx_orbit.append(B_orbit[0])
    By_orbit.append(B_orbit[1])
    Bz_orbit.append(B_orbit[2])
    vx_sun_orbit.append(vsun_orbit[0])
    vy_sun_orbit.append(vsun_orbit[1])
    vz_sun_orbit.append(vsun_orbit[2])

q0_e2o = np.array(q0_e2o)
q1_e2o = np.array(q1_e2o)
q2_e2o = np.array(q2_e2o)
q3_e2o = np.array(q3_e2o)
Bx_orbit =np.array(Bx_orbit)
By_orbit =np.array(By_orbit)
Bz_orbit =np.array(Bz_orbit)
vx_sun_orbit = np.array(vx_sun_orbit)
vy_sun_orbit = np.array(vy_sun_orbit)
vz_sun_orbit = np.array(vz_sun_orbit)

#%% q y w inicial real y q TRIAD

# Condiciones iniciales dadas del vector estado (cuaternion y velocidad angular)
q = np.array([-0.7071/np.sqrt(3),0.7071/np.sqrt(3),-0.7071/np.sqrt(3),0.7071])
w= np.array([0.0001,0.0001,0.0001])

#Obtener fuerzas magneticas de la Tierra inicial orbit y BODY
Bi_quat = [Bx_IGRF[0],By_IGRF[0],Bz_IGRF[0],0]
inv_qi_b = functions.inv_q(q)
Bi_body = functions.quat_mult(functions.quat_mult(q,Bi_quat),inv_qi_b)
Bi_body_n = np.array([Bi_body[0],Bi_body[1],Bi_body[2]])
Bi_body = Bi_body_n / np.linalg.norm(Bi_body_n)

#vector sol en orbit y BODY
vsuni_quat = [vsun_x[0],vsun_y[0],vsun_z[0],0]
inv_qi_s = functions.inv_q(q)
vsuni_body = functions.quat_mult(functions.quat_mult(q,vsuni_quat),inv_qi_s)
vsuni_body_n = np.array([vsuni_body[0],vsuni_body[1],vsuni_body[2]]) 
vsuni_body =  vsuni_body_n / np.linalg.norm(vsuni_body_n)

#Creacion de la solucion TRIAD como matriz de rotacion en orbit-body
DCM = functions.TRIAD(Bi_body,vsuni_body,Bi_quat[:3],vsuni_quat[:3])
q_TRIAD = Rotation.from_matrix(DCM).as_quat()

# Matriz de rotacion ECI2ORBIT
q_e2o = np.array([q0_e2o[0],q1_e2o[0],q2_e2o[0],q3_e2o[0]])
R_ECI_orbit = functions.quaternion_to_dcm(q_e2o)

# Matriz de rotacion ORBIT2BODY
R_orbit_B = np.dot(np.transpose(DCM),R_ECI_orbit)
q_orbit_B= Rotation.from_matrix(R_orbit_B).as_quat()

RPY_TRIAD = functions.quaternion_to_euler(q_TRIAD)

#%% Listas para guardar las variables de estado reales, TRIAD y body

I_x = 0.037
I_y = 0.036
I_z = 0.006

Bx_bodys = [Bi_body[0]]
By_bodys = [Bi_body[1]]
Bz_bodys = [Bi_body[2]]
vsun_bodys_x = [vsuni_body[0]]
vsun_bodys_y = [vsuni_body[1]]
vsun_bodys_z = [vsuni_body[2]]

q0_rot = [q_orbit_B[0]]
q1_rot = [q_orbit_B[1]]
q2_rot = [q_orbit_B[2]]
q3_rot = [q_orbit_B[3]]

q0_TRIADS = [q_TRIAD[0]]
q1_TRIADS = [q_TRIAD[1]]
q2_TRIADS = [q_TRIAD[2]]
q3_TRIADS = [q_TRIAD[3]]

w0_values = [w[0]]
w1_values = [w[1]]
w2_values = [w[2]]

#%% Control lineal PD

w0_o = 0.00163
w0_oi = np.array([w0_o,0,0])

w0_test = 0
w1_test = 0
w2_test = 0

A = functions.A_PD(I_x,I_y,I_z,w0_o, w0_test,w1_test,w2_test)
A= np.where(np.abs(A) == 0, 0, A)

B_body_ctrl = np.array([1617.3,4488.9,46595.8])*1e-9

B = functions.B_PD(I_x,I_y,I_z,B_body_ctrl)

# # quiza para magnetorquer medio
# Kp_x =  -0.0261
# Kp_y = 0.0027 
# Kp_z = -0.6306
# Kd_x = -17.2535
# Kd_y = -10.6473 
# Kd_z =  138.2978

# Kp_x =  -241.799276978197
# Kp_y = 43.3569691036660
# Kp_z = -3059.34072509731
# Kd_x = -560.886975412642
# Kd_y = -155.273483672191
# Kd_z =  -5161.73551572602

Kp_x =  -11.7126815151457
Kp_y = 0.0215395552140476
Kp_z = -1.94092971659118
Kd_x = 2.23100994690840
Kd_y = -0.0591645752827404
Kd_z =  -473.824574185555
																																																									
K_Ex_app = functions.K(Kp_x, Kp_y, Kp_z, Kd_x, Kd_y, Kd_z)
   
K_Ps = K_Ex_app[:,:3]
K_Ds = K_Ex_app[:,3:6]

#%% sensores
# Características del magnetómetro bueno
rango = 800000  # nT
ruido = 0.1*1e-9  # nT/√Hz

#Caracteristicas del magnetometro malo
rango_bad = 75000 #nT
ruido_bad = 1.18*1e-9 #nT/√Hz

#caracteristicas del sun sensor malo
acc_bad = 5 #°
sigma_bad = functions.sigma_sensor(acc_bad)

#Caracteristicas del sun sensor intermedio
acc_med = 1 #°
sigma_med = functions.sigma_sensor(acc_med)

#caracteristicas del sun sensor bueno
sigma_good = 0.05

# Datos del giroscopio malo
bias_instability_bad = 0.05 / 3600 *np.pi/180 # <0.10°/h en radianes por segundo
noise_rms_bad = 0.12*np.pi/180 # 0.12 °/s en radianes por segundo

#Datos del giroscopio medio
bias_instability_med = 0.03 / 3600 * np.pi/180  # <0.06°/h en radianes por segundo
noise_rms_med = 0.050 *np.pi/180 # 0.12 °/s rms en radianes por segundo

#Datos del giroscopio bueno
bias_instability_good = 0 # <0.06°/h en radianes por segundo
noise_rms_good= 0.033 *np.pi/180  # 0.050 °/s rms en radianes por segundo

deltat = 2 #cambiable segun otro archivo .py
h = 0.1
#%% Propagacion del control PD

np.random.seed(42)

for i in range(len(t_aux)-1):
    q_TRIAD_PD = np.array([q0_TRIADS[-1],q1_TRIADS[-1],q2_TRIADS[-1],q3_TRIADS[-1]])
    w_TRIAD_PD = np.array([w0_values[-1],w1_values[-1],w2_values[-1]])
    #w_TRIAD_PD = w_in_NL - np.dot(quaternion_to_dcm(q_TRIAD_PD),w0_oi)

    dx_PD_NL = np.hstack((np.transpose(q_TRIAD_PD[:3]), np.transpose(w_TRIAD_PD)))
    u_PD_NL = np.dot(K_Ex_app,dx_PD_NL)
    
    q_in_PD = np.array([q0_rot[-1],q1_rot[-1],q2_rot[-1],q3_rot[-1]])
    x_PD_REAL = np.hstack((np.transpose(q_in_PD[:3]), np.transpose(w_TRIAD_PD)))
    u_PD_REAL =np.dot(K_Ps,q_in_PD[:3]) + np.dot(K_Ds,w_TRIAD_PD)

    for j in range(int(deltat/h)):
        q_rot,w_new = functions.rk4_step_PD(functions.dynamics, x_PD_REAL, A, B,u_PD_NL, h)
        if  1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2 < 0:
            q_rot = q_rot / np.linalg.norm(q_rot)
            dx_PD_NL = np.hstack((np.transpose(q_rot), np.transpose(w_new)))
            q3s_rot = 0

        else:
            dx_PD_NL = np.hstack((np.transpose(q_rot), np.transpose(w_new)))
            q3s_rot = np.sqrt(1-q_rot[0]**2-q_rot[1]**2-q_rot[2]**2)
            
    q0_rot.append(q_rot[0])
    q1_rot.append(q_rot[1])
    q2_rot.append(q_rot[2])
    q3_rot.append(q3s_rot)
    q_rot_ind = np.array([q0_rot[-1],q1_rot[-1],q2_rot[-1],q3_rot[-1]])
    
    w_gyros_bad = functions.simulate_gyros_reading(w_new,0, 0)
    w0_values.append(w_gyros_bad[0])
    w1_values.append(w_gyros_bad[1])
    w2_values.append(w_gyros_bad[2])
    
    B_quat_ECI = [Bx_IGRF[i+1],By_IGRF[i+1],Bz_IGRF[i+1],0]
    B_quat = [Bx_orbit[i+1],By_orbit[i+1],Bz_orbit[i+1],0]
    inv_q_b = functions.inv_q(q_rot_ind)
    B_body = functions.quat_mult(functions.quat_mult(q_rot_ind,B_quat),inv_q_b)
    B_body_n = np.array([B_body[0],B_body[1],B_body[2]])
    B_body = B_body_n / np.linalg.norm(B_body_n)
    B_magn_bad = functions.simulate_magnetometer_reading(B_body, ruido_bad)
    Bx_bodys.append(B_magn_bad[0])
    By_bodys.append(B_magn_bad[1])
    Bz_bodys.append(B_magn_bad[2])
    
    vsun_quat_ECI = [vsun_x[i+1],vsun_y[i+1],vsun_z[i+1],0]
    vsun_quat = [vx_sun_orbit[i+1],vy_sun_orbit[i+1],vz_sun_orbit[i+1],0]
    inv_q_s = functions.inv_q(q_rot_ind)
    vsun_body = functions.quat_mult(functions.quat_mult(q_rot_ind,vsun_quat),inv_q_s)
    vsun_body_n = np.array([vsun_body[0],vsun_body[1],vsun_body[2]]) 
    vsun_body = vsun_body_n / np.linalg.norm(vsun_body_n)
    vsun_bad = functions.simulate_sunsensor_reading(vsun_body, sigma_med)
    vsun_bodys_x.append(vsun_bad[0])
    vsun_bodys_y.append(vsun_bad[1])
    vsun_bodys_z.append(vsun_bad[2])
    
    delta_q = np.random.rand(4)*0.35#np.linalg.norm(np.array([0.1,ruido_bad,sigma_bad]))
    q_TRIADA_n = q_rot_ind + delta_q
    q_TRIADA = q_TRIADA_n / np.linalg.norm(q_TRIADA_n)
    
    # DCM_PD = functions.TRIAD(B_body,vsun_body,B_quat_ECI[:3],vsun_quat_ECI[:3])
    # q_TRIADAECI = Rotation.from_matrix(DCM_PD).as_quat()
    
    # # Matriz de rotacion ECI2ORBIT
    # q_eci2orbit = np.array([q0_e2o[i+1],q1_e2o[i+1],q2_e2o[i+1],q3_e2o[i+1]])
    # R_ECI2orbit = functions.quaternion_to_dcm(q_eci2orbit)

    # # Matriz de rotacion ORBIT2BODY
    # R_orbit2Body = np.dot(np.transpose(DCM_PD),R_ECI2orbit)
    # q_TRIADA = Rotation.from_matrix(R_orbit2Body).as_quat()
    
    q0_TRIADS.append(q_TRIADA[0])
    q1_TRIADS.append(q_TRIADA[1])
    q2_TRIADS.append(q_TRIADA[2])
    q3_TRIADS.append(q_TRIADA[3])

Bx_bodys = np.array(Bx_bodys)
By_bodys = np.array(By_bodys)
Bz_bodys = np.array(Bz_bodys)

vsun_bodys_x = np.array(vsun_bodys_x)
vsun_bodys_y = np.array(vsun_bodys_y)
vsun_bodys_z = np.array(vsun_bodys_z)

q0_TRIADS = np.array(q0_TRIADS)
q1_TRIADS = np.array(q1_TRIADS) 
q2_TRIADS = np.array(q2_TRIADS)
q3_TRIADS = np.array(q3_TRIADS)

q0_rot = np.array(q0_rot)
q1_rot = np.array(q1_rot)
q2_rot = np.array(q2_rot)
q3_rot = np.array(q3_rot)

w0_values = np.array(w0_values)
w1_values = np.array(w1_values)
w2_values = np.array(w2_values)
    
q_k_all_id = np.vstack((q0_rot,q1_rot,q2_rot,q3_rot))
q_k_all_id_t = np.transpose(q_k_all_id)
RPY_all_id = []

for i in range(len(t_aux)):
    RPY_EKF_id = functions.quaternion_to_euler(q_k_all_id_t[i,:])
    RPY_all_id.append(RPY_EKF_id)
    
RPY_all_id = np.array(RPY_all_id)

#%% Guardar datos para graficar en otro codigo

# Nombre del archivo
archivo_c = "control_bad_o.csv"

# Abrir el archivo en modo escritura
with open(archivo_c, 'w') as f:
    # Escribir los encabezados
    f.write("t_aux, Roll,Pitch,Yaw, q0_rot, q1_rot, q2_rot, q3_rot, q0_TRIAD, q1_TRIAD, q2_TRIAD, q3_TRIAD, w0_values, w1_values, w2_values _z\n")

    # Escribir los datos en filas
    for i in range(len(t_aux)):
        f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
            t_aux[i], RPY_all_id[i,0], RPY_all_id[i,1],RPY_all_id[i,2],q0_rot[i],q1_rot[i],q2_rot[i],
            q3_rot[i],q0_TRIADS[i],q1_TRIADS[i], q2_TRIADS[i], q3_TRIADS[i],
            w0_values[i], w1_values[i], w2_values[i]
        ))

print("Vectores guardados en el archivo:", archivo_c)