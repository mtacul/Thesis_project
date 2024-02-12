from skyfield.positionlib import Geocentric
import numpy as np
from datetime import datetime
import math
from scipy.signal import butter, lfilter

def eci2lla(posicion, fecha):
    from skyfield.api import Distance, load, utc, wgs84
    ts = load.timescale()
    fecha = fecha.replace(tzinfo=utc)
    t = ts.utc(fecha)
    d = [Distance(m=i).au for i in (posicion[0]*1000, posicion[1]*1000, posicion[2]*1000)]
    p = Geocentric(d,t=t)
    g = wgs84.subpoint(p)
    latitud = g.latitude.degrees
    longitud = g.longitude.degrees
    altitud = g.elevation.m
    return latitud, longitud, altitud

#%%inversa de un cuaternion

def inv_q(q):
    inv_q = np.array([-q[0],-q[1],-q[2],q[3]])
    return inv_q

#%% Para vector sol

def datetime_to_jd2000(fecha):
    t1 = (fecha - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
    t2 = 86400  # Número de segundos en un día
    jd2000 = t1 / t2
    return jd2000

def sun_vector(jd2000):
    M_sun = 357.528 + 0.9856003*jd2000
    M_sun_rad = M_sun * np.pi/180
    lambda_sun = 280.461 + 0.9856474*jd2000 + 1.915*np.sin(M_sun_rad)+0.020*np.sin(2*M_sun_rad)
    lambda_sun_rad = lambda_sun * np.pi/180
    epsilon_sun = 23.4393 - 0.0000004*jd2000
    epsilon_sun_rad = epsilon_sun * np.pi/180
    X_sun = np.cos(lambda_sun_rad)
    Y_sun = np.cos(epsilon_sun_rad)*np.sin(lambda_sun_rad)
    Z_sun = np.sin(epsilon_sun_rad)*np.sin(lambda_sun_rad)
    return X_sun, Y_sun, Z_sun

#%% TRIAD solution

def TRIAD(V1,V2,W1,W2):
    r1 = V1
    r2 = np.cross(V1,V2) / np.linalg.norm(np.cross(V1,V2))
    r3 = np.cross(r1,r2)
    M_obs = np.array([r1,r2,r3])
    s1 = W1
    s2 = np.cross(W1,W2) / np.linalg.norm(np.cross(W1,W2))
    s3 = np.cross(s1,s2)
    M_ref = np.array([s1,s2,s3])
    
    A = np.dot(M_obs,np.transpose(M_ref))
    return A

#%% de cuaternion a angulos de euler
def quaternion_to_euler(q):
    # Extracción de los componentes del cuaternión
    x, y, z, w = q

    # Cálculo de ángulos de Euler en radianes
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
     
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
     
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
     
    # Convierte los ángulos a grados si lo deseas
    roll_deg = np.degrees(roll_x)
    pitch_deg = np.degrees(pitch_y)
    yaw_deg = np.degrees(yaw_z)

    return roll_deg, pitch_deg, yaw_deg


def quat_mult(qk_priori,dqk):
    
    dqk_n = dqk 

# Realizar la multiplicación de cuaterniones
    result = np.array([
    qk_priori[3]*dqk_n[0] + qk_priori[0]*dqk_n[3] + qk_priori[1]*dqk_n[2] - qk_priori[2]*dqk_n[1],  # Componente i
    qk_priori[3]*dqk_n[1] + qk_priori[1]*dqk_n[3] + qk_priori[2]*dqk_n[0] - qk_priori[0]*dqk_n[2],  # Componente j
    qk_priori[3]*dqk_n[2] + qk_priori[2]*dqk_n[3] + qk_priori[0]*dqk_n[1] - qk_priori[1]*dqk_n[0],  # Componente k
    qk_priori[3]*dqk_n[3] - qk_priori[0]*dqk_n[0] - qk_priori[1]*dqk_n[1] - qk_priori[2]*dqk_n[2]   # Componente escalar
    ])
    return result

#%%Funciones para sensores

def simulate_magnetometer_reading(B_eci, ruido):
    # np.random.seed(42)  # Puedes cambiar el número 42 a cualquier otro número entero

    # Simular el ruido gaussiano
    noise = np.random.normal(0, ruido, 1)
        
    # Simular la medición del magnetómetro con ruido
    measurement = B_eci + noise


    return measurement

# Obtener la desviacion estandar del sun sensor
def sigma_sensor(acc):
    sigma = acc/(2*3)
    return sigma

# Funcion para generar realismo del sun sensor
def simulate_sunsensor_reading(vsun,sigma):
    # np.random.seed(42)  # Puedes cambiar el número 42 a cualquier otro número entero

    sigma_rad = sigma*np.pi/180
        
    # Simulación de la medición con error
    error = np.random.normal(0, sigma_rad, 1)  # Genera un error aleatorio dentro de la precisión del sensor
        
    measured_vsun = vsun + error

    return measured_vsun

# Funcion para generar realismo del giroscopio
def simulate_gyros_reading(w,ruido,bias):
    # np.random.seed(42)  # Puedes cambiar el número 42 a cualquier otro número entero

    #aplicar el ruido del sensor
    noise = np.random.normal(0, ruido, 1)
        
    #Simular la medicion del giroscopio
    measurement = w + noise + bias
        
    return measurement

#%% matrices A y B control PD lineal

def A_PD(I_x,I_y,I_z,w0_O, w0,w1,w2):
    A1 = np.array([0, 0.5*w2, -0.5*w1, 0.5, 0,0])
    A2 = np.array([-0.5*w2,0,0.5*w0,0,0.5,0])
    A3 = np.array([0.5*w1,-0.5*w0,0,0,0,0.5])
    A4 = np.array([6*w0_O**2*(I_x-I_y), 0, 0, 0, w2*(I_y-I_z)/I_x, w1*(I_y-I_z)/I_x])
    A5 = np.array([0, 6*w0_O**2*(I_z-I_y), 0, w2*(I_x-I_z)/I_y,0, (w0+w0_O)*(I_x-I_z)/I_y + I_y*w0_O])
    A6 = np.array([0, 0, 0, w1*(I_y-I_x)/I_z, (w0+w0_O)*(I_y-I_x)/I_z - I_z*w0_O, 0])
    
    A_k = np.array([A1,A2,A3,A4,A5,A6])
    
    return A_k    

def B_PD(I_x,I_y,I_z,B_magnet):
    b_norm = np.linalg.norm(B_magnet)
    B123 = np.zeros((3,3))
    B4 = np.array([(-(B_magnet[2]**2)-B_magnet[1]**2)/(b_norm*I_x), B_magnet[1]*B_magnet[0]/(b_norm*I_x), B_magnet[2]*B_magnet[0]/(b_norm*I_x)])
    B5 = np.array([B_magnet[0]*B_magnet[1]/(b_norm*I_y), (-B_magnet[2]**2-B_magnet[0]**2)/(b_norm*I_y), B_magnet[2]*B_magnet[1]/(b_norm*I_y)])
    B6 = np.array([B_magnet[0]*B_magnet[2]/(b_norm*I_z), B_magnet[1]*B_magnet[2]/(b_norm*I_z), (-B_magnet[1]**2-B_magnet[0]**2)/(b_norm*I_z)])
    
    B_k = np.vstack((B123,B4,B5,B6))
    #B_k = np.array([B123,B4,B5,B6])

    return B_k

def rk4_step_PD(dynamics, x, A, B, u, h):
    k1 = h * dynamics(A, x, B, u)
    k2 = h * dynamics(A, x + 0.5 * k1, B, u)
    k3 = h * dynamics(A, x + 0.5 * k2, B, u)
    k4 = h * dynamics(A, x + k3, B, u)
        
    # Update components of q
    q0_new = x[0] + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
    q1_new = x[1] + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
    q2_new = x[2] + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
    
    q_new_real = np.array([q0_new, q1_new, q2_new])

    # Update components of w
    w0_new = x[3] + (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6
    w1_new = x[4] + (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]) / 6
    w2_new = x[5] + (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5]) / 6
    w_new = np.array([w0_new, w1_new, w2_new])

    return q_new_real, w_new

def dynamics(A, x, B, u):
    return np.dot(A, x) - np.dot(B, u)

def K(Kp_x, Kp_y, Kp_z, Kd_x, Kd_y, Kd_z):
    K_gain = np.hstack((np.diag([Kp_x,Kp_y,Kp_z]), np.diag([Kd_x,Kd_y,Kd_z])))
    return K_gain

def quaternion_to_dcm(q):
    x, y, z, w = q
    dcm = np.array([
        [w**2 + x*2 + 2*y**2 + 2*z**2, 2*x*y + 2*w*z, 2*x*z - 2*w*y],
        [2*x*y - 2*w*z, w**2-x**2+y**2+z**2, 2*y*z + 2*w*x],
        [2*x*z + 2*w*y, 2*y*z - 2*w*x, w**2-x**2-y**2+z**2]
    ])
    return dcm

def high_pass_filter(signal, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def low_pass_filter(signal, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal