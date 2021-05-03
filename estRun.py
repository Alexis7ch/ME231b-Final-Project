import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

def estRun(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement):
    # In this function you implement your estimator. The function arguments
    # are:
    #  time: current time in [s] 
    #  dt: current time step [s]
    #  internalStateIn: the estimator internal state, definition up to you. 
    #  steeringAngle: the steering angle of the bike, gamma, [rad] 
    #  pedalSpeed: the rotational speed of the pedal, omega, [rad/s] 
    #  measurement: the position measurement valid at the current time step
    #
    # Note: the measurement is a 2D vector, of x-y position measurement.
    #  The measurement sensor may fail to return data, in which case the
    #  measurement is given as NaN (not a number).
    #
    # The function has four outputs:
    #  x: your current best estimate for the bicycle's x-position
    #  y: your current best estimate for the bicycle's y-position
    #  theta: your current best estimate for the bicycle's rotation theta
    #  internalState: the estimator's internal state, in a format that can be understood by the next call to this function

    # Example code only, you'll want to heavily modify this.
    # this internal state needs to correspond to your init function:
    
    #Extract the previous estimations/initialization
    x = internalStateIn[0]
    y = internalStateIn[1]
    theta = internalStateIn[2]
    r = internalStateIn[3]
    B = internalStateIn[4]

    v = 5 * r * pedalSpeed #Linear velocity of the bicycle
    
    def q(x, dt, v, steeringAngle): #Define the dynamics of the system x' = q(x)
       v1 = np.random.normal(0,np.sqrt(0.0213))
       v2 = np.random.normal(0,np.sqrt(0.0213))
       v3 = np.random.normal(0,np.sqrt(0.1))
       out_q = [x[0] + v * np.cos(x[2])*dt, 
              x[1] + v * np.sin(x[2])*dt,
              x[2] + v * np.tan(steeringAngle)*dt/B]  
       return out_q
    
    def h(x): #Measurement function
       out_h = [x[0] + 0.5 * B * np.cos(x[2]),
              x[1] + 0.5 * B * np.sin(x[2])]
       return out_h
       
    if type(x) == int: #Only used for the first step
       xm = np.array([0, 0, np.pi/4])
    else:
       xm = np.array([x.item(), y.item(), theta.item()]) 
   
    Pm = internalStateIn[5]
    if type(x) == int or np.linalg.det(Pm) <= 0.1:
       Pm = np.eye(3)
    else: 
       Pm = internalStateIn[5]
    
    #Beginning of the UKF
    chol = np.linalg.cholesky(Pm) #Cholesky decomposition used to create the 6-sigma points
    
    #n = 3 here (dimension of the state)
    
    #Prior update
    sxm = [xm + np.sqrt(3) * chol[:,0], xm + np.sqrt(3) * chol[:,1], xm + np.sqrt(3)*chol[:,2], 
           xm - np.sqrt(3) * chol[:,0], xm - np.sqrt(3) * chol[:,1], xm - np.sqrt(3)*chol[:,2]]

    sxp = [q(sxm[0], dt, v, steeringAngle),
           q(sxm[1], dt, v, steeringAngle),
           q(sxm[2], dt, v, steeringAngle),
           q(sxm[3], dt, v, steeringAngle),
           q(sxm[4], dt, v, steeringAngle),
           q(sxm[5], dt, v, steeringAngle)] 

    #Prior statistics
    xp = (1/6) * np.sum(sxp, axis=0) #Estimation of the mean thanks to the 6-sigma points
    xp = np.reshape(xp , (3,1)) 

    Pp = 0
    for i in range(6): 
       sxp_i = np.reshape(np.array(sxp[i]), (3,1))
       Pp += (sxp_i - xp) @ (sxp_i - xp).T
    Pp = Pp/6 #Variance
    
    #Measurement update step
    
    sz = [h(sxp[0]),
          h(sxp[1]),
          h(sxp[2]),
          h(sxp[3]),
          h(sxp[4]),
          h(sxp[5])]
    
    zp = (1/6) * np.sum(sz, axis = 0) #Compute z^(k)
    zp = np.reshape(zp, (2,1))
    
    #Covariance Pzz(k) and cross covariance Pxz(k)
    Pzz = 0
    Pxz = 0
    for i in range(6): 
       sz_i = np.reshape(np.array(sz[i]), (2,1))
       sxp_i = np.reshape(np.array(sxp[i]), (3,1))
       Pzz += (sz_i - zp) @ (sz_i - zp).T
       Pxz += (sxp_i - xp) @ (sz_i - zp).T
    Pzz = (1/6) * Pzz + np.array([[1.089, 0], [0, 2.988]]) #Covariance Pzz(k) with the noise
    Pxz = (1/6) * Pxz #Compute the cross covariance Pxz(k)
    
    
    ###Now we can compute K(k), and estimate Pm(k) & xm(k)
    K = Pxz @ np.linalg.inv(Pzz)
    
    #Measurements from run_000
    wx = np.random.normal(-0.019, 1.09)
    wy = np.random.normal(1.627, 2.99)
    
    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
       x = measurement[0] 
       y = measurement[1]
        
       zmeas = np.array([[x], [y]])
       xm = xp + K @ (zmeas - zp)
       Pm = Pp - K @ Pzz @ K.T 

    else: 
       xm = xp #Update without measurement, we only use the prediction step
    
    x = xm[0]
    y = xm[1]
    theta = xm[2]

    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function
    # at next run), must obviously be compatible with the format of
    # internalStateIn:
    internalStateOut = [x,
                     y,
                     theta,
                     r,
                     B,
                     Pm]
      

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut 


