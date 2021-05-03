import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

def estInitialize():
    # Fill in whatever initialization you'd like here. This function generates
    # the internal state of the estimator at time 0. You may do whatever you
    # like here, but you must return something that is in the format as may be
    # used by your estRun() function as the first returned variable.
    #
    # The second returned variable must be a list of student names.
    # 
    # The third return variable must be a string with the estimator type

    #we make the internal state a list, with the first three elements the position
    # x, y; the angle theta; and our favorite color. 
    x = 0
    y = 0
    theta = np.pi/4
    
    #We say B and r are uniformely distributed around their nominal values
    #The length does not change during the run so we only initialize one value here
    u_r = np.random.uniform(0.95, 1.05)
    u_B = np.random.uniform(0.9, 1.1)
    r = 0.425*u_r #Tire radius in m
    B = 0.8*u_B #Baseline in m
    Pm = np.eye(3)
    xm = np.array([x, y, theta])
    
    # note that there is *absolutely no prescribed format* for this internal state.
    # You can put in it whatever you like. Probably, you'll want to keep the position
    # and angle, and probably you'll remove the color.
    internalState = [x,
                     y,
                     theta,
                     r,
                     B,
                     Pm]
                     

    # replace these names with yours. Delete the second name if you are working alone.
    studentNames = ['Henry FONG', 'James LIAO', 'Alexis RUIZ']

    
    # replace this with the estimator type. Use one of the following options:
    #  'EKF' for Extended Kalman Filter
    #  'UKF' for Unscented Kalman Filter
    #  'PF' for Particle Filter
    #  'OTHER: XXX' if you're using something else, in which case please
    #                 replace "XXX" with a (very short) description
    estimatorType = 'UKF' 
    
    return internalState, studentNames, estimatorType

