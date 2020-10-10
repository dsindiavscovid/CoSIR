import numpy as np

def sir_equations(y, t, N, beta, gamma):
    """
    Setup the differential equations defining the SIR model
    
    Args:
        y : (S, I, R) - tuple of S, I, R variables
        t : time variable
        beta : Beta parameter of the model
        gamma : Infectious period
    """
    S, I, R = y
    dSdt = -beta*S*I/N
    dIdt = beta*S*I/N - gamma*I
    dRdt = gamma*I
    return dSdt, dIdt, dRdt

def lv_equations(y, t, r, e, b, d):
    """
    Setup the differential equations defining the LV system
    
    Args:
        y - tuple of (X, Y) prey-predator variables
        r : reproduction rate of prey
        e : rate at which prey is consumed per predator
        b : birth rate of predator
        d : death rate of predator per prey
    """
    X, Y = y
    dXdt = r*X - e*X*Y
    dYdt = b*X*Y - d*Y
    return dXdt, dYdt

def sir_lv_equations(y, t, N, r, e, gamma):
    """
    Setup the differential equations defining the SIR model
    formulated as an LV system
    
    Args:
        y : (S, I, R, beta) - tuple of S, I, R variables
        t : time variable
        beta : Beta parameter of the model
        gamma : Infectious period
    """
    S, I, R, beta = y
    dSdt = -beta*S*I/N
    dIdt = beta*S*I/N - gamma*I 
    dRdt = gamma*I
    dbetadt = r*beta - e*beta*I + (beta**2)*I/N
    return dSdt, dIdt, dRdt, dbetadt


def compute_w(beta, S, I, gamma, N, r, e):
    Jstar = gamma*N
    Istar = r/e
    W = beta*S/N - gamma*np.log(beta*S/Jstar) + e*I - r*np.log(I/Istar)
    return W

def sir_lv_control_equations(y, t, N, r, e, gamma, eta):
    """
    Setup the differential equations defining the SIR model
    formulated as an LV system. Apply control action on beta
    to achieve a desired W() = gamma + r

    Args:
        y : (S, I, R, beta) - tuple of S, I, R, beta variables
        t : time variable
        N : Total population
        beta : Beta parameter of the model
        gamma : Infectious period
        eta : Learning rate for the SG update.
    """
    S, I, R, beta = y
    W = compute_w(beta, S, I, gamma, N, r, e)
    Wstar = gamma + r
    u = -eta*(W - Wstar)*((beta*S)/(gamma*N) - 1)
    dSdt = -beta*S*I/N
    dIdt = beta*S*I/N - gamma*I
    dRdt = gamma*I
    dbetadt = r*beta - e*beta*I + (beta**2)*I/N + beta*u
    return dSdt, dIdt, dRdt, dbetadt


