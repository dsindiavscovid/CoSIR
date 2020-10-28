from scipy.integrate import odeint
import numpy as np

from models import sir_equations
from models import lv_equations
from models import sir_lv_equations
from models import sir_lv_control_equations


def solve(equations, init, t, args):
    """
    General version of the solver.
    Takes the equations, initial conditions and args
    as inputs.
    However, it is not easy to make the inputs easy
    to parse and hence this version is not used.

    TODO: Check if we can make the order of initial
    values and arguments apparent in the function signature.
    """
    ret = odeint(equations, init, t, args=args)
    return ret.T

def solve_sir(S0, I0, R0, N, beta, gamma, t):
    """
    S0, I0, R0 - Initial values on the S, I, R buckets
    N - Total population size
    beta, gamma : Model parameters
    t - Grid of time points (in days)
    """
    y0 = S0, I0, R0
    ret = odeint(sir_equations, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return S, I, R

def solve_lv(X0, Y0, r, e, b, d, t):
    """
    X0, Y0 - Initial number of prey and predator
    r, e, b, d - Model parameters
    t - Grid of time points (in days)
    """
    ret = odeint(lv_equations, (X0, Y0), t, args=(r, e, b, d))
    X, Y = ret.T
    return X, Y

def solve_sir_lv(S0, I0, R0, beta0, N, r, e, gamma, t):
    """
    S0, I0, R0 - Initial values on the S, I, R buckets
    beta0 - Initial value of beta0
    N - Total population size
    gamma : Gamma parameter
    t - Grid of time points (in days)
    """
    y0 = S0, I0, R0, beta0
    ret = odeint(sir_lv_equations, y0, t, args=(N, r, e, gamma))
    S, I, R, beta = ret.T
    return S, I, R, beta

def solve_sir_lv_control(S0, I0, R0, beta0, N, r, e, gamma, eta, t):
    """
    S0, I0, R0 - Initial values on the S, I, R buckets
    beta0 - Initial value of beta0
    N - Total population size
    gamma : Gamma parameter
    eta : Learning rate for SG updates
    t - Grid of time points (in days)
    """
    y0 = S0, I0, R0, beta0
    ret = odeint(sir_lv_control_equations, y0, t, args=(N, r, e, gamma, eta))
    S, I, R, beta = ret.T
    return S, I, R, beta

def solve_discrete_delayedSir(init, beta, gamma, N, timesteps, delta_t=0.001, tau = 4):
    """
        Discrete version of SIR model.

        init : Initial conditions
        beta : Sequence of beta values (constant or array)
        gamma : Gamma parameter of SIR model
        timesteps : Number of timesteps to consider (in units of days)
        delta_t : Spacing between points in the discretised version
    """
    S0, I0, R0 = init
    tauIdx = int(tau/delta_t)
    #Checks on beta
    if isinstance(beta, np.ndarray):
        assert beta.shape[0] == timesteps
    elif isinstance(beta, float):
        beta = np.repeat(beta, timesteps)
    else:
       raise ValueError('Incorrect argument to beta parameter') 

    num_repetitions = np.int(np.ceil(1/delta_t))
    num_points = num_repetitions*timesteps #1 day will have 1/delta_t timesteps
    S, I, R = np.zeros(num_points), np.zeros(num_points), np.zeros(num_points)

    S[0], I[0], R[0] = S0, I0, R0

    # Replicate the same beta across the timesteps introduced
    # between days
    beta = np.repeat(beta, num_repetitions)

    for t in range(num_points-1):
        rate_s2i = beta[t]*S[t]*I[t]/N # beta*S*I/N
        rate_i2r = gamma*I[t]          # gamma*I
        S[t+1] = S[t] + delta_t*(-rate_s2i)
        R[t+1] = R[t] + delta_t*(rate_i2r)
        if(t<=tauIdx):
            rate_incomingI = 0
        else:
            rate_incomingI = beta[t-tauIdx]*S[t-tauIdx]*I[t-tauIdx]/N
        I[t+1] = I[t] + delta_t*(rate_incomingI - rate_i2r)
            

    # Pick the elements based on days as time-steps
    S = S[::num_repetitions]
    I = I[::num_repetitions]
    R = R[::num_repetitions]

    return S, I, R

def solve_discrete_SEIR(init, beta, gamma, sigma, N, timesteps, delta_t=0.001):
    """
        Discrete version of SIR model.

        init : Initial conditions
        beta : Sequence of beta values (constant or array)
        gamma : Gamma parameter of SIR model
        timesteps : Number of timesteps to consider (in units of days)
        delta_t : Spacing between points in the discretised version
    """
    S0, E0, I0, R0 = init
    #Checks on beta
    if isinstance(beta, np.ndarray):
        assert beta.shape[0] == timesteps
    elif isinstance(beta, float):
        beta = np.repeat(beta, timesteps)
    else:
       raise ValueError('Incorrect argument to beta parameter') 

    num_repetitions = np.int(np.ceil(1/delta_t))
    num_points = num_repetitions*timesteps #1 day will have 1/delta_t timesteps
    S, E, I, R = np.zeros(num_points), np.zeros(num_points), np.zeros(num_points), np.zeros(num_points)

    S[0], E[0], I[0], R[0] = S0, E0, I0, R0

    # Replicate the same beta across the timesteps introduced
    # between days
    beta = np.repeat(beta, num_repetitions)

    for t in range(num_points-1):
        rate_s2e = beta[t]*S[t]*I[t]/N # beta*S*I/N
        rate_e2i = sigma*E[t]
        rate_i2r = gamma*I[t]          # gamma*I

        #dSdt = -beta*S*I/N
        S[t+1] = S[t] + delta_t*(-rate_s2e)
        
        E[t+1] = E[t] + delta_t*(rate_s2e  - rate_e2i)

        I[t+1] = I[t] + delta_t*(rate_e2i - rate_i2r)

        #dRdt = gamma*I
        R[t+1] = R[t] + delta_t*(rate_i2r)

    # Pick the elements based on days as time-steps
    S = S[::num_repetitions]
    E = E[::num_repetitions]
    I = I[::num_repetitions]
    R = R[::num_repetitions]

    return S, E, I, R

def solve_discrete_sir(init, beta, gamma, N, timesteps, delta_t=0.001):
    """
        Discrete version of SIR model.

        init : Initial conditions
        beta : Sequence of beta values (constant or array)
        gamma : Gamma parameter of SIR model
        timesteps : Number of timesteps to consider (in units of days)
        delta_t : Spacing between points in the discretised version
    """
    S0, I0, R0 = init

    #Checks on beta
    if isinstance(beta, np.ndarray):
        assert beta.shape[0] == timesteps
    elif isinstance(beta, float):
        beta = np.repeat(beta, timesteps)
    else:
       raise ValueError('Incorrect argument to beta parameter') 

    num_repetitions = np.int(np.ceil(1/delta_t))
    num_points = num_repetitions*timesteps #1 day will have 1/delta_t timesteps
    S, I, R = np.zeros(num_points), np.zeros(num_points), np.zeros(num_points)

    S[0], I[0], R[0] = S0, I0, R0

    # Replicate the same beta across the timesteps introduced
    # between days
    beta = np.repeat(beta, num_repetitions)

    for t in range(num_points-1):
        rate_s2i = beta[t]*S[t]*I[t]/N # beta*S*I/N
        rate_i2r = gamma*I[t]          # gamma*I

        #dSdt = -beta*S*I/N
        S[t+1] = S[t] + delta_t*(-rate_s2i)

        #dIdt = beta*S*I/N - gamma*I
        I[t+1] = I[t] + delta_t*(rate_s2i - rate_i2r)

        #dRdt = gamma*I
        R[t+1] = R[t] + delta_t*(rate_i2r)

    # Pick the elements based on days as time-steps
    S = S[::num_repetitions]
    I = I[::num_repetitions]
    R = R[::num_repetitions]

    return S, I, R


def compute_w(beta, S, I, gamma, N, r, e):
    Jstar = gamma*N
    Istar = r/e
    W = beta*S/N - gamma*np.log(beta*S/Jstar) + e*I - r*np.log(I/Istar)
    return W



# def solve_discrete_delayedSIR_LV_Control(init, gamma, N, timesteps, r, e, delta_t=0.001, tau = 5, eta=1):
#     """
#         Discrete version of SIR model.

#         init : Initial conditions
#         beta : Sequence of beta values (constant or array)
#         gamma : Gamma parameter of SIR model
#         timesteps : Number of timesteps to consider (in units of days)
#         delta_t : Spacing between points in the discretised version
#     """
#     S0, I0, R0, beta0 = init
#     tauIdx = int(tau/delta_t)


#     num_repetitions = np.int(np.ceil(1/delta_t))
#     num_points = num_repetitions*timesteps #1 day will have 1/delta_t timesteps
#     S, I, R, beta, Ws = np.zeros(num_points), np.zeros(num_points), np.zeros(num_points), np.zeros(num_points), np.zeros(num_points)

#     S[0], I[0], R[0], beta[0] = S0, I0, R0, beta0
#     Wstar = gamma + r
#     for t in range(num_points-1):
#         W = compute_w(beta[t], S[t], I[t], gamma, N, r, e)
#         Ws[t] = W
#         u = -eta*(W - Wstar)*((beta[t]*S[t])/(gamma*N) - 1)
#         if(t<tauIdx+1):
#             rate_incomingI = 0
#         else:
#             rate_incomingI = beta[t - tauIdx]*S[t - tauIdx]*I[t - tauIdx]/N
#         rate_s2i = beta[t]*S[t]*I[t]/N 
#         rate_i2r = gamma*I[t]          
#         rate_beta = r*beta[t] - e*beta[t]*I[t] + (beta[t]**2)*I[t]/N + beta[t]*u
#         S[t+1] = S[t] + delta_t*(-rate_s2i)
#         beta[t+1] = beta[t] + delta_t*rate_beta
#         I[t+1] = I[t] + delta_t*(rate_incomingI - rate_i2r)


#         R[t+1] = R[t] + delta_t*(rate_i2r)
            
#     # Pick the elements based on days as time-steps
#     S = S[::num_repetitions]
#     I = I[::num_repetitions]
#     R = R[::num_repetitions]
#     beta = beta[::num_repetitions]
#     Ws = Ws[::num_repetitions]
#     return S, I, R, beta, Ws


def solve_discrete_delayedSIR_LV_Control(init, gamma, N, timesteps, r, e, delta_t=0.001, tau = 5, eta=1):
    """
        Discrete version of SIR model.

        init : Initial conditions
        beta : Sequence of beta values (constant or array)
        gamma : Gamma parameter of SIR model
        timesteps : Number of timesteps to consider (in units of days)
        delta_t : Spacing between points in the discretised version
    """
    S0, I0, R0, beta0 = init
    tauIdx = int(tau/delta_t)
    epsilon = 1e-4

    num_repetitions = np.int(np.ceil(1/delta_t))
    num_points = num_repetitions*timesteps #1 day will have 1/delta_t timesteps
    S, I, R, beta, Ws, Us = np.zeros(num_points), np.zeros(num_points), np.zeros(num_points), np.zeros(num_points), np.zeros(num_points), np.zeros(num_points)

    S[0], I[0], R[0], beta[0] = S0, I0, R0, beta0
    Wstar = gamma + r
    Istar = r/e
    Jstar = gamma*N
    
    for t in range(num_points-1):
        W = compute_w(beta[t], S[t], I[t], gamma, N, r, e)
        Ws[t] = W
#         u = -eta*(W - Wstar)*((beta[t]*S[t])/(gamma*N) - 1)
        if(t<tauIdx+1):
            u = 0
            rate_incomingI = 0
        else:
            y = I[t]/Istar
            x = beta[t]*S[t]/Jstar
            yprime = I[t-tauIdx]/Istar
            xprime = beta[t-tauIdx]*S[t-tauIdx]/Jstar
            if(np.abs(x-1)<epsilon):
                u = 0
            else:
                u = -eta*(W - Wstar)*(x-1) - r*(y-1)*(xprime*yprime - x*y)/(x*y-y)
            rate_incomingI = beta[t - tauIdx]*S[t - tauIdx]*I[t - tauIdx]/N
#             print(f't:{t}, u: {u}, x: {x}, y:{y}, beta:{beta[t]}, W: {Ws[t]}, denom:{(x*y-y)}')
            
        Us[t] = u
        rate_s2i = beta[t]*S[t]*I[t]/N 
        rate_i2r = gamma*I[t]          
        rate_beta = r*beta[t] - e*beta[t]*I[t] + (beta[t]**2)*I[t]/N + beta[t]*u
        S[t+1] = S[t] + delta_t*(-rate_s2i)
        beta[t+1] = beta[t] + delta_t*rate_beta
        I[t+1] = I[t] + delta_t*(rate_incomingI - rate_i2r)


        R[t+1] = R[t] + delta_t*(rate_i2r)
            
    # Pick the elements based on days as time-steps
#     S = S[::num_repetitions]
#     I = I[::num_repetitions]
#     R = R[::num_repetitions]
#     beta = beta[::num_repetitions]
#     Ws = Ws[::num_repetitions]
#     Us = Us[::num_repetitions]
    return S, I, R, beta, Ws, Us

def solve_discrete_sir_jump(init, beta, gamma, N, timesteps, spikes, delta_t=0.001):
    """
    Discrete version of SIR model.

    init : Initial conditions
    beta : Sequence of beta values (constant or array)
    gamma : Gamma parameter of SIR model
    timesteps : Number of timesteps to consider (in units of days)
    delta_t : Spacing between points in the discretised version
    """
    S0, I0, R0 = init

    #Checks on beta
    if isinstance(beta, np.ndarray):
        assert beta.shape[0] == timesteps
    elif isinstance(beta, float):
        beta = np.repeat(beta, timesteps)
    else:
       print(type(beta))
       raise ValueError('Incorrect argument to beta parameter') 

    num_repetitions = np.int(np.ceil(1/delta_t))
    num_points = num_repetitions*timesteps #1 day will have 1/delta_t timesteps
    tspikeIdxs = [spike[0] * num_repetitions for spike in spikes]
    print(spikes)
    S, I, R = np.zeros(num_points), np.zeros(num_points), np.zeros(num_points)

    S[0], I[0], R[0] = S0, I0, R0

    # Replicate the same beta across the timesteps introduced
    # between days
    beta = np.repeat(beta, num_repetitions)
    spikesDone = 0
    for t in range(num_points-1):
        rate_s2i = beta[t]*S[t]*I[t]/N # beta*S*I/N
        rate_i2r = gamma*I[t]          # gamma*I

        #dSdt = -beta*S*I/N
        S[t+1] = S[t] + delta_t*(-rate_s2i)

        #dIdt = beta*S*I/N - gamma*I
        I[t+1] = I[t] + delta_t*(rate_s2i - rate_i2r)
        if(spikesDone< len(spikes) and t == tspikeIdxs[spikesDone]):
            I[t+1] = I[t+1] + spikes[spikesDone][1]
            print("At {} I goes up by {}".format(t/num_repetitions, spikes[spikesDone][1]))
            if(spikes[spikesDone][1] > 0):
                S[t+1] = S[t+1] - spikes[spikesDone][1]
                print("At {} S goes down by {}".format(t/num_repetitions, spikes[spikesDone][1]))
            else:
                R[t+1] = R[t+1] - spikes[spikesDone][1]
                print("At {} R goes down by {}".format(t/num_repetitions, spikes[spikesDone][1]))
            spikesDone += 1

        #dRdt = gamma*I
        R[t+1] = R[t] + delta_t*(rate_i2r)

    # Pick the elements based on days as time-steps
    S = S[::num_repetitions]
    I = I[::num_repetitions]
    R = R[::num_repetitions]

    return S, I, R
