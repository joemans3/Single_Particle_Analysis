import numpy as np
import matplotlib.pyplot as plt

def MCMC_state_selection(initial_state_index:int, 
                         transition_matrix:np.ndarray,
                         possible_states:np.ndarray,
                         n:int):
    '''Markov Chain Monte Carlo state selection

    Parameters:
    -----------
    initial_state_index : int
        Initial state index, this is the index of the initial state in the possible states
    transition_matrix : np.ndarray
        Transition matrix, this is the stocastic rate constants with units 1/dt (time step which is the iteration step)
    possible_states : np.ndarray
        possible states
    n : int
        Number of iterations
    
    Returns:
    --------
    np.ndarray
        State selection at each iteration
    '''
    #initialize the state selection
    state_selection = np.zeros(n)
    #initialize the current state
    current_state = possible_states[initial_state_index]
    current_state_index = initial_state_index
    #find the total rate constant for each state (cache the total rate constant given a state so it does not have to be recalculated)
    total_rate_constant = np.zeros(len(possible_states))
    for i in range(len(possible_states)):
        total_rate_constant[i] = np.sum(transition_matrix[i])
    t = 0
    while t < n:
        #find the rate constant for transitioning from the current state
        tot_rate_transition = total_rate_constant[current_state_index] - transition_matrix[current_state_index][current_state_index]
        #if tot_rate_transition is 0 then the current state is the only state viable based on the transition matrix
        if tot_rate_transition == 0:
            state_selection[int(t):n] = current_state
            break
        #find the time to transition
        tau = -np.log(np.random.rand())/tot_rate_transition
        #find the next state
        next_state_index = np.random.choice(len(possible_states), p=transition_matrix[current_state_index]/total_rate_constant[current_state_index])
        next_state = possible_states[next_state_index]
        #for the duration of the transition, the state is the previous state
        state_selection[int(t):int(t+tau)] = current_state
        #update the current state
        current_state = next_state
        current_state_index = next_state_index
        #update the time
        t += tau
    return state_selection

class FBM_BP:
    def __init__(self,
                 n:int,
                 dt:float,
                 diffusion_parameters:np.ndarray,
                 hurst_parameters:np.ndarray,
                 diffusion_parameter_transition_matrix:np.ndarray,
                 hurst_parameter_transition_matrix:np.ndarray,
                 state_probability_diffusion:np.ndarray,
                 state_probability_hurst:np.ndarray,
                 space_lim:float):
        self.n = n
        self.dt = dt#ms
        self.diffusion_parameter = diffusion_parameters
        self.hurst_parameter = hurst_parameters
        self.diffusion_parameter_transition_matrix = diffusion_parameter_transition_matrix #stocastic rate constants with units 1/dt
        self.hurst_parameter_transition_matrix = hurst_parameter_transition_matrix #stocastic rate constants with units 1/dt
        self.state_probability_diffusion = state_probability_diffusion #probability of the initial state, this approximates the population distribution
        self.state_probability_hurst = state_probability_hurst #probability of the initial state, this approximates the population distribution
        self.space_lim = space_lim
    def _autocovariance(self,k,dt,hurst,diff_a):
        '''Autocovariance function for fGn

        Parameters:
        -----------
        k : int
            Lag
        dt : float
            Time step
        hurst : float
            Hurst parameter
        diff_a : float
            Diffusion coefficient related to the Hurst parameter
        
        Returns:
        --------
        float
            Autocovariance function
        '''
        return diff_a*(dt**(2*hurst))*(abs(k - 1) ** (2 * hurst) - 2 * abs(k) ** (2 * hurst) + abs(k + 1) ** (2 * hurst))
    def fbm(self):
        fgn = np.zeros(self.n)
        fbm_store = np.zeros(self.n)
        phi = np.zeros(self.n)
        psi = np.zeros(self.n)
        diff_a_n = np.zeros(self.n)
        hurst_n = np.zeros(self.n)

        #using the state probability to choose the diffusion and hurst parameter to start
        diff_a_start = np.random.choice(self.diffusion_parameter, p=self.state_probability_diffusion)
        hurst_start = np.random.choice(self.hurst_parameter, p=self.state_probability_hurst)
        diff_a_n[0] = diff_a_start
        hurst_n[0] = hurst_start
        #use the MC state selection to choose the diffusion and hurst parameter at each iteration
        diff_a_n[1:] = MCMC_state_selection(np.where(self.diffusion_parameter == diff_a_start)[0][0], self.diffusion_parameter_transition_matrix, self.diffusion_parameter, self.n-1)
        hurst_n[1:] = MCMC_state_selection(np.where(self.hurst_parameter == hurst_start)[0][0], self.hurst_parameter_transition_matrix, self.hurst_parameter, self.n-1)
        #generate the autocovariance matrix using the chosen diffusion and hurst parameter
        self._cov = np.zeros(self.n)
        for i in range(self.n):
            self._cov[i] = self._autocovariance(i, self.dt, hurst_n[i], diff_a_n[i])

        #construct a gaussian noise vector
        gn = np.random.normal(0, 1, self.n)
        #catch is all hurst are 0.5 then use the gaussian noise vector corresponding to the scale defined by the diffusion parameter
        if np.all(hurst_n == 0.5):
            #each gn is then pulled from a normal distribution with mean 0 and standard deviation diff_a_n
            gn = gn * np.sqrt(2*diff_a_n)
            #ignore the fbm calculations but keep the reflection
            for i in range(1, self.n):
                fbm_candidate = fbm_store[i - 1] + gn[i]
                #check if this is outside the space limit in either direction of 0
                if abs(fbm_candidate) > self.space_lim:
                    if fbm_candidate > self.space_lim:
                        #if the candidate is greater than the space limit then reflect the difference back into the space limit
                        fbm_store[i] = self.space_lim - (fbm_candidate - self.space_lim)
                    else:
                        #if the candidate is less than the negative space limit then reflect the difference back into the space limit
                        fbm_store[i] = -self.space_lim + (-fbm_candidate - self.space_lim)
                else:
                    fbm_store[i] = fbm_candidate
            return fbm_store


        fbm_store[0] = 0
        fgn[0] = gn[0]
        v = 1
        phi[0] = 0

        for i in range(1, self.n):
            phi[i - 1] = self._cov[i]
            for j in range(i - 1):
                psi[j] = phi[j]
                phi[i - 1] -= psi[j] * self._cov[i - j - 1]
            phi[i - 1] /= v
            for j in range(i - 1):
                phi[j] = psi[j] - phi[i - 1] * psi[i - j - 2]
            v *= 1 - phi[i - 1] * phi[i - 1]
            for j in range(i):
                fgn[i] += phi[j] * fgn[i - j - 1]
            #catch the error if v is negative
            if v < 0:   
                print('v is negative: ', "value of v: ", v, "iteration: ", i, "phi: ", phi[i - 1], "fgn: ", fgn[i], "gn: ", gn[i])
            fgn[i] += np.sqrt(v) * gn[i]
            #add to the fbm
            fbm_candidate = fbm_store[i - 1] + fgn[i]
            #check if this is outside the space limit in either direction of 0
            if abs(fbm_candidate) > self.space_lim:
                #reflect the difference back into the space limit
                if fbm_candidate > self.space_lim:
                    #if the candidate is greater than the space limit then reflect the difference back into the space limit
                    fbm_store[i] = self.space_lim - (fbm_candidate - self.space_lim)
                    #update the fgn based on the new difference
                    fgn[i] = fbm_store[i] - fbm_store[i - 1]
                else:
                    #if the candidate is less than the negative space limit then reflect the difference back into the space limit
                    fbm_store[i] = -self.space_lim + (-fbm_candidate - self.space_lim)
                    #update the fgn based on the new difference
                    fgn[i] = fbm_store[i] - fbm_store[i - 1]
            else:
                fbm_store[i] = fbm_candidate
        print("fbm: ", fbm_store,"length: ", len(fbm_store))
        print("fgn: ", fgn,"length: ", len(fgn))
        print("cumsum: ", np.cumsum(fgn),"length: ", len(np.cumsum(fgn)))
        return fbm_store 


#run tests if this is the main module

if __name__ == "__main__":

    # # test the FBM_BP class
    # n = 100
    # dt = 1
    # diffusion_parameters = np.array([0.1, 10])
    # hurst_parameters = np.array([0.9, 0.9])
    # diffusion_parameter_transition_matrix = np.array([[0.01, 0.01],
    #                                                 [0.01, 0.01]])
    # hurst_parameter_transition_matrix = np.array([[0.9, 0.1],
    #                                             [0.1, 0.9]])
    # state_probability_diffusion = np.array([1, 0])
    # state_probability_hurst = np.array([0.5, 0.5])
    # space_lim = 1000
    # fbm_bp = FBM_BP(n, dt, diffusion_parameters, hurst_parameters, diffusion_parameter_transition_matrix, hurst_parameter_transition_matrix, state_probability_diffusion, state_probability_hurst, space_lim)
    # # test the fbm method
    # fbm = fbm_bp.fbm()
    # # plot the fbm
    # plt.plot(fbm, linestyle='--')
    # plt.xlabel('Iteration')
    # plt.ylabel('Value')
    # plt.title('Fractional Brownian motion')
    # plt.show()

    # # test the MCMC_state_selection function
    # # initialize the transition matrix
    # transition_matrix = np.array([[0.9, 0.1],
    #                             [0.1, 0.9]])
    # # initialize the possible states
    # possible_states = np.array([1, 2])
    # # initialize the number of iterations
    # n = 10000
    # # initialize the initial state index
    # initial_state_index = 1 

    # # test the MCMC_state_selection function
    # state_selection = MCMC_state_selection(initial_state_index, transition_matrix, possible_states, n)
    # # plot the state selection
    # plt.plot(state_selection)
    # plt.xlabel('Iteration')
    # plt.ylabel('State')
    # plt.title('State selection')
    # plt.show()

    # # plot the probability of each state
    # state_probability = np.zeros(len(possible_states))
    # for i in range(len(possible_states)):
    #     state_probability[i] = np.sum(state_selection == possible_states[i])/n

    # # compare the population distribution with the state probability distribution
    # total_rate = np.sum(transition_matrix)
    # # add the column of the transition matrix and divide by the total rate
    # true_state_probability = np.sum(transition_matrix, axis=0)/total_rate
    # plt.bar(possible_states, state_probability, label='State probability distribution', alpha=0.5)
    # plt.bar(possible_states, true_state_probability, label='Population distribution', alpha=0.5)
    # plt.xlabel('State')
    # plt.ylabel('Probability')
    # plt.title('State probability distribution')
    # plt.legend()
    # plt.show()

    pass