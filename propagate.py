import numpy as np
#import time as time

from .npropagateS15 import npropagateS15


def runge_kutta(t, efields, n_recorded, hamiltonian):
    """ Evolves the hamiltonian in time using the runge_kutta method.

    Parameters
    ----------
    t : 1-D array of float
        Time points, equally spaced array.
        Shape T, number of timepoints
    efields : ndarray <Complex>
        Time dependant electric fields for all pulses.
        SHape M x T where M is number of electric fields, T is number of time points.
    n_recorded : int
        Number of timesteps to record at the end of the simulation.
    hamiltonian : Hamiltonian
        The hamiltonian object which contains the inital conditions and the 
            function to use to obtain the matrices.

    Returns
    -------
    ndarray : <Complex>
        2-D array of recorded density vector elements for each time step in n_recorded.
    """
    
    # can only call on n_recorded and t after efield_object.E is called
    dt = np.abs(t[1]-t[0])
    # extract attributes of the system
    rho_emitted = np.empty((len(hamiltonian.recorded_indices), n_recorded), dtype=np.complex128)
    # index to keep track of elements of rho_emitted
    emitted_index = 0
    # H has 3 dimensions: time and the 2 matrix dimensions
    H = hamiltonian.matrix(efields, t)
    rho_i = hamiltonian.rho.copy()

      
    
    # cython code block test starts here.
      
    # rho_tmp is an np.float64 array construct to provide a memoryview for npropagatS15.   It needs to be
    # equal in shape to [all_rhos, n_recorded] = [9, n_recorded].  Must be created as float64 not complex.
    
    #begin = time.perf_counter()
    rho_tmp = np.zeros((9,n_recorded), dtype=np.float64)
    rho_o= npropagateS15(H.real,H.imag,rho_i.real,rho_i.imag,dt,len(t),n_recorded, rho_tmp)
    
    for ind,k2 in enumerate(hamiltonian.recorded_indices):
        rho_emitted[ind, :] = rho_o[k2,:]
    #print(time.perf_counter()-begin) 
    return rho_emitted

    #end cython code block test
    
 
