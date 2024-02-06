import numpy as np
# from tqdm import tqdm
from numba import njit
from constants import ITERATIONS, LENGHT_SEQUENCE, WIDTH, HEIGHT

############################ Parameters #############################

# Flow parameters
REYNOLDS_NUMBER = 220.0
L = HEIGHT//9                       # Length
uLB = 0.04                      # Velocity (in lattice units)
nulb = uLB*L/REYNOLDS_NUMBER    # Kinematic Viscoscity (in lattice units)
omega = 1 / (3*nulb+0.5)        # Relaxation parameter

# Lattice Constants
# Velocities
v = np.array([
    [1,1],[1,0],[1,-1],[0,1],[0,0], 
    [0,-1],[-1,1],[-1,0],[-1,-1] 
])
# Weights
t = np.array([1/36,1/9,1/36,1/9,4/9,1/9,1/36,1/9,1/36])

col1 = np.array([0, 1, 2])
col2 = np.array([3, 4, 5])
col3 = np.array([6, 7, 8])

################################################################################

@njit
def macroscopic(fin):
    """
    Calculates the macroscopic density and velocity
    """
    density = np.sum(fin, axis=0)
    velocity = np.zeros((2, WIDTH, HEIGHT))
    for i in range(9):
        velocity[0,:,:] += v[i,0] * fin[i,:,:]
        velocity[1,:,:] += v[i,1] * fin[i,:,:]
    velocity /= density
    return density, velocity

@njit
def equilibrium(density, velocity):
    """
    Calculates the equilibrium distribution
    """
    usqr = 3/2 * (velocity[0]**2 + velocity[1]**2)
    feq = np.zeros((9,WIDTH,HEIGHT))
    for i in range(9):
        cu = 3 * (v[i,0]*velocity[0,:,:] + v[i,1]*velocity[1,:,:])
        feq[i,:,:] = density*t[i] * (1 + cu + 0.5*cu**2 - usqr)
    return feq

@njit
def inlet_velocity(d, x, y):
    ly = HEIGHT-1
    return (1-d) * uLB * (1 + 1e-4*np.sin(y/ly*2*np.pi))

def cfd_step(sequence, fin, vel, obstacle, time):
    # Outflow boundary at right wall
    fin[col3,-1,:] = fin[col3,-2,:] 

    # Compute macroscopic variables
    density, velocity = macroscopic(fin)

    # Inflow at left wall
    velocity[:,0,:] = vel[:,0,:]
    density[0,:] = 1/(1-velocity[0,0,:]) * (np.sum(fin[col2,0,:], axis=0) +
                                            2*np.sum(fin[col3,0,:], axis=0))
    # Compute equilibrium
    feq = equilibrium(density, velocity)
    fin[[0,1,2],0,:] = feq[[0,1,2],0,:] + fin[[8,7,6],0,:] - feq[[8,7,6],0,:]

    # Collision
    fout = fin - omega * (fin - feq)

    # Bounceback for obstacle
    for i in range(9):
        fout[i, obstacle] = fin[8-i, obstacle]

    # Stream
    for i in range(9):
        fin[i,:,:] = np.roll(
                            np.roll(fout[i,:,:], v[i,0], axis=0),
                            v[i,1], axis=1 
                        )

    # Save state in dataset
    if (time%100==0):
        frame_id = time//100
        frame = np.sqrt(velocity[0]**2+velocity[1]**2)
        sequence[frame_id, : ] = frame


def run_cfd(obstacle):
    sequence = np.zeros((LENGHT_SEQUENCE, WIDTH, HEIGHT))
    
    # Initialize simulation

    vel = np.fromfunction(inlet_velocity, (2,WIDTH,HEIGHT))
    fin = equilibrium(1, vel)

    # for time in tqdm(range(interations)):
    for time in range(ITERATIONS):
        cfd_step(sequence, fin, vel, obstacle, time)

    return sequence
