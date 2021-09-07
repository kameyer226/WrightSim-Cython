#! /usr/bin/env python3
"""
creates an (d2, w1, w2) "movie" scan of a TRIVE process.

This script utilizes ffmpeg and celluloid to snap frames of w1, w2 as a function of d2.  
The frames are then added and saved as a viewable movie in an MP4 container.  ffmpeg and 
celluloid must be installed.  The MP4 file named at the end of the script will be overwritten
if a file is already located in the path.

In this example, the generic TRIVE.ini file containing default parameters for a TRIVE 
run is modified to numbers below.  

"""


# --- import --------------------------------------------------------------------------------------

import os
import sys
import time

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation as anim

import WrightTools as wt
from WrightSim import experiment as experiment
from WrightSim import hamiltonian as hamiltonian

from celluloid import Camera

Writer = anim.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me')) #, bitrate=1800)

# --- define --------------------------------------------------------------------------------------

here = os.path.abspath(os.path.dirname(__file__))

dt = 50.  # pulse duration (fs)
slitwidth = 120.  # mono resolution (wn)


nw = 16  # number of frequency points (w1 and w2)

nt = 20 # number of delay points (d2)


# --- workspace -----------------------------------------------------------------------------------
if __name__ == "__main__":

    # create experiment
    exp = experiment.builtin('trive')
    exp.w1.points = np.linspace(-2.5, 2.5, nw) * 4 * np.log(2) / dt * 1 / (2 * np.pi * 3e-5)
    exp.w2.points = np.linspace(-2.5, 2.5, nw) * 4 * np.log(2) / dt * 1 / (2 * np.pi * 3e-5)

    exp.d2.points = np.linspace(-2 * dt, 8 * dt, nt)
    exp.w1.active = exp.w2.active = exp.d2.active = True

    exp.timestep = 2.
    exp.early_buffer = 100.0
    exp.late_buffer  = 400.0

    # create hamiltonian
    ham = hamiltonian.Hamiltonian(w_central=0.)
    #ham.time_orderings = [5]
    ham.recorded_elements = [7,8]

    # do scan
    begin = time.perf_counter()
    scan = exp.run(ham, mp='')
    print(time.perf_counter()-begin)
    gpuSig = scan.sig.copy()

    # create an mp4 file for testing and validating results
    
    plt.close('all')
    # measure and plot
    fig, gs = wt.artists.create_figure(cols=[1, 'cbar'])
    ax = plt.subplot(gs[0, 0])
    xi = exp.active_axes[0].points
    yi = exp.active_axes[1].points
    zi = np.sum(np.abs(np.sum(scan.sig, axis=-2)), axis=-1).T
    ax.set_xlabel(exp.active_axes[0].name)
    ax.set_ylabel(exp.active_axes[1].name)

    camera=Camera(fig)

    for i in range(0,nt-1,1):
        coll = ax.pcolor(xi,yi,zi[i][:][:],cmap='default')
        coll = ax.contour(xi,yi,zi[i][:][:],colors='k')
        cax=plt.subplot(gs[0,1])
        wt.artists.plot_colorbar(label='amplitude')
        #plt.show()
        camera.snap()


    ani=camera.animate(repeat=True)
    ani.save('animate.mp4', writer=writer)
    


