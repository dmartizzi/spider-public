# spider-public
Cosmic Web classifier for the IllustrisTNG simulations based on the deformation tensor approach (Hahn et al. 2007, Forero-Romero et al. 2009). 

These tools were developed by Davide Martizzi to perform published work in computational astrophysics. Please, reference [Martizzi et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.3766M/abstract) if you use this code. 

Cosmic Web classification is performed with two Python scripts:
- `spider_make_mass_grid.py` interpolates the mass of the particles in the simulation box into a regular Cartesian grid. 
- `spider_make_web.py` computes the deformation tensor at each node of the Cartesian grid, and performs Cosmic Web classification using the local eigenvalues of the deformation tensor.

The `io_utils.py` module contains a few functions for I/O and for operations on Cartesian grids; some of this code relies on the `h5py` package and on Mark Vogelsberger's [`python-cosmo` tools](https://wwwmpa.mpa-garching.mpg.de/svn/cosmo-group/Arepo/tools/Python). The I/O functions can be modified to import output from any simulation code. 

The scripts are parallelized with `mpi4py`.

## Instructions: 

Let us assume that we want to run the scripts on 4 cores.

- Step 1: make a mass grid and print it in hdf5 format:

  `mpirun -np 4 python spider_make_mass_grid.py`
  
- Step 2: run the Cosmic Web classifier and print results in hdf5 format:

  `mpirun -np 4 python spider_make_web.py 0 4 0.3`
  
  where the command line arguments mean that the script runs at redshift z=0, that the cosmic density field will be smoothed with a filter of radius R = 4 Mpc/h, and that the Cosmic Web classification threshold for the eigenvalues of the deformation tensor is lambda_th=0.3 (see [Martizzi et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.3766M/abstract) for details).
  
