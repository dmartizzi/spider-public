import sys
import numpy as np
import readsnapHDF5 as rs
import mpi4py
from mpi4py import MPI
import h5py
from numpy import linalg as LA
import scipy.ndimage.filters
from io_utils import *

### DAVIDE MARTIZZI - NOV 2, 2017
### THIS SCRIP COMPUTES THE TIDAL 
### TENSOR FROM A CARTESIAN GRID
### THEN CLASSIFIES EACH CELL 
### AS PART OF HALOS, SHEETS, 
### FILAMENTS

if __name__ == "__main__":

    # MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Redshift, smoothing scale and lambda
    zz_str = sys.argv[1]
    rr_str = sys.argv[2]
    ll_str = sys.argv[3]
    ll = float(ll_str)
    print 'Redshift = ', zz_str
    print 'Smoothing scale = ', rr_str
    print 'Lambda_th = ', ll

    # Read from a pre-generated hdf5 grid 
    MGrid = CartesianGrid() # this grid contains the mass in each cell
    MGrid.read_from_hdf5("mass_grid_512_z"+zz_str+".hdf5")
    #MGrid.read_from_hdf5("mass_grid_256.hdf5")
    #MGrid.read_from_hdf5("mass_grid_512_z0.hdf5")
    #MGrid.read_from_hdf5("mass_grid_1024.hdf5")
    #MGrid.read_from_hdf5("mass_grid_512_z02.hdf5")
    #MGrid.read_from_hdf5("mass_grid_512_z05.hdf5")
    #MGrid.read_from_hdf5("mass_grid_512_z1.hdf5")
    #MGrid.read_from_hdf5("mass_grid_512_z2.hdf5")
    #MGrid.read_from_hdf5("mass_grid_512_z3.hdf5")
    #MGrid.read_from_hdf5("mass_grid_512_z4.hdf5")
    #MGrid.read_from_hdf5("mass_grid_512_z5.hdf5")
    #MGrid.read_from_hdf5("mass_grid_512_z6.hdf5")
    #MGrid.read_from_hdf5("mass_grid_512_z8.hdf5")


    if rank == 0: print "Data read..."

    # Parameters
    n_x = MGrid.n_x
    n_y = MGrid.n_y
    n_z = MGrid.n_z
    L = MGrid.L
    aexp = MGrid.time
    hubble = MGrid.hubble
    omega0 = MGrid.omega0
    omegaL = MGrid.omegaL

    # Smoothing
    #n_smooth = 4000.0*n_x/L # Number of cells used for smoothing
    n_smooth = float(rr_str)*1000.0*n_x/L
    MGrid.grid = scipy.ndimage.filters.uniform_filter(MGrid.grid,n_smooth)

    # Compute overdensity
    Mtot = MGrid.grid.sum()/n_x/n_y/n_z
    rho_mean = n_x*n_y*n_z*Mtot*1e10*2e33/(L*3.08e21)**3
    MGrid.grid = MGrid.grid/Mtot - 1.0

    # Compute the FFT and the frequencies along each axis
    M_fft = np.fft.fftn(MGrid.grid)
    k_x_lin = np.fft.fftfreq(MGrid.n_x)
    k_y_lin = np.fft.fftfreq(MGrid.n_y)
    k_z_lin = np.fft.fftfreq(MGrid.n_z)

    print M_fft.shape
    print k_x_lin.shape
    print k_y_lin.shape
    print k_z_lin.shape

    if rank == 0: print"FFT of rho field computed..."

    # These meshes contain the components of k at each node of 
    # the 3D k-space
    k_x_mesh, k_y_mesh, k_z_mesh = np.meshgrid(k_x_lin,k_y_lin,k_z_lin)
    k_x_mesh = k_x_mesh.astype(complex)
    k_y_mesh = k_y_mesh.astype(complex)
    k_z_mesh = k_z_mesh.astype(complex)
    k_mesh = (k_x_mesh**2 + k_y_mesh**2 + k_z_mesh**2 + 1.0e-20+1.e-20j)**0.5
    k_mesh = k_mesh.astype(complex)
    k2_mesh = k_x_mesh**2 + k_y_mesh**2 + k_z_mesh**2 + 1.0e-20+1.e-20j
    k2_mesh = k2_mesh.astype(complex)

    # Top-hat smoothing
    #r = 2.0*np.pi*4000.0/MGrid.L #4 kpc/h normalized by boxsize
    #WK = 3/(k_mesh*r)**3 * (np.sin(k_mesh*r)-k_mesh*r*np.cos(k_mesh*r))
    
    # Gaussian smoothing
    #r = 2.0*np.pi*4000.0/MGrid.L #4 kpc/h normalized by boxsize
    #WK = np.exp(-(k_mesh*r)**2/2)
    
    # Use Poisson equation in k-space to compute the potential 
    # k^2 * \Phi = 4 pi G \rho 
    #t_scale2 = 1.0/(4.0*np.pi*6.66e-8*rho_mean)
    Phi_fft = M_fft/k2_mesh
    del M_fft

    if rank == 0: print"FFT of Phi field computed..."
    
    # Compute components of tidal tensor in k-space
    # it's a symmetric tensor, so only 6 components 
    # should be actually computed
    Tid_fft = np.empty((6,len(k_x_lin),len(k_y_lin),len(k_z_lin)))
    Tid_fft = Tid_fft.astype(complex)
    Tid_fft[0] = k_x_mesh*k_x_mesh*Phi_fft
    Tid_fft[1] = k_x_mesh*k_y_mesh*Phi_fft
    Tid_fft[2] = k_x_mesh*k_z_mesh*Phi_fft
    Tid_fft[3] = k_y_mesh*k_y_mesh*Phi_fft
    Tid_fft[4] = k_y_mesh*k_z_mesh*Phi_fft
    Tid_fft[5] = k_z_mesh*k_z_mesh*Phi_fft
    del Phi_fft, k_x_mesh, k_y_mesh, k_z_mesh
    del k_mesh, k2_mesh

    # Parallelize inverse FFT if needed
    n_components = 6
    n_cores = size
    Tid = np.zeros((3,3,len(k_x_lin),len(k_y_lin),len(k_z_lin)))
    # The FFTs will be divided in chunks for each core
    l_chunk = int(n_components/n_cores)
    l_chunk_arr = np.zeros((n_cores))
    for ii in range(0,n_cores):
        l_chunk_arr[ii] = l_chunk
    n_left = n_components-n_cores*l_chunk
    for ii in range(0,n_left):
        l_chunk_arr[ii] = l_chunk_arr[ii] + 1 
    indstart = np.empty((n_cores))
    indend = np.empty((n_cores))
    indold = 0
    for ii in range(0,n_cores):
        indstart[ii] = indold
        indend[ii] = indold + l_chunk_arr[ii]
        indold = indend[ii] + 1
    for i in range(0,3):
        for j in range(0,3):
            ind = (i)*3+j # index between 0 and 8 
            if 3 <= ind and ind <= 5:
                ind = ind - 1
            if 6 <= ind and ind <= 8:
                ind = ind - 3
            if j >= i and indstart[rank] <= ind and ind <= indend[rank]:
                print i, j, ind
                Tid[i,j,:,:,:] = np.fft.ifftn(Tid_fft[ind])
                Tid[j,i,:,:,:] = Tid[i,j,:,:,:]
    
    del Tid_fft

    # MPI COMMUNICATION
    Tid_total = np.zeros((3,3,len(k_x_lin),len(k_y_lin),len(k_z_lin)))
    if size > 1:    
        comm.Allreduce(Tid, Tid_total, op = MPI.SUM)
        comm.Barrier()
        del Tid
    else :
        Tid_total = Tid
        del Tid

    # CLASSIFICATION
    cla = CartesianGrid(L,n_x,n_y,n_z,aexp,hubble,omega0,omegaL)
    # 3 = halo
    # 2 = filament
    # 1 = sheet
    # 0 = void
    #threshold = 0.070 # From Lee & White 2016
    threshold = ll

    # Break the operations in chunks for each core
    n_cores = size
    l_chunk = int(n_x*n_y*n_z/n_cores)
    l_chunk_arr = np.zeros((n_cores))
    for ii in range(0,n_cores):
        l_chunk_arr[ii] = l_chunk
    n_left = n_x*n_y*n_z-n_cores*l_chunk
    for ii in range(0,n_left):
        l_chunk_arr[ii] = l_chunk_arr[ii] + 1    
    indstart = np.empty((n_cores))
    indend = np.empty((n_cores))
    indold = 0    
    for ii in range(0,n_cores):
        indstart[ii] = indold
        indend[ii] = indold + l_chunk_arr[ii]
        indold = indend[ii] + 1    
    
    # Find eigenvalues of tidal tensor in each cell and 
    # use classification scheme
    for i in range(0,n_x):
        for j in range(0,n_y):
            for k in range(0,n_z):
                ind = k*n_x*n_y + j*n_x + i
                if indstart[rank] <= ind and ind <= indend[rank]:
                    eigenval = LA.eigvalsh(Tid_total[:,:,i,j,k])
                    counter = 0
                    for lamb in eigenval:
                        if lamb > threshold: counter += 1
                        cla.grid[i,j,k] = int(counter)
    
    # MPI COMMUNICATION
    cla_total = CartesianGrid(L,n_x,n_y,n_z,aexp,hubble,omega0,omegaL)
    if size > 1:
        comm.Allreduce(cla.grid, cla_total.grid, op = MPI.SUM)
        comm.Barrier()
        del cla
    else :
        cla_total = cla
        del cla
    
    if rank == 0:
        cla_total.write_to_hdf5("web_grid_R"+rr_str+"_512_z"+zz_str+"_lth"+ll_str+".hdf5")
        #cla_total.write_to_hdf5("web_grid_R4_256.hdf5")
        #cla_total.write_to_hdf5("web_grid_R4_512_z0.hdf5")
        #cla_total.write_to_hdf5("web_grid_R4_1024.hdf5")
        #cla_total.write_to_hdf5("web_grid_R4_512_z02.hdf5")
        #cla_total.write_to_hdf5("web_grid_R4_512_z05.hdf5")
        #cla_total.write_to_hdf5("web_grid_R4_512_z1.hdf5")
        #cla_total.write_to_hdf5("web_grid_R4_512_z2.hdf5")
        #cla_total.write_to_hdf5("web_grid_R4_512_z3.hdf5")
        #cla_total.write_to_hdf5("web_grid_R4_512_z4.hdf5")
        #cla_total.write_to_hdf5("web_grid_R4_512_z5.hdf5")
        #cla_total.write_to_hdf5("web_grid_R4_512_z6.hdf5")
        #cla_total.write_to_hdf5("web_grid_R4_512_z8.hdf5")
