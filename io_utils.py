import numpy as np
import readsnapHDF5 as rs
import mpi4py
from mpi4py import MPI
import h5py

### DAVIDE MARTIZZI - NOV 2, 2017
### THIS MODULE CONTAINS USEFUL
### FUNCTIONS AND CLASSES TO 
### READ THE ILLUSTRIS DATASET 
### AND TO OPERATE ON CARTESIAN 
### GRIDS. 

class CartesianGrid:
    # a class that generates cartesian grids
    def __init__(self, L=1.0, n_x=1, n_y=1, n_z=1, time=1.0, hubble=1.0, omega0=0.3, omegaL=0.7):
        self.L = L
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.time = time
        self.hubble = hubble
        self.omega0 = omega0
        self.omegaL = omegaL
        self.grid = np.zeros((n_x,n_y,n_z))

    def CIC(self,pos,var,iblock):
        # PERFORM CIC INTERPOLATION
        L = self.L
        n_x = self.n_x
        n_y = self.n_y
        n_z = self.n_z

        # indices of the particles
        xp = np.empty(pos.shape)
        xp[:,0] = pos[:,0]*n_x/L
        xp[:,1] = pos[:,1]*n_y/L
        xp[:,2] = pos[:,2]*n_z/L
        # indices of the particles
        xp = np.empty(pos.shape)
        xp[:,0] = pos[:,0]*n_x/L
        xp[:,1] = pos[:,1]*n_y/L
        xp[:,2] = pos[:,2]*n_z/L

        temp_grid = np.zeros((n_x,n_y,n_z))

        # Perform CIC
        for x, y, z, v in zip(xp[:,0], xp[:,1], xp[:,2], var[:]):
            #Weights for CIC
            dx = x-np.floor(x)
            dy = y-np.floor(y)
            dz = z-np.floor(z)
            tx = -(dx-1.0)
            ty = -(dy-1.0)
            tz = -(dz-1.0)

            ix = int(x)
            iy = int(y)
            iz = int(z)
            temp_grid[ix,iy,iz] += tx*ty*tz*v

            ix = int(x)+1
            iy = int(y)
            iz = int(z)
            if ix == n_x: ix = 0 # periodic BC
            temp_grid[ix,iy,iz] += dx*ty*tz*v

            ix = int(x)
            iy = int(y)+1
            iz = int(z)
            if iy == n_y: iy = 0 # periodic BC
            temp_grid[ix,iy,iz] += tx*dy*tz*v

            ix = int(x)+1
            iy = int(y)+1
            iz = int(z)
            if ix == n_x: ix = 0 # periodic BC
            if iy == n_y: iy = 0 # periodic BC
            temp_grid[ix,iy,iz] += dx*dy*tz*v

            ix = int(x)
            iy = int(y)
            iz = int(z)+1
            if iz == n_z: iz = 0 # periodic BC
            temp_grid[ix,iy,iz] += tx*ty*dz*v

            ix = int(x)+1
            iy = int(y)
            iz = int(z)+1
            if ix == n_x: ix = 0 # periodic BC
            if iz == n_z: iz = 0 # periodic BC
            temp_grid[ix,iy,iz] += dx*ty*dz*v

            ix = int(x)
            iy = int(y)+1
            iz = int(z)+1
            if iy == n_y: iy = 0 # periodic BC
            if iz == n_z: iz = 0 # periodic BC
            temp_grid[ix,iy,iz] += tx*dy*dz*v

            ix = int(x)+1
            iy = int(y)+1
            iz = int(z)+1
            if ix == n_x: ix = 0 # periodic BC
            if iy == n_y: iy = 0 # periodic BC
            if iz == n_z: iz = 0 # periodic BC
            temp_grid[ix,iy,iz] += dx*dy*dz*v

        self.grid = self.grid + temp_grid
        del temp_grid

        print "CIC INTERPOLATION DONE FOR BLOCK ", iblock

    def write_to_hdf5(self,fname):
        ### WRITE OUTPUT IN HDF5
        f = h5py.File(fname, "w")

        time_out = f.create_dataset("time", (1,), dtype='f')
        time_out[...] = self.time

        hubble_out = f.create_dataset("hubble", (1,), dtype='f')
        hubble_out[...] = self.hubble

        omega0_out = f.create_dataset("omega0", (1,), dtype='f')
        omega0_out[...] = self.omega0

        omegaL_out = f.create_dataset("omegaL", (1,), dtype='f')
        omegaL_out[...] = self.omegaL

        L_out = f.create_dataset("boxsize", (1,), dtype='f')
        L_out[...] = self.L

        gridshape = np.array([self.n_x, self.n_y, self.n_z])
        gridshape_out = f.create_dataset("gridshape", data = gridshape, dtype='i')

        MGrid_out = f.create_dataset("MGrid", data = self.grid, dtype='f')

        f.close()

    def read_from_hdf5(self,fname):
        f = h5py.File(fname, "r")
        #for name in f:
        #    print name
        self.time = f["time"][0]
        self.hubble = f["hubble"][0]
        self.omega0 = f["omega0"][0]
        self.omegaL = f["omegaL"][0]
        self.L = f["boxsize"][0]
        self.n_x = f["gridshape"][0]
        self.n_y = f["gridshape"][1]
        self.n_z = f["gridshape"][2]
        
        self.grid = f["MGrid"][...]
        
        f.close()


    def find_class(self,x,y,z):
        nx = self.n_x
        ny = self.n_y
        nz = self.n_z
        L = self.L
        dx = L/nx
        dy = L/ny
        dz = L/nz
        i = np.floor(x/dx)
        j = np.floor(y/dy)
        k = np.floor(z/dz)
        i = i.astype(int)
        j = j.astype(int)
        k = k.astype(int)
        iii = i > nx-1 
        i[iii] = i[iii]-nx
        jjj = j > ny-1
        j[jjj] = j[jjj]-ny
        kkk = k > nz-1 
        k[kkk] = k[kkk]-nz
        class_w = np.empty((len(x)))
        class_w[:] = self.grid[i[:],j[:],k[:]]
        return class_w

#### READ ILLUSTRIS USING MARK VOGELSBERGER'S LIBRARY
def read_block_vars(fname,ptype):
    if ptype == 0: print "Processing GAS PARTICLES"
    if ptype == 1: print "Processing DM PARTICLES"
    if ptype == 4: print "Processing STARS+WIND PARTICLES"
    if ptype == 5: print "Processing BH PARTICLES"
    # position
    rrr = rs.read_block(fname, "POS ", parttype = ptype)
    # potential
    #pot = rs.read_block(fname, "POT ", parttype = ptype)
    mmm = rs.read_block(fname, "MASS", parttype = ptype)
    return rrr, mmm

### FUNCTION AND CLASSES END
