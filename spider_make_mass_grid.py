import numpy as np
import readsnapHDF5 as rs
import mpi4py
from mpi4py import MPI
import h5py
from io_utils import *

### DAVIDE MARTIZZI - NOV 2, 2017
### THIS SCRIPT GENERATES A 3D CUBE WITH MASSES AT THE
### NODES OF A CARTESIAN GRID. PARTICLES ARE INTERPOLATED
### TO THE GRID USING CIC INTERPOLATION.

if __name__ == "__main__":

    # MPI parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # IllustrisTNG100 Snapshot 099 (z=0)
    # path_tng100_z0 = "/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG/output/snapdir_099/snap_099"
    # path_tng100_z02 = "/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG/output/snapdir_084/snap_084"
    # path_tng100_z05 = "/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG/output/snapdir_067/snap_067"
    # path_tng100_z1 = "/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG/output/snapdir_050/snap_050"
    # path_tng100_z2 = "/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG/output/snapdir_033/snap_033"
    # path_tng100_z3 = "/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG/output/snapdir_025/snap_025"
    # path_tng100_z4 = "/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG/output/snapdir_021/snap_021"
    path_tng100_z5 = (
        "/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG/output/snapdir_017/snap_017"
    )
    # path_tng100_z6 = "/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG/output/snapdir_013/snap_013"
    # path_tng100_z8 = "/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG/output/snapdir_008/snap_008"

    # Read header
    # header = rs.snapshot_header(path_tng100_z0+".1.hdf5")
    # header = rs.snapshot_header(path_tng100_z02+".1.hdf5")
    # header = rs.snapshot_header(path_tng100_z05+".1.hdf5")
    # header = rs.snapshot_header(path_tng100_z1+".1.hdf5")
    # header = rs.snapshot_header(path_tng100_z2+".1.hdf5")
    # header = rs.snapshot_header(path_tng100_z3+".1.hdf5")
    # header = rs.snapshot_header(path_tng100_z4+".1.hdf5")
    header = rs.snapshot_header(path_tng100_z5 + ".1.hdf5")
    # header = rs.snapshot_header(path_tng100_z6+".1.hdf5")
    # header = rs.snapshot_header(path_tng100_z8+".1.hdf5")
    if rank == 0:
        print("Time = ", header.time)
        print("H0 = ", header.hubble)
        print("Omega0 = ", header.omega0)
        print("OmegaLambda = ", header.omegaL)
        print("Box Size = ", header.boxsize)
        print("Number of files = ", header.filenum)

    # Interpolation grid parameters
    # nx = 256
    # ny = 256
    # nz = 256
    nx = 512
    ny = 512
    nz = 512
    # nx = 1024
    # ny = 1024
    # nz = 1024
    time = header.time
    hubble = header.hubble
    omega0 = header.omega0
    omegaL = header.omegaL
    L = header.boxsize
    MGrid = CartesianGrid(L, nx, ny, nz, time, hubble, omega0, omegaL)

    ### START LOOP ON SNAPSHOT BLOCKS ###
    nblocks = header.filenum  # number of HDF5 blocks
    n_chunks = size  # we split the number of blocks in n_chunks = size of the MPI job
    l_chunks = int(
        nblocks / size
    )  # average length of each chunk = number of HDF5 blocks per chunk
    n_left = nblocks - n_chunks * l_chunks  # leftovers
    l_chunks_arr = np.empty(n_chunks)  # length of each chunk + leftovers
    l_chunks_arr[rank] = l_chunks
    for i in range(0, n_left):  # redistribute leftover chunks among the other cores
        l_chunks_arr[rank] = int(l_chunks_arr[rank] + 1)
    istart = int(rank * l_chunks)  # starting point of loop on each cpu
    for iblock in range(istart, istart + int(l_chunks_arr[rank])):
        ### STARTING OPERATIONS ON i-TH BLOCK
        if iblock < istart + l_chunks:
            iii = iblock
        if iblock == istart + l_chunks:
            iii = (
                n_chunks * l_chunks + rank
            )  # this only exists for a chunk that received a leftover

        print("BLOCK NO. ", iii)
        ### LOOP ON MASSIVE PARTICLE TYPES
        for ptype in [0, 1, 4]:  # , 5]:
            ### READ POSITION AND POTENTIAL
            # fname = path_tng100_z0+"."+str(iii)+".hdf5"
            # fname = path_tng100_z02+"."+str(iii)+".hdf5"
            # fname = path_tng100_z05+"."+str(iii)+".hdf5"
            # fname = path_tng100_z1+"."+str(iii)+".hdf5"
            # fname = path_tng100_z2+"."+str(iii)+".hdf5"
            # fname = path_tng100_z3+"."+str(iii)+".hdf5"
            # fname = path_tng100_z4+"."+str(iii)+".hdf5"
            fname = path_tng100_z5 + "." + str(iii) + ".hdf5"
            # fname = path_tng100_z6+"."+str(iii)+".hdf5"
            # fname = path_tng100_z8+"."+str(iii)+".hdf5"
            rrr, mmm = read_block_vars(fname, ptype)
            # print rrr.shape
            # print mmm.shape
            # print rrr.min(),rrr.max()

            # INTERPOLATE TO GRID USING CIC
            MGrid.CIC(rrr, mmm, iii)
            # print MGrid.grid

            del rrr, mmm

        ### END OF LOOP ON MASSIVE PARTICLE TYPES

    ### END OF OPERATIONS ON i-TH BLOCK

    ### MPI COMMUNICATION TO GET FINAL GRID
    MGrid_total = CartesianGrid(L, nx, ny, nz, time, hubble, omega0, omegaL)
    if size > 1:
        comm.Allreduce(MGrid.grid, MGrid_total.grid, op=MPI.SUM)
        comm.Barrier()
        del MGrid
    else:
        MGrid_total.grid = MGrid.grid
        del MGrid

    if rank == 0:
        print(MGrid_total)

        ### WRITE OUTPUT IN HDF5
        # MGrid_total.write_to_hdf5("mass_grid_256.hdf5")
        # MGrid_total.write_to_hdf5("mass_grid_512.hdf5")
        # MGrid_total.write_to_hdf5("mass_grid_1024.hdf5")
        # MGrid_total.write_to_hdf5("mass_grid_512_z02.hdf5")
        # MGrid_total.write_to_hdf5("mass_grid_512_z05.hdf5")
        # MGrid_total.write_to_hdf5("mass_grid_512_z1.hdf5")
        # MGrid_total.write_to_hdf5("mass_grid_512_z2.hdf5")
        # MGrid_total.write_to_hdf5("mass_grid_512_z3.hdf5")
        # MGrid_total.write_to_hdf5("mass_grid_512_z4.hdf5")
        MGrid_total.write_to_hdf5("mass_grid_512_z5.hdf5")
        # MGrid_total.write_to_hdf5("mass_grid_512_z6.hdf5")
        # MGrid_total.write_to_hdf5("mass_grid_512_z8.hdf5")
