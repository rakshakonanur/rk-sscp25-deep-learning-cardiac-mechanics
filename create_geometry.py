from pathlib import Path

# Some issue with gmsh in CONDA, see https://github.com/conda-forge/gmsh-feedstock/issues/101
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from mpi4py import MPI
import cardiac_geometries as cg


def main():
    comm = MPI.COMM_WORLD
    mode = -1
    outdir = Path("data") / f"mode_{mode}"
    outdir.mkdir(parents=True, exist_ok=True)
    cg.mesh.ukb(
        outdir=outdir,
        comm=comm,
        mode=-1,
        case="ED",
        char_length_max=10.0,
        char_length_min=10.0,
        fiber_angle_endo=60,
        fiber_angle_epi=-60,
        fiber_space="DG_0",
        clipped=True,
    )


if __name__ == "__main__":
    main()
