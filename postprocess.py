import numpy as np
from pathlib import Path
import pandas as pd
import meshio
import dolfinx
import adios4dolfinx
from mpi4py import MPI
import cardiac_geometries


def main(
    mode: int = -1,
    datadir: Path = Path("data-clipped"),
    resultsdir: Path = Path("results-clipped"),
):
    geodir = Path(datadir) / f"mode_{mode}"
    outdir = Path(resultsdir) / f"mode_{mode}"

    comm = MPI.COMM_WORLD
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)
    V = dolfinx.fem.functionspace(geo.mesh, ("CG", 2, (3,)))
    u = dolfinx.fem.Function(V)

    # Read the displacement from the last time step of the simulation
    timestamps = adios4dolfinx.read_timestamps(
        comm=comm, filename=outdir / "u_checkpoint.bp", function_name="displacement"
    )
    adios4dolfinx.read_function(
        outdir / "u_checkpoint.bp", u, time=timestamps[-1], name="displacement"
    )

    # Load the original mesh points for the surfaces
    # points = {}
    # for label in ["EPI", "LV", "RV"]:
    #     points[label] = meshio.read(geodir / f"{label}_ED.stl").points

    points = {}
    for label, fname in [
        ("EPI", "epi_clipped.ply"),
        ("LV", "lv_clipped.ply"),
        ("RV", "rv_clipped.ply"),
    ]:
        points[label] = meshio.read(geodir / fname).points

    fdim = geo.mesh.topology.dim - 1
    geo.mesh.topology.create_connectivity(fdim, 0)
    ftree = dolfinx.geometry.bb_tree(geo.mesh, fdim, padding=0.1)
    bb_tree = dolfinx.geometry.bb_tree(geo.mesh, geo.mesh.topology.dim)
    for label, coords in points.items():
        # Find the corresponding facets from the mesh for a given surface label
        entities = geo.ffun.find(geo.markers[label][0])
        # Create a midpoint tree for the entities
        mid_tree = dolfinx.geometry.create_midpoint_tree(geo.mesh, fdim, entities)
        # Compute the closest entity for each point
        entity = dolfinx.geometry.compute_closest_entity(
            ftree, mid_tree, geo.mesh, coords
        )
        # Now find the coordinates of the midpoints of the entities
        midpoint_coords = dolfinx.mesh.compute_midpoints(geo.mesh, 2, entity)

        # Now evaluate the function at the midpoints
        # (need to find the cells that contain the midpoints)
        potential_colliding_cells = dolfinx.geometry.compute_collisions_points(
            bb_tree, midpoint_coords
        )
        colliding_cells = dolfinx.geometry.compute_colliding_cells(
            geo.mesh, potential_colliding_cells, midpoint_coords
        )

        cells = []
        for i, coord in enumerate(midpoint_coords):
            if len(colliding_cells.links(i)) > 0:
                cells.append(colliding_cells.links(i)[0])
            else:
                raise ValueError(f"Point {i} at coordinate {coord} not found in mesh")

        cells = np.array(cells, dtype=np.int32)
        u_values = u.eval(midpoint_coords, cells)

        df = pd.DataFrame(
            {
                "ux": u_values[:, 0],
                "uy": u_values[:, 1],
                "uz": u_values[:, 2],
                "mesh_x": midpoint_coords[:, 0],
                "mesh_y": midpoint_coords[:, 1],
                "mesh_z": midpoint_coords[:, 2],
                "surface_x": coords[:, 0],
                "surface_y": coords[:, 1],
                "surface_z": coords[:, 2],
            }
        )
        df.to_csv(outdir / f"{label}_displacement.csv", index=False)


if __name__ == "__main__":
    main()
