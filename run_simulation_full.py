from pathlib import Path

from mpi4py import MPI

import adios4dolfinx
import dolfinx
import fenicsx_pulse
import numpy as np
from dolfinx import log
import cardiac_geometries
import cardiac_geometries.geometry


def main(
    mode: int = -1,
    case: str = "ED",
    datadir: Path = Path("data-full"),
    resultsdir: Path = Path("results-full"),
):
    geodir = Path(datadir) / f"mode_{mode}" / case
    outdir = Path(resultsdir) / f"mode_{mode}" / case
    outdir.mkdir(parents=True, exist_ok=True)

    log.set_log_level(log.LogLevel.INFO)
    geo = cardiac_geometries.geometry.Geometry.from_folder(
        comm=MPI.COMM_WORLD, folder=geodir
    )
    geometry = fenicsx_pulse.Geometry.from_cardiac_geometries(
        geo, metadata={"quadrature_degree": 4}
    )

    material_params = fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters()
    material = fenicsx_pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore

    Ta = fenicsx_pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa"
    )
    # Add 30% transverse active stress
    active_model = fenicsx_pulse.ActiveStress(geo.f0, activation=Ta, eta=0.3)

    comp_model = fenicsx_pulse.Compressible()

    model = fenicsx_pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    lvp = fenicsx_pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa"
    )
    neumann_lv = fenicsx_pulse.NeumannBC(traction=lvp, marker=geometry.markers["LV"][0])

    rvp = fenicsx_pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa"
    )
    neumann_rv = fenicsx_pulse.NeumannBC(traction=rvp, marker=geometry.markers["RV"][0])

    def dirichlet_bc(
        V: dolfinx.fem.FunctionSpace,
    ) -> list[dolfinx.fem.bcs.DirichletBC]:
        mesh = geometry.mesh
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

        bcs = []
        for marker in ["MV", "TV", "AV", "PV"]:
            facets = geo.ffun.find(geometry.markers[marker][0])
            dofs = dolfinx.fem.locate_dofs_topological(V, 2, facets)
            u_fixed = dolfinx.fem.Function(V)
            u_fixed.x.array[:] = 0.0
            bcs.append(dolfinx.fem.dirichletbc(u_fixed, dofs))

        return bcs

    # Could alternatively use a Robin BC
    pericardium = fenicsx_pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(1e5)), "Pa / m"
    )
    robin_per = fenicsx_pulse.RobinBC(
        value=pericardium, marker=geometry.markers["EPI"][0]
    )

    robin = (robin_per,)
    # We collect all the boundary conditions

    bcs = fenicsx_pulse.BoundaryConditions(
        neumann=(neumann_lv, neumann_rv), robin=robin, dirichlet=(dirichlet_bc,)
    )

    problem = fenicsx_pulse.StaticProblem(
        model=model,
        geometry=geometry,
        bcs=bcs,
        parameters={"u_space": "P_2", "mesh_unit": "cm"},
    )

    problem.solve()

    vtx = dolfinx.io.VTXWriter(
        geometry.mesh.comm, outdir / "displacement.bp", [problem.u], engine="BP4"
    )
    vtx.write(0.0)

    PLV_ES = 15.0
    PRV_ES = 3.0
    Ta_ES = 120.0
    # Number of steps to take. Just pick a high enough number so that we end without divergence
    N = 200

    for i, (plv, prv, tai) in enumerate(
        zip(
            np.linspace(0, PLV_ES, N),
            np.linspace(0, PRV_ES, N),
            np.linspace(0, Ta_ES, N),
        )
    ):
        if geometry.mesh.comm.rank == 0:
            print(f"i: {i}, plv: {plv}, prv: {prv}, Ta: {tai}", flush=True)

        adios4dolfinx.write_function(
            outdir / "u_checkpoint.bp", problem.u, time=float(i), name="displacement"
        )

        lvp.assign(plv)
        rvp.assign(prv)
        Ta.assign(tai)
        problem.solve()
        vtx.write(float(i))
    vtx.close()


if __name__ == "__main__":
    main()
