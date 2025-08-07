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
    PLV: np.array = [30.0, 40.0],
    PRV: np.array = [6.0, 8.0],
    TA: np.array = [0.0, 120.0],
    N: np.array = [500, 200],
    eta: float = 0.3,
    a: np.array = [0.5, 2.280],
    a_f: float = 1.685,
):
    geodir = Path(datadir) / f"mode_{mode}" / case
    outdir = Path(resultsdir) / f"mode_{mode}" / case
    outdir.mkdir(parents=True, exist_ok=True)

    # Print fenicsx_pulse path
    # print fenicsx_pulse location
    print(f"fenicsx_pulse location: {fenicsx_pulse.__file__}")

    log.set_log_level(log.LogLevel.INFO)
    geo = cardiac_geometries.geometry.Geometry.from_folder(
        comm=MPI.COMM_WORLD, folder=geodir  
    )
    geometry = fenicsx_pulse.Geometry.from_cardiac_geometries(
        geo, metadata={"quadrature_degree": 4}
    )

    material_params = fenicsx_pulse.HolzapfelOgden.transversely_isotropic_parameters(a_val = a, a_f_val = a_f)
    material = fenicsx_pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore

    Ta = fenicsx_pulse.Variable(
        dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0)), "kPa"
    )
    # Add 30% transverse active stress
    active_model = fenicsx_pulse.ActiveStress(geo.f0, activation=Ta, eta=eta)

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

    # for i, (plv, prv, tai) in enumerate(
    #     zip(
    #         np.concatenate(np.linspace(0, PLV[0], N[0]), np.linspace(PLV[0], PLV[1], N[1])),
    #         np.concatenate(np.linspace(0, PRV[0], N[0]), np.linspace(PRV[0], PRV[1], N[1])),
    #         np.concatenate(np.linspace(0, TA[0], N[0]), np.linspace(TA[0], TA[1], N[1])),
    #     )
    # ):
        
    # for i, (plv, prv, tai) in enumerate(zip(
    #     np.concatenate([
    #         np.linspace(0, PLV[0], N[0]),
    #         np.linspace(PLV[0], PLV[1], N[1])
    #     ]),
    #     np.concatenate([
    #         np.linspace(0, PRV[0], N[0]),
    #         np.linspace(PRV[0], PRV[1], N[1])
    #     ]),
    #     np.concatenate([
    #         np.linspace(0, TA[0],  N[0]),
    #         np.linspace(TA[0],  TA[1], N[1])
    #     ])
    # )):
    #     if geometry.mesh.comm.rank == 0:
    #         print(f"i: {i}, plv: {plv}, prv: {prv}, Ta: {tai}", flush=True)

    #     # adios4dolfinx.write_function(
    #     #     outdir / "u_checkpoint.bp", problem.u, time=float(i), name="displacement"
    #     # )

    #     lvp.assign(plv)
    #     rvp.assign(prv)
    #     Ta.assign(tai)
    #     problem.solve()
    #     vtx.write(float(i))
    # vtx.close()

    # --- Define targets ---
    PLV_ED_vals = np.array([5, 10, 20])
    PRV_ED_vals = np.array([1, 1.5, 4])
    PLV_ES_vals = np.array([5.5, 16, 30])
    PRV_ES_vals = np.array([1.5, 2.67, 8])

    # Generate full PLV, PRV, TA profiles
    plv_vals = np.concatenate([
        np.linspace(0, PLV[0], N[0]),
        np.linspace(PLV[0], PLV[1], N[1])
    ])
    prv_vals = np.concatenate([
        np.linspace(0, PRV[0], N[0]),
        np.linspace(PRV[0], PRV[1], N[1])
    ])
    ta_vals = np.concatenate([
        np.linspace(0, TA[0],  N[0]),
        np.linspace(TA[0],  TA[1], N[1])
    ])

    # --- Find closest-matching indices ---
    def find_closest_indices(signal, targets):
        return [np.argmin(np.abs(signal - val)) for val in targets]

    PLV_ED_inds = find_closest_indices(plv_vals[:N[0]], PLV_ED_vals)
    PRV_ED_inds = find_closest_indices(prv_vals[:N[0]], PRV_ED_vals)
    PLV_ES_inds = find_closest_indices(plv_vals[N[0]:], PLV_ES_vals)
    PRV_ES_inds = find_closest_indices(prv_vals[N[0]:], PRV_ES_vals)
    # print(f"Indices: {PLV_ED_inds}, {PRV_ED_inds}, {PLV_ES_inds}, {PRV_ES_inds}")

    # Shift ES indices since they start at N[0]
    PLV_ES_inds = [i + N[0] for i in PLV_ES_inds]
    PRV_ES_inds = [i + N[0] for i in PRV_ES_inds]

    # Merge unique write indices
    write_indices = set(PLV_ED_inds + PRV_ED_inds + PLV_ES_inds + PRV_ES_inds)

    # --- Main loop ---
    for i, (plv, prv, tai) in enumerate(zip(plv_vals, prv_vals, ta_vals)):
        if geometry.mesh.comm.rank == 0:
            print(f"i: {i}, plv: {plv}, prv: {prv}, Ta: {tai}", flush=True)

        lvp.assign(plv)
        rvp.assign(prv)
        Ta.assign(tai)
        problem.solve()

        # Write only if i is closest to one of the targets
        if i in write_indices:
            print(f"Saving pressures: plv: {plv}, prv: {prv}, Ta: {tai}", flush=True)
            vtx.write(float(i))
            adios4dolfinx.write_function(
            outdir / "u_checkpoint.bp", problem.u, time=float(i), name="displacement"
        )

if __name__ == "__main__":
    main()
