"""Synthetic l0-SNMF run of iGGBPL and GGBPL."""

from ggbpl.snmf import is_feasible, reconstruction, run_experiment


def main():
    result = run_experiment()
    instance = result["instance"]
    inertial = result["iggbpl"]
    plain = result["ggbpl"]
    X = instance["X"]
    s1, s2 = instance["s1"], instance["s2"]
    j0 = float(inertial["objectives"][0])
    j_i = float(inertial["objectives"][-1])
    j_g = float(plain["objectives"][-1])
    recon0 = float(reconstruction(X, instance["U0"], instance["V0"]))
    recon_i = float(reconstruction(X, inertial["U"], inertial["V"]))
    if not (j_i < j0 and j_g < j0 and recon_i < recon0):
        raise SystemExit(
            f"objective did not decrease: J0={j0}, iGGBPL={j_i}, GGBPL={j_g}, "
            f"recon0={recon0}, recon_i={recon_i}"
        )
    for k in range(inertial["U_hist"].shape[0]):
        if not is_feasible(inertial["U_hist"][k], inertial["V_hist"][k], s1, s2):
            raise SystemExit(f"infeasible iGGBPL iterate {k}")
        if not is_feasible(plain["U_hist"][k], plain["V_hist"][k], s1, s2):
            raise SystemExit(f"infeasible GGBPL iterate {k}")
    print(f"iGGBPL final J: {j_i:.8e}")
    print(f"GGBPL final J: {j_g:.8e}")


if __name__ == "__main__":
    main()
