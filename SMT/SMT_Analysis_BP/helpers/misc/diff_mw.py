import numpy as np
import matplotlib.pyplot as plt


def mw_to_diff(mw, n, T=310.15, micron=True):
    """
    Documentation:
    Given the molecular weight, returns the diffusion coefficent assuming a spherical shape of the moleucle while its moving in solution

    From: A Novel Correlation for Protein Diffusion Coefficients Based on Molecular Weight and Radius of Gyration, He and Niemeyer, 2003, Biotechnol. Prog. 19, 544-548

    Parameters:
    -----------
    mw = molecular weight in g*mol^-1

    n = solution viscocity in c*P

    T = Temperature of the solution in K (default is 310.15K = 37C)

    micron = if True:
                             return diffusion coefficient value as um^2*s^-1
                     if False:
                             return diffusion coefficient value as m^2*s^-1


    Return:
    -------
    Diffusion Coefficient in m^2*s^-1 if mircon is False

    """

    coeff = 8.34 * (10**-12)
    under = n * (mw ** (1.0 / 3.0))

    dif_coeff = coeff * (T / under)

    if micron:
        return dif_coeff * (10**12)
    else:
        return dif_coeff


# RPOC mw
# https://www.uniprot.org/uniprot/P0A8T7
mw_RPOC = 155160  # Dalton

# NUSA
# https://www.uniprot.org/uniprot/P0AFF6
mw_NUSA = 54871  # Dalton

# LACI
# https://www.uniprot.org/uniprot/P03023
mw_LACI = 38590  # Dalton

# RNAP holoenzyme
# https://bionumbers.hms.harvard.edu/bionumber.aspx?id=111558&ver=1&trm=rna+polymerase&org=
mw_RNAP = 350000  # Dalton

# Viscocity of Ecoli in cP
# Measuring the Viscosity of the Escherichia coli Plasma Membrane Using Molecular Rotors (Mika et al, 2016, Biophys J.)
vis_ecoli = 950  # cP

# cytoplasmic viscocity
# https://pubs.acs.org/doi/abs/10.1021/acs.jpcb.8b07362
vis_cyto_ecoli = 2.82  # cP

dif_RPOC = mw_to_diff(mw_RPOC, vis_cyto_ecoli)
dif_NUSA = mw_to_diff(mw_NUSA, vis_cyto_ecoli)
dif_LACI = mw_to_diff(mw_LACI, vis_cyto_ecoli)
dif_RNAP = mw_to_diff(mw_RNAP, vis_cyto_ecoli)
print("RPOC Diffusion Coefficient: ", dif_RPOC)
print("NUSA Diffusion Coefficient: ", dif_NUSA)
print("LACI Diffusion Coefficient: ", dif_LACI)
print("RNAP Diffusion Coefficient: ", dif_RNAP)

if __name__ == "__main__":
    plt.plot(
        ["RPOC", "NUSA", "LACI", "RNAP_holo"],
        np.log10(np.array([dif_RPOC, dif_NUSA, dif_LACI, dif_RNAP])),
        "b.",
    )
    plt.ylabel("Diffusion Coefficient (um^2/s)")
    plt.show()
