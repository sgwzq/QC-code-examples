#!/usr/bin/env python3
# HF Code Example
# Author: Peng Bao <baopeng@iccas.ac.cn>
# Modified by: Ziqiu Wang <3417847429@qq.com>

import numpy as np
import scipy.linalg
from pyscf import gto, scf
from pyscf.data.nist import HARTREE2EV

mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='cc-pvdz')

# To show details, set DEBUG=True.
DEBUG = False

############### HF in PySCF to compare ##############
mf = scf.RHF(mol)
mf.diis = False

if DEBUG:
    mf.verbose = 4
else:
    mf.verbose = 1

mf.kernel()
print('Reference HF total energy =', mf.e_tot)
#####################################################

#################### HF Code Example ####################
# RHF. Only need structure information of molecule and electronic integrals 
  
# one-electron Hamiltonian integral hcore_ij = <i|h|j>
hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
# one-electron electron-overlap integral s1e_ij = <i|j>
s1e = mol.intor_symmetric('int1e_ovlp')
# number of space orbitals by given basis
nao = hcore.shape[0]
# two-electron integrals(Electron Repulsion Integrals)
eri = mol.intor('int2e').reshape(nao,nao,nao,nao)
# number of occupied space orbitals
nocc = mol.nelectron // 2
nvir = nao - nocc
mo_occ = np.array([2]*nocc + [0]*nvir)

def energy_nuc(mol):
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    rr = np.linalg.norm(coords.reshape(-1,1,3) - coords, axis=2)
    rr[np.diag_indices_from(rr)] = 1e200
    e = np.einsum('i,ij,j->', charges, 1./rr, charges) * .5
    return e

# density matrix of initial guess in pyscf.scf.hf is doubled for there are two
# electrons in one space orbital in fact, dm in pyscf is made through make_rdm
# subroutine, so we divide the initial guess dm by 2
dm = mf.init_guess_by_minao(mol) / 2
vhf = np.einsum('ijkl,ji->kl', eri, dm)*2 - np.einsum('ijkl,jk->il', eri, dm) 
# e_tot = 2tr(hcore) + tr(vhf) = tr(2 * hcore + vhf)
e_tot = np.einsum('ij, ji->', 2 * hcore + vhf , dm) + energy_nuc(mol)
if DEBUG:
    print("Initial_e: ", e_tot)

def get_grad(mo_coeff, mo_occ, fock_ao) -> np.ndarray:
    '''RHF orbital gradients from PySCF

    Args:
        mo_coeff : 2D ndarray
            Obital coefficients
        mo_occ : 1D ndarray
            Orbital occupancy
        fock_ao : 2D ndarray
            Fock matrix in AO representation

    Returns:
        Gradients in MO representation.  It's a num_occ*num_vir vector.
    '''
    occidx = mo_occ > 0
    viridx = ~occidx
    g = 2 * (mo_coeff[:,viridx].conj().T @ fock_ao @ mo_coeff[:,occidx])
    return g.ravel()

scf_conv = False
cycle = 0
while not scf_conv and cycle < 50:
    dm_last = dm
    last_hf_e = e_tot

    fock = hcore + vhf

    # Diagonalize fock matrix to get mo_energy and mo_coefficients
    mo_energy, mo_coeff = scipy.linalg.eigh(fock, s1e)
    # Take occupied orbitals and form the new density matrix
    dm = mo_coeff[:,:nocc] @ mo_coeff[:,:nocc].T

    # Since new density matrix corresponds to new V_hf, we should update Fock
    # matrix before calculating total energy, otherwise there will be error
    # introduced though it does not affect the final result 
    # (dm and dm_last will finally be identical)
    vhf = (
            np.einsum('ijkl,ji->kl', eri, dm) * 2 
           - np.einsum('ijkl,jk->il', eri, dm) )

    # Calculate total energy using 'e_tot = tr(2 * hcore + vhf)'
    e_homo = mo_energy[nocc - 1]
    e_lumo = mo_energy[nocc]
    e_tot = np.einsum('ij,ji->', hcore * 2 + vhf, dm)  + energy_nuc(mol)

    # judge if converged
    delta_e = e_tot - last_hf_e
    norm_ddm = np.linalg.norm(dm-dm_last)
    norm_gorb = np.linalg.norm(get_grad(mo_coeff, mo_occ, fock))
    if abs(delta_e) < 1.0E-10 and norm_gorb < 1.0E-5:
        scf_conv = True
    cycle += 1

    if DEBUG:
        print('cycle= {:^3d} E= {:.15g}  delta_E= {:^6.3g}  '
              .format(cycle, e_tot, delta_e), 
              '|g|= {:^6.3g}  |ddm|= {:^6.3g}' .format(norm_gorb, norm_ddm)
              )
        print('  HOMO = {:.15f}  LUMO = {:.15f} gap/eV = {:.5f}'
              .format(e_homo, e_lumo, HARTREE2EV * (e_lumo-e_homo)))

print(
        'HF Code Example total energy = {}, Cycle number = {}, Converged? {}'
        .format(e_tot, cycle, scf_conv)
        )

