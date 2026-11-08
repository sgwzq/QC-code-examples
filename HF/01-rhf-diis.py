#!/usr/bin/env python3
# HF-DIIS Code Example
# Author: Peng Bao <baopeng@iccas.ac.cn>
# Modified by: Ziqiu Wang <sgwzq0810@gmail.com>

import numpy as np
import scipy.linalg
from pyscf import gto, scf
from pyscf.scf.hf import get_grad, init_guess_by_minao
from pyscf.data.nist import HARTREE2EV

# ------- Options -------
DEBUG = False
DIIS_DEBUG = False
DIIS_MAX_VOLUME = 5
DIIS_MAXERR_THRESHOLD = 0.1 # Unit: Energy, Hartree.
# --- End of Options ---

mol = gto.M(atom='''
 H          2.031885   -1.178162    0.0000
 H          2.727235    0.545776   -0.0000
 C          1.856825   -0.104575   -0.0000
 C          0.568770    0.416868   -0.0000
 H          0.432847    1.496788   -0.0000
 H         -0.432847   -1.496788    0.0000
 C         -0.568770   -0.416868    0.0000
 C         -1.856825    0.104575    0.0000
 H         -2.727235   -0.545776    0.0000
 H         -2.031885    1.178162   -0.0000
''', basis='cc-pvdz')
if DEBUG:
    mol.verbose = 4
else:
    mol.verbose = 0
############### HF in PySCF to compare ##############
mf = scf.RHF(mol)
mf.kernel()
print('Reference HF total energy =', mf.e_tot)
#####################################################

#################### HF-DIIS Code Example ####################
# RHF. Only need structure information of molecule and electronic integrals 
hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
s1e = mol.intor_symmetric('int1e_ovlp')
nao = hcore.shape[0]
eri = mol.intor('int2e').reshape(nao,nao,nao,nao)
nocc = mol.nelectron // 2
nvir = nao - nocc
mo_occ = np.array([2]*nocc + [0]*nvir)

# diis ref:
# 1. P. Pulay, J. Comp. Chem. 3, 556 (1982).
# 2. P. Pulay, Chem. Phys. Lett. 73, 393 (1980). 
# DIIS Initialization
_err_vec = [None] * DIIS_MAX_VOLUME
_fock_vec = [None] * DIIS_MAX_VOLUME
quad_mat = np.zeros((DIIS_MAX_VOLUME+1,) * 2)
quad_mat[1:, 0] = -1
quad_mat[0, 1:] = -1
right_vec = np.zeros((DIIS_MAX_VOLUME + 1,))
right_vec[0] = -1
cnt = 0
# DIIS updating
def diis(fock, err):
    global DIIS_MAX_VOLUME, DIIS_MAXERR_THRESHOLD, right_vec
    global cnt, _err_vec, _fock_vec, quad_mat
    err_max = err.max()
    enter_diis = err_max < DIIS_MAXERR_THRESHOLD
    if DIIS_DEBUG:
        print(f'Maximum of err matrix: {err_max: .6f}, entering DIIS? '
              f'{enter_diis}')
    if not enter_diis:
        return fock
    index = cnt % DIIS_MAX_VOLUME # Circular assignment. First in First out.
    _fock_vec[index] = fock
    _err_vec[index] = err
    mat_idx = index + 1
    for i, err_i in enumerate(_err_vec):
        if err_i is None: break
        mat_idx_i = i + 1
        mat_element = np.trace(err_i.T @ err)
        quad_mat[mat_idx_i, mat_idx] = mat_element
        quad_mat[mat_idx, mat_idx_i] = mat_element
    cnt += 1
    vec_vol = min(cnt, DIIS_MAX_VOLUME)
    diis_coeff = np.linalg.solve(
            quad_mat[:vec_vol+1,:vec_vol+1], right_vec[:vec_vol+1])
    if DIIS_DEBUG:
        print('DIIS coefficients of Fock matrix: '
              + ', '.join(f'{coeff: .5f}' for coeff in diis_coeff[1:]))
    new_fock = np.zeros_like(fock)
    for i in range(vec_vol):
        new_fock += diis_coeff[i+1] * _fock_vec[i]
    return new_fock

def energy_nuc(mol):
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    rr = np.linalg.norm(coords.reshape(-1,1,3) - coords, axis=2)
    rr[np.diag_indices_from(rr)] = float('inf')
    e = np.einsum('i,ij,j->', charges, 1./rr, charges) * .5
    return e

def get_vhf(dm: np.ndarray, eri) -> np.ndarray:
    return (
            np.einsum('ijkl,ji->kl', eri, dm)*2 
            - np.einsum('ijkl,jk->il', eri, dm) )

dm = init_guess_by_minao(mol) / 2.
vhf = get_vhf(dm, eri)
e_tot = np.einsum('ij, ji->', 2 * hcore + vhf , dm) + energy_nuc(mol)

# ------- The Main Process -------
scf_conv = False
cycle = 0
while not scf_conv and cycle < 50:
    dm_last = dm
    last_hf_e = e_tot

    fock = hcore + vhf

    #defalt: Pulay, P. J. Comput. Chem. 3, 556 (1982).
    sdf = s1e@dm@fock
    err = sdf.T - sdf   

    fock = diis(fock, err)

    mo_energy, mo_coeff = scipy.linalg.eigh(fock, s1e)
    dm = mo_coeff[:,:nocc] @ mo_coeff[:,:nocc].T
    vhf = get_vhf(dm, eri)

    e_homo = mo_energy[nocc - 1]
    e_lumo = mo_energy[nocc]
    e_tot = np.einsum('ij,ji->', hcore*2 + vhf, dm)  + energy_nuc(mol)

    # judge if converged
    delta_e = e_tot - last_hf_e
    norm_ddm = np.linalg.norm(dm-dm_last)
    norm_gorb = np.linalg.norm(get_grad(mo_coeff, mo_occ, fock))
    if abs(delta_e) < 1.0E-10 and norm_gorb < 1.0E-5:
        scf_conv = True
    cycle += 1

    if DEBUG:
        print('cycle= {:^3d} E= {:.15g}  delta_E= {:^6.3g}  '
              '|g|= {:^6.3g}  |ddm|= {:^6.3g}' 
              .format(cycle, e_tot, delta_e, norm_gorb, norm_ddm)
              )
        print('  HOMO = {:.15f}  LUMO = {:.15f} gap/eV = {:.5f}'
              .format(e_homo, e_lumo, HARTREE2EV * (e_lumo-e_homo)))

print(
    'HF Code Example total energy = {}, Cycle number = {}, Converged? {}'
    .format(e_tot, cycle, scf_conv)
    )

