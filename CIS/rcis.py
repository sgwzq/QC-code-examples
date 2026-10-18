#!/usr/bin/env python3
# rcis.py
# Spin-Orbital Configuration Interaction Singles
# Author: Ziqiu Wang

import numpy
import scipy.linalg
import itertools
from public import mf, print_excited_states
mol = mf.mol

# Setting variables
nao = len(mol.ao_labels())
nelec = mol.nelectron
nvirt = 2 * nao - nelec
nsingles = nelec * nvirt

# CIS procedure
# 1. Getting Fock Matrix from linear combination of Gaussian orbitals
#    to Molecular Orbitals

# (2-)Electron Repulsion Integral
eri = mol.intor('int2e').reshape(nao,nao,nao,nao)

# Transform ERI from Atomic basis to Molecular basis
mo_eri = numpy.einsum(
        'abcd,ai,bj,ck,dl->ijkl',
        eri, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff)

# 2. Calculating Hamiltonian Matrix
def get_H_single(i, a, j, b):
    # <\Psi i->a|H|\Psi j->b>
    # i a j b are all spin molecular orbitals
    # supposing closed shell
    # even(0, 2, 4...) number indexes represents for alpha-spin, odd for beta
    ene = 0.
    if i == j and a == b:
        # one-electron energy for a and i
        ene = mf.mo_energy[(nelec + a) // 2] - mf.mo_energy[i // 2]
    # two-electron integrals -<ja||ib> or -([ji|ab] - [jb|ia])
    if not (i & 1)^(j & 1) and not (a & 1)^(b & 1):
        # i and j are of the same spin, a and b also are
        ene -= mo_eri[j // 2][i // 2][(a + nelec) // 2][(b + nelec) // 2]
    if not (j & 1)^(b & 1) and not (i & 1)^(a & 1):
        ene += mo_eri[j // 2][(b + nelec) // 2][i // 2][(a + nelec) // 2]
    return ene

H = numpy.zeros((nsingles, nsingles))
ridx, cidx = 0, 0
for j, i in itertools.product(range(nelec), range(nvirt)):
    cidx = 0
    for l, k in itertools.product(range(nelec), range(nvirt)):
        H[ridx][cidx] = get_H_single(j, i, l, k)
        cidx += 1
    ridx += 1

e_cis, co_cis = scipy.linalg.eigh(H)

print_excited_states(e_cis, title='CIS Correlation Energy')
print('CIS First Excited State(S1) Energy = ', e_cis[0] + mf.e_tot)

