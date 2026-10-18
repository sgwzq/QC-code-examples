#!/usr/bin/env python3
# rcis-spin-adapted.py
# Restricted Configuration Interaction Singles
# Author: Ziqiu Wang

import numpy
import scipy.linalg
import itertools
from pyscf import ao2mo
from public import mf, print_excited_states
mol = mf.mol

# Setting variables
nao = len(mol.ao_labels())
noccmo = mol.nelectron // 2
nvirmo = nao - noccmo

# CIS procedure
# 1. Getting Fock Matrix from linear combination of Gaussian orbitals
#    to Molecular Orbitals
mo_eri = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape(nao, nao, nao, nao)

# 2. Build Hamiltonian Matrix
H_size = noccmo * nvirmo
H_singlets = numpy.zeros((H_size, H_size))
H_triplets = numpy.zeros((H_size, H_size))

ridx, cidx = 0, 0
# i, j, a, b are spatial molecular orbitals
# i, j range from (0, number of occupied mo)
# a, b range from (0, number of virtual mo)
# so the index of virt mo is (a + noccmo)
for i, a in itertools.product(range(noccmo), range(nvirmo)):
    cidx = 0
    for j, b in itertools.product(range(noccmo), range(nvirmo)):
        if i == j and a == b:
            e_gap = mf.mo_energy[a + noccmo] - mf.mo_energy[i]
            H_singlets[ridx][cidx] += e_gap
            H_triplets[ridx][cidx] += e_gap
        H_singlets[ridx][cidx] +=(
                2* mo_eri[j][b + noccmo][i][a + noccmo]
                - mo_eri[j][i][a + noccmo][b + noccmo])
        H_triplets[ridx][cidx] -= mo_eri[j][i][a + noccmo][b + noccmo]
        cidx += 1
    ridx += 1

e_cis_singlets, co_cis_singlets = scipy.linalg.eigh(H_singlets)
e_cis_triplets, co_cis_triplets = scipy.linalg.eigh(H_triplets)

print_excited_states(e_cis_singlets, title='CIS singlets Energy')
print_excited_states(e_cis_triplets, title='CIS triplets Energy')
