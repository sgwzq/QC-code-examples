#!/usr/bin/env python
# MP2 Code Example
# Author: Peng Bao <baopeng@iccas.ac.cn>
# Modified by: Ziqiu Wang <sgwzq0810@gmail.com>

import numpy
import itertools
from pyscf import gto, scf, mp

############### MP2 in PySCF to compare   #############
mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='sto-3g')

mf = scf.RHF(mol)
mf.kernel()

mp2 = mp.RMP2(mf)
mp2.kernel()
######################################################

#################### MP2 Code Example ####################
mo=mf.mo_coeff
mo_energy=mf.mo_energy

nocc = mol.nelectron // 2
nvir = mo.shape[0] - nocc
mo_occ = mo[:, :nocc]
mo_vir = mo[:, nocc:]

ao_eri = mol.intor('int2e')
mo_eri = numpy.einsum(
        'ui,va,xj,yb, uvxy->iajb',
        mo_occ, mo_vir, mo_occ, mo_vir, ao_eri,
        )

Jintm = 0.0
Kintm = 0.0
# Unwinding nested for loop with itertools.product function.
# See https://docs.python.org/3/library/itertools.html#itertools.product
for i, a, j, b in itertools.product(range(nocc), range(nvir), repeat = 2):
    devisor = (
            mo_energy[i] + mo_energy[j]
            - mo_energy[a+nocc] - mo_energy[b+nocc])
    Jintm += mo_eri[i,a,j,b] * mo_eri[i,a,j,b] / devisor
    Kintm += mo_eri[i,a,j,b] * mo_eri[j,a,i,b] / devisor

E_corr = 2*Jintm - Kintm
print('Simple MP2', 'E_corr = ', E_corr)
