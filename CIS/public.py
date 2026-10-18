from pyscf import gto, scf
import pyscf.data.nist # NIST Physical Constants

############### HF calc in PySCF ##############
# Supposing closed shell configuration
mol = gto.M(
        atom=
        '''
        O 0. 0. 0.
        H 0.86681633521 0.67723114352 0.
        H -0.86681633521 0.67723114352 0.
        ''', 
        basis='sto-3g',
        )
mf = scf.RHF(mol)
mf.kernel()

def print_excited_states(e_excited_states, title=None,
                         tablewidth=10, accuracy=6,
                         ) -> None:
    HARTREE2EV = pyscf.data.nist.HARTREE2EV
    ncolumns = 3
    if title is None: title = ''
    print('{:*^{width}}'.format(title, width = ncolumns*(tablewidth + 1)))
    n_es = len(e_excited_states)
    e_es = e_excited_states
    print(
            '{:^{width}},{:>{width}},{:>{width}},'
            .format('INDEX', 'Hartree', 'eV', width = tablewidth)
            )
    for i in range(n_es):
        print(
                '{:^{width}n},{:>{width}.{accuracy}f},{:>{width}.{accuracy}f},'
                .format(i, e_es[i], e_es[i] * HARTREE2EV,
                        width = tablewidth, accuracy = accuracy)
                )

    print('{:*^{width}}'
          .format('End of ' + title, width = ncolumns*(tablewidth + 1)))
