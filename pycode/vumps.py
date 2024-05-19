"""
mainVUMPS.py
---------------------------------------------------------------------
Script file for initializing the Hamiltonian and MPS tensors before passing to
the VUMPS routine to optimize for the ground state.

by Glen Evenbly (c) for www.tensors.net, (v1.0) - last modified 07/2020
"""

import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import LinearOperator, eigs
from ncon import ncon
from doVUMPS import doVUMPS


""" set algorithm options """
m = 32  # MPS bond dimension
update_mode = 'polar'  # set update decomposition ('polar' or 'svd')
ev_tol = 1e-5  # tolerance for convergence of eighs
model = 'heisenberg'  # set model ('heisenberg', 'XX' or 'ising')
num_iter = 20  # number of iterations to perform


""" define the Hamiltonian """
d = 4
sX = np.array([[0, 1.0], [1.0, 0]])
sY = np.array([[0, -1.0 * 1j], [1.0 * 1j, 0]])
sZ = np.array([[1.0, 0], [0, -1.0]])
sI = np.array([[1.0, 0], [0, 1.0]])
if model == 'heisenberg':
  h_temp = (np.real(np.kron(sX, sX) + np.kron(sY, sY) + np.kron(sZ, sZ)))
  h0 = (0.5 * np.kron(h_temp, np.eye(4)) + 0.5 * np.kron(np.eye(4), h_temp) +
        np.kron(np.eye(2), np.kron(h_temp, np.eye(2)))).reshape(d, d, d, d)
  en_exact = (1 - 4 * np.log(2))
elif model == 'XX':
  h_temp = (np.real(np.kron(sX, sX) + np.kron(sY, sY)))
  h0 = (0.5 * np.kron(h_temp, np.eye(4)) + 0.5 * np.kron(np.eye(4), h_temp) +
        np.kron(np.eye(2), np.kron(h_temp, np.eye(2)))).reshape(d, d, d, d)
  en_exact = -4 / np.pi
elif model == 'ising':
  h_temp = (-np.real(np.kron(sX, sX) + 0.5 * np.kron(sZ, sI) +
                     0.5 * np.kron(sI, sZ)))
  h0 = (0.5 * np.kron(h_temp, np.eye(4)) + 0.5 * np.kron(np.eye(4), h_temp) +
        np.kron(np.eye(2), np.kron(h_temp, np.eye(2)))).reshape(d, d, d, d)
  en_exact = -4 / np.pi

""" initialize the MPS tensors """
C = np.random.rand(m)
C = C / LA.norm(C)
AL = (LA.svd(np.random.rand(m * d, m), full_matrices=False)[0]).reshape(m, d, m)
AR = (LA.svd(np.random.rand(m * d, m), full_matrices=False)[0]).reshape(m, d, m)
HL = np.zeros([m, m])
HR = np.zeros([m, m])


""" run the optimization algorithm """
AL, C, AR, HL, HR = doVUMPS(AL, C, AR, h0, HL=HL, HR=HR, m=m,
                            update_mode=update_mode, num_iter=num_iter,
                            ev_tol=ev_tol, en_exact=en_exact)


""" increase bond dim, reduce tolerance, then continue updates """
m = 64
ev_tol = 1e-7
AL, C, AR, HL, HR = doVUMPS(AL, C, AR, h0, HL=HL, HR=HR, m=m,
                            update_mode=update_mode, num_iter=num_iter,
                            ev_tol=ev_tol, en_exact=en_exact)


""" compute the local 2-site reduced density matrix """
# define linear operator for contraction from the right
def RightDensity(rhoR):
  m = AL.shape[2]
  tensors = [AL, AL.conj(), rhoR.reshape(m, m)]
  connects = [[-2, 1, 2], [-1, 1, 3], [3, 2]]
  con_order = [2, 1, 3]
  return ncon(tensors, connects, con_order).flatten()


RightDensityOp = LinearOperator((m**2, m**2), matvec=RightDensity,
                                dtype=np.float64)
# solve eigenvalue problem for r.h.s density matrix
rhoR_temp = eigs(RightDensityOp, k=1, which='LM', v0=np.diag(C**2).flatten(),
                 ncv=None, maxiter=None, tol=0)[1]
rhoR_temp = np.real(rhoR_temp.reshape(m, m))
rhoR_temp = rhoR_temp + rhoR_temp.T.conj()
rhoR = rhoR_temp / np.trace(rhoR_temp)

# contract for the 2-site local reduced density matrix
tensors = [AL, AL, AL.conj(), AL.conj(), rhoR]
connects = [[3, -1, 4], [4, -2, 1], [3, -3, 5], [5, -4, 2], [1, 2]]
rho_two = ncon(tensors, connects)

en_final = ncon([h0, rho_two], [[1, 2, 3, 4], [1, 2, 3, 4]]) / 2
en_error = en_final - en_exact
print('final energy: %4.4f, en-error: %2.2e' % (en_final, en_error))
