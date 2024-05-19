"""
doVUMPS.py
---------------------------------------------------------------------
Module for optimizing an infinite MPS using the VUMPS algorithm.

by Glen Evenbly (c) for www.tensors.net, (v1.0) - last modified 07/2020
"""

import numpy as np
from numpy import linalg as LA
from ncon import ncon
from scipy.sparse.linalg import LinearOperator, eigsh, gmres
from scipy.linalg import polar
from typing import List, Optional


def doVUMPS(AL: np.ndarray,
            C: np.ndarray,
            AR: np.ndarray,
            h0: np.ndarray,
            HL: Optional[np.ndarray] = None,
            HR: Optional[np.ndarray] = None,
            m: Optional[int] = None,
            update_mode: Optional[str] = 'polar',
            num_iter: Optional[int] = 1000,
            ev_tol: Optional[float] = 1e-12,
            en_exact: Optional[float] = 0.0):
  """
  Implementation the VUMPS algorithm to optimize an infinite MPS with a 1-site
  unit cell.
  Args:
    AL: the rank-3 MPS tensor in the left orthogonal gauge.
    C: vector of the singular weights of the MPS.
    AR: the rank-3 MPS tensor in the right orthogonal gauge.
    h0: two-site Hamiltonian coupling.
    HL: the left block Hamiltonian (if previously computed).
    HR: the right block Hamiltonian (if previously computed).
    m: the MPS bond dimension. If the input MPS tensors are of smaller
      dimension, then they will be exapnded to match `m`.
    update_mode: set updates via 'polar' decomposition (more stable) of the
      'svd' (less stable).
    num_iter: number of iterations to perform.
    ev_tol: tolerance for convergence of eighs.
    en_exact: the exact energy (if known).
  Returns:
    np.ndarray: MPS tensor AL
    np.ndarray: MPS singular weights C
    np.ndarray: MPS tensor AR
    np.ndarray: block Hamiltonian HL
    np.ndarray: block Hamiltonian HR
  """

  # Initialise tensors
  d = h0.shape[0]
  h = h0.copy()
  if HL is None:
    HL = np.zeros([m, m])
  if HR is None:
    HR = np.zeros([m, m])

  if m is None:
    m = AL.shape[2]
  else:
    if m > AL.shape[2]:
      # Expand tensors to new dimension
      AL = Orthogonalize(TensorExpand(AL, [m, d, m]), 2)
      C = TensorExpand(C, [m])
      AR = Orthogonalize(TensorExpand(AR, [m, d, m]), 2)
      HL = TensorExpand(HL, [m, m])
      HR = TensorExpand(HR, [m, m])
  AC = ncon([AL, np.diag(C)], [[-1, -2, 1], [1, -3]])

  # Begin variational update iterations
  for p in range(num_iter):
    """ Evaluate the energy """
    tensors = [AL, AL, h0, AL.conj(), AL.conj()]
    connects = [[7, 1, 6], [6, 2, -1], [3, 4, 1, 2], [7, 3, 5], [5, 4, -2]]
    con_order = [7, 1, 3, 6, 2, 5, 4]
    hL0 = ncon(tensors, connects, con_order)
    energyL = np.trace(np.diag(C**2) @ hL0)

    tensors = [AR, AR, h0, AR.conj(), AR.conj()]
    connects = [[6, 4, 1], [1, 3, -1], [5, 7, 3, 4], [6, 7, 2], [2, 5, -2]]
    con_order = [6, 4, 7, 1, 3, 2, 5]
    hR0 = ncon(tensors, connects, con_order)
    energyR = np.trace(np.diag(C**2) @ hR0)

    energy = 0.25 * (energyL + energyR)
    en_error = energy - en_exact
    print('iteration: %d of %d, dim: %d, energy: %4.4f, '
          'en-error: %2.2e' % (p, num_iter, m, energy, en_error))

    """ Contract the MPS from the left """
    # Shift left edge Hamiltonian terms to set energy to zero
    tensors = [AL, AL, h, AL.conj(), AL.conj()]
    connects = [[7, 1, 6], [6, 2, -1], [3, 4, 1, 2], [7, 3, 5], [5, 4, -2]]
    con_order = [7, 1, 3, 6, 2, 5, 4]
    hL = ncon(tensors, connects, con_order)

    en_density = np.dot(np.diag(hL), C**2)
    hL -= en_density * np.eye(m)
    h -= en_density * np.eye(d**2).reshape(d, d, d, d)
    HL -= np.dot(np.diag(HL), C**2) * np.eye(m)

    # Solve left edge Hamiltonian eigenvector
    def LeftEdge(HL):
      m = AL.shape[0]
      d = AL.shape[1]
      HL_T = AL.reshape(m * d, m).T @ (HL.reshape(m, m) @ AL.reshape(m, d * m)
                                       ).reshape(m * d, m)
      return HL.flatten() - HL_T.flatten()
    LeftOp = LinearOperator((m**2, m**2), matvec=LeftEdge, dtype=np.float64)
    HL_temp, is_conv = gmres(LeftOp, hL.flatten(), x0=HL.flatten(), tol=1e-10,
                             restart=None, maxiter=5, atol=None)
    HL_temp = HL_temp.reshape(m, m)
    HL = 0.5 * (HL_temp + HL_temp.T)

    """ Contract the MPS from the right """
    # Shift right edge Hamiltonian terms to set energy to zero
    tensors = [AR, AR, h, AR.conj(), AR.conj()]
    connects = [[6, 4, 1], [1, 3, -1], [5, 7, 3, 4], [6, 7, 2], [2, 5, -2]]
    con_order = [6, 4, 7, 1, 3, 2, 5]
    hR = ncon(tensors, connects, con_order)
    en_density = np.dot(np.diag(hR), C**2)

    hR -= en_density * np.eye(m)
    h -= np.dot(np.diag(hR), C**2) * np.eye(d**2).reshape(d, d, d, d)
    HR -= np.dot(np.diag(HR), C**2) * np.eye(m)

    # Solve right edge Hamiltonian eigenvector
    def RightEdge(HR):
      m = AR.shape[0]
      d = AR.shape[1]
      HR_T = AR.reshape(m * d, m).T @ (HR.reshape(m, m) @ AR.reshape(m, d * m)
                                       ).reshape(m * d, m)
      return HR.flatten() - HR_T.flatten()
    RightOp = LinearOperator((m**2, m**2), matvec=RightEdge, dtype=np.float64)
    HR_temp, is_conv = gmres(RightOp, hR.flatten(), x0=HR.flatten(), tol=1e-10,
                             restart=None, maxiter=5, atol=None)
    HR_temp = HR_temp.reshape(m, m)
    HR = 0.5 * (HR_temp + HR_temp.T)

    """ Update the MPS singular values """
    # define function for applying the block Hamiltonian
    def MidWeights(C_mat):
      m = AL.shape[2]
      C_mat = C_mat.reshape(m, m)
      tensors = [AL, h, AL.conj(), AR, AR.conj(), C_mat]
      connects = [[3, 1, 8], [2, 7, 1, 5], [3, 2, -1], [6, 5, 4], [6, 7, -2],
                  [8, 4]]
      con_order = [4, 8, 1, 5, 6, 7, 3, 2]
      return (ncon(tensors, connects, con_order) + (HL @ C_mat) +
              (C_mat @ HR)).flatten()

    WeightOp = LinearOperator((m**2, m**2), matvec=MidWeights, dtype=np.float64)
    C_temp = eigsh(WeightOp, k=1, which='SA', v0=np.diag(C).flatten(),
                   ncv=None, maxiter=None, tol=ev_tol)[1]

    # Change to diagonal gauge
    ut, C, vt = LA.svd(C_temp.reshape(m, m))
    AL = ncon([ut.T, AL, ut], [[-1, 1], [1, -2, 2], [2, -3]])
    AR = ncon([vt, AR, vt.T], [[-1, 1], [1, -2, 2], [2, -3]])
    HL = ut.T @ HL @ ut
    HR = vt @ HR @ vt.T

    # Contract left/middle Hamiltonian term
    tensors = [AL, h0, AL.conj()]
    connects = [[3, 1, -1], [2, -4, 1, -2], [3, 2, -3]]
    con_order = [2, 3, 1]
    hL_mid = ncon(tensors, connects, con_order).reshape(d * m, d * m)

    # Contract right/middle Hamiltonian term
    tensors = [AR, h0, AR.conj()]
    connects = [[2, 1, -2], [-3, 3, -1, 1], [2, 3, -4]]
    con_order = [2, 1, 3]
    hR_mid = ncon(tensors, connects, con_order).reshape(d * m, d * m)

    """ Update the MPS tensors """
    # define function for applying the block Hamiltonian
    def MidTensor(AC):
      m = AL.shape[2]
      d = AL.shape[1]
      return ((HL @ AC.reshape(m, d * m)).flatten() +
              (hL_mid.reshape(m * d, m * d) @ AC.reshape(m * d, m)).flatten() +
              (AC.reshape(m * d, m) @ HR).flatten() +
              (AC.reshape(m, d * m) @ hR_mid.reshape(d * m, d * m)).flatten())

    TensorOp = LinearOperator((d * m**2, d * m**2),
                              matvec=MidTensor, dtype=np.float64)
    AC = (eigsh(TensorOp, k=1, which='SA', v0=AC.flatten(),
                ncv=None, maxiter=None, tol=ev_tol)[1]).reshape(m, d, m)

    if update_mode == 'polar':
      AL = (polar(AC.reshape(m * d, m))[0]).reshape(m, d, m)
      AR = (polar(AC.reshape(m, d * m), side='left')[0]
            ).reshape(m, d, m).transpose(2, 1, 0)
    elif update_mode == 'svd':
      ut, _, vt = LA.svd(AC.reshape(m * d, m) @ np.diag(C), full_matrices=False)
      AL = (ut @ vt).reshape(m, d, m)
      ut, _, vt = LA.svd(np.diag(C) @ AC.reshape(m, d * m), full_matrices=False)
      AR = (ut @ vt).reshape(m, d, m).transpose(2, 1, 0)

  return AL, C, AR, HL, HR


def TensorExpand(A: np.ndarray, chivec: List):
  """ expand tensor dimension by padding with zeros """

  if [*A.shape] == chivec:
    return A
  else:
    for k in range(len(chivec)):
      if A.shape[k] != chivec[k]:
        indloc = list(range(-1, -len(chivec) - 1, -1))
        indloc[k] = 1
        A = ncon([A, np.eye(A.shape[k], chivec[k])], [indloc, [1, -k - 1]])

    return A


def Orthogonalize(A: np.ndarray, pivot: int):
  """ orthogonalize an array with respect to a pivot """

  A_sh = A.shape
  ut, st, vht = LA.svd(
      A.reshape(np.prod(A_sh[:pivot]), np.prod(A_sh[pivot:])),
      full_matrices=False)
  return (ut @ vht).reshape(A_sh)
