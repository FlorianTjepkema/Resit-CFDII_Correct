"""
Author: Suyash Shrestha
License: MIT
Copyright (c) 2024, Suyash Shrestha
All rights reserved.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
# from concurrent.futures import ThreadPoolExecutor

class gaussLobatto():
	def __init__(self) -> None:
		pass

	def gllNodes(self, n):
		c = np.zeros(n);
		c[-1] = 1;
		roots = np.polynomial.legendre.legroots(np.polynomial.legendre.legder(c));
		xi = np.concatenate(([-1], roots, [1]));
		return xi;

	def gllWeights(self, xi):
		p = xi.shape[0];
		A = np.ones((p, p));
		for i in range(1, p):
			A[i, :] = xi**i;

		B = np.zeros(p);
		for i in range(1, p + 1):
			B[i - 1] = (xi[-1]**i - xi[0]**i)/i;
		w = np.linalg.solve(A, B);
		return w;

class mimeticBasis():
	def __init__(self) -> None:
		pass

	def lagrangeBasis(self, i, GLnodes, xi):
		l = np.ones_like(xi);
		for j in range(GLnodes.shape[0]):
			if i != j:
				l *= (xi - GLnodes[j])/(GLnodes[i] - GLnodes[j]);
		return l;

	def ddx_lagrangeBasis(self, i, GLnodes, xi):
		lPrime = np.zeros_like(xi);
		for m in range(GLnodes.shape[0]):
			if i != m:
				t = np.ones_like(xi);
				for j in range(GLnodes.shape[0]):
					if i != j and j != m:
						t *= (xi - GLnodes[j])/(GLnodes[i] - GLnodes[j]);
				lPrime += 1/(GLnodes[i] - GLnodes[m])*t;
		return lPrime;

	def lagrangeEdgeBasis(self, i, GLnodes, xi):
		e = np.zeros_like(xi);
		for j in range(i):
			e += self.ddx_lagrangeBasis(j, GLnodes, xi);
		e *= -1;
		return e;

	def extendedEdgeBasis(self, GLnodes, xi):
		p = GLnodes.shape[0] - 1;
		eLR = np.zeros((2, xi.shape[0]));
		L = np.polynomial.legendre.legval(xi, np.concatenate((np.zeros(p), np.ones(1))));
		dLdx = np.polynomial.legendre.legval(xi, np.polynomial.legendre.legder(np.concatenate((np.zeros(p), np.ones(1))), 1));
		eLR[1, :] = 0.5*((1 + xi)*L - (1 - xi**2)*dLdx/(p*(p + 1)));
		eLR[0, :] = 0.5*(-1)**(p)*((1 - xi)*L + (1 - xi**2)*dLdx/(p*(p + 1)));
		return eLR;

	def evalBasis(self, basis_k, GLnodes, xi):
		if basis_k == 0:
			psi = np.zeros((GLnodes.shape[0], xi.shape[0]));
			for i in range(GLnodes.shape[0]):
				psi[i, :] = self.lagrangeBasis(i, GLnodes, xi);
		elif basis_k == 1:
			psi = np.zeros((GLnodes.shape[0] - 1, xi.shape[0]));
			for i in range(1, GLnodes.shape[0]):
				psi[i - 1, :] = self.lagrangeEdgeBasis(i, GLnodes, xi);
		elif basis_k == 101:
			psi = np.zeros((GLnodes.shape[0] + 1, xi.shape[0]));
			for i in range(1, GLnodes.shape[0]):
				psi[i - 1, :] = self.lagrangeEdgeBasis(i, GLnodes, xi);
			psi[-2:, :] = self.extendedEdgeBasis(GLnodes, xi);
		return psi;

class sparseLinAlgTools():

	def sparseElimZeros(A: sp.csr_matrix, tol: np.float64):
		A.data[np.abs(A.data)/np.abs(A.data).max() <= tol] = 0;
		A.eliminate_zeros();

	def shurComplementInv(Ainv: sp.csr_matrix, B: sp.csr_matrix, C: sp.csr_matrix, D: sp.csr_matrix):
		inv_D_CAinv_B = la.spsolve(D - C@Ainv@B, sp.eye(D.shape[0], format = 'csr'));
		sysinv0 = Ainv + Ainv@B@inv_D_CAinv_B@C@Ainv;
		sysinv1 = -Ainv@B@inv_D_CAinv_B;
		sysinv2 = -inv_D_CAinv_B@C@Ainv;
		sysinv3 = inv_D_CAinv_B;
		return sysinv0, sysinv1, sysinv2, sysinv3;

	# def parallel_apply_inv(LU, RHS, max_workers = 8):
	# 	apply_inv = lambda lu, column: lu.solve(column);
	# 	result = np.zeros_like(RHS);
	# 	with ThreadPoolExecutor(max_workers = max_workers) as executor:
	# 		futures = {executor.submit(apply_inv, LU, RHS[:, j]): j for j in range(RHS.shape[1])};
	# 		for future in futures:
	# 			j = futures[future];
	# 			result[:, j] = future.result();
	# 	return result;


