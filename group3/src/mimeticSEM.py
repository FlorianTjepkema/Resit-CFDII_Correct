"""
Author: Suyash Shrestha
License: MIT
Copyright (c) 2024, Suyash Shrestha
All rights reserved.
"""
import numpy as np
import scipy.sparse as sp
import src.mimeticMachinery as miMc
import matplotlib.pyplot as plt
from src.mesh import mesh2D
from incidenceMatrices import assemble_element_E10, assemble_element_E21
from massMatrices import assemble_element_M0, assemble_element_M1, assemble_element_M2

class SEM2D():
	def __init__(self, mesh: mesh2D) -> None:
		self.mesh = mesh;
		self.p = self.mesh.p
		self.N = self.mesh.N
		self.M = self.mesh.M
		self.basisFunction = miMc.mimeticBasis();
	
	def tensorProd2D(self, psi_i, psi_j):
		return np.einsum('ijk, jm -> ijkm', psi_j[:, np.newaxis], psi_i).reshape(-1, psi_j.shape[1], psi_i.shape[1]);

	def basisEval(self, basis_k, xi, eta):
		GLnodes_xi, GLnodes_eta = self.mesh.xi, self.mesh.eta;
		if basis_k == 0:
			psi_i, psi_j = self.basisFunction.evalBasis(0, GLnodes_xi, xi), self.basisFunction.evalBasis(0, GLnodes_eta, eta);
			psi = self.tensorProd2D(psi_i, psi_j);
			return psi;
		elif basis_k == 1:
			psi_i_u, psi_j_u = self.basisFunction.evalBasis(0, GLnodes_xi, xi), self.basisFunction.evalBasis(1, GLnodes_eta, eta);
			psi_i_v, psi_j_v = self.basisFunction.evalBasis(1, GLnodes_xi, xi), self.basisFunction.evalBasis(0, GLnodes_eta, eta);
		
			psi_u = self.tensorProd2D(psi_i_u, psi_j_u);
			psi_v = self.tensorProd2D(psi_i_v, psi_j_v);
			return psi_u, psi_v;
		elif basis_k == 2:
			psi_i, psi_j = self.basisFunction.evalBasis(1, GLnodes_xi, xi), self.basisFunction.evalBasis(1, GLnodes_eta, eta);
			psi = self.tensorProd2D(psi_i, psi_j);
			return psi;
		elif basis_k == 202:
			psi_i, psi_j = self.basisFunction.evalBasis(0, self.mesh.txi, xi), self.basisFunction.evalBasis(0, self.mesh.teta, eta);
			psi_i_tnode, psi_j_tnode = self.basisFunction.evalBasis(0, self.mesh.txi, xi[[0, -1]]), self.basisFunction.evalBasis(0, self.mesh.teta, eta[[0, -1]]);
			psi_1D_tnode_x, psi_1D_tnode_y  = self.basisFunction.evalBasis(0, self.mesh.txi[1:-1], xi[[0, -1]]), self.basisFunction.evalBasis(0, self.mesh.teta[1:-1], eta[[0, -1]]);
			psi_base = self.tensorProd2D(psi_i[1:-1, :], psi_j[1:-1, :]);
			psi_corner = np.einsum('ik, jl -> klji', psi_i[[0, -1], :].T, psi_j[[0, -1], :].T).reshape(4, -1).T;
			psi_LR = np.einsum('ik, jl -> klji', psi_i[[0, -1], :].T, psi_j[1:-1, :].T).reshape(2*self.p, -1).T;
			psi_BT = np.einsum('ik, jl -> klij', psi_j[[0, -1], :].T, psi_i[1:-1, :].T).reshape(2*self.p, -1).T;
			psi_LR_corner_interp = np.einsum('ik, jl -> klij', psi_i_tnode[[0, -1], :].T, psi_1D_tnode_y.T).reshape(2*self.p, -1).T;
			psi_BT_corner_interp = np.einsum('ik, jl -> klji', psi_j_tnode[[0, -1], :].T, psi_1D_tnode_x.T).reshape(2*self.p, -1).T;
			return psi_base, psi_corner, psi_LR, psi_BT, psi_LR_corner_interp, psi_BT_corner_interp;

	def E10(self):
		E10 = assemble_element_E10(self.p);
		E10 = sp.kron(sp.eye(self.N*self.M), E10, format = 'csr');
		return E10;

	def E21(self):
		E21 = assemble_element_E21(self.p);
		E21 = sp.kron(sp.eye(self.N*self.M), E21, format = 'csr');
		return E21;

	def massMatrix(self, basis_k, inv = False, hiDOP = False):
		if hiDOP:
			xi, eta = self.mesh.xi_hiDOP, self.mesh.eta_hiDOP;
			wx, wy = self.mesh.wx_hiDOP, self.mesh.wy_hiDOP;
			det_J = self.mesh.detJ_hiDOP;
			J = self.mesh.J_hiDOP;
		else:
			xi, eta = self.mesh.xi, self.mesh.eta;
			wx, wy = self.mesh.wx, self.mesh.wy;
			det_J = self.mesh.detJ;
			J = self.mesh.J;
		if basis_k == 1:
			psi_xi_i, psi_eta_i = self.basisEval(basis_k, xi, eta);
			psi_xi_j, psi_eta_j = self.basisEval(basis_k, xi, eta);
			elemMats = np.zeros((self.N*self.M, 2*psi_xi_i.shape[0], 2*psi_eta_i.shape[0]));
			for i in range(self.N*self.M):
				elemMat = assemble_element_M1(psi_xi_i, psi_eta_i, psi_xi_j, psi_eta_j, wx, wy, det_J[i, ...], J[i, 0, ...], J[i, 1, ...], J[i, 2, ...], J[i, 3, ...]);
				if inv: elemMats[i, ...] = np.linalg.inv(elemMat);
				else: elemMats[i, ...] = elemMat;
			M = sp.block_diag(elemMats, format = 'csr');
		elif basis_k == 0:
			psi_i = self.basisEval(basis_k, xi, eta);
			psi_j = self.basisEval(basis_k, xi, eta);
			elemMats = np.zeros((self.N*self.M, psi_i.shape[0], psi_j.shape[0]));
			for i in range(self.N*self.M):
				elemMat = assemble_element_M0(psi_i, psi_j, wx, wy, det_J[i, ...], J[i, 0, ...], J[i, 1, ...], J[i, 2, ...], J[i, 3, ...]);
				if inv: elemMats[i, ...] = np.linalg.inv(elemMat);
				else: elemMats[i, ...] = elemMat;
			M = sp.block_diag(elemMats, format = 'csr');
		elif basis_k == 2:
			psi_i = self.basisEval(basis_k, xi, eta);
			psi_j = self.basisEval(basis_k, xi, eta);
			elemMats = np.zeros((self.N*self.M, psi_i.shape[0], psi_j.shape[0]));
			for i in range(self.N*self.M):
				elemMat = assemble_element_M2(psi_i, psi_j, wx, wy, det_J[i, ...], J[i, 0, ...], J[i, 1, ...], J[i, 2, ...], J[i, 3, ...]);
				if inv: elemMats[i, ...] = np.linalg.inv(elemMat);
				else: elemMats[i, ...] = elemMat;
			M = sp.block_diag(elemMats, format = 'csr');
		return M;

	def k_formBasis(self, basis_k, xi, eta):
		if basis_k == 0:
			basis = np.vstack((self.basisEval(basis_k, xi, eta).transpose(2, 1, 0).transpose(1, 0, 2)));
			return sp.kron(sp.eye(self.N*self.M, format = 'csr'), basis, format = 'csr');
		elif basis_k == 1:
			basis_u, basis_v = self.basisEval(1, xi, eta);
			basis_u, basis_v = np.vstack((basis_u.transpose(2, 1, 0).transpose(1, 0, 2))), np.vstack((basis_v.transpose(2, 1, 0).transpose(1, 0, 2)));
			psi_u = np.zeros((self.mesh.N*self.mesh.M, basis_u.shape[0], 2*basis_u.shape[1]));
			psi_v = np.zeros((self.mesh.N*self.mesh.M, basis_v.shape[0], 2*basis_v.shape[1]));
			XI, ETA = np.meshgrid(xi, eta);
			J = np.zeros((4, *XI.shape));
			for j in range(self.mesh.M):
				for i in range(self.mesh.N):
					J[:, ...] = self.mesh.get_J(XI, ETA, i, j);
					detJ = self.mesh.get_detJ(J);
					psi_u[i + self.N*j, ...] = np.hstack(((1/detJ*J[0, ...]).flatten()[:, np.newaxis]*basis_u, (1/detJ*J[1, ...]).flatten()[:, np.newaxis]*basis_v));
					psi_v[i + self.N*j, ...] = np.hstack(((1/detJ*J[2, ...]).flatten()[:, np.newaxis]*basis_u, (1/detJ*J[3, ...]).flatten()[:, np.newaxis]*basis_v));
			psi_u = sp.block_diag(psi_u, format = 'csr');
			psi_v = sp.block_diag(psi_v, format = 'csr');
			return psi_u, psi_v;
		elif basis_k == 2:
			basis = np.vstack((self.basisEval(basis_k, xi, eta).transpose(2, 1, 0).transpose(1, 0, 2)));
			psi = np.zeros((self.mesh.N*self.mesh.M, basis.shape[0], basis.shape[1]));
			XI, ETA = np.meshgrid(xi, eta);
			J = np.zeros((4, *XI.shape));
			for j in range(self.mesh.M):
				for i in range(self.mesh.N):
					J[:, ...] = self.mesh.get_J(XI, ETA, i, j);
					detJ = self.mesh.get_detJ(J);
					psi[i + self.N*j, ...] = 1/detJ.flatten()[:, np.newaxis]*(basis);
			psi = sp.block_diag(psi, format = 'csr');
			return psi;

	def L2innerProduct(self, psi_i, psi_j, w_mat):
		M = (w_mat@psi_i).T@psi_j;
		return M;

	def Elambda1(self):
		nEdges = self.mesh.nEdges_per_elem;
		M_lst2 = np.repeat(np.arange(0, self.M - 1, dtype = int), self.N*self.p);
		N_lst2 = np.tile(np.repeat(np.arange(0, self.N, dtype = int), self.p), self.M - 1);
		p_lst = np.tile(np.arange(self.p, 0, -1, dtype = int), self.N*(self.M - 1));

		M_lst = np.repeat(np.arange(0, self.M, dtype = int), (self.N - 1)*self.p);
		N_lst = np.tile(np.repeat(np.arange(1, self.N, dtype = int), self.p), self.M);
		q_lst = np.tile(np.arange(1, self.p + 1, dtype = int), self.M*(self.N - 1));

		v_idx = np.vstack((self.N*M_lst*nEdges + (N_lst - 1)*nEdges + q_lst*(self.p + 1) - 1, self.N*M_lst*nEdges + N_lst*nEdges + (q_lst - 1)*(self.p + 1))).T.flatten();
		h_idx = np.vstack((nEdges - p_lst + N_lst2*nEdges + M_lst2*self.N*nEdges, nEdges - p_lst + N_lst2*nEdges + M_lst2*self.N*nEdges + (self.N)*nEdges - self.p*self.p)).T.flatten();

		M_arr = np.repeat(np.arange(self.M), self.p);
		N_arr = np.repeat(np.arange(self.N), self.p);

		Left = np.tile(np.arange(0, self.p*(self.p + 1), self.p + 1), self.M) + M_arr*self.N*nEdges;
		Right = np.tile(np.arange((self.N - 1)*nEdges + self.p, (self.N - 1)*nEdges + self.p*(self.p + 1), self.p + 1), self.M) + M_arr*self.N*nEdges;
		Bot = np.tile(np.arange((self.p + 1)*self.p, (self.p + 1)*self.p + self.p), self.N) + N_arr*nEdges;
		Top = np.tile(np.arange(nEdges - self.p + (self.M - 1)*self.N*nEdges, nEdges + (self.M - 1)*self.N*nEdges), self.N) + N_arr*nEdges;

		LR = np.vstack((Left, Right)).T.flatten();
		j = np.concatenate((v_idx, h_idx, LR, Bot, Top));
		i = np.concatenate((np.repeat(np.arange(0, v_idx.shape[0]//2, dtype = int), 2), v_idx.shape[0]//2 + np.repeat(np.arange(0, h_idx.shape[0]//2, dtype = int), 2), v_idx.shape[0]//2 + h_idx.shape[0]//2 + np.repeat(np.arange(0, LR.shape[0]//2, dtype = int), 2), v_idx.shape[0]//2 + h_idx.shape[0]//2 + LR.shape[0]//2 + np.arange(0, Top.shape[0] + Bot.shape[0], dtype = int)));
		v = np.concatenate((np.vstack((-np.ones(v_idx.shape[0]//2), np.ones(v_idx.shape[0]//2))).T.flatten(), np.vstack((-np.ones(h_idx.shape[0]//2), np.ones(h_idx.shape[0]//2))).T.flatten(), np.vstack((-np.ones(LR.shape[0]//2), np.ones(LR.shape[0]//2))).T.flatten(), -np.ones_like(Bot), np.ones_like(Top)));
		Elambda1 = sp.csr_matrix((v, (i, j)));
		constraints = np.zeros(Elambda1.shape[0]);
		return Elambda1, constraints;

	def Egamma0(self):
		nPoints = self.mesh.nNodes_per_elem;
		M_lst2 = np.repeat(np.arange(0, self.M - 1, dtype = int), self.N*(self.p + 1));
		N_lst2 = np.tile(np.repeat(np.arange(0, self.N, dtype = int), self.p + 1), self.M - 1);
		p_lst = np.tile(np.arange(self.p + 1, 0, -1, dtype = int), self.N*(self.M - 1));

		M_lst = np.repeat(np.arange(0, self.M, dtype = int), (self.N - 1)*(self.p + 1));
		N_lst = np.tile(np.repeat(np.arange(1, self.N, dtype = int), self.p + 1), self.M);
		q_lst = np.tile(np.arange(1, self.p + 2, dtype = int), self.M*(self.N - 1));

		v_idx = np.vstack((self.N*M_lst*nPoints + (N_lst - 1)*nPoints + q_lst*(self.p + 1) - 1, self.N*M_lst*nPoints + N_lst*nPoints + (q_lst - 1)*(self.p + 1))).T.flatten();
		h_idx = np.vstack((nPoints - p_lst + N_lst2*nPoints + M_lst2*self.N*nPoints + (self.N)*nPoints - self.p*(self.p + 1), nPoints - p_lst + N_lst2*nPoints + M_lst2*self.N*nPoints)).T.flatten();

		M_arr = np.repeat(np.arange(self.M), self.p + 1);
		N_arr = np.repeat(np.arange(self.N), self.p + 1);

		Left = np.tile(np.arange(0, (self.p + 1)*(self.p + 1), self.p + 1), self.M) + M_arr*self.N*nPoints;
		Right = np.tile(np.arange((self.N - 1)*nPoints + self.p, (self.N - 1)*nPoints + (self.p + 1)*(self.p + 1), self.p + 1), self.M) + M_arr*self.N*nPoints;
		Bot = np.tile(np.arange(0, self.p + 1, 1, dtype = int), self.N) + N_arr*nPoints
		Top = np.tile(np.arange(nPoints - (self.p + 1) + (self.M - 1)*self.N*nPoints, nPoints + (self.M - 1)*self.N*nPoints), self.N) + N_arr*nPoints;
		
		bc_i = np.concatenate((Bot, Top));
		bc_j = np.arange(0, 2*self.mesh.N*(self.mesh.p + 1));
		bc_v = np.concatenate((np.ones_like(Bot), -np.ones_like(Top)));
		weak_bc_mat = sp.csr_matrix((bc_v, (bc_i, bc_j)), shape = (nPoints*self.N*self.M, np.max(bc_j) + 1));

		LR = np.vstack((Right, Left)).T.flatten();
		j = np.concatenate((v_idx, h_idx, LR));
		i = np.concatenate((np.repeat(np.arange(0, v_idx.shape[0]//2, dtype = int), 2), v_idx.shape[0]//2 + np.repeat(np.arange(0, h_idx.shape[0]//2, dtype = int), 2), v_idx.shape[0]//2 + h_idx.shape[0]//2 + np.repeat(np.arange(0, LR.shape[0]//2, dtype = int), 2)));
		v = np.concatenate((np.vstack((-np.ones(v_idx.shape[0]//2), np.ones(v_idx.shape[0]//2))).T.flatten(), np.vstack((-np.ones(h_idx.shape[0]//2), np.ones(h_idx.shape[0]//2))).T.flatten(), np.vstack((-np.ones(LR.shape[0]//2), np.ones(LR.shape[0]//2))).T.flatten()));
		Egamma0 = sp.csr_matrix((v, (i, j)), shape = (np.max(i) + 1, nPoints*self.N*self.M));
		constraints = np.zeros(Egamma0.shape[0]);
		return Egamma0, weak_bc_mat, constraints;

	def Ethetagamma(self):
		N_arr = np.tile(np.arange(1, self.N), self.M - 1);
		M_arr = np.repeat(np.arange(1, self.M), self.N - 1);
		south = (N_arr - 1)*(self.p + 1) + self.p + (M_arr - 1)*(self.p + 1)*(self.N - 1);
		north = (self.p + 1)*(self.N - 1) + (N_arr - 1)*(self.p + 1) + (M_arr - 1)*(self.p + 1)*(self.N - 1);
		west = (self.p + 1)*(self.N - 1)*self.M + (N_arr - 1)*(self.p + 1) + self.p + (M_arr - 1)*(self.p + 1)*(self.N);
		east = (self.p + 1)*(self.N - 1)*self.M + (N_arr - 1)*(self.p + 1) + self.p + (M_arr - 1)*(self.p + 1)*(self.N) + 1;
		j = np.vstack((south, north, west, east)).T.flatten();
		i = np.repeat(np.arange(0, south.shape[0], dtype = int), 4);
		v = np.vstack((-np.ones_like(south, dtype = float), np.ones_like(north, dtype = float), \
						-np.ones_like(west, dtype = float), np.ones_like(east, dtype = float))).T.flatten();
		shp = (np.max(i) + 1, (self.N - 1)*self.M*(self.p + 1) + (self.M - 1)*self.N*(self.p + 1));
   
		Left = np.arange(0, self.M*(self.p + 1), self.p + 1) + (self.N - 1)*self.M*(self.p + 1) + (self.M - 1)*self.N*(self.p + 1);
		Bot = np.arange(0, self.N*(self.p + 1), self.p + 1) + Left[-1] + self.p + 1;

		N_arr = np.arange(1, self.N);
		M_arr = np.arange(1, self.M);
		south_B = Left[:-1] + self.p;
		north_B = Left[1:];
		west_B = (self.p + 1)*(self.N - 1)*self.M + (self.N - 1)*(self.p + 1) + self.p + (M_arr - 1)*(self.p + 1)*(self.N);
		east_B = (self.p + 1)*(self.N - 1)*self.M  + self.p + (M_arr - 1)*(self.p + 1)*(self.N) + 1 - (self.p + 1);
		boundaries_j = np.vstack((south_B, north_B, west_B, east_B)).T.flatten();
		boundaries_i = np.repeat(np.arange(0, south_B.shape[0], dtype = int), 4);
		boundaries_v = np.vstack((-np.ones_like(south_B, dtype = float), np.ones_like(north_B, dtype = float), \
						-np.ones_like(west_B, dtype = float), np.ones_like(east_B, dtype = float))).T.flatten();

		i = np.concatenate((i, boundaries_i + south.shape[0]));
		j = np.concatenate((j, boundaries_j));
		v = np.concatenate((v, boundaries_v));
		shp = (np.max(i) + 1, 2*(self.N)*self.M*(self.p + 1) - self.N*(self.p + 1));
		
		Ethetagamma = sp.csr_matrix((v, (i, j)), shape = shp);
		constraints = np.zeros(Ethetagamma.shape[0]);
		return Ethetagamma, constraints;

	def showcsrMatrix(self, A):
		fig = plt.figure();
		ax = fig.add_subplot(111);
		im = ax.imshow(A.toarray());
		fig.colorbar(im);
		plt.tight_layout();
		plt.show();
		return 0;

