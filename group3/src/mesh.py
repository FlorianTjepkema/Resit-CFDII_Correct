"""
Author: Suyash Shrestha
License: MIT
Copyright (c) 2024, Suyash Shrestha
All rights reserved.
"""
import numpy as np
import src.mimeticMachinery as miMc
import scipy.sparse as sp
import matplotlib.pyplot as plt

class mesh2D():
	def __init__(self, X0, X1, N, M, p, Phi, dPhi, mapping_parms, p_hiDOP) -> None:
		self.X0 = X0;
		self.X1 = X1;
		self.p = p;
		self.p_hiDOP = p_hiDOP;
		self.N = N;
		self.M = M;
		self.Phi = Phi;
		self.dPhi = dPhi;
		self.mapping_parms = mapping_parms;
		self.nNodes_per_elem = (p + 1)**2;
		self.nEdges_per_elem = 2*(p + 1)*p;
		self.nSurfs_per_elem = p**2;
		self.nNodes = N*M*self.nNodes_per_elem;
		self.nEdges = N*M*self.nEdges_per_elem;
		self.nSurfs = N*M*self.nSurfs_per_elem;
		self.nPlot_x = None;
		self.nPlot_y = None;
		self.edges = None;
		self.leftNodes = None;
		self.connectMat = None;
		self.hx = None;
		self.thx = None;
		self.hy = None;
		self.thy = None;
		self.tx = None;
		self.ty = None;
		self.x = None;
		self.y = None;
		self.x_hat = None;
		self.y_hat = None;
		self.XI = None;
		self.ETA = None;
		self.xi = None;
		self.eta = None;
		self.wx = None;
		self.wy = None;

	def get_mapping(self, xi, eta, i, j): return self.Phi(xi, eta, self.X0, self.X1, self.N, self.M, *self.mapping_parms, i, j);
	def get_J(self, xi, eta, i, j): return self.dPhi(xi, eta, self.X0, self.X1, self.N, self.M, *self.mapping_parms, i, j);
	def get_detJ(self, J): return J[0, ...]*J[3, ...] - J[1, ...]*J[2, ...];

	def buildMesh(self):
		gll = miMc.gaussLobatto();
		self.xi = gll.gllNodes(self.p + 1);
		self.eta = gll.gllNodes(self.p + 1);
		self.wx = gll.gllWeights(self.xi);
		self.wy = gll.gllWeights(self.eta);

		self.txi = np.concatenate(([-1.0], np.cumsum(self.wx) - 1));
		self.teta = np.concatenate(([-1.0], np.cumsum(self.wy) - 1));

		self.xi_hiDOP = gll.gllNodes((self.p_hiDOP + 1));
		self.eta_hiDOP = gll.gllNodes((self.p_hiDOP + 1));
		self.wx_hiDOP = gll.gllWeights(self.xi_hiDOP);
		self.wy_hiDOP = gll.gllWeights(self.eta_hiDOP);

		self.XI, self.ETA = np.meshgrid(self.xi, self.eta);

		self.tXI, self.tETA = np.meshgrid(self.txi, self.teta);

		self.XI_hiDOP, self.ETA_hiDOP = np.meshgrid(self.xi_hiDOP, self.eta_hiDOP);

		self.x = np.zeros((self.N*self.M, self.p + 1, self.p + 1), dtype = float);
		self.y = np.zeros_like(self.x);
		self.x_hat = np.zeros((self.N*self.M, self.p + 1, self.p + 1), dtype = float);
		self.y_hat = np.zeros_like(self.x_hat);
		self.x_hiDOP = np.zeros((self.N*self.M, (self.p_hiDOP + 1), (self.p_hiDOP + 1)), dtype = float);
		self.y_hiDOP = np.zeros_like(self.x_hiDOP);
		self.tx = np.zeros((self.N*self.M, self.p + 2, self.p + 2), dtype = float);
		self.ty = np.zeros((self.N*self.M, self.p + 2, self.p + 2), dtype = float);
		self.tx_hat = np.zeros((self.N*self.M, self.p + 2, self.p + 2), dtype = float);
		self.ty_hat = np.zeros((self.N*self.M, self.p + 2, self.p + 2), dtype = float);
		self.J = np.zeros((self.M*self.N, 4, self.p + 1, self.p + 1), dtype = float);
		self.detJ = np.zeros((self.M*self.N, self.p + 1, self.p + 1), dtype = float);
		self.J_hiDOP = np.zeros((self.M*self.N, 4, (self.p_hiDOP + 1), (self.p_hiDOP + 1)), dtype = float);
		self.detJ_hiDOP = np.zeros((self.M*self.N, (self.p_hiDOP + 1), (self.p_hiDOP + 1)), dtype = float);

		for j in range(self.M):
			for i in range(self.N):
				self.x_hat[i + self.N*j, ...], self.y_hat[i + self.N*j, ...], self.x[i + self.N*j, ...], self.y[i + self.N*j, ...] = self.get_mapping(self.XI, self.ETA, i, j);
				self.tx_hat[i + self.N*j, ...], self.ty_hat[i + self.N*j, ...], self.tx[i + self.N*j, ...], self.ty[i + self.N*j, ...] = self.get_mapping(self.tXI, self.tETA, i, j);
				_, _, self.x_hiDOP[i + self.N*j, ...], self.y_hiDOP[i + self.N*j, ...] = self.get_mapping(self.XI_hiDOP, self.ETA_hiDOP, i, j);
				self.J[i + self.N*j, ...] = self.get_J(self.XI, self.ETA, i, j);
				self.detJ[i + self.N*j, ...] = self.get_detJ(self.J[i + self.N*j, ...]);
				self.J_hiDOP[i + self.N*j, ...] = self.get_J(self.XI_hiDOP, self.ETA_hiDOP, i, j);
				self.detJ_hiDOP[i + self.N*j, ...] = self.get_detJ(self.J_hiDOP[i + self.N*j, ...]);
		
		self.xPlot = self.gridit(self.x_hiDOP);
		self.yPlot = self.gridit(self.y_hiDOP);

		W = self.wx[:, np.newaxis]@self.wy[np.newaxis, :];
		self.W = np.einsum('ikl, kl -> ikl', self.detJ, W).flatten();
		self.w_mat = sp.diags(self.W, format = 'csr');

		# # For high DOP integration
		W_hiDOP = self.wx_hiDOP[:, np.newaxis]@self.wy_hiDOP[np.newaxis, :];
		self.W_hiDOP = np.einsum('ikl, kl -> ikl', self.detJ_hiDOP, W_hiDOP).flatten();
		self.w_mat_hiDOP = sp.diags(self.W_hiDOP, format = 'csr');
		
		self.h_x = self.x[:, :-1, 1:] - self.x[:, :-1, :-1];
		self.h_y = self.y[:, 1:, :-1] - self.y[:, :-1, :-1];

		self.left_boundary_edg_h = self.h_y[::self.N, :, 0].flatten();
		self.right_boundary_edg_h = self.h_y[self.N - 1::self.N, :, -1].flatten();
		self.bottom_boundary_edg_h = self.h_x[:self.N, 0, :].flatten();
		self.top_boundary_edg_h = self.h_x[self.N*self.M - self.N:, -1, :].flatten();

		self.th_x = self.tx[:, 1:-1, 1:] - self.tx[:, 1:-1, :-1];
		self.th_y = self.ty[:, 1:, 1:-1] - self.ty[:, :-1, 1:-1];

		self.left_boundary_tedg_h = self.th_y[::self.N, :, 0].flatten();
		self.right_boundary_tedg_h = self.th_y[self.N - 1::self.N, :, -1].flatten();
		self.bottom_boundary_tedg_h = self.th_x[:self.N, 0, :].flatten();
		self.top_boundary_tedg_h = self.th_x[self.N*self.M - self.N:, -1, :].flatten();

		self.thx = np.concatenate((self.tx[0, 0, :], [self.X0[1]])) - np.concatenate(([self.X0[0]], self.tx[0, 0, :]));
		self.nPlot_x = (self.p_hiDOP + 1);
		self.nPlot_y = (self.p_hiDOP + 1);
		return 0;
	
	def gridit(self, X):
		return np.hstack(np.hstack((X[np.arange(self.N*self.M).reshape(self.M, self.N), ...])));
	
	def plotMesh(self):
		fig = plt.figure();
		ax = fig.add_subplot(111);
		ax.set_aspect('equal');
		for i in range(self.x.shape[0]):
			ax.plot(self.x[i, ...], self.y[i, ...], 'k-', linewidth = 0.5);
			ax.plot(self.x[i, ...].T, self.y[i, ...].T, 'k-', linewidth = 0.5);
			ax.plot(self.x[i, 0, :], self.y[i, 0, :], 'k-', linewidth = 2);
			ax.plot(self.x[i, -1, :], self.y[i, -1, :], 'k-', linewidth = 2);
			ax.plot(self.x[i, :, 0], self.y[i, :, 0], 'k-', linewidth = 2);
			ax.plot(self.x[i, :, -1], self.y[i, :, -1], 'k-', linewidth = 2);
		ax.set_xlabel(r'$x$');
		ax.set_ylabel(r'$y$');
		ax.set_aspect('equal');
		# plt.savefig('./plots/advectionDiffusion2D_anisoV2/Mesh_skew.pdf', dpi = 300, bbox_inches = 'tight');
		# plt.show();
		return ax;