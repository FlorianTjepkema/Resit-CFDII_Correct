"""
Author: Suyash Shrestha
License: MIT
Copyright (c) 2024, Suyash Shrestha
All rights reserved.
"""
import src.mesh as mesh
import src.mimeticSEM as mimeticSEM
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
import numpy as np
import matplotlib.colors as colors
from src.geometryMaps import cylinder
import warnings
warnings.filterwarnings('ignore')
## ============================================== ##
## ============ Main Solver function ============ ##
## ============================================== ##
def stokesSolver(msh: mesh.mesh2D, sem: mimeticSEM.SEM2D, fx, fy, velocity_inner = -1.5, velocity_outer = 3.0):
	# Call global Incidence and Mass matrices
	print('-----------Assembling global matrices');
	E10 = sem.E10();
	E21 = sem.E21();
	M0 = sem.massMatrix(0);
	M1 = sem.massMatrix(1);
	M2 = sem.massMatrix(2);
	# M1inv = sem.massMatrix(1, inv = True);
	# Call trace matrices
	Elambda1, constraints_normal_velocity = sem.Elambda1();
	Egamma0, weak_bc_mat, constraints_tangentail_velocity = sem.Egamma0();
	Ethetagamma, constraints_corners = sem.Ethetagamma();
	psi1x, psi1y = sem.k_formBasis(1, msh.xi, msh.eta);
	# Assemble global system
	print('-----------Initialising global system and rhs');
	A = sp.bmat([[M0, E10.T@M1, None, None, Egamma0.T, None], \
				[M1@E10, None, E21.T@M2, Elambda1.T, None, None], \
					[None, M2@E21, None, None, None, None], \
					[None, Elambda1, None, None, None, None], \
					[Egamma0, None, None, None, None, Ethetagamma.T], \
					[None, None, None, None, Ethetagamma, None]], format = 'csr');
	# Build RHS
	x_hat_top = msh.x_hat[-msh.N:, -1, :].flatten();
	x_hat_bot = msh.x_hat[:msh.N, 0, :].flatten();
	top_wall_velocity = velocity_outer*np.pi*(0.5*(1 + np.tanh(100*(x_hat_top - 0.5))));
	bottom_wall_velocity = velocity_inner*np.pi*(0.5*(1 - np.tanh(100*(x_hat_bot - 0.5))));
	weak_wall_tangentail_velocity = weak_bc_mat@np.concatenate((bottom_wall_velocity*(msh.wx*np.sqrt((msh.J[:msh.N, 0, 0, :]**2 + msh.J[:msh.N, 2, 0, :]**2))).flatten(), top_wall_velocity*(msh.wx*np.sqrt((msh.J[-msh.N:, 0, -1, :]**2 + msh.J[-msh.N:, 2, -1, :]**2))).flatten()));
	f_hat = sem.L2innerProduct(psi1x, fx(msh.x.flatten(), msh.y.flatten()), msh.w_mat) + sem.L2innerProduct(psi1y, fy(msh.x.flatten(), msh.y.flatten()), msh.w_mat);
	RHS = np.concatenate((-weak_wall_tangentail_velocity, f_hat, np.zeros(msh.nSurfs), constraints_normal_velocity, constraints_tangentail_velocity, constraints_corners));
	# Solve system
	print('-----------Solving system');
	U = la.spsolve(A, RHS);
	# Retrieve degrees of freedom
	omega = -U[:msh.nNodes];
	u = U[msh.nNodes:msh.nNodes + msh.nEdges];
	pressure = U[msh.nNodes + msh.nEdges:msh.nNodes + msh.nEdges + msh.nSurfs];
	# lambdas = U[msh.nNodes + msh.nEdges + msh.nSurfs:msh.nNodes + msh.nEdges + msh.nSurfs + Elambda1.shape[0]];
	streamFunc = la.spsolve(sp.bmat([[E10.T@E10, Egamma0.T, None], [Egamma0, None, Ethetagamma.T], [None, Ethetagamma, None]], format = 'csr'), np.concatenate((E10.T@u, np.zeros(Egamma0.shape[0]), np.zeros(Ethetagamma.shape[0]))))[:E10.shape[1]];
	DIVu = E21@u;
	CURLomega = E10@omega;
	# GRADp = -M1inv@(E21.T@M2@pressure + Elambda1.T@lambdas);
	print('-----------Solve completed');
	return omega, u, pressure, streamFunc, DIVu, CURLomega;
## ============================================== ##
## ============================================== ##

if __name__ == '__main__':
	## ============================================== ##
	## =================== Inputs =================== ##
	## ============================================== ##
	# Polynomial degree
	p = 4;
	# Number of elements in azimuthal-direction
	N = 24;
	# Number of elements in radial-direction
	M = 4;
	# Number of nodes to plot in
	pRefined = p + 2;
	# Geometry definition
	geomMap = cylinder();
	# Source term of PDE
	fx = lambda x, y: np.zeros_like(x);
	fy = lambda x, y: np.zeros_like(x);
	## ============================================== ##
	## ============================================== ##

	## ============================================== ##
	## ============== Assemble objects ============== ##
	## ============================================== ##
	X0, X1 = geomMap.X0, geomMap.X1;
	mapping_parms = geomMap.mapping_parms;
	Phi, dPhi = geomMap.Phi, geomMap.dPhi;
	msh = mesh.mesh2D(X0, X1, N, M, p, Phi, dPhi, mapping_parms, pRefined);
	sem = mimeticSEM.SEM2D(msh);
	msh.buildMesh();
	## ============================================== ##
	## ============================================== ##

	## ============================================== ##
	## ============= Solve Stokes system ============ ##
	## ============================================== ##
	omega, u, pressure, streamFunc, DIVu, CURLomega = stokesSolver(msh, sem, fx, fy);
	## ============================================== ##
	## ============================================== ##

	## ============================================== ##
	## ============ Post process solution =========== ##
	## ============================================== ##
	# Call basis functions
	psi0_plot = sem.k_formBasis(0, msh.xi_hiDOP, msh.eta_hiDOP);
	psi1x_plot, psi1y_plot = sem.k_formBasis(1, msh.xi_hiDOP, msh.eta_hiDOP);
	psi2_plot = sem.k_formBasis(2, msh.xi_hiDOP, msh.eta_hiDOP);

	# Reconstruct solution 
	omega_Reconstruct = msh.gridit((psi0_plot@omega).reshape(-1, msh.nPlot_x, msh.nPlot_y));
	streamFunc_Reconstruct = msh.gridit((psi0_plot@streamFunc).reshape(-1, msh.nPlot_x, msh.nPlot_y));
	DIVu_Reconstruct = msh.gridit((psi2_plot@DIVu).reshape(-1, msh.nPlot_x, msh.nPlot_y));
	p_Reconstruct = msh.gridit((psi2_plot@pressure).reshape(-1, msh.nPlot_x, msh.nPlot_y));
	u_Reconstruct = msh.gridit((psi1x_plot@u).reshape(-1, msh.nPlot_x, msh.nPlot_y));
	v_Reconstruct = msh.gridit((psi1y_plot@u).reshape(-1, msh.nPlot_x, msh.nPlot_y));
	
	msh.plotMesh();

	# Plot of velocity
	velocity_isoValues = np.linspace(0, 9, 25);
	fig = plt.figure();
	ax = fig.add_subplot(111);
	cax = ax.contourf(msh.xPlot, msh.yPlot, np.sqrt(u_Reconstruct**2 + v_Reconstruct**2), levels = velocity_isoValues, extend = 'both', cmap = plt.cm.inferno);
	ax.set_aspect('equal');
	ax.set_xlabel(r'$x$');
	ax.set_ylabel(r'$y$');
	fig.colorbar(cax, orientation = 'vertical');

	# Now implement your own plotting routine for the streamlines, the divergence plots, etc
	plt.show();
	## ============================================== ##
	## ============================================== ##
