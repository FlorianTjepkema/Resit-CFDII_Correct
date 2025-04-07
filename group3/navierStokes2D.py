"""
Author: Suyash Shrestha
License: MIT
Copyright (c) 2024, Suyash Shrestha
All rights reserved.
"""
import src.mesh as mesh
import src.mimeticSEM as mimeticSEM
from src.mimeticMachinery import sparseLinAlgTools as splaTools
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
import numpy as np
import matplotlib.colors as colors
from src.geometryMaps import cylinder
import sys
import matplotlib.animation as pltani
import pickle
import warnings
warnings.filterwarnings('ignore')
## ============================================== ##
## ============ Main Solver function ============ ##
## ============================================== ##
def navierStokesSolver(msh: mesh.mesh2D, sem: mimeticSEM.SEM2D, dt, Ndt, Re, fx, fy, saveData = False):
	# Call global Incidence and Mass matrices
	print('-----------Assembling global matrices');
	t_arr = np.arange(0, dt*(Ndt + 1), dt);
	E10 = sem.E10();
	E21 = sem.E21();
	M0 = sem.massMatrix(0);
	M1 = sem.massMatrix(1);
	M1inv = sem.massMatrix(1, inv = True);
	M2inv = sem.massMatrix(2, inv = True);
	# Call trace matrices
	Elambda1, _ = sem.Elambda1();
	Egamma0, weak_bc_mat, _ = sem.Egamma0();
	Ethetagamma, _ = sem.Ethetagamma();
	psi0 = sem.k_formBasis(0, msh.xi, msh.eta);
	psi1x, psi1y = sem.k_formBasis(1, msh.xi, msh.eta);
	shp0, shp1 = psi1x.shape;
	zeros_12 = np.zeros((msh.nNodes, msh.nSurfs));
	# Assemble global system
	print('-----------Initialising global system and rhs');
	NSsys = sp.bmat([[M0, -E10.T@M1, None, None, Egamma0.T, None], \
				[1/(2*Re)*M1@E10, 1/dt*M1, E21.T, Elambda1.T, None, None], \
					[None, E21, None, None, None, None], \
					[None, Elambda1, None, None, None, None], \
					[Egamma0, None, None, None, None, Ethetagamma.T], \
					[None, None, None, None, Ethetagamma, None]], format = 'csc');
	# Build RHS
	x_hat_top = msh.x_hat[-msh.N:, -1, :].flatten();
	x_hat_bot = msh.x_hat[:msh.N, 0, :].flatten();
	top_wall_velocity = 3.0*np.pi*(0.5*(1 + np.tanh(100*(x_hat_top - 0.5))));
	bottom_wall_velocity = -1.5*np.pi*(0.5*(1 - np.tanh(100*(x_hat_bot - 0.5))));
	weak_wall_tangentail_velocity = weak_bc_mat@np.concatenate((bottom_wall_velocity*(msh.wx*np.sqrt((msh.J[:msh.N, 0, 0, :]**2 + msh.J[:msh.N, 2, 0, :]**2))).flatten(), top_wall_velocity*(msh.wx*np.sqrt((msh.J[-msh.N:, 0, -1, :]**2 + msh.J[-msh.N:, 2, -1, :]**2))).flatten()));
	# Invert sytem using Shur Complement approach
	dinv00, dinv01, dinv10, dinv11 = splaTools.shurComplementInv(dt*M1inv, E21.T, E21, np.zeros((sem.mesh.nSurfs, sem.mesh.nSurfs)));
	dinv = sp.bmat([[dinv00, dinv01], [dinv10, dinv11]]);
	Ainv11, Ainv10, Ainv01, Ainv00 = splaTools.shurComplementInv(dinv, sp.bmat([[1/(2*Re)*M1@E10], [zeros_12.T]], format = 'csc'), sp.bmat([[-E10.T@M1, zeros_12]], format = 'csc'), M0);
	Ainv = sp.bmat([[Ainv00, Ainv01], [Ainv10, Ainv11]], format = 'csc');
	B = NSsys[:sem.mesh.nNodes + sem.mesh.nEdges + sem.mesh.nSurfs, sem.mesh.nNodes + sem.mesh.nEdges + sem.mesh.nSurfs:];
	C = NSsys[sem.mesh.nNodes + sem.mesh.nEdges + sem.mesh.nSurfs:, :sem.mesh.nNodes + sem.mesh.nEdges + sem.mesh.nSurfs];
	D = NSsys[sem.mesh.nNodes + sem.mesh.nEdges + sem.mesh.nSurfs:, sem.mesh.nNodes + sem.mesh.nEdges + sem.mesh.nSurfs:];

	inv_D_C_Ainv_B = la.splu(D - C@Ainv@B);

	RHSx1 = np.vstack((np.zeros((sem.mesh.nNodes, shp0)), psi1x.T.toarray(), np.zeros((sem.mesh.nSurfs, shp0))));
	RHSy1 = np.vstack((np.zeros((sem.mesh.nNodes, shp0)), psi1y.T.toarray(), np.zeros((sem.mesh.nSurfs, shp0))));
	RHS2 = np.zeros((Elambda1.shape[0] + Egamma0.shape[0] + Ethetagamma.shape[0], shp0));

	Ux_lm = inv_D_C_Ainv_B.solve(RHS2 - C@Ainv@RHSx1);
	Uy_lm = inv_D_C_Ainv_B.solve(RHS2 - C@Ainv@RHSy1);

	Ux = Ainv@RHSx1 - Ainv@B@Ux_lm;
	Uy = Ainv@RHSy1 - Ainv@B@Uy_lm;

	RHS_bc = np.concatenate((weak_wall_tangentail_velocity, np.zeros(msh.nEdges + msh.nSurfs)));
	RHS_bc2 = np.zeros(Elambda1.shape[0] + Egamma0.shape[0] + Ethetagamma.shape[0]);
	weak_bc_lm = inv_D_C_Ainv_B.solve(RHS_bc2 - C@Ainv@RHS_bc);
	Ubc = np.concatenate((Ainv@RHS_bc - Ainv@B@weak_bc_lm, weak_bc_lm));

	t_psi_H_curlx = Ux[:sem.mesh.nNodes, :]@msh.w_mat.T;
	t_psi_H_divx = Ux[sem.mesh.nNodes:sem.mesh.nNodes + shp1, :]@msh.w_mat.T;
	t_psi_H1x = Ux[sem.mesh.nNodes + shp1:sem.mesh.nNodes + shp1 + sem.mesh.nSurfs, :]@msh.w_mat.T;
	# t_lambdax = Ux[:Elambda1.shape[0], :]@msh.w_mat.T;
	# t_gammax = Ux[Elambda1.shape[0]:Elambda1.shape[0] + Egamma0.shape[0], :].T;
	# t_thetax = Ux[Elambda1.shape[0] + Egamma0.shape[0]:, :].T;

	t_psi_H_curly = Uy[:sem.mesh.nNodes, :]@msh.w_mat.T;
	t_psi_H_divy = Uy[sem.mesh.nNodes:sem.mesh.nNodes + shp1, :]@msh.w_mat.T;
	t_psi_H1y = Uy[sem.mesh.nNodes + shp1:sem.mesh.nNodes + shp1 + sem.mesh.nSurfs, :]@msh.w_mat.T;
	# t_lambday = Uy[:Elambda1.shape[0], :]@msh.w_mat.T;
	# t_gammay = Uy[Elambda1.shape[0]:Elambda1.shape[0] + Egamma0.shape[0], :].T;
	# t_thetay = Uy[Elambda1.shape[0] + Egamma0.shape[0]:, :].T;

	weak_bc_H_curl = Ubc[:sem.mesh.nNodes];
	weak_bc_H_div = Ubc[sem.mesh.nNodes:sem.mesh.nNodes + shp1];
	weak_bc_H1 = Ubc[sem.mesh.nNodes + shp1:sem.mesh.nNodes + shp1 + sem.mesh.nSurfs];
	# weak_bc_lambda = Ubc[sem.mesh.nNodes + shp1 + sem.mesh.nSurfs:sem.mesh.nNodes + shp1 + sem.mesh.nSurfs + Elambda1.shape[0]];

	# Initialise solution vectors
	omega = np.zeros((t_psi_H_curlx.shape[0], Ndt));
	u = np.zeros((t_psi_H_divx.shape[0], Ndt));
	pressure = np.zeros((t_psi_H1x.shape[0], Ndt));
	# lambdas = np.zeros((t_lambdax.shape[0], Ndt));

	# Solve system
	print('-----------Solving system');
	n_correctors = 2;
	for t_idx in range(Ndt - 1):
		for i in range(n_correctors):
			rhsx = fx(msh.x.flatten(), msh.y.flatten(), t_idx*dt + dt/2) + psi1x@(1/dt*u[:, t_idx] - 1/(2*Re)*E10@omega[:, t_idx]) - (0.5**2)**(i > 0)*((psi0@(omega[:, t_idx] + omega[:, t_idx + 1]))*(-psi1y@(u[:, t_idx] + u[:, t_idx + 1])));
			rhsy = fy(msh.x.flatten(), msh.y.flatten(), t_idx*dt + dt/2) + psi1y@(1/dt*u[:, t_idx] - 1/(2*Re)*E10@omega[:, t_idx]) - (0.5**2)**(i > 0)*((psi0@(omega[:, t_idx] + omega[:, t_idx + 1]))*(psi1x@(u[:, t_idx] + u[:, t_idx + 1])));
			omega[:, t_idx + 1] = weak_bc_H_curl + t_psi_H_curlx@rhsx + t_psi_H_curly@rhsy;
			u[:, t_idx + 1] = weak_bc_H_div + t_psi_H_divx@rhsx + t_psi_H_divy@rhsy;
			pressure[:, t_idx + 1] = weak_bc_H1 + t_psi_H1x@rhsx + t_psi_H1y@rhsy;
			# lambdas[:, t_idx + 1] = weak_bc_lambda + t_lambdax@rhsx + t_lambday@rhsy;
		sys.stdout.write('\rt = %0.4f s,	max(DIV_u^{n + 1}) = %0.3e,	max((u^{n + 1} - u^{n})/dt) = %0.3e'%(t_arr[t_idx + 1], np.abs(E21@(u[:, t_idx + 1])).max(), np.abs(u[:, t_idx + 1] - u[:, t_idx]).max()/dt));
		# sys.stdout.flush();
	sys.stdout.write('\n');

	# Retrieve degrees of freedom
	DIVu = E21@u;
	streamFunc = la.spsolve(sp.bmat([[E10.T@E10, Egamma0.T, None], [Egamma0, None, Ethetagamma.T], [None, Ethetagamma, None]], format = 'csr'), np.vstack((E10.T@u, np.zeros((Egamma0.shape[0], Ndt)), np.zeros((Ethetagamma.shape[0], Ndt)))))[:E10.shape[1], :];
	pressure = M2inv@pressure;
	if (saveData):
		print('-----------Saving data');
		with open('./savedData/solution_p_%i_NxM_%ix%i_Re_%0.1f_dt_%0.2e.pkl'%(p, N, M, Re, dt), 'wb') as file:
			pickle.dump({'t_arr': t_arr, 'omega': omega, 'u': u, 'pressure': pressure, 'streamFunc': streamFunc, 'DIVu': DIVu}, file);
			file.close();
	print('-----------Solve completed');
	return t_arr, omega, u, pressure, streamFunc, DIVu;
## ============================================== ##
## ============================================== ##

if __name__ == '__main__':
	## ============================================== ##
	## =================== Inputs =================== ##
	## ============================================== ##
	# Polynomial degree
	p = 4;
	# Number of elements in azimuthal-direction
	N = 30;
	# Number of elements in radial-direction
	M = 4;
	# Number of nodes to plot in
	pRefined = p + 2;
	# Time step size
	dt = 1e-3;
	# Number of time steps
	Ndt = 10001;
	# Reynolds number
	Re = 50;
	# Geometry definition
	geomMap = cylinder();
	# Source term of PDE
	fx = lambda x, y, t: np.zeros_like(x);
	fy = lambda x, y, t: np.zeros_like(x);
	# Boolien flag to save data
	saveData = False;
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
	## ========= Solve Navier-Stokes system ========= ##
	## ============================================== ##
	t_arr, omega, u, pressure, streamFunc, DIVu = navierStokesSolver(msh, sem, dt, Ndt, Re, fx, fy, saveData = saveData);

	# with open('./savedData/solution_p_%i_NxM_%ix%i_Re_%0.1f_dt_%0.2e.pkl'%(p, N, M, Re, dt), 'rb') as file:
	# 		data = pickle.load(file);
	# 		file.close();
	# t_arr, omega, u, pressure, lambdas, DIVu = data['t_arr'], data['omega'], data['u'], data['pressure'], data['lambdas'], data['DIVu'];
	## ============================================== ##
	## ============================================== ##

	## ============================================== ##
	## ============ Post process solution =========== ##
	## ============================================== ##
	# Call basis functions
	psi0_plot = sem.k_formBasis(0, msh.xi_hiDOP, msh.eta_hiDOP);
	psi1x_plot, psi1y_plot = sem.k_formBasis(1, msh.xi_hiDOP, msh.eta_hiDOP);
	psi2_plot = sem.k_formBasis(2, msh.xi_hiDOP, msh.eta_hiDOP);

	# Compute integral of vorticity over the domain
	integral_omega = msh.W@omega[:, 1:];

	# Reconstruct solution (Note, this reconstruction can be very memory intensive for fine meshes with many time steps.
	# To save cost, you can slice the array so as to reconstruct only a select few time steps instead of all the time steps)
	omega_Reconstruct = msh.gridit((psi0_plot@omega).reshape(-1, msh.nPlot_x, msh.nPlot_y, Ndt));
	streamFunc_Reconstruct = msh.gridit((psi0_plot@streamFunc).reshape(-1, msh.nPlot_x, msh.nPlot_y, Ndt));
	DIVu_Reconstruct = msh.gridit((psi2_plot@DIVu).reshape(-1, msh.nPlot_x, msh.nPlot_y, Ndt));
	# p_Reconstruct = msh.gridit((psi2_plot@pressure).reshape(-1, msh.nPlot_x, msh.nPlot_y, Ndt));
	u_Reconstruct = msh.gridit((psi1x_plot@u).reshape(-1, msh.nPlot_x, msh.nPlot_y, Ndt));
	v_Reconstruct = msh.gridit((psi1y_plot@u).reshape(-1, msh.nPlot_x, msh.nPlot_y, Ndt));
	u_magnitude = np.sqrt(u_Reconstruct**2 + v_Reconstruct**2)

	msh.plotMesh();

	skip_frames = 20;
	u_magnitude_plot = u_magnitude[..., ::skip_frames];
	t_arr_plot = t_arr[::skip_frames];
	levels = np.linspace(0, 9, 25);
	fig = plt.figure(figsize = (10, 8));
	ax = fig.add_subplot(111);
	ax.set_xlabel(r'$x$');
	ax.set_ylabel(r'$y$');
	ax.set_aspect('equal');
	cax = ax.contourf(msh.xPlot, msh.yPlot, u_magnitude_plot[..., 1], levels = levels, cmap = plt.cm.inferno, extend = 'both');
	text = ax.text(0.05, 1.05, '', transform = ax.transAxes, usetex = False);
	fig.colorbar(cax, orientation = 'vertical');
	
	def init():
		return cax, text;

	def step(i):
		for c in ax.collections: c.remove();
		cax = ax.contourf(msh.xPlot, msh.yPlot, u_magnitude_plot[..., i + 1], levels = levels, cmap = plt.cm.inferno, extend = 'both');		
		text.set_text('time = %.3fs'%(t_arr_plot[i]));
		return cax, text;

	ani = pltani.FuncAnimation(fig, step, u_magnitude_plot.shape[-1] - 1, init_func = init, interval = 30);
	# ani.save('%s.mp4'%('./savedData/geom4_solution_anim_p_%i_NxM_%ix%i_Re_%0.1f_dt_%0.2e.pkl'%(p, N, M, Re, dt)), writer = 'ffmpeg');
	plt.show();

	static_timeStamps = [0, Ndt - 1]; #(np.linspace(0, 1, 5)*(Ndt - 1)).astype(int);
	for i in static_timeStamps[1:]:
		# Plot of vorticity
		fig = plt.figure('t = %0.1f'%t_arr[i]);
		ax = fig.add_subplot(111);
		ax.set_xlabel(r'$x$');
		ax.set_ylabel(r'$y$');
		ax.set_aspect('equal');
		cax = ax.contourf(msh.xPlot, msh.yPlot, u_magnitude[..., i], levels = levels, cmap = plt.cm.inferno, extend = 'both');
		fig.colorbar(cax, orientation = 'vertical');

		# Now implement your own plotting routine for the streamlines, the divergence plots, etc
	plt.show();
	
	## ============================================== ##
	## ============================================== ##

	# static_timeStamps = (np.linspace(0, 1, 5) * (Ndt - 1)).astype(int);
	# for i in static_timeStamps[1:]:
	# 	# Plot of vorticity
	# 	fig = plt.figure('t = %0.1f' % t_arr[i]);
	# 	ax = fig.add_subplot(111);
	# 	ax.set_xlabel(r'$x$');
	# 	ax.set_ylabel(r'$y$');
	# 	ax.set_aspect('equal');
	# 	ax.set_title('Contour plot for the vorticity')
	# 	cax = ax.contourf(msh.xPlot, msh.yPlot, omega_Reconstruct[..., i], levels=levels, cmap=plt.cm.twilight_shifted,
	# 					  extend='both');
	# 	fig.colorbar(cax, orientation='vertical', label='Vorticity');
	#
	# 	# Now implement your own plotting routine for the streamlines, the divergence plots, etc
	# 	# Plot for streamlines
	# 	fig, ax = plt.subplots()
	# 	fig.suptitle('t = %0.1f' % t_arr[i])  # Add a title with time
	# 	ax.set_aspect('equal')
	# 	cax = ax.contour(msh.xPlot, msh.yPlot, streamFunc_Reconstruct[..., i], levels=50, cmap='viridis')
	# 	ax.set_xlabel(r'$x$')
	# 	ax.set_ylabel(r'$y$')
	# 	ax.set_title('Contour plot for the streamlines')
	# 	fig.colorbar(cax, orientation='vertical', label='Stream Function')
	#
	# 	# Plot for pointwise divergence
	# 	fig, ax = plt.subplots()
	# 	fig.suptitle('t = %0.1f' % t_arr[i])  # Add a title with time
	# 	cmap = plt.cm.RdBu  # Diverging color map for divergence
	# 	cax = ax.pcolor(msh.xPlot, msh.yPlot, DIVu_Reconstruct[..., i], cmap=cmap, shading='auto')
	# 	ax.set_aspect('equal')
	# 	ax.set_xlabel(r'$x$')
	# 	ax.set_ylabel(r'$y$')
	# 	ax.set_title('Contour plot for the pointwise divergence')
	# 	fig.colorbar(cax, orientation='vertical', label='Divergence')
	#
	# plt.show();
	# Only plot the last timestep
	last_timestep = static_timeStamps[-1]  # Get the last time step

	# Plot of vorticity
	fig = plt.figure('t = %0.1f' % t_arr[last_timestep]);
	ax = fig.add_subplot(111);
	ax.set_xlabel(r'$x$');
	ax.set_ylabel(r'$y$');
	ax.set_aspect('equal');
	ax.set_title('Contour plot for the vorticity')
	cax = ax.contourf(msh.xPlot, msh.yPlot, omega_Reconstruct[..., last_timestep], cmap=plt.cm.twilight_shifted, extend='both');
	fig.colorbar(cax, orientation='vertical', label='Vorticity');

	# Plot for streamlines
	fig, ax = plt.subplots()
	fig.suptitle('t = %0.1f' % t_arr[last_timestep])  # Add a title with time
	ax.set_aspect('equal')
	cax = ax.contour(msh.xPlot, msh.yPlot, streamFunc_Reconstruct[..., last_timestep], levels=50, cmap='viridis')
	ax.set_xlabel(r'$x$')
	ax.set_ylabel(r'$y$')
	ax.set_title('Contour plot for the streamlines')
	fig.colorbar(cax, orientation='vertical', label='Stream Function')

	# Plot for pointwise divergence
	fig, ax = plt.subplots()
	fig.suptitle('t = %0.1f' % t_arr[last_timestep])  # Add a title with time
	cmap = plt.cm.RdBu  # Diverging color map for divergence
	cax = ax.pcolor(msh.xPlot, msh.yPlot, DIVu_Reconstruct[..., last_timestep], cmap=cmap, shading='auto')
	ax.set_aspect('equal')
	ax.set_xlabel(r'$x$')
	ax.set_ylabel(r'$y$')
	ax.set_title('Contour plot for the pointwise divergence')
	fig.colorbar(cax, orientation='vertical', label='Divergence')

	plt.show();

