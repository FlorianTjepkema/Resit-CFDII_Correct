import numpy as np

def assemble_element_M0(psi0_i, psi0_j, w_xi, w_eta, det_F, dx_dxi, dx_deta, dy_dxi, dy_deta):
	""" 
		This function is called internally to generate the mass matrix
		M0 for a single element with polynomial degree 'p'.
		The function must return a numpy array of size 
		((p + 1)*(p + 1)) x ((p + 1)*(p + 1)).
		The input arguments 'psi0_i', 'psi0_j' are the ((p + 1)*(p + 1)) 
		basis functions evaluated at the Gauss-Lobatto quadrature nodes.
		'w_xi', 'w_eta' are the Gauss-Lobatto weights, and 'det_F, dx_dxi, 
		dx_deta, dy_dxi, dy_deta' are the Jacobian of the mapping along 
		with its components. 

		psi0_i, psi0_j are 3D numpy arrays of shape (((p + 1)*(p + 1)), (p + 1), (p + 1))
		det_F, dx_dxi, dx_deta, dy_dxi, dy_deta are 2D numpy arrays of shape ((p + 1), (p + 1))
		w_xi, w_eta are 1D numpy arrays of shape (p + 1)
	"""
	num_basis = psi0_i.shape[0]  # Number of basis functions
	M0 = np.zeros((num_basis, num_basis))  # Initialize M0

	# Loop over quadrature points in xi and eta
	for q_eta in range(len(w_eta)):
		for q_xi in range(len(w_xi)):
			# Extract weights and Jacobian determinant
			weight = w_eta[q_eta] * w_xi[q_xi]
			jacobian_det = det_F[q_eta, q_xi]

			# Accumulate contributions to the mass matrix
			for i in range(num_basis):
				for j in range(num_basis):
					M0[i, j] += (
							psi0_i[i, q_eta, q_xi] *
							psi0_j[j, q_eta, q_xi] *
							weight *
							jacobian_det
					)
	return M0

def assemble_element_M1(psi1_xi_i, psi1_eta_i, psi1_xi_j, psi1_eta_j, w_xi, w_eta, det_F, dx_dxi, dx_deta, dy_dxi, dy_deta):
	"""
		This function is called internally to generate the mass matrix
		M1 for a single element with polynomial degree 'p'.
		The function must return a numpy array of size
		(2*p*(p + 1)) x (2*p*(p + 1)).
		The input arguments 'psi1_xi_i, psi1_eta_i, psi1_xi_j, psi1_eta_j' are the
		basis functions evaluated at the Gauss-Lobatto quadrature nodes. 'w_xi', 'w_eta'
		are the Gauss-Lobatto weights, and 'det_F, dx_dxi, dx_deta, dy_dxi, dy_deta' are the Jacobian
		of the mapping along with its components.

		psi_xi_i, psi_eta_i, psi_xi_j, psi_eta_j are 3D numpy arrays of shape (p*(p + 1), (p + 1), (p + 1))
		det_F, dx_dxi, dx_deta, dy_dxi, dy_deta are 2D numpy arrays of shape ((p + 1), (p + 1))
		w_xi, w_eta are 1D numpy arrays of shape (p + 1)
	"""
	num_basis = psi1_xi_i.shape[0]  # Number of basis functions
	M1_xi_xi = np.zeros((num_basis, num_basis))  # xi-xi block
	M1_xi_eta = np.zeros((num_basis, num_basis))  # xi-eta block
	M1_eta_xi = np.zeros((num_basis, num_basis))  # eta-xi block
	M1_eta_eta = np.zeros((num_basis, num_basis))  # eta-eta block

	# Loop over quadrature points
	for q_eta in range(len(w_eta)):
		for q_xi in range(len(w_xi)):
			weight = w_eta[q_eta] * w_xi[q_xi]
			jacobian_det = det_F[q_eta, q_xi]

			# Compute the metric tensor F^T F
			FTF = np.array([
				[dx_dxi[q_eta, q_xi] ** 2 + dy_dxi[q_eta, q_xi] ** 2,
				 dx_dxi[q_eta, q_xi] * dx_deta[q_eta, q_xi] + dy_dxi[q_eta, q_xi] * dy_deta[q_eta, q_xi]],
				[dx_dxi[q_eta, q_xi] * dx_deta[q_eta, q_xi] + dy_dxi[q_eta, q_xi] * dy_deta[q_eta, q_xi],
				 dx_deta[q_eta, q_xi] ** 2 + dy_deta[q_eta, q_xi] ** 2]
			])

			# Accumulate contributions for each block of M1
			for i in range(num_basis):
				for j in range(num_basis):
					M1_xi_xi[i, j] += psi1_xi_i[i, q_eta, q_xi] * FTF[0, 0] * psi1_xi_j[
						j, q_eta, q_xi] * weight / jacobian_det
					M1_xi_eta[i, j] += psi1_xi_i[i, q_eta, q_xi] * FTF[0, 1] * psi1_eta_j[
						j, q_eta, q_xi] * weight / jacobian_det
					M1_eta_xi[i, j] += psi1_eta_i[i, q_eta, q_xi] * FTF[1, 0] * psi1_xi_j[
						j, q_eta, q_xi] * weight / jacobian_det
					M1_eta_eta[i, j] += psi1_eta_i[i, q_eta, q_xi] * FTF[1, 1] * psi1_eta_j[
						j, q_eta, q_xi] * weight / jacobian_det

	# Combine blocks into the full M1 matrix
	M1 = np.block([[M1_xi_xi, M1_xi_eta], [M1_eta_xi, M1_eta_eta]])
	return M1

def assemble_element_M2(psi2_i, psi2_j, w_xi, w_eta, det_F, dx_dxi, dx_deta, dy_dxi, dy_deta):
	"""
		This function is called internally to generate the mass matrix
		M2 for a single element with polynomial degree 'p'.
		The function must return a numpy array of size
		(p*p) x (p*p).
		The input arguments 'psi2_i', 'psi2_j' are the basis functions evaluated
		at the Gauss-Lobatto quadrature nodes. 'w_xi', 'w_eta' are the Gauss-Lobatto
		weights, and 'det_F, dx_dxi, dx_deta, dy_dxi, dy_deta' are the Jacobian
		of the mapping along with its components.

		psi_i, psi_j are 3D numpy arrays of shape (p*p, (p + 1), (p + 1))
		det_F, dx_dxi, dx_deta, dy_dxi, dy_deta are 2D numpy arrays of shape ((p + 1), (p + 1))
		w_xi, w_eta are 1D numpy arrays of shape (p + 1)
	"""
	num_basis = psi2_i.shape[0]  # Number of basis functions
	M2 = np.zeros((num_basis, num_basis))  # Initialize M2

	# Loop over quadrature points
	for q_eta in range(len(w_eta)):
		for q_xi in range(len(w_xi)):
			weight = w_eta[q_eta] * w_xi[q_xi]
			jacobian_det = det_F[q_eta, q_xi]

			# Accumulate contributions to M2
			for i in range(num_basis):
				for j in range(num_basis):
					M2[i, j] += (
							psi2_i[i, q_eta, q_xi] *
							psi2_j[j, q_eta, q_xi] *
							weight / jacobian_det
					)
	return M2