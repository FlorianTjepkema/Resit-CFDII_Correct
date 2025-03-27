import numpy as np
import scipy.sparse as sp

def assemble_element_E10(p):
	""" 
		This function is called internally to 
		generate the incidence matrix E10 for a 
		single element with polynomial degree 'p'.
		The function must return a numpy array 
		or scipy sparse array of size 
		(2*p*(p + 1)) x ((p + 1)*(p + 1))
	"""
	## ============================================================== ##
	## ++++++++++++++ Implement the incidence matrix E10 ++++++++++++ ##
	## +++++++++++++++++++++++++++ below ++++++++++++++++++++++++++++ ##
	## ============================================================== ##
	# raise NotImplementedError
	E10_1 = np.zeros((p * (p + 1), (p + 1) * (p + 1)))
	for i in range(0,p*(p + 1)+1):
		if i < (p*(p+1)):
			E10_1[i,i] 	= -1
			E10_1[i,i+p+1] 	= 1

	E10_2 = np.zeros((p * (p + 1), (p + 1) * (p + 1)))
	shift = 0
	for i in range(0, (p * (p + 1))):
		E10_2[i, i + shift] = 1
		E10_2[i, i + 1 + shift] = -1

		# After every p rows, shift the pattern by 1 column
		if (i + 1) % p == 0:
			shift += 1
		E10 = np.vstack((E10_1,E10_2))
	return E10

def assemble_element_E21(p):
	"""
		This function is called internally to
		generate the incidence matrix E21 for a
		single element with polynomial degree 'p'.
		The function must return a numpy array
		or scipy sparse array of size
		(p*p) x (2*p*(p + 1))
	"""
	## ============================================================== ##
	## ++++++++++++++ Implement the incidence matrix E21 ++++++++++++ ##
	## +++++++++++++++++++++++++++ below ++++++++++++++++++++++++++++ ##
	## ============================================================== ##
	# raise NotImplementedError
	E21_1 = np.zeros((p**2 , p * (p + 1)))
	shift = 0
	for i in range(0, p**2):
		E21_1[i, i + shift] = -1
		E21_1[i, i + 1 + shift] = 1

		# After every p rows, shift the pattern by 1 column
		if (i + 1) % p == 0:
			shift += 1

	E21_2 = np.zeros((p**2 , p * (p + 1)))
	for i in range(0,p**2):
		E21_2[i,i] = -1
		E21_2[i,i+p] = 1

	E21 = np.hstack((E21_1,E21_2))
	return E21

print(assemble_element_E10(2))
print(assemble_element_E21(2))
