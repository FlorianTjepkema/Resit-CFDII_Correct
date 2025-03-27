"""
Author: Suyash Shrestha
License: MIT
Copyright (c) 2024, Suyash Shrestha
All rights reserved.
"""
import numpy as np

class cylinder():
	def __init__(self):
		self.X0 = [0.0, 1.0];
		self.X1 = [0.0, 1.0];
		self.mapping_parms = ();
		self.r_in = 0.5;
		self.r_out = 0.5;

		self.Fx_L = lambda y_hat: np.zeros_like(y_hat);
		self.Fy_L = lambda y_hat: self.r_out*(y_hat) + self.r_in;
		self.Fx_R = lambda y_hat: np.zeros_like(y_hat);
		self.Fy_R = lambda y_hat: self.r_out*(y_hat) + self.r_in;
		self.Fx_B = lambda x_hat: self.r_in*np.sin(2*np.pi*x_hat);
		self.Fy_B = lambda x_hat: self.r_in*np.cos(2*np.pi*x_hat);
		self.Fx_T = lambda x_hat: (self.r_out + self.r_in)*np.sin(2*np.pi*x_hat);
		self.Fy_T = lambda x_hat: (self.r_out + self.r_in)*np.cos(2*np.pi*x_hat);

		self.dFx_L = lambda y_hat: np.zeros_like(y_hat);
		self.dFy_L = lambda y_hat: self.r_out*np.ones_like(y_hat);
		self.dFx_R = lambda y_hat: np.zeros_like(y_hat);
		self.dFy_R = lambda y_hat: self.r_out*np.ones_like(y_hat);
		self.dFx_B = lambda x_hat: self.r_in*2*np.pi*np.cos(2*np.pi*x_hat);
		self.dFy_B = lambda x_hat: -self.r_in*2*np.pi*np.sin(2*np.pi*x_hat);
		self.dFx_T = lambda x_hat: (self.r_out + self.r_in)*2*np.pi*np.cos(2*np.pi*x_hat);
		self.dFy_T = lambda x_hat: -(self.r_out + self.r_in)*2*np.pi*np.sin(2*np.pi*x_hat);

	def Phi(self, xi, eta, X0, X1, Kx, Ky, i, j):
		r = X0[0] + ((X0[1] - X0[0])/Kx)*(i + (xi + 1)/2);
		s = X1[0] + ((X1[1] - X1[0])/Ky)*(j + (eta + 1)/2);
		x = (1 - s)*self.Fx_B(r) + s*self.Fx_T(r) + (1 - r)*self.Fx_L(s) + r*self.Fx_R(s) - (r*s*self.Fx_T(1) + r*(1 - s)*self.Fx_B(1) + s*(1 - r)*self.Fx_T(0) + (1 - r)*(1 - s)*self.Fx_B(0));
		y = (1 - s)*self.Fy_B(r) + s*self.Fy_T(r) + (1 - r)*self.Fy_L(s) + r*self.Fy_R(s) - (r*s*self.Fy_T(1) + r*(1 - s)*self.Fy_B(1) + s*(1 - r)*self.Fy_T(0) + (1 - r)*(1 - s)*self.Fy_B(0));
		return r, s, x, y;

	def dPhi(self, xi, eta, X0, X1, Kx, Ky, i, j):
		r = X0[0] + ((X0[1] - X0[0])/Kx)*(i + (xi + 1)/2);
		s = X1[0] + ((X1[1] - X1[0])/Ky)*(j + (eta + 1)/2);
		dx_xi = ((X0[1] - X0[0])/Kx)/2*((1 - s)*self.dFx_B(r) + s*self.dFx_T(r) + (-1)*self.Fx_L(s) + 1*self.Fx_R(s) - (1*s*self.Fx_T(1) + 1*(1 - s)*self.Fx_B(1) + s*(-1)*self.Fx_T(0) + (-1)*(1 - s)*self.Fx_B(0)));
		dx_eta = ((X1[1] - X1[0])/Ky)/2*((-1)*self.Fx_B(r) + 1*self.Fx_T(r) + (1 - r)*self.dFx_L(s) + r*self.dFx_R(s) - (r*1*self.Fx_T(1) + r*(-1)*self.Fx_B(1) + 1*(1 - r)*self.Fx_T(0) + (1 - r)*(-1)*self.Fx_B(0)));
		dy_xi = ((X0[1] - X0[0])/Kx)/2*((1 - s)*self.dFy_B(r) + s*self.dFy_T(r) + (-1)*self.Fy_L(s) + 1*self.Fy_R(s) - (1*s*self.Fy_T(1) + 1*(1 - s)*self.Fy_B(1) + s*(-1)*self.Fy_T(0) + (-1)*(1 - s)*self.Fy_B(0)));
		dy_eta = ((X1[1] - X1[0])/Ky)/2*((-1)*self.Fy_B(r) + 1*self.Fy_T(r) + (1 - r)*self.dFy_L(s) + r*self.dFy_R(s) - (r*1*self.Fy_T(1) + r*(-1)*self.Fy_B(1) + 1*(1 - r)*self.Fy_T(0) + (1 - r)*(-1)*self.Fy_B(0)));
		return dx_xi, dx_eta, dy_xi, dy_eta;
