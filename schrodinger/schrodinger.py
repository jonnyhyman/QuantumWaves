from pathlib import Path

from numpy import ( sin, cos, exp, pi, tan, log, sinh, cosh, tanh, sinc,
					sqrt, cbrt, angle, real, imag, abs,
					arcsin, arccos, arctan, arcsinh, arccosh, arctanh)
from numpy import pi, e
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation


import scipy.linalg
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

from numba import njit

from schrodinger import util
import sys

from time import time

""" Original french comments from
https://github.com/Azercoco/Python-2D-Simulation-of-Schrodinger-Equation

Le programme simule le comportement d'un paquet d'onde gaussien suivant
l'équation de Schrödinger. L'algorithme utilisé est la méthode
Alternating direction implicit method.

La simulation permet de configurer un potentiel constant avec le temps
ainsi que la présence d'obstacles (qui sont gérés comme des barrières
de potentiel très élévées).

La fonction d'onde complexe est affichée en convertissant les nombres
complexes en format de couleur HSV.

x , y : Les positions de départ du paquet d'onde
Kx, Ky : Ses nombres d'onde
Ax, Ay : Ses facteurs d'étalements selon x et y
V : L'expression du potentiel
O : L'expression de la présence d'obstacles

Le potentiel et la présence d'obstacle doivent être exprimés comme des
expressions Python valides dépendant de x et y (valant respectivement
un float et un boolean) car le progamme utilise la fonction Python
eval() pour les évaluer.

"""

""" Translated by Google Translate
https://github.com/Azercoco/Python-2D-Simulation-of-Schrodinger-Equation

The program simulates the behavior of a Gaussian wave packet following the
Schrödinger's equation. The algorithm used is the method
Alternating direction implicit method.

The simulation makes it possible to configure a constant potential over time
as well as the presence of obstacles (which are managed as barriers
very high potential).

Complex wave function is displayed by converting numbers
complex in HSV color format.

x, y: The starting positions of the wave packet
Kx, Ky: The numbers of the wave
Ax, Ay: Its spreading factors along x and y
V: The expression of potential
O: The expression of the presence of obstacles

The potential and the presence of obstacles must be expressed as
valid Python expressions depending on x and y (respectively
a float and a boolean) because the program uses the Python function
eval () to evaluate them.

"""

class Field:

	def __init__(self):
		self.potential_expr = None
		self.obstacle_expr = None

	def setPotential(self, expr):
		self.potential_expr = expr
		self.test_pot_expr()

	def setObstacle(self, expr):
		self.obstacle_expr = expr
		self.test_obs_expr()


	def test_pot_expr(self):

		# required for eval()
		x = 0
		y = 0

		try:
			a = eval(self.potential_expr)
		except:
			print(self.potential_expr)
			print('Potential calculation error: set to 0 by default')
			self.potential_expr = '0'

	def test_obs_expr(self):

		# required for eval()
		x = 0
		y = 0

		try:
			a = eval(self.obstacle_expr)
		except:
			print('Error setting obstacle: Set to False by default')
			self.obstacle_expr = 'False'

	def isObstacle(self, x, y):

		a = False

		try:
			a = eval(self.obstacle_expr)
		except:
			print(f'Invalid obstacle: {self.obstacle_expr}')

		return a

	def getPotential(self, x, y):

		a = 0 + 0j

		try:
			a = eval(self.potential_expr)
		except:
			print(f'Invalid potential: {self.potential_expr}')

		return a

def solve(wf, V_x, V_y, HX, HY, N, step, delta_t):

	vector_wrt_x = util.x_concatenate(wf, N)
	vector_derive_y_wrt_x = util.x_concatenate(util.dy_square(wf, N, step), N)
	U_wrt_x = vector_wrt_x + (1j*delta_t/2 )*(vector_derive_y_wrt_x - V_x*vector_wrt_x)
	U_wrt_x_plus = scipy.sparse.linalg.spsolve(HX, U_wrt_x)

	wf = util.x_deconcatenate(U_wrt_x_plus, N)

	vector_wrt_y = util.y_concatenate(wf, N)
	vector_derive_x_wrt_y = util.y_concatenate(util.dx_square(wf, N, step), N)
	U_wrt_y = vector_wrt_y  + (1j*delta_t/2 )*(vector_derive_x_wrt_y - V_y *vector_wrt_y)
	U_wrt_y_plus = scipy.sparse.linalg.spsolve(HY, U_wrt_y)

	wf = util.y_deconcatenate(U_wrt_y_plus, N)

	return wf

class Simulate:
	SIZE = 10 # simulation self.size

	# wavefunction collision
	FPS = 60
	DURATION = 5 # duration in seconds
	DELTA_T = 0.005 # 0.125 #time elapsed per second of video

	# wavefunction collapse
	FPS = 60
	DURATION = 5 # duration in seconds
	DELTA_T = 0.01 # 0.125 #time elapsed per second of video

	# wavefunction collapse 2 & 3
	FPS = 60
	DURATION = 5 # duration in seconds
	DELTA_T = 0.03 # 0.125 #time elapsed per second of video

	# wavefunction collapse 4
	FPS = 60
	DURATION = 5 # duration in seconds
	DELTA_T = 0.005 # 0.125 #time elapsed per second of video

	# entanglement1
	FPS = 60
	DURATION = 5 # duration in seconds
	DELTA_T = 0.02 # 0.125 #time elapsed per second of video

	# wavefunction movement
	FPS = 60
	DURATION = 5 # duration in seconds
	DELTA_T = 0.005 # 0.125 #time elapsed per second of video

	def __init__(self, N, collapse=False):

		self.N = N # dimension in number of points of the simulation

		self.FRAMES = self.DURATION * self.FPS

		self.field = Field()

		#Potential as a function of x and y
		self.field.setPotential("0") # Ex: x**2+y**2"

		#Obstacle: boolean expression in fct of x and y
		# (set to False if you do not want an obstacle)

		obstacles = ("(x > 0.5 and x < 1 and not "
					"((y > 0.25 and y < 0.75) or "
					"(y < -0.25 and y > -0.75)))")

		obstacles = "False"

		self.collapse = collapse

		self.field.setObstacle(obstacles)
		self.size = self.SIZE

		#self.dataset = np.zeros((self.FRAMES,self.N,self.N), dtype='c16')

		print(16*self.N*self.N*1e-9, 'GB of memory')

		#if self.dataset.nbytes > 100e9:
		#	raise(Exception("TOO MUCH DATA FOR MEMORY"))

		self.simulation_initialize()

	""" ------ INITIAL CONDITIONS FOR WAVEFUNCTION COLLISION
	x0 = [0, 0],
	y0 = [0,1],

	#number of waves
	k_x = [0, 0],#5000
	k_y = [0, 90000],#2500,

	#spreading
	a_x = [.2, .2], #.2, #.1,#.33,#0.05#.33
	a_y = [.2, .2], #.2, #.1,#.33,#0.05#.33
	"""

	""" ------ INITIAL CONDITIONS FOR WAVEFUNCTION COLLISION 1
	x0 = [0,0],
	y0 = [0,1.5],

	#number of waves
	k_x = [10, 0],#5000
	k_y = [0, 90000],#2500,

	#spreading
	a_x = [.15, .15], #.2, #.1,#.33,#0.05#.33
	a_y = [.15, .15], #.2, #.1,#.33,#0.05#.33
	"""

	""" ------ INITIAL CONDITIONS FOR MOVEMENT SHOTS
	x0 = [0],
	y0 = [0],

	#number of waves
	k_x = [5000],
	k_y = [2500],#2500,

	#spreading
	a_x = [.2], #.2, #.1,#.33,#0.05#.33
	a_y = [.2], #.2, #.1,#.33,#0.05#.33
	"""

	""" ------ INITIAL CONDITIONS FOR WAVEFUNCTION COLLAPSE
	x0 = [0],#0],
	y0 = [0],

	#number of waves
	k_x = [50],
	k_y = [25],#2500,

	#spreading
	a_x = [.25], #.2, #.1,#.33,#0.05#.33
	a_y = [.25], #.2, #.1,#.33,#0.05#.33
	"""

	""" ------ INITIAL CONDITIONS FOR WAVEFUNCTION COLLAPSE 3
	x0 = [0],#0],
	y0 = [0],

	#number of waves
	k_x = [50],
	k_y = [25],#2500,

	#spreading
	a_x = [.28], #.2, #.1,#.33,#0.05#.33
	a_y = [.28], #.2, #.1,#.33,#0.05#.33
	"""

	""" ------ INITIAL CONDITIONS FOR ENTANGLEMENT

	x0 = [0, 0],
	y0 = [1,-1],

	#number of waves
	k_x = [0, 0],#5000
	k_y = [-3000, 3000],#2500,

	#spreading
	a_x = [.15, .15], #.2, #.1,#.33,#0.05#.33
	a_y = [.15, .15], #.2, #.1,#.33,#0.05#.33
	"""


	def simulation_initialize(self,
			#characteristics of the wave packet gaussian 2D
			#centre
			x0 = [0],
			y0 = [0],

			#number of waves
			k_x = [5000],
			k_y = [2500],#2500,

			#spreading
			a_x = [.2], #.2, #.1,#.33,#0.05#.33
			a_y = [.2], #.2, #.1,#.33,#0.05#.33

			# keep below the same
			wall_potential = 1e10,
		):
		""" initialize the wave packet """

		N = self.N
		step = self.SIZE/self.N
		delta_t = self.DELTA_T/self.FPS

		self.counter = 0

		# create points at all xy coordinates in meshgrid
		self.x_axis = np.linspace(-self.size/2, self.size/2, N)
		self.y_axis = np.linspace(-self.size/2, self.size/2, N)
		X, Y = np.meshgrid(self.x_axis, self.y_axis)

		n = 0
		phase = np.exp( 1j*(X*k_x[n] + Y*k_y[n]))
		px = np.exp( - ((x0[n] - X)**2)/(4*a_x[n]**2))
		py = np.exp( - ((y0[n] - Y)**2)/(4*a_y[n]**2))

		wave_function = phase*px*py
		norm = np.sqrt(util.integrate(np.abs(wave_function)**2, N, step))
		self.wave_function = wave_function/norm

		for n in range(1,len(x0)):
			phase = np.exp( 1j*(X*k_x[n] + Y*k_y[n]))
			px = np.exp( - ((x0[n] - X)**2)/(4*a_x[n]**2))
			py = np.exp( - ((y0[n] - Y)**2)/(4*a_y[n]**2))

			wave_function = phase*px*py
			norm = np.sqrt(util.integrate(np.abs(wave_function)**2, N, step))
			self.wave_function += wave_function/norm

		LAPLACE_MATRIX = sp.sparse.lil_matrix(-2*sp.sparse.identity(N*N))
		for i in range(N):
			for j in range(N-1):
				k = i*N + j
				LAPLACE_MATRIX[k,k+1] = 1
				LAPLACE_MATRIX[k+1,k] = 1

		self.V_x = np.zeros(N*N, dtype='c16')

		for j in range(N):
			for i in range(N):
				xx = i
				yy = N*j
				if self.field.isObstacle(self.x_axis[j], self.y_axis[i]):
					self.V_x[xx+yy] = wall_potential
				else:
					self.V_x[xx+yy] = self.field.getPotential(self.x_axis[j],
																self.y_axis[i])

		self.V_y = np.zeros(N*N, dtype='c16')

		for j in range(N):
			for i in range(N):
				xx = j*N
				yy = i
				if self.field.isObstacle(self.x_axis[i], self.y_axis[j]):
					self.V_y[xx+yy] = wall_potential
				else:
					self.V_y[xx+yy] = self.field.getPotential(self.x_axis[i],
																self.y_axis[j])

		self.V_x_matrix = sp.sparse.diags([self.V_x], [0])
		self.V_y_matrix = sp.sparse.diags([self.V_y], [0])

		LAPLACE_MATRIX = LAPLACE_MATRIX/(step ** 2)

		self.H1 = (1*sp.sparse.identity(N*N) - 1j*(delta_t/2)*(LAPLACE_MATRIX))
		self.H1 = sp.sparse.dia_matrix(self.H1)

		self.HX = (1*sp.sparse.identity(N*N) - 1j*(delta_t/2)*(LAPLACE_MATRIX - self.V_x_matrix))
		self.HX = sp.sparse.dia_matrix(self.HX)

		self.HY = (1*sp.sparse.identity(N*N) - 1j*(delta_t/2)*(LAPLACE_MATRIX - self.V_y_matrix))
		self.HY = sp.sparse.dia_matrix(self.HY)

		self.start_time = time()
		self.i_time = time()

	def simulate_frames(self):

		for f in range(self.FRAMES):
			start=time()
			simulate_frame(f)
			print('>>>',time()-start)

		#dataname = f"C:/data/sim_{N}x{N}.npz"
		#np.savez(dataname, self.dataset)

	def simulate_frame(self, save=False, debug=True):
		""" evolve according to schrodinger equation """

		N = self.N
		step = self.SIZE/self.N
		delta_t = self.DELTA_T/self.FPS

		self.wave_function = solve(self.wave_function,
									self.V_x, self.V_y,
									self.HX, self.HY,
									N, step, delta_t)

		if save:
			self.save_wave(self.wave_function)

		if debug:
			self.print_update()

		self.counter += 1

		#if self.counter == self.FRAMES:
		#	self.simulation_initialize()

		return self.wave_function

	def collapse_wavefunction(self):

		dist=np.abs(self.wave_function)**2 # joint pmf
		dist/=dist.sum() # it has to be normalized

		# generate the set of all x,y pairs represented by the pmf
		pairs=np.indices(dimensions=(self.N, self.N)).T # here are all of the x,y pairs

		# make n random selections from the flattened pmf without replacement
		# whether you want replacement depends on your application
		n=1
		inds=np.random.choice(np.arange(self.N**2),
									p=dist.reshape(-1),
									size=n,
									replace=False)

		# inds is the set of n randomly chosen indicies into the flattened dist array...
		# therefore the random x,y selections
		# come from selecting the associated elements
		# from the flattened pairs array
		selection_place = pairs.reshape(-1,2)[inds][0]

		# convert to sim coordinates
		selection = (selection_place/self.N -.5) * (self.size)

		selection = [selection[0], selection[1], 0, 0]
		# collapsewidth
		cw = 10 / self.N
		print(">>> COLLAPSED TO =", selection, cw)

		self.simulation_initialize(x0=[selection[0]],
									y0=[selection[1]],

									k_x = [selection[2]],
									k_y = [selection[3]],

									a_x=[cw], a_y=[cw])

		return selection_place

	def dual_collapse_wavefunction(self):

		dist=np.abs(self.wave_function)**2 # joint pmf
		dist/=dist.sum() # it has to be normalized

		# generate the set of all x,y pairs represented by the pmf
		pairs=np.indices(dimensions=(self.N, self.N)).T # here are all of the x,y pairs

		# make n random selections from the flattened pmf without replacement
		# whether you want replacement depends on your application
		n=10
		inds=np.random.choice(np.arange(self.N**2),
									p=dist.reshape(-1),
									size=n,
									replace=False)

		# inds is the set of n randomly chosen indicies into the flattened dist array...
		# therefore the random x,y selections
		# come from selecting the associated elements
		# from the flattened pairs array
		selection_place = pairs.reshape(-1,2)[inds][0]

		# convert to sim coordinates
		selection = (selection_place/self.N -.5) * (self.size)


		momx, momy = 700, 1000

		selection1 = [selection[0], selection[1], momx, momy]
		# collapsewidth
		cw = 10 / self.N
		print(">>> COLLAPSED TO =", selection1, cw)

		for i in range(1, n):


			selection_place = pairs.reshape(-1,2)[inds][i]

			# convert to sim coordinates
			selection = (selection_place/self.N -.5) * (self.size)

			normto1 = np.linalg.norm(selection - np.array([selection1[0],selection1[1]]))

			if normto1 < 2:
				print("CONTINUE, dist:", normto1)
				continue
			else:
				print("FOUND IT!, dist:", normto1)
				break

		selection2 = [selection[0], selection[1],  -momx, -momy]

		# collapsewidth
		cw = 10 / self.N
		print(">>> COLLAPSED TO =", selection2, cw)

		self.simulation_initialize(x0=[selection1[0], selection2[0]],
									y0=[selection1[1], selection2[1]],

									k_x = [selection1[2], selection2[2]],
									k_y = [selection1[3], selection2[3]],

									a_x=[cw, cw], a_y=[cw, cw])

		return selection_place

	def save_wave(self, data):
		self.dataset[self.counter,:,:] = data

	def print_update(self):

		N = self.N
		step = self.SIZE/self.N
		delta_t = self.DELTA_T/self.FPS

		NORM = np.sqrt(util.integrate(np.abs(self.wave_function)**2, N, step))
		report = self.counter/(self.DURATION*self.FPS)

		M = 20
		k = int(report*M)
		l = M - k
		to_print = '[' + k*'#' + l*'-'+ ']   {0:.3f} %'

		d_time = time() - self.start_time

		ETA = (time()-self.i_time) * (self.FRAMES-self.counter) # (time / frame) * frames remaining
		ETA = (ETA / 60) # sec to min ... / 60 # seconds to hours
		ETA = np.modf(ETA)
		ETA = int(ETA[1]), int(round(ETA[0]*60))
		ETA = str(ETA[0]) + ":" + str(ETA[1]).zfill(2)

		self.i_time = time()

		print('--- Simulation in progress ---')
		print(to_print.format(report*100))
		print('Time elapsed : {0:.1f} s'.format(d_time))
		print(f'Estimated time remaining : {ETA}')
		print('Function standard : {0:.3f} '.format(NORM))
