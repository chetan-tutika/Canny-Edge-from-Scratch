'''
File name: edgeLink.py
Author:
Date created:
'''

'''
File clarification:
Use hysteresis to link edges based on high and low magnitude thresholds
- Input M: H x W logical map after non-max suppression
- Input Mag: H x W matrix represents the magnitude of gradient
- Input Ori: H x W matrix represents the orientation of gradient
- Output E: H x W binary matrix represents the final canny edge detection map
'''

import numpy as np
import cv2
from scipy import signal
import utils
import math
from interp import interp2

from PIL import Image
import matplotlib.pyplot as plt
import scipy

import findDerivatives as fd 
from findDerivatives import findDerivatives
from nonMaxSup import nonMaxSup

def find_edge_in_dir(strong_edge, weak_edge, filter_dir, filter_dir_opp, Ori_dir):
	strong_edge_mask1=strong_edge*(Ori_dir)
	#print(strong_edge_mask.shape)
	weak_edge_mask1=weak_edge*(Ori_dir)


	strong_edge_shift = signal.convolve2d(strong_edge_mask1, filter_dir, mode='same')
	strong_edge_shift_opp = signal.convolve2d(strong_edge_mask1, filter_dir, mode='same')
	edge1 = np.logical_or(strong_edge_mask1, np.logical_and(strong_edge_shift, weak_edge_mask1))
	edge2 = np.logical_or(strong_edge_mask1, np.logical_and(strong_edge_shift, weak_edge_mask1))
	edge = np.logical_or(edge1, edge2)

	return edge



def edgeLink(M, Mag, Ori):
	# TODO: your code here

	nms_clean = M * Mag	

	nms_non_zero=nms_clean[nms_clean!=0]

	mean = np.median(nms_non_zero)
	std = np.std(nms_non_zero)

	#low_thresh=mean*0.66
	#high_thresh=mean*1.5

	low_thresh = mean/(10**3)
	high_thresh = mean + 2*std

	strong_edge = (nms_clean > high_thresh).astype(int)
	weak_edge = np.logical_and(nms_clean > low_thresh, nms_clean <= high_thresh).astype(int)


	Ori = Ori + (np.pi/2)
	Ori = np.arctan2(np.sin(Ori), np.cos(Ori))

	Ori[Ori<0]=np.pi+ Ori[Ori<0]
	Ori[Ori>7*(np.pi/8)]=np.pi-Ori[Ori>7*(np.pi/8)]

	Ori[np.logical_and(Ori>=0,Ori<np.pi/8)]=0.0
	Ori[np.logical_and(Ori>=np.pi/8,Ori<3*np.pi/8)]=np.pi/4
	Ori[np.logical_and(Ori>=3*np.pi/8,Ori<5*(np.pi/8))]=np.pi/2
	Ori[np.logical_and(Ori>=5*np.pi/8,Ori<7*np.pi/8)]=3*np.pi/4

	# Ori[np.logical_and(Ori>=-np.pi/8, Ori < np.pi/8)] = 0
	# Ori[np.logical_and(Ori>=np.pi/8 ,  Ori < 3*np.pi/8)]= np.pi/4
	# Ori[np.logical_and(Ori>=3*np.pi/8  ,  Ori < 5*np.pi/8)] = np.pi/2
	# Ori[np.logical_and(Ori>=5*np.pi/8  ,  Ori < 7*np.pi/8)] = 3*np.pi/4
	# Ori[np.logical_and(Ori>=7*np.pi/8  , Ori < -7*np.pi/8)] = np.pi
	# Ori[np.logical_and(Ori>=-7*np.pi/8 ,  Ori< -5*np.pi/8)] = -np.pi/4
	# Ori[np.logical_and(Ori>=-5*np.pi/8  ,  Ori< -3*np.pi/8)]= -np.pi/2
	# Ori[np.logical_and(Ori>=-3*np.pi/8  ,  Ori< -1*np.pi/8)] = -3*np.pi/4


	filter0 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
	filter180 = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
	filter90 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
	filter270 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
	filter45 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
	filter225 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
	filter135 = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
	filter335 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])


	plt.figure(num='initial strong_edge')
	plt.imshow(strong_edge, cmap='gray')
	result = Image.fromarray((strong_edge * 255).astype(np.uint8))
	result.save('initial_strong_edge1.bmp')

	initial_strong_edges= np.sum(strong_edge)
	print('Number of strong edges initially, ',initial_strong_edges)


	strong_edge_mask1=strong_edge.copy()

	for i in range(300):

		final_edge_mask1 = find_edge_in_dir(strong_edge, weak_edge, filter0, filter180, Ori == 0)
		#print(final_edge_mask.shape)
		strong_edge_mask1=np.logical_or(strong_edge_mask1,final_edge_mask1)

		final_edge_mask2 = find_edge_in_dir(strong_edge, weak_edge, filter45, filter225,Ori == np.pi/4)
		strong_edge_mask1=np.logical_or(strong_edge_mask1,final_edge_mask2)

		final_edge_mask7 = find_edge_in_dir(strong_edge, weak_edge, filter90, filter270, Ori == -np.pi/2)
		strong_edge_mask1=np.logical_or(strong_edge_mask1,final_edge_mask7)

		final_edge_mask4 = find_edge_in_dir(strong_edge, weak_edge, filter135, filter335, Ori == 3*np.pi/4)
		strong_edge_mask1=np.logical_or(strong_edge_mask1,final_edge_mask4)

		# final_edge_mask5 = find_edge_in_dir(strong_edge, weak_edge, filter180, Ori == np.pi)
		# strong_edge_mask1=np.logical_or(strong_edge_mask1,final_edge_mask5)

		# final_edge_mask6 = find_edge_in_dir(strong_edge, weak_edge, filter225, Ori == -np.pi/4)
		# strong_edge_mask1=np.logical_or(strong_edge_mask1,final_edge_mask6)


		# final_edge_mask3 = find_edge_in_dir(strong_edge, weak_edge, filter270,	Ori == np.pi/2)
		# strong_edge_mask1=np.logical_or(strong_edge_mask1,final_edge_mask3)

		
		# final_edge_mask6 = find_edge_in_dir(strong_edge, weak_edge, filter225, Ori == -np.pi/4)
		# strong_edge_mask1=np.logical_or(strong_edge_mask1,final_edge_mask6)

		

		# final_edge_mask8 = find_edge_in_dir(strong_edge, weak_edge, filter335, Ori == -3*np.pi/4)
		# strong_edge_mask1=np.logical_or(strong_edge_mask1,final_edge_mask8)

		strong_edge=strong_edge_mask1.copy()
		

	final_strong_edges = np.sum(strong_edge)
	
	print('Number of strong edges finally, ', final_strong_edges)
	print('Increase in Edges after 300 iterations, ', final_strong_edges - initial_strong_edges)

	plt.figure(num='final_strong_edges')

	plt.imshow(strong_edge, cmap='gray')
	#print('sedge',strong_edge.shape)
	result = Image.fromarray((strong_edge * 255).astype(np.uint8))
	result.save('initial_strong_edge2.bmp')



	return strong_edge


if __name__ == "__main__":

	I = np.array(Image.open('118035.jpg').convert('RGB'))
	I_gray = utils.rgb2gray(I)
	Mag, Magx, Magy, Ori = findDerivatives(I_gray)

	plt.figure(num='Mag')
	plt.imshow(Mag, cmap='gray')

	M = nonMaxSup(Mag, Ori)

	plt.figure(num='nms')
	plt.imshow(M*Mag, cmap='gray')


	finaledge=edgeLink(M, Mag, Ori)


	plt.show()




