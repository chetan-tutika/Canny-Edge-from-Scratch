'''
  File name: findDerivatives.py
  Author:
  Date created:
'''

'''
  File clarification:
    Compute gradient information of the input grayscale image
    - Input I_gray: H x W matrix as image
    - Output Mag: H x W matrix represents the magnitude of derivatives
    - Output Magx: H x W matrix represents the magnitude of derivatives along x-axis
    - Output Magy: H x W matrix represents the magnitude of derivatives along y-axis
    - Output Ori: H x W matrix represents the orientation of derivatives
'''

import numpy as np
from scipy import signal
import utils
import math

from PIL import Image
import matplotlib.pyplot as plt
import scipy

def findDerivatives(I_gray):

  g=utils.GaussianPDF_2D(0.1,1,5,5)
  #print(g)

  x=np.array([[0,0,0],[1,0,-1],[0,0,0]])
  y=np.array([[0,1,0],[0,0,0],[0,-1,0]])
  #print(x.shape)
  #I_gray1=signal.convolve2d(I_gray,g,mode='same')


  dx=signal.convolve2d(g,x,mode='same')
  dy=signal.convolve2d(g,y,mode='same')

  i_gx=signal.convolve2d(I_gray,dx,'same')
  i_gy=signal.convolve2d(I_gray,dy,'same')


  #plt.imshow(i_gy,cmap='gray')

  M=np.sqrt(i_gx*i_gx + i_gy*i_gy)
  ang1=np.arctan2(i_gy,i_gx)
  return M,i_gx,i_gy,ang1






