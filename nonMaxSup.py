'''
  File name: nonMaxSup.py
  Author:
  Date created:
'''

'''
  File clarification:
    Find local maximum edge pixel using NMS along the line of the gradient
    - Input Mag: H x W matrix represents the magnitude of derivatives
    - Input Ori: H x W matrix represents the orientation of derivatives
    - Output M: H x W binary matrix represents the edge map after non-maximum suppression
'''

import numpy as np
from scipy import signal
import utils
import math
import interp
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import findDerivatives as fd


def nonMaxSup(Mag, Ori):
  # TODO: your code here
  '''
	Ori[Ori<0]=np.pi+ Ori[Ori<0]
	Ori[Ori>7*(np.pi/8)]=np.pi-Ori[Ori>7*(np.pi/8)]

	Ori[np.logical_and(Ori>=0,Ori<np.pi/8)]=0.0
	Ori[np.logical_and(Ori>=np.pi/8,Ori<3*np.pi/8)]=np.pi/4
	Ori[np.logical_and(Ori>=3*np.pi/8,Ori<5*(np.pi/8))]=np.pi/2
	Ori[np.logical_and(Ori>=5*np.pi/8,Ori<7*np.pi/8)]=3*np.pi/4

	#plt.imshow(ang1,cmap='gray')
	#print(ang1)


	fx=np.array([[-1,2,-1]])
	fy=np.array([[-1],[2],[-1]])
	fxy=np.array([[0,0,-1],[0,2,0],[-1,0,0]])
	fyx=np.array([[-1,0,0],[0,2,0],[0,0,-1]])


	ang1x=(Ori==0)
	ang1y=(Ori==np.pi/2)
	ang1xy=(Ori==np.pi/4)
	ang1yx=(Ori==3*np.pi/4)


	#edgex1=np.logical_and(signal.convolve2d(M,np.asarray([[-1,1,0]]),'same')>0, signal.convolve2d(M,np.asarray([[0,1,-1]]),'same')>0).astype(int)


	edgex=np.logical_and(signal.convolve2d(Mag,np.asarray([[-1,1.1,0]]),'same')>0, signal.convolve2d(Mag,np.asarray([[0,1.1,-1]]),'same')>0).astype(int)
	#edgex[edgex>0]=1.0
	#edgex[edgex<=0]=0.0
	edgex=edgex*ang1x
	edgex=edgex*Mag
	#print(edgex1)


	#edgey=signal.convolve2d(M,fy,'same')
	edgey=np.logical_and(signal.convolve2d(Mag,np.asarray([[-1],[1.1],[0]]),'same')>0, signal.convolve2d(Mag,np.asarray([[0],[1.1],[-1]]),'same')>0).astype(int)
	#edgey[edgey>0]=1.0
	#edgey[edgey<=0]=0.0
	edgey=edgey*ang1y
	edgey=edgey*Mag


	#edgexy1=np.logical_and(signal.convolve2d(M,np.asarray([[0,0,-1],[0,1,0],[0,0,0]]),'same'), signal.convolve2d(M,np.asarray([[0,0,0],[0,1,0],[-1,0,0]]),'same')).astype(int)

	edgexy=np.logical_and(signal.convolve2d(Mag,np.asarray([[0,0,-1],[0,1.1,0],[0,0,0]]),'same')>0, signal.convolve2d(Mag,np.asarray([[0,0,0],[0,1.1,0],[-1,0,0]]),'same')>0).astype(int)
	#edgexy=signal.convolve2d(M,fxy,'same')
	#edgexy[edgexy>0]=1.0
	#edgexy[edgexy<=0]=0.0
	edgexy=edgexy*ang1xy
	edgexy=edgexy*Mag

	#print(edgexy1)

	edgeyx=np.logical_and(signal.convolve2d(Mag,np.asarray([[-1,0,0],[0,1.1,0],[0,0,0]]),'same')>0, signal.convolve2d(Mag,np.asarray([[0,0,0],[0,1.1,0],[0,0,-1]]),'same')>0).astype(int)

	#edgeyx=signal.convolve2d(M,fyx,'same')
	#edgeyx[edgeyx>0]=1.0
	#edgeyx[edgeyx<0]=0.0
	edgeyx=edgeyx*ang1yx
	edgeyx=edgeyx*Mag

	#plt.imshow(edgex,cmap='gray')

	edge=edgex+edgey+edgexy+edgeyx
	return edge
	'''

  #X, Y = np.meshgrid(np.arange(0, Mag.shape[1], 1), np.arange(0, Mag.shape[0], 1))
  X, Y = np.meshgrid(np.arange(Mag.shape[1]), np.arange(Mag.shape[0]))
  Ori=+1 * Ori

  Xc = X + np.cos(Ori)
  Ys = Y + np.sin(Ori)

  Xc_neg = X - np.cos(Ori)
  Ys_neg = Y - np.sin(Ori)

  M_interp_pos = interp.interp2(Mag,Xc,Ys)
  M_interp_neg = interp.interp2(Mag,Xc_neg,Ys_neg)


  
  Magx = Mag > M_interp_pos
  Magy = Mag > M_interp_neg


  Mag1=np.logical_and(Magx,Magy)

  return Mag1
 








  #print(Xc,Ys)
if __name__== '__main__':
  im=np.asarray(Image.open('55067.jpg').convert('RGB'))

  I_gray=utils.rgb2gray(im)
  [Mag,Magx,Magy,Ori]=fd.findDerivatives(I_gray)
  M = nonMaxSup(Mag,Ori)
  M1=M*Mag
  plt.figure(num="nms")
  plt.imshow(M,cmap='gray')
  plt.figure()
  plt.imshow(Mag,cmap='gray')
  plt.figure()
  plt.imshow(M1,cmap='gray')


plt.show()



