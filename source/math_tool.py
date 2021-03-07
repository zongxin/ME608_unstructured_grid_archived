import os
import sys
import numpy as np
import scipy as sp 
from pdb import set_trace



#################################################### 
#    Author:Danish                                 #
#    Function: correct the order of point in space #
####################################################

def PolyVertSort(x, y):
  cx = sum(x)/len(x)
  cy = sum(y)/len(y)
  # create a new list of corners which includes angles
  cornersWithAngles = []
  for xx, yy in zip(x,y):
    dx = xx - cx
    dy = yy - cy
    an = np.arctan(dy/dx)
    if dy < 0:
      an = an +np.pi 
    cornersWithAngles.append((xx, yy, an))
  # sort it using the angles
  cornersWithAngles.sort(key = lambda tup: tup[2])
  x = np.array(cornersWithAngles)[:,0]
  y = np.array(cornersWithAngles)[:,1]
  return x,y

#################################################### 
#    Author:Danish                                 #
#    Function: compute the area of tri or rect     #
#    Name :Shoelace formula                        #
####################################################
def PolyArea(x,y):
    # shoelace formula: https://en.wikipedia.org/wiki/Shoelace_formula
    x,y = PolyVertSort(x,y)       # sort vertices to form closed polygon
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def Gauss_Seidel(A,q,w):
  n=len(q)
  x=np.zeros(n)
  # A =D+L+U
  #np.triu()/np.tril()/np.diag()/np.identity(3)
  D = np.diag(A)*np.identity(n)
  L = np.tril(A)-D
  U = np.triu(A)-D

  #(D+wL)xn+1=[D-w(D+U)]xn+w*q

  x_new  = np.zeros(n)
  r      = np.zeros(n)
  norm_r = {}
  dat_ite={}
  r_0  = A.dot(x_new)-q
  norm_r[0] = np.sum(r_0**2)**0.5
  ii = 0
  tol =10**(-7) 
  D_inv = np.linalg.inv(D+w*L)

  while ((ii==0) or (norm_r[ii]/norm_r[0]>tol)):
    ii = ii+1
    x  = x_new*1.0

    x_new = D_inv.dot(((D-w*(D+U)).dot(x)+w*q))

    r = A.dot(x_new)-q
    norm_r[ii] = np.sum(r**2)**0.5
    dat_ite[str(ii)] = x_new
    
  return x_new,norm_r,dat_ite