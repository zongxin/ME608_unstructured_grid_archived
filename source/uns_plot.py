import os
import sys
import numpy as np 
import scipy as sp 
from pdb import set_trace
from matplotlib import rc as matplotlibrc
import math_tool as ma
import scipy.sparse as scysparse
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PatchCollection
import matplotlib as mpl


def PolyVertSort(x, y):
  cx = sum(x)/len(x)
  cy = sum(y)/len(y)
  # create a new list of corners which includes angles
  cornersWithAngles = []
  for xx, yy in zip(x,y):
    dx = xx - cx
    dy = yy - cy
    an = np.arctan(dy/(dx+0.000001))
    if (dy < 0):
      an= an +np.pi 
    cornersWithAngles.append((xx, yy, an))
  # sort it using the angles
  cornersWithAngles.sort(key = lambda tup: tup[2])
  x = np.array(cornersWithAngles)[:,0]
  y = np.array(cornersWithAngles)[:,1]
  return x,y



def Ave_operator(noocv,nno):

  ncv     = len(noocv)
  An2cv     = scysparse.lil_matrix((ncv,nno),dtype=np.float64)

  for icv, nolist in enumerate(noocv):
    ele = 1.0/(len(nolist))
    for i, note in enumerate(nolist):
      An2cv[icv,note] = ele*1.0 

  return An2cv.tocsr()

def Ave_operator_fa(faocv,nfa):

  ncv     = len(faocv)
  Afa2cv     = scysparse.lil_matrix((ncv,nfa),dtype=np.float64)

  for icv, falist in enumerate(faocv):
    ele = 1.0/(len(falist))
    for i, face in enumerate(falist):
      Afa2cv[icv,face] = ele*1.0 

  return Afa2cv.tocsr()





#base=1 :vertex based
#base=2: face based 
#base=3: cell based
def contour_plot(xy_no,noocv):
  if True:

    patches=[]
    for icv, nolist in enumerate(noocv):
      if True:
        temp=nolist
        xx=np.zeros(len(temp))
        yy=np.zeros(len(temp))
        for i, ino in enumerate(temp):
          xx[i]=xy_no[ino,0]
          yy[i]=xy_no[ino,1]
          xx1,yy1=PolyVertSort(xx, yy)
        Path = mpath.Path
        if len(temp)==3:
          path_data = [
             (Path.MOVETO, (xx1[0],yy1[0])),
             (Path.LINETO, (xx1[1],yy1[1])),
             (Path.LINETO, (xx1[2],yy1[2])),
             (Path.CLOSEPOLY, (xx1[0],yy1[0])),
             ]
        if len(temp)==4:
          path_data = [
             (Path.MOVETO, (xx1[0],yy1[0])),
             (Path.LINETO, (xx1[1],yy1[1])),
             (Path.LINETO, (xx1[2],yy1[2])),
             (Path.LINETO, (xx1[3],yy1[3])),
             (Path.CLOSEPOLY, (xx1[0],yy1[0])),
             ]

        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)

        patch = mpatches.PathPatch(path, ec='w', alpha=1)

        patches.append(patch)          

  return patches

#  cm = plt.get_cmap('viridis')
#  fig_width = 10
#  fig_height = 10
#  textFontSize   = 10
#  gcafontSize    = 26
#  lineWidth      = 2  
#  Plot_Node_Labels = True
#  Plot_Face_Labels = True
#  Plot_CV_Labels   = True

  #the following enables LaTeX typesetting, which will cause the plotting to take forever..  
#  matplotlibrc('text.latex', preamble='\usepackage{color}')
#  matplotlibrc('text',usetex=True)
#  matplotlibrc('font', family='serif') #

#  fig, ax = plt.subplots()
#  fig = plt.figure(figsize=(10, 10))#
#

#  lowcolor=[69,5,88]
#  highcolor=[250,230,34]
#  midcolor =[32,141,139]
#  up=1.6
#  down=-3
#  mid =(up+down)*0.5
#  barlable=r'$Phi$'#

#  nno=xy_no.shape[0]
#  mgplx = 0.05*np.abs(max(xy_no[:,0])-min(xy_no[:,0]))
#  mgply = 0.05*np.abs(max(xy_no[:,1])-min(xy_no[:,1]))
#  xlimits = [min(xy_no[:,0])-mgplx,max(xy_no[:,0])+mgplx]
#  ylimits = [min(xy_no[:,1])-mgply,max(xy_no[:,1])+mgply] #

#  left,width=0.14,0.77
##  bottom,height=0.11,0.77
##  bottom_h=bottom+height+0.04
#  bottom_h =0.11
#  height_h =0.005
#  height=0.77
#  bottom=bottom_h+height_h+0.12
#  rect_line1=[left,bottom,width,height]
#  rect_line2=[left,bottom_h,width,0.05]
#  ax=plt.axes(rect_line1)
#  ax2=plt.axes(rect_line2)
#  ax.set_xlim(xlimits)
#  ax.set_ylim(ylimits)
#  plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#  plt.setp(ax.get_yticklabels(),fontsize=gcafontSize) #
#

#  if base ==1:
#    An2cv=Ave_operator(noocv,nno)
#    Tn=T
#    Tcv=An2cv.dot(Tn)
#  if base ==2:
#    Afa2cv=Ave_operator_fa(faocv,nfa)
#    Tfa=T
#    Tcv=Afa2cv.dot(Tfa)#

#  if False:
#    for icv, nolist in enumerate(noocv):
#      if True:
#        temp=nolist
#        xx=np.zeros(len(temp))
#        yy=np.zeros(len(temp))
#        for i, ino in enumerate(temp):
#          xx[i]=xy_no[ino,0]
#          yy[i]=xy_no[ino,1]
#          xx1,yy1=PolyVertSort(xx, yy)
#        Path = mpath.Path
#        if len(temp)==3:
#          path_data = [
#             (Path.MOVETO, (xx1[0],yy1[0])),
#             (Path.LINETO, (xx1[1],yy1[1])),
#             (Path.LINETO, (xx1[2],yy1[2])),
#             (Path.CLOSEPOLY, (xx1[0],yy1[0])),
#             ]
#        if len(temp)==4:
#          path_data = [
#             (Path.MOVETO, (xx1[0],yy1[0])),
#             (Path.LINETO, (xx1[1],yy1[1])),
#             (Path.LINETO, (xx1[2],yy1[2])),
#             (Path.LINETO, (xx1[3],yy1[3])),
#             (Path.CLOSEPOLY, (xx1[0],yy1[0])),
#             ]
#        if Tcv[icv] < mid:
#          r=lowcolor[0]+  (midcolor[0]-lowcolor[0])/(mid-down)*(Tcv[icv]-down)
#          g=lowcolor[1]+  (midcolor[1]-lowcolor[1])/(mid-down)*(Tcv[icv]-down)
#          b=lowcolor[2]+  (midcolor[2]-lowcolor[2])/(mid-down)*(Tcv[icv]-down)
#        if Tcv[icv] >= mid:
#          r=midcolor[0]+  (highcolor[0]-midcolor[0])/(up-mid)*(Tcv[icv]-mid)
#          g=midcolor[1]+  (highcolor[1]-midcolor[1])/(up-mid)*(Tcv[icv]-mid)
#          b=midcolor[2]+  (highcolor[2]-midcolor[2])/(up-mid)*(Tcv[icv]-mid)   #

#        r=np.round(r)/256
#        g=np.round(g)/256
#        b=np.round(b)/256#
##        RGB[icv,:]=[r,g,b]
#        codes, verts = zip(*path_data)
#        path = mpath.Path(verts, codes)#

#        patch = mpatches.PathPatch(path, facecolor=(r,g,b), alpha=1)#

#        ax.add_patch(patch)
#       #

#        # plot control points and connecting lines
#        x, y = zip(*path.vertices)
#        line, = ax.plot(x, y, 'k-',linewidth=1)        #

#        ax.grid()
#        ax.axis('equal')
# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.

#  norm = mpl.colors.Normalize(vmin=down, vmax=up)#
#

#  cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cm,
#                                norm=norm,
#                                orientation='horizontal')
#  cb1.set_label(barlable,fontsize=gcafontSize)#

#  cb1.set_ticks(np.linspace(down, up, 3))  
#  cb1.set_ticklabels( (str(down),str(mid),str(up)))
#  #cb1.ax2.tick_params(labelsize=gcafontSize)  
#  plt.setp(ax2.get_xticklabels(),fontsize=gcafontSize)
#  plt.savefig('../report/figures/iteration/'+ figurename)




