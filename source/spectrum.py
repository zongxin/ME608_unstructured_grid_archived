import os
import sys
import numpy as np
import scipy as sp 
import scipy.sparse as scysparse
import scipy.sparse.linalg as splinalg
from scipy.io import savemat,loadmat
import pylab as plt
from pdb import set_trace
from scipy.interpolate import griddata
from matplotlib import rc as matplotlibrc
import umesh_reader,copy
import math_tool as ma
import Operator as Op
import bivariate_fit as fit
import time
import uns_plot as uns
from matplotlib.collections import PatchCollection
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

one_D =False
mesh  = False
two_D =False
contour=False
three_d=False

Nj=15
N =Nj-1
base=np.arange(Nj)*1.0
#base=np.array(base)
base2=np.arange(Nj-1)*1.0

xl=0.5*(1-np.cos(np.pi*base/N))
xg=0.5*(1-np.cos(np.pi*(2*base2+1)/(2*N)))
yl=0.5*(1-np.cos(np.pi*base/N))
yg=0.5*(1-np.cos(np.pi*(2*base2+1)/(2*N)))
# base function for any single value
def base(xg,x):
	h=np.ones((len(xg)))
	for j in range(len(xg)):
		for i in range(len(xg)):
			if (i!=j):
				m=(x-xg[i])/(xg[j]-xg[i])
				h[j]=h[j]*m
	return h
# first order base func need orign y_value

def der_base(xl,x):
	k={}
	h=np.ones((len(xl)))

	temp=np.zeros(len(xl))

	for j in range(len(xl)):
		b=np.ones((len(xl)-1))

		xx=np.ones((len(xl)-1))*x
		for i in range(len(xl)):
			k=list(copy.deepcopy(xl))
			bb=1
			k.remove(xl[j])
			b=xl[j]-k
			t=xx-k
			tt=np.ones((len(xl)-1))			
			for z,value in enumerate(t):
				bb=bb*b[z]#over
				for zz in range(len(t)):
					if (zz!=z):
						tt[z]=tt[z]*t[zz]
		temp[j]=np.sum(tt)
		h[j]=temp[j]/bb
		#set_trace()
	return h

def der_base2_0(xl,x):
	k={}
	#h=np.ones((len(xl)))

	#temp=np.zeros(len(xl))
	j=0
	if j==0:
		b=np.ones((len(xl)-1))

		xx=np.ones((len(xl)-1))*x
		for i in range(len(xl)):
			k=list(copy.deepcopy(xl))
			bb=1
			k.remove(xl[j])
			b=xl[j]-k
			t=xx-k
			tt=np.ones((len(xl)-1))			
			for z,value in enumerate(t):
				bb=bb*b[z]#over
				for zz in range(len(t)):
					if (zz!=z):
						tt[z]=tt[z]*t[zz]
		temp=np.sum(tt)
		h=temp/bb
		#set_trace()
	return h

def x_g2l(xg,xl):
	I=np.zeros((len(xl),len(xg)))
	for i,ixl in enumerate(xl):
		I[i,:]=base(xg,xl[i])
	return I

#def y_g2l(xg,xl):
#	I=np.zeros((len(xg),len(xl)))
#	for i,ixl in enumerate(xl):
#		I[:,i]=base(xg,xl[i])
#	return I


g=3
ug=np.zeros(N)
#ug = np.sin(2.*np.pi*xg)
I=x_g2l(xg,xl)
ul=I.dot(ug)
#ul[0]=g
#set_trace()
l_flux=np.zeros((len(ug),len(ul)))
for i in range(len(ug)):
	l_flux[i]=der_base(xl,xg[i])


#k=l_flux.dot(ul)
g_flux=np.zeros((len(ug),len(ug)))
for i in range(len(ug)):
	g_flux[i]=der_base(xg,xg[i])

GX,GY=np.meshgrid(xg,xg)
LX,LY=np.meshgrid(xl,xl)



xdelt=x_g2l(xg,xl)


lxx_flux=np.zeros((len(ug),len(ul)))
for i in range(len(ug)):
	lxx_flux[i]=der_base(xl,xg[i])

hxx_flux=np.zeros((len(ug),len(ug)))
for i in range(len(ug)):
	hxx_flux[i]=der_base(xg,xg[i])

#lxy_flux=np.zeros((len(ul),len(ug)))
#for i in range(len(ug)):
#	lxy_flux[:,i]=der_base(xl,xg[i])

ly_flux=np.zeros((len(ug),len(ul)))
for i in range(len(ug)):
	ly_flux[i]=der_base(xl,yg[i])

if two_D:


	ugx=np.zeros((len(ug),len(ug)))
	#ugx=np.sin(GX*2*np.pi)*np.sin(GY*2*np.pi)*0.1
	#set_trace()
	phi_in=ugx
	phi_in_new=ugx

	ii=0
	dx = xg[1:] - xg[:-1]
	dx_min = np.min(dx)
	#dt = 0.1*dx_min*0.5
	dt = 0.1*dx_min**2
	#while (ii==0 or \
	#       np.sum(np.abs(phi_in - phi_in_new)**2)**0.5>0.01):#	
	time = 0.0
	print dt
	while (ii<10000):
		ii=ii+1
		phi_in=phi_in_new*1.0

		ulx=phi_in.dot(xdelt.T)
		uly=xdelt.dot(phi_in)

		ulx[:,0] = np.sin(5*time)
		uly[0,:] = np.sin(5000*time)*30.0
		ulx[:,0] = np.sin(5000*time)*30.0
		#ulx[:,0] = ulx[:,-1]
		#uly[0,:] = uly[-1,:]
		
		F=ulx.dot(lxx_flux.T)
		Q=lxx_flux.dot(uly)
		Fx=F.dot(hxx_flux.T)
		Qy=hxx_flux.dot(Q)		

		m = phi_in+dt*(Fx+Qy)
		phi_in_new = np.array(m)	
	
		time += dt
		print np.max(phi_in_new)
		print np.min(phi_in_new)

		if contour:						
			fig_width = 10
			fig_height = 10
			textFontSize   = 10
			gcafontSize    = 32
			lineWidth      = 2  
			fig = plt.figure(0,figsize=(fig_width,fig_height))
			ax = fig.add_subplot(111, aspect='equal')		
			c=ax.pcolormesh(GX,GY,phi_in_new,shading='gouraud')		

			cbar= fig.colorbar(c)
			cbar.ax.tick_params(labelsize=gcafontSize)								

			c.set_clim(0, 30)
					 
			cbar.ax.set_ylabel(r'$/phi$', fontsize=1.2*gcafontSize)
			cl = plt.getp(cbar.ax, 'ymajorticklabels')
			plt.setp(cl, fontsize=gcafontSize) 					

			ax.set_xlabel(r'$x$',fontsize=1.2*gcafontSize)
			ax.set_ylabel(r'$y$',fontsize=1.2*gcafontSize)
			#plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
			#plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)	
			ax.set_xticks([])
			ax.set_yticks([])
			fig.tight_layout()
			fig_name = 'what'+str(ii)+'.png'
			figure_path = '../report/figures/'
			fig_fullpath = figure_path + fig_name
			plt.savefig(fig_fullpath)
			plt.close()
			print fig_name+' saved!'		


	if three_d:
		fig = plt.figure()
		ax = Axes3D(fig)

		ax.plot_surface(GX, GY,phi_in_new, rstride=1, cstride=1, cmap=cm.viridis)
		ax.plot_surface(GX, GY,Fx, rstride=1, cstride=1, cmap=cm.viridis)
		#plt.ylim([-2.0, 2.0])
		fig_name ='diffu'+str(ii)+'_2D.png'
		figure_path = '../report/figures/'
		fig_fullpath = figure_path + fig_name
		#if ii ==20:
			#set_trace()
			#plt.show()
		plt.savefig(fig_fullpath)
		plt.close()
		print fig_name+'saved!'
if mesh:
	fig = plt.figure()
	ax = fig.add_subplot(111)	##
	#set_trace()
	#ax.axhline(3,color='k',linestyle='--')
	#ax.axhline(g_mesh[3,0],color='k',linestyle='--')
	for i in range(GX.shape[0]):
		ax.axvline(GY[i,0],color='k',linestyle='-')
		ax.axhline(GX[0,i],color='k',linestyle='-')
	for i in range(LX.shape[0]):
		ax.axvline(LY[i,0],color='r',linestyle='--')
		ax.axhline(LX[0,i],color='r',linestyle='--')
	ax.set_xlim([0.0, 1.0])
	ax.set_ylim([0.0, 1.0])
	plt.show()

if one_D:
	phi_in_new=ug
	phi_in =ug
	ii=0
	dx = xg[1:] - xg[:-1]
	dx_min = np.min(dx)
	#dt=0.002
	dt = 0.5*dx_min
	#while (ii==0 or \
	#       np.sum(np.abs(phi_in - phi_in_new)**2)**0.5>0.01):#	
	time = 0.0

	while (ii<10000):
		ii=ii+1
		phi_in=phi_in_new*1.0
		#set_trace()
		ul=I.dot(phi_in)
		#ul[0]=ul[-1]#periodic
		ul[0] = 10.0*np.sin(0.1*time)
		#set_trace()
		#phi_in_new = splinalg.spsolve(l_flux.dot(I),g*np.zeros(xg.shape), permc_spec=None, use_umfpack=True)
		k1=l_flux.dot(ul)
		m = phi_in-dt*k1
		phi_in_new = np.array(m)	
		time += dt

#	phi_temp_2=k1*dt*0.5+phi_in
#	k2=l_flux.dot(phi_temp_2)
#	phi_temp_3=k2*dt*0.5+phi_in
#	k3=l_flux.dot(phi_temp_3)
#	phi_temp_4=k3*dt+phi_in
#	k4=l_flux.dot(phi_temp_4)
#	m = phi_in+dt*(k1+k2*2.0+k3*2.0+k4)/6.0
#	phi_in_new = np.array(m)
	#set_trace()
	#print phi_in_new
		print 'iteration:',ii, 'time: ', time, 'sin:', np.sin(time)
	#print "phi_new i:s:",phi_in_new
	#print "phi -old is: ", phi_in
	#print 'difference:',np.sum(np.abs(phi_in - phi_in_new)**2)**0.5
	#print 'max_v:',np.max(phi_in_new)
		fig = plt.figure()
		ax = fig.add_subplot(111)	##
		plt.plot(xg,phi_in_new)
		plt.ylim([-2.0, 2.0])
		fig_name =str(ii)+'_1D.png'
		figure_path = '../report/figures/1D_frames/'
		fig_fullpath = figure_path + fig_name
		plt.savefig(fig_fullpath)
		plt.close()
		print fig_name+'saved!'
	
