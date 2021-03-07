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
import spectrum as sp
#the following enables LaTeX typesetting, which will cause the plotting to take forever..  
matplotlibrc('text.latex', preamble='\usepackage{color}')
matplotlibrc('text',usetex=True)
matplotlibrc('font', family='serif')

Problem1 = True
streamline= False
contour=False
along =True


fig_width = 10
fig_height = 10
textFontSize   = 10
gcafontSize    = 26
lineWidth      = 2  
icemcfd_project_folder = 'mesh_folder/'

####################################################################################
####################################  Problem1  ####################################
####################################################################################
if Problem1:

	Config 	= True	
	P1_2 		= True  # need config


	filename = ['cavity1.msh','cavity2.msh','cavity3.msh']
	NCV=np.zeros(3)
	C_RMS=np.zeros(3)
	Phi_dict={}
	#for kk in range(3):
	kk=2
	Phi_dict={}	
	#if kk ==2:
	for ll in range(3):
		mshfile_fullpath = icemcfd_project_folder + filename[kk]
		part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = \
						umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)

		
		#ll=0
		#if ll==0:
		if True:
			############################################################
			########################    Group   ########################
			############################################################
			B_fa = {}			

			i = 0
			nno = xy_no.shape[0]
			ncv = xy_cv.shape[0]
			nfa = xy_fa.shape[0]
			phi  = np.zeros(nno)			
	
			for ifa, part in enumerate(partofa):
				if part != 'FLUID':
					B_fa[i] = ifa
					i = i+1

			B_fa=list(B_fa.values())			

			B_cv  = {}
			B_no  = {}
			B_noo = {}
			B_cv0 = {}
			B_cvv = {}
			for i, ifa in enumerate(B_fa):
				B_noo[i] = noofa[ifa]
				B_no = list(set(B_no).union(set(B_noo[i])))
				B_cvv[i] = cvofa[ifa]
				B_cv0 = list(set(B_cv0).union(set(B_cvv[i])))			

			B_cv = list(B_cv0) # makes a copy
			# B_cvnew= B_cv # does NOT make a copy, it create a new variable that
			#                 points to the same list.. 
			B_cv.remove(-1)			

			#Obtain noocv 
			noocv = {}
			for icv, face in enumerate(faocv):
				temp  = list(face)
				temp2 = {}
				for i, iface in enumerate(temp):
					temp2 = list(set(temp2).union(set(noofa[iface])))
				noocv[icv] = temp2
			noocv=list(noocv.values())			

			#Obtain cv of inner notes: cvono_in
			In_no = range(nno)
			for i, b_no in enumerate(B_no):
				In_no.remove(b_no)			

			cvono_in = {}
			for i, i_no in enumerate(In_no):
				temp  = list(faono[i_no])
				temp2 = {}
				for i, iface in enumerate(temp):
					temp2 = list(set(temp2).union(set(cvofa[iface])))
				cvono_in[i_no] = temp2
			cvono_in=list(cvono_in.values())	

			In_fa = range(nfa)
			for i, b_fa in enumerate(B_fa):
				In_fa.remove(b_fa)			

			nno_in=len(In_no)	
			nno_B=len(B_no)
			nfa_in=len(In_fa)	
			nfa_B=len(B_fa)
			#Obtain ncx and ncy with the same order of In_no
			#Obtain the surounding note numbers around one note,
			# store in the dictionary Sur_no
			ncx=np.zeros(len(In_no))
			ncy=np.zeros(len(In_no))
			Sur_no ={}	
			lengh=np.zeros(len(In_no))

			for i, ino in enumerate(In_no):			

				Nc = xy_no[ino]
				ncx[i] = Nc[0]
				ncy[i] = Nc[1]			

				sur_no = []
				temp   =list([])
				lengh[i]=len(faono[ino])
				for inoo, ifa in enumerate(faono[ino]):			

					temp = list(noofa[ifa])
					temp.remove(ino)
					sur_no.append(temp[0])			

				Sur_no[str(ino)]=sur_no
				INO= np.argmax(lengh)
					
			#Obtain the coordinate of surounding note group
			#The coordinates of center point is stored at the last line of xy_sur			

			xy_sur={}
			nfaoin_no=np.zeros(len(In_no))			

			for nc,nsurlist in Sur_no.items():			

				temp=np.zeros((len(nsurlist)+1,3))
				temp[len(nsurlist),0]  =int(nc)
				temp[-1,0]  =int(nc)
				temp[-1,1:3]=xy_no[int(nc)]
				for i, nsur in enumerate(nsurlist)  :			

					temp[i,1:3] = xy_no[nsur]
					temp[i,0]= nsur			

				xy_sur[str(nc)]=temp

			#instore all nodes!!!
			#Obtain the surounding note numbers around one note,
			# store in the dictionary SSur_no
			llist=range(nno)
			#set_trace()
			nncx=xy_no[:,0]
			nncy=xy_no[:,1]
			SSur_no ={}	
			lengh=np.zeros(nno)

			for i, ino in enumerate(llist):			
	

				sur_no = []
				temp   =list([])
				lengh[i]=len(faono[ino])
				for inoo, ifa in enumerate(faono[ino]):			

					temp = list(noofa[ifa])
					temp.remove(ino)
					sur_no.append(temp[0])			

				SSur_no[str(ino)]=sur_no
				INO= np.argmax(lengh)
			
					
			#Obtain the coordinate of surounding note group
			#The coordinates of center point is stored at the last line of xy_sur			

			XXYY_sur={}
			nfaoin_no=np.zeros(len(In_no))			

			for nc,nsurlist in SSur_no.items():			

				temp=np.zeros((len(nsurlist)+1,3))
				temp[len(nsurlist),0]  =int(nc)
				temp[-1,0]  =int(nc)
				temp[-1,1:3]=xy_no[int(nc)]
				for i, nsur in enumerate(nsurlist)  :			

					temp[i,1:3] = xy_no[nsur]
					temp[i,0]= nsur			

				XXYY_sur[str(nc)]=temp
			#set_trace()

			cvofa_in={}
			for i,ifa in enumerate(In_fa):
				no1=noofa[ifa][0]
				no2=noofa[ifa][1]
				falist=np.hstack((faono[no1],faono[no2]))
				temp={}
				for j,face in enumerate(falist):
					temp = list(set(temp).union(set(cvofa[face])))
				if ((-1)in temp):
					temp.remove(-1)
				cvofa_in[ifa] = temp
			cvofa_in=list(cvofa_in.values())	

			XY_cv2fa={}
			for ifa,cvlist in enumerate(cvofa_in):
				xy_cv2fa=np.zeros((len(cvlist)+1,4))
				xy_cv2fa[-1,0]=ifa
				xy_cv2fa[-1,1]=xy_fa[ifa,0]
				xy_cv2fa[-1,2]=xy_fa[ifa,1]
				for i,icv in enumerate(cvlist):
					xy_cv2fa[i,0]=icv
					xy_cv2fa[i,1]=xy_cv[icv,0]
					xy_cv2fa[i,2]=xy_cv[icv,1]		
				XY_cv2fa[ifa]=xy_cv2fa			

				XY_cv2no={}
				for ino,cvlist in enumerate(cvono_in):
					xy_cv2no=np.zeros((len(cvlist)+1,3))
					xy_cv2no[-1,0]=ino
					xy_cv2no[-1,1]=xy_no[ino,0]
					xy_cv2no[-1,2]=xy_no[ino,1]
					for i,icv in enumerate(cvlist):
						xy_cv2no[i,0]=icv
						xy_cv2no[i,1]=xy_cv[icv,0]
						xy_cv2no[i,2]=xy_cv[icv,1]		
					XY_cv2no[ino]=xy_cv2no							
			
			############################################################
			####################    configer      ####################
			############################################################

		if Config:

			Ulid = -1.0
			vis  = 100.0

			U_new  = np.zeros(nno_in)
			U_in   = np.zeros(nno_in)
			U_bc   = np.zeros(nno_B)
			Ufa_B = np.zeros(nfa_B)
			U_star  = np.zeros(nno_in)
			#Ufa_in_ = np.zeros(nfa_in)
			Ufa_in_star =np.zeros(nfa_in)
			Uno_wave	= np.zeros(nno_in)

			V_new  = np.zeros(nno_in)
			V_in   = np.zeros(nno_in)
			V_bc   = np.zeros(nno_B)		
			Vfa_in = np.zeros(nfa_in)
			Vfa_B = np.zeros(nfa_B)			
			V_star  = np.zeros(nno_in)
			Vfa_in_star =np.zeros(nfa_in)	
			Vno_wave	= np.zeros(nno_in)

			#phi_in_new = np.zeros(nno_in)
			#phi_in     = np.zeros(nno_in)
			#phi_bc = np.zeros(nno_B)

			for i, ifa in enumerate(B_fa):
				temp = list(noofa[ifa])
				no0=B_no.index(temp[0])
				no1=B_no.index(temp[1])
				nn0=temp[0]
				nn1=temp[1]
				#set_trace()
				if partofa[ifa]=='LID':
					U_bc[no0] = Ulid
					U_bc[no1] = Ulid
					V_bc[no0] = 0
					V_bc[no1] = 0

					Ufa_B[i] = Ulid
					Vfa_B[i] = 0					
				else:
					U_bc[no0] = 0
					U_bc[no1] = 0
					V_bc[no0] = 0
					V_bc[no1] = 0
			Gx_no2no,Gy_no2no,Gxq_no2no,Gyq_no2no = Op.Grad_no2no(xy_sur,In_no,B_no)
			Lap_biv,Lap_biv_q = Op.Lap_grad(XXYY_sur,In_no,B_no)
			Ano2infa,Aqno2infa=Op.Ave_no2infa(noofa,In_fa,B_no,In_no)
			Afa2no,Aqfa2no=Op.Ave_infa2inno(faono,In_fa,B_fa,In_no)
			Dx_fa2cv,Dy_fa2cv,Dqx_fa2cv,Dqy_fa2cv,area=\
							Op.Div_fa2cv(faocv,noocv,noofa,In_fa,B_fa,xy_no,xy_fa,xy_cv)
			Gx_cv2fa,Gy_cv2fa=Op.Grad_cv2fa(XY_cv2fa,In_fa,ncv)
			Gx_cv2no,Gy_cv2no=Op.Grad_cv2no(XY_cv2no,In_no,ncv)				

			AAA=Dx_fa2cv.dot(Gx_cv2fa)+Dy_fa2cv.dot(Gy_cv2fa)		
			BBB=copy.deepcopy(AAA.toarray())
			BBB[36,:]=copy.deepcopy(np.ones(ncv)*1.0)			
			############################################################
			####################    Prob   1.2      ####################
			############################################################

		if P1_2 :
			a  = vis
			tlist=[0.000001/0.13*0.1,0.000001/0.5*0.1,0.000001/1.8*0.1]
			dt = tlist[kk]
			cx = 0
			cy = 0
			area = np.zeros(ncv)
			c_abs = np.zeros(ncv)
			k     = np.zeros(ncv)
			CFL   = np.zeros(ncv)			
			# BC treatment
			qu1_bc = -a*Lap_biv_q.dot(U_bc.T)
			qu2_bc = Gxq_no2no.dot(U_bc.T)*cx+Gyq_no2no.dot(U_bc.T)*cy
			qv1_bc = -a*Lap_biv_q.dot(V_bc.T)
			qv2_bc = Gxq_no2no.dot(V_bc.T)*cx+Gyq_no2no.dot(V_bc.T)*cy			
			qu_bc  = -(qu1_bc+qu2_bc)
			qv_bc  = -(qv1_bc+qv2_bc)
			A = -a*Lap_biv+Gx_no2no*cx+Gy_no2no*cy


			ii=0
			for icv,falist in enumerate(faocv):			
				xx = {}
				yy = {}
				for i, note in enumerate(noocv[icv]):
					xx[i]=xy_no[note,0]
					yy[i]=xy_no[note,1]

				xx = np.array(xx.values())
				yy = np.array(yy.values())
				area[icv] = ma.PolyArea(xx,yy)				

				cx=xx/(yy+2)
				cy=-np.log(yy+2)
				c_abs[icv] = (np.average(cx)**2+np.average(cy)**2)**0.5
				k[icv]=c_abs[icv]/(area[icv]**0.5)

			CFL1=4*a*dt/area
			CFL2=k*dt

			for i in range(ncv):
				if CFL1[i]>=CFL2[i]:
					CFL[i]=CFL1[i]
				else:
					CFL[i]=CFL2[i]
			print 'dt:',dt
			print 'CFLmax:',np.max(CFL)

			#set_trace()
			time2=0
			if True: # First order
				#while(ii<1000000):
					########################################		
					##########  Begin Loop   #############						
					########################################				
				while (np.sum(np.abs(U_new))==0 or \
							 np.sum(np.abs(U_in - U_new)**2)**0.5>10**(-6)):#	
					ii=ii+1
					time2=time2+dt
					U_in=U_new*1.0
					m = qu_bc*dt+(np.eye(len(In_no))-dt*A).dot(U_in)
					U_star = np.array(m)
					U_star = U_star[0,:]

					Ufa_in_star=Ano2infa.dot(U_star)+Aqno2infa.dot(U_bc)

					V_in=V_new*1.0
					m = qu_bc*dt+(np.eye(len(In_no))-dt*A).dot(V_in)
					V_star = np.array(m)
					V_star = V_star[0,:]
					Vfa_in_star=Ano2infa.dot(V_star)+Aqno2infa.dot(V_bc)

					DIV_star=Dx_fa2cv.dot(Ufa_in_star)+Dy_fa2cv.dot(Vfa_in_star)+\
									Dqx_fa2cv.dot(Ufa_B)+Dqy_fa2cv.dot(Vfa_B)
					qq=DIV_star/dt		
			
					qq[36] =0		
					phi_cv=splinalg.spsolve(BBB,qq, permc_spec=None, use_umfpack=True)	
					########################################		
					##########  Three method   #############						
					########################################
					if ll==0:
						Ufa_wave=Ufa_in_star-dt*Gx_cv2fa.dot(phi_cv)
						Vfa_wave=Vfa_in_star-dt*Gy_cv2fa.dot(phi_cv)
						#set_trace()
						U_new=Afa2no.dot(Ufa_wave)+Aqfa2no.dot(Ufa_B)
						V_new=Afa2no.dot(Vfa_wave)+Aqfa2no.dot(Vfa_B)

					if ll==1:
						Uno_in_star=Afa2no.dot(Ufa_in_star)+Aqfa2no.dot(Ufa_B)
						Vno_in_star=Afa2no.dot(Vfa_in_star)+Aqfa2no.dot(Vfa_B)
						Uno_wave=Uno_in_star-dt*Gx_cv2no.dot(phi_cv)
						Vno_wave=Vno_in_star-dt*Gy_cv2no.dot(phi_cv)
						U_new=Uno_wave
						V_new=Vno_wave
					if ll==2:
						Uno_in_star=Afa2no.dot(Ufa_in_star)+Aqfa2no.dot(Ufa_B)
						Vno_in_star=Afa2no.dot(Vfa_in_star)+Aqfa2no.dot(Vfa_B)
						Gx_fano=Afa2no.dot(Gx_cv2fa.dot(phi_cv))+Aqfa2no.dot(np.zeros(Ufa_B.shape))
						Gy_fano=Afa2no.dot(Gy_cv2fa.dot(phi_cv))+Aqfa2no.dot(np.zeros(Ufa_B.shape))
						Uno_wave=Uno_in_star-dt*Gx_fano
						Vno_wave=Vno_in_star-dt*Gy_fano
						U_new=Uno_wave
						V_new=Vno_wave						

					print 'dt:',dt
					print 'CFLmax:',np.max(CFL)					
					print 'iteration:',ii
					print 'difference:',np.sum(np.abs(U_in - U_new)**2)**0.5
					print 'max_v:',np.max(U_new)

					V_phi=np.zeros(nno)	
					for i in range(nno):
						#if (ii==2):
						#		set_trace()							
						if (i in In_no):
							k=In_no.index(i)
							V_phi[i]=V_new[k]
						else:
							k=B_no.index(i)
							V_phi[i]=V_bc[k]	

					U_phi=np.zeros(nno)	
					for i in range(nno):
						#if (ii==2):
						#		set_trace()							
						if (i in In_no):
							k=In_no.index(i)
							U_phi[i]=U_new[k]
						else:
							k=B_no.index(i)
							U_phi[i]=U_bc[k]	
					#VOTX,VOTY=Op.Grad_for_Lap(XXYY_sur,In_no,B_no)
					#w=VOTX.dot(V_phi)-VOTY.dot(U_phi)

			#		#P_phi=np.zeros(nno)	
					#for i in range(nno):
					#	#if (ii==2):
					#	#		set_trace()							
					#	if (i in In_no):
					#		k=In_no.index(i)
					#		P_phi[i]=w[k]
					#	else:
					#		k=B_no.index(i)
					#		P_phi[i]=w[k]	

					print 'max_U',np.max(U_phi)
					print 'min_U',np.min(U_phi)
					print 'max_V',np.max(V_phi)
					print 'min_V',np.min(V_phi)
					#print 'max_w',np.max(P_phi)
					#print 'min_w',np.min(P_phi)
					print '************'					

					mm='u'+str(ll)
					nn='v'+str(ll)

					Phi_dict[mm]=U_phi
					Phi_dict[nn]=V_phi	
		
					#################################################################################
					#################################################################################
					#################################################################################
			#if (ii%100==0):
				if True:					
					if streamline:
						#set_trace()
						x = xy_no[:,0]
						y = xy_no[:,1]#
						phi = P_phi					
						n=500
						xg = np.linspace(x.min(),x.max(),n)
						yg = np.linspace(y.min(),y.max(),n)
						X,Y = np.meshgrid(xg,yg)	
						# interpolate Z values on defined grid
						Un = griddata(np.vstack((x.flatten(),y.flatten())).T, U_phi.flatten().T,(X,Y),method='linear').reshape(X.shape)
						Vn = griddata(np.vstack((x.flatten(),y.flatten())).T, V_phi.flatten().T,(X,Y),method='linear').reshape(X.shape)				

						# Generating the streamline plot
						fig=plt.figure(0,figsize=(fig_width,fig_height))
						ax = fig.add_subplot(111, aspect='equal')	
						c = ax.streamplot(X,Y,Un,Vn)

						ax.set_xlabel(r'$x$',fontsize=1.2*gcafontSize)
						ax.set_ylabel(r'$y$',fontsize=1.2*gcafontSize)
						#plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
						#plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)	
						name='Streamline\_mesh'+str(kk+1)
						ax.set_title(name, fontsize=1.2*gcafontSize)
						ax.set_xticks([])
						ax.set_yticks([])
						fig.tight_layout()
						fig_name = str(kk)+'Streamline.png'
						figure_path = '../report/figures/test/'
						fig_fullpath = figure_path + fig_name
						plt.savefig(fig_fullpath)
						plt.close()
						print fig_name+' saved!'													
					#if (ii==1 or ii%100==0):
					if contour:
						#set_trace()
						x = xy_no[:,0]
						y = xy_no[:,1]#
						phi = V_phi					
						n=50
						xg = np.linspace(x.min(),x.max(),n)
						yg = np.linspace(y.min(),y.max(),n)
						X,Y = np.meshgrid(xg,yg)
						# interpolate Z values on defined grid
						Z = griddata(np.vstack((x.flatten(),y.flatten())).T, phi.flatten().T, \
												(X,Y), method='linear').reshape(X.shape)
						# mask nan values, so they will not appear on plot
						Zm = np.ma.masked_where(np.isnan(Z),Z)	


						fig = plt.figure(0,figsize=(fig_width,fig_height))
						ax = fig.add_subplot(111, aspect='equal')		
						c=ax.pcolormesh(X,Y,Zm,shading='gouraud')
						for inos_of_fa in noofa:
							ax.plot(xy_no[inos_of_fa,0], xy_no[inos_of_fa,1], 'k-', linewidth = 0.2*lineWidth)
						cbar= fig.colorbar(c)
						cbar.ax.tick_params(labelsize=gcafontSize)							

						c.set_clim(-0.35, 0.35)
				 
						cbar.ax.set_ylabel('V(m/s)', fontsize=1.2*gcafontSize)
						cl = plt.getp(cbar.ax, 'ymajorticklabels')
						plt.setp(cl, fontsize=gcafontSize) 				

						ax.set_xlabel(r'$x$',fontsize=1.2*gcafontSize)
						ax.set_ylabel(r'$y$',fontsize=1.2*gcafontSize)
						#plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
						#plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)	
						k='Velocity\_V'
						ax.set_title(k, fontsize=1.2*gcafontSize)
						ax.set_xticks([])
						ax.set_yticks([])
						fig.tight_layout()
						fig_name = 'VelocityV05.png'
						figure_path = '../report/figures/'
						fig_fullpath = figure_path + fig_name
						plt.savefig(fig_fullpath)
						plt.close()
						print fig_name+' saved!'	


						phi = U_phi					
						n=50
						# interpolate Z values on defined grid
						Z = griddata(np.vstack((x.flatten(),y.flatten())).T, phi.flatten().T, \
												(X,Y), method='linear').reshape(X.shape)
						# mask nan values, so they will not appear on plot
						Zm = np.ma.masked_where(np.isnan(Z),Z)	


						fig = plt.figure(0,figsize=(fig_width,fig_height))
						ax = fig.add_subplot(111, aspect='equal')		
						c=ax.pcolormesh(X,Y,Zm,shading='gouraud')
						#plt.contour(c, colors='k')
						#c=ax.pcolormesh(X,Y,Zm,shading='gouraud')	
						for inos_of_fa in noofa:
							ax.plot(xy_no[inos_of_fa,0], xy_no[inos_of_fa,1], 'k-', linewidth = 0.2*lineWidth)
						cbar= fig.colorbar(c)
						cbar.ax.tick_params(labelsize=gcafontSize)							

						c.set_clim(-1, 0.25)
				 
						cbar.ax.set_ylabel('U(m/s)', fontsize=1.2*gcafontSize)
						cl = plt.getp(cbar.ax, 'ymajorticklabels')
						plt.setp(cl, fontsize=gcafontSize) 				

						ax.set_xlabel(r'$x$',fontsize=1.2*gcafontSize)
						ax.set_ylabel(r'$y$',fontsize=1.2*gcafontSize)
						#plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
						#plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)	
						k='Uelocity\_U'
						ax.set_title(k, fontsize=1.2*gcafontSize)
						ax.set_xticks([])
						ax.set_yticks([])
						fig.tight_layout()
						fig_name = 'UelocityU05.png'
						figure_path = '../report/figures/'
						fig_fullpath = figure_path + fig_name
						plt.savefig(fig_fullpath)
						plt.close()
						print fig_name+' saved!'	

						phi = P_phi					
						n=50
						# interpolate Z values on defined grid
						Z = griddata(np.vstack((x.flatten(),y.flatten())).T, phi.flatten().T, \
												(X,Y), method='linear').reshape(X.shape)
						# mask nan values, so they will not appear on plot
						Zm = np.ma.masked_where(np.isnan(Z),Z)	

						if False:	
							fig = plt.figure(0,figsize=(fig_width,fig_height))
							ax = fig.add_subplot(111, aspect='equal')		
							#c = plt.contourf(X,Y,Zm,15)
							#plt.contour(c, colors='k')
							c=ax.pcolormesh(X,Y,Zm,shading='gouraud')	
							for inos_of_fa in noofa:
								ax.plot(xy_no[inos_of_fa,0], xy_no[inos_of_fa,1], 'k-', linewidth = 0.2*lineWidth)
							cbar= fig.colorbar(c)
							cbar.ax.tick_params(labelsize=gcafontSize)							

							c.set_clim(-7, 26)
				 
							cbar.ax.set_ylabel('w(/s)', fontsize=1.2*gcafontSize)
							cl = plt.getp(cbar.ax, 'ymajorticklabels')
							plt.setp(cl, fontsize=gcafontSize) 				

							ax.set_xlabel(r'$x$',fontsize=1.2*gcafontSize)
							ax.set_ylabel(r'$y$',fontsize=1.2*gcafontSize)
							#plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
							#plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)	
							k='Vorticity'
							ax.set_title(k, fontsize=1.2*gcafontSize)
							ax.set_xticks([])
							ax.set_yticks([])
							fig.tight_layout()
							fig_name = 'Vorticity.png'
							figure_path = '../report/figures/'
							fig_fullpath = figure_path + fig_name
							plt.savefig(fig_fullpath)
							plt.close()
							print fig_name+' saved!'	
					
	if along:
					x = xy_no[:,0]
					y = xy_no[:,1]#
						

					n = 20
					xg = np.linspace(0,1,n)
					yg = np.linspace(0,1,n)
					one=np.ones(xg.shape)*0.5			

					ue = [1.0, 0.84123, 0.78871, 0.73722, 0.687171, 0.23151, 0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150,\
							 -0.064634, -0.04775, -0.04192, -0.03717, 0.0]
					ye = [1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, \
							0.0625, 0.0547, 0.0]			

					ve = [0.0, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077,\
							0.12317, 0.10890, 0.10091, 0.09233, 0.0]
					xe = [1.0, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703,\
							0.0625, 0.0]			
					#set_trace()
					# interpolate Z values on defined grid
					Z0 = griddata(np.vstack((x.flatten(),y.flatten())).T, Phi_dict['u0'].flatten().T, \
													(one,yg), method='cubic').reshape(xg.shape)
					Z1 = griddata(np.vstack((x.flatten(),y.flatten())).T, Phi_dict['u1'].flatten().T, \
													(one,yg), method='cubic').reshape(xg.shape)		
					Z2 = griddata(np.vstack((x.flatten(),y.flatten())).T, Phi_dict['u2'].flatten().T, \
													(one,yg), method='cubic').reshape(xg.shape)			
			

					Z00 = griddata(np.vstack((x.flatten(),y.flatten())).T, Phi_dict['v0'].flatten().T, \
													(xg,one), method='cubic').reshape(xg.shape)
					Z11 = griddata(np.vstack((x.flatten(),y.flatten())).T, Phi_dict['v1'].flatten().T, \
													(xg,one), method='cubic').reshape(xg.shape)		
					Z22 = griddata(np.vstack((x.flatten(),y.flatten())).T, Phi_dict['v2'].flatten().T, \
													(xg,one), method='cubic').reshape(xg.shape)


					fig = plt.figure(0,figsize=(fig_width,fig_height))
					ax = fig.add_subplot(111)	
					#aa,=ax.plot(xg,-Z0,'r')
					#bb,=ax.plot(xg,-Z1,'g')
					#cc,=ax.plot(xg,-Z2,'b')
					aa,=ax.plot(-Z0,xg,'ro')
					bb,=ax.plot(-Z1,xg,'g*')
					cc,=ax.plot(-Z2,xg,'b>')					
					dd,=ax.plot(np.array(ue),ye,'k+--')
					#plt.xlabel(r'$ncv$',fontsize=1.5*gcafontSize)
					ax.set_ylim([0,1])
					plt.ylabel(r'$y$',fontsize=1.2*gcafontSize)
					plt.xlabel(r'$U$',fontsize=1.2*gcafontSize)
					plt.legend([aa,bb,cc,dd], ('method\_1','method\_2','method\_3','ref'),loc='best',fontsize=gcafontSize)
					plt.grid(True, which='both')					
					plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
					plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
					ax.set_title('mesh\_3', fontsize=1.2*gcafontSize)
					fig_name ='3vsu.png'
					figure_path = '../report/figures/test/'
					fig_fullpath = figure_path + fig_name
					plt.savefig(fig_fullpath)
					plt.close()
					print fig_name+' saved!'			




					fig = plt.figure(0,figsize=(fig_width,fig_height))
					ax = fig.add_subplot(111)	
					aa,=ax.plot(xg,-Z00,'ro')
					bb,=ax.plot(xg,-Z11,'g*')
					cc,=ax.plot(xg,-Z22,'b>')
					dd,=ax.plot(np.array(xe),np.array(ve),'k+--')
					#plt.xlabel(r'$ncv$',fontsize=1.5*gcafontSize)
					ax.set_xlim([0,1])
					plt.ylabel(r'$V$',fontsize=1.2*gcafontSize)
					plt.xlabel(r'$x$',fontsize=1.2*gcafontSize)
					plt.legend([aa,bb,cc,dd], ('method\_1','method\_2','method\_3','ref'),loc='best',fontsize=gcafontSize)
					plt.grid(True, which='both')					
					plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
					plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
					ax.set_title('mesh\_3', fontsize=1.2*gcafontSize)
					fig_name ='3vsv.png'
					figure_path = '../report/figures/test/'
					fig_fullpath = figure_path + fig_name
					plt.savefig(fig_fullpath)
					plt.close()
					print fig_name+' saved!'