import os
import sys
import numpy as np 
import scipy as sp 
from pdb import set_trace
from matplotlib import rc as matplotlibrc
import math_tool as ma
import scipy.sparse as scysparse
import bivariate_fit as fit
from scipy.interpolate import griddata
###############################################################	
############  average the value at notes to cv  ##############	
###############################################################	


def Ave_operator(noocv,cvono_in,nno):

	ncv 		= len(noocv)
	nno_int = len(cvono_in)

	An2cv			= scysparse.lil_matrix((ncv,nno),dtype=np.float64)
	Acv2n_int = scysparse.lil_matrix((nno_int,ncv),dtype=np.float64)

	for icv, nolist in enumerate(noocv):
		ele = 1.0/(len(nolist))
		for i, note in enumerate(nolist):
			An2cv[icv,note] = ele*1.0	
	
	for ino, cvlist in enumerate(cvono_in):
		ele = 1.0/(len(cvlist))
		for i, icv in enumerate(cvlist):
			Acv2n_int[ino,icv] = ele*1.0

	return An2cv.tocsr(),Acv2n_int.tocsr()

###############################################################	
############  average the value at notes to fa  ##############	
###############################################################	


def Ave_no2infa(noofa,In_fa,B_no,In_no):

	nno_bc 	= len(B_no)
	nno_int = len(In_no)
	nfa_int = len(In_fa)
	Ano2infa			= scysparse.lil_matrix((nfa_int,nno_int),dtype=np.float64)
	Aqno2infa = scysparse.lil_matrix((nfa_int,nno_bc),dtype=np.float64)

	for ifa, face in enumerate(In_fa):
		temp = list(noofa[face])
		if temp[0] in In_no:
			no0=In_no.index(temp[0])
			Ano2infa[ifa,no0]=0.5
		if temp[1] in In_no:
			no1=In_no.index(temp[1])
			Ano2infa[ifa,no1]=0.5			
		if temp[0] in B_no:
			no0=B_no.index(temp[0])
			Aqno2infa[ifa,no0]=0.5
		if temp[1] in B_no:
			no1=B_no.index(temp[1])
			Aqno2infa[ifa,no1]=0.5

	return Ano2infa,Aqno2infa

###############################################################	
############  average the value at fa to no  ##############	
###############################################################	


def Ave_infa2inno(faono,In_fa,B_fa,In_no):


	nno_int = len(In_no)
	nfa_bc  = len(B_fa)
	nfa_int = len(In_fa)
	Afa2no	= scysparse.lil_matrix((nno_int,nfa_int),dtype=np.float64)
	Aqfa2no = scysparse.lil_matrix((nno_int,nfa_bc),dtype=np.float64)

	for i, ino in enumerate(In_no):
		temp  = list(faono[ino])
		nffa=len(temp)
		weight=1.0/nffa
		for j, iface in enumerate(temp):
			if iface in In_fa:
				ifa=In_fa.index(iface)
				Afa2no[i,ifa]=weight
			if iface in B_fa:
				ifa=B_fa.index(iface)
				Aqfa2no[i,ifa]=weight

	return Afa2no,Aqfa2no
###############################################################	
##############  from notes to cv. divergence  #################
###############################################################	


def Div_operator(faocv,noocv,noofa,xy_no,xy_fa,xy_cv):
	ncv=len(faocv)
	nfa=len(noofa)
	nno=xy_no.shape[0]
	area = np.zeros(ncv)
	nor_fa = np.zeros((nfa,2))
	DIV    = np.zeros(ncv)
	cc=0
	t_fa = np.zeros((nfa,2))
	Dx_n2cv= scysparse.lil_matrix((ncv,nno),dtype=np.float64)
	Dy_n2cv= scysparse.lil_matrix((ncv,nno),dtype=np.float64)

	for icv,falist in enumerate(faocv):	

		xx = {}
		yy = {}
		for i, note in enumerate(noocv[icv]):
			xx[i]=xy_no[note,0]
			yy[i]=xy_no[note,1]
		xx = list(xx.values())
		yy = list(yy.values())
		area[icv] = ma.PolyArea(xx,yy)	

		for i, face in enumerate(falist):
			no0 = noofa[face,0]
			no1 = noofa[face,1]	

			if np.sum(abs(t_fa[face,:])) == 0:
				t_fa[face,:] = xy_no[no0]-xy_no[no1]	

			nor_fa[face,0] = t_fa[face,1]
			nor_fa[face,1] = -t_fa[face,0]
			out_direc = xy_fa[face,:]-xy_cv[icv,:]
			if out_direc.dot(nor_fa[face,:]) < 0 :
				nor_fa[face,:] = -nor_fa[face,:]*1.0	

			Dx_n2cv[icv,no0] = nor_fa[face,0]*0.5+Dx_n2cv[icv,no0]
			Dy_n2cv[icv,no0] = nor_fa[face,1]*0.5+Dy_n2cv[icv,no0]
			Dx_n2cv[icv,no1] = nor_fa[face,0]*0.5+Dx_n2cv[icv,no1]
			Dy_n2cv[icv,no1] = nor_fa[face,1]*0.5+Dy_n2cv[icv,no1]	

		Dx_n2cv[icv,:] = Dx_n2cv[icv,:]/area[icv] 
		Dy_n2cv[icv,:] = Dy_n2cv[icv,:]/area[icv] 

	return Dx_n2cv.tocsr(),Dy_n2cv.tocsr()


###############################################################	
#################  from fa to cv. divergence  #################
###############################################################	

def Div_fa2cv(faocv,noocv,noofa,In_fa,B_fa,xy_no,xy_fa,xy_cv):

	ncv=len(faocv)
	nfa=len(noofa)
	nfa_in=len(In_fa)
	nfa_B =len(B_fa)
	nno=xy_no.shape[0]
	area = np.zeros(ncv)
	nor_fa = np.zeros((nfa,2))
	DIV    = np.zeros(ncv)
	cc=0
	t_fa = np.zeros((nfa,2))
	Dx_fa2cv= scysparse.lil_matrix((ncv,nfa_in),dtype=np.float64)
	Dy_fa2cv= scysparse.lil_matrix((ncv,nfa_in),dtype=np.float64)
	Dqx_fa2cv= scysparse.lil_matrix((ncv,nfa_B),dtype=np.float64)
	Dqy_fa2cv= scysparse.lil_matrix((ncv,nfa_B),dtype=np.float64)
	for icv,falist in enumerate(faocv):	
		if True:	

			xx = {}
			yy = {}
			for i, note in enumerate(noocv[icv]):
				xx[i]=xy_no[note,0]
				yy[i]=xy_no[note,1]
			xx = list(xx.values())
			yy = list(yy.values())
			area[icv] = ma.PolyArea(xx,yy)		

			for i, face in enumerate(falist):
				no0 = noofa[face,0]
				no1 = noofa[face,1]		

				if np.sum(abs(t_fa[face,:])) == 0:
					t_fa[face,:] = xy_no[no0]-xy_no[no1]		

				nor_fa[face,0] = t_fa[face,1]
				nor_fa[face,1] = -t_fa[face,0]
				out_direc = xy_fa[face,:]-xy_cv[icv,:]
				if out_direc.dot(nor_fa[face,:]) < 0 :
					nor_fa[face,:] = -nor_fa[face,:]*1.0		

				if face in In_fa:
					ifa=In_fa.index(face)
					Dx_fa2cv[icv,ifa] = nor_fa[face,0]
					Dy_fa2cv[icv,ifa] = nor_fa[face,1]	
					
				else:
					ifa=B_fa.index(face)
					Dqx_fa2cv[icv,ifa] = nor_fa[face,0]
					Dqy_fa2cv[icv,ifa] = nor_fa[face,1]	
							
		Dx_fa2cv[icv,:]  = Dx_fa2cv[icv,:]/area[icv] 
		Dy_fa2cv[icv,:]  = Dy_fa2cv[icv,:]/area[icv] 
		Dqx_fa2cv[icv,:] = Dqx_fa2cv[icv,:]/area[icv] 
		Dqy_fa2cv[icv,:] = Dqy_fa2cv[icv,:]/area[icv] 

	return Dx_fa2cv.tocsr(),Dy_fa2cv.tocsr(),Dqx_fa2cv.tocsr(),Dqy_fa2cv.tocsr(),area

def Grad_cv2fa(XY_cv2fa,In_fa,ncv):

	nfa_in=len(In_fa)
	if True:
		dx		 = {}
		dy 		 ={}
		Gx_cv2fa= scysparse.lil_matrix((nfa_in,ncv),dtype=np.float64)
		Gy_cv2fa= scysparse.lil_matrix((nfa_in,ncv),dtype=np.float64)
		w={}
		a=np.zeros(nfa_in)
		b=np.zeros(nfa_in)
		c=np.zeros(nfa_in)
		M={}
		N={}
		A={}
		AA={}
		B={}
		BB={}

		for i,ifa in enumerate(In_fa):
			dx[i]= XY_cv2fa[ifa][0:-1,1]-XY_cv2fa[ifa][-1,1]
			dy[i]= XY_cv2fa[ifa][0:-1,2]-XY_cv2fa[ifa][-1,2]
			w[i] = 1/((dx[i]**2+dy[i]**2)**0.5)
			nn=len(dx[i])

			a[i] = np.sum(w[i]**2*(dx[i]**2))
			b[i] = np.sum(w[i]**2*dx[i]*dy[i])
			c[i] = np.sum(w[i]**2*(dy[i]**2))

			M[i] = (w[i]**2)*dx[i]
			N[i] = (w[i]**2)*dy[i]

			Px=c[i]/(a[i]*c[i]-b[i]**2)
			Qx=b[i]/(a[i]*c[i]-b[i]**2)
			Py=-b[i]/(a[i]*c[i]-b[i]**2)
			Qy=-a[i]/(a[i]*c[i]-b[i]**2)

			A[i] = Px*M[i]-Qx*N[i]
			B[i] = Py*M[i]-Qy*N[i]
			d_=np.identity(nn)-np.ones((nn,nn))/nn
			AA[i]=A[i].dot(d_)
			BB[i]=B[i].dot(d_)
			

			for j,icv in enumerate(XY_cv2fa[ifa][0:-1,0]):
				Gx_cv2fa[i,icv] = AA[i][j]
				Gy_cv2fa[i,icv] = BB[i][j]
	return Gx_cv2fa,Gy_cv2fa

def Grad_cv2no(XY_cv2no,In_no,ncv):

	nno_in=len(In_no)
	if True:
		dx		 = {}
		dy 		 ={}
		Gx_cv2no= scysparse.lil_matrix((nno_in,ncv),dtype=np.float64)
		Gy_cv2no= scysparse.lil_matrix((nno_in,ncv),dtype=np.float64)
		w={}
		a=np.zeros(nno_in)
		b=np.zeros(nno_in)
		c=np.zeros(nno_in)
		M={}
		N={}
		A={}
		AA={}
		B={}
		BB={}

		for i,ino in enumerate(In_no):
			dx[i]= XY_cv2no[ino][0:-1,1]-XY_cv2no[ino][-1,1]
			dy[i]= XY_cv2no[ino][0:-1,2]-XY_cv2no[ino][-1,2]
			w[i] = 1/((dx[i]**2+dy[i]**2)**0.5)
			nn=len(dx[i])

			a[i] = np.sum(w[i]**2*(dx[i]**2))
			b[i] = np.sum(w[i]**2*dx[i]*dy[i])
			c[i] = np.sum(w[i]**2*(dy[i]**2))

			M[i] = (w[i]**2)*dx[i]
			N[i] = (w[i]**2)*dy[i]

			Px=c[i]/(a[i]*c[i]-b[i]**2)
			Qx=b[i]/(a[i]*c[i]-b[i]**2)
			Py=-b[i]/(a[i]*c[i]-b[i]**2)
			Qy=-a[i]/(a[i]*c[i]-b[i]**2)

			A[i] = Px*M[i]-Qx*N[i]
			B[i] = Py*M[i]-Qy*N[i]
			d_=np.identity(nn)-np.ones((nn,nn))/nn
			AA[i]=A[i].dot(d_)
			BB[i]=B[i].dot(d_)
			
			for j,icv in enumerate(XY_cv2no[ino][0:-1,0]):
				Gx_cv2no[i,icv] = AA[i][j]
				Gy_cv2no[i,icv] = BB[i][j]
	return Gx_cv2no,Gy_cv2no

def Grad_no2no(xy_sur,In_no,B_no):

	nno_in=len(In_no)
	nno_B=len(B_no)
	if True:
		dx		 = {}
		dy 		 ={}
		Gx_no2no= scysparse.lil_matrix((nno_in,nno_in),dtype=np.float64)
		Gy_no2no= scysparse.lil_matrix((nno_in,nno_in),dtype=np.float64)
		Gxq_no2no= scysparse.lil_matrix((nno_in,nno_B),dtype=np.float64)
		Gyq_no2no= scysparse.lil_matrix((nno_in,nno_B),dtype=np.float64)
		w={}
		a=np.zeros(nno_in)
		b=np.zeros(nno_in)
		c=np.zeros(nno_in)
		M={}
		N={}
		A={}
		AA={}
		B={}
		BB={}
		#set_trace()
		#for i,ifa in enumerate(In_fa):
		for i,ino in enumerate(In_no):
			dx[i]= xy_sur[str(ino)][0:-1,1]-xy_sur[str(ino)][-1,1]
			dy[i]= xy_sur[str(ino)][0:-1,2]-xy_sur[str(ino)][-1,2]
			w[i] = 1/((dx[i]**2+dy[i]**2)**0.5)
			nn=len(dx[i])

			a[i] = np.sum(w[i]**2*(dx[i]**2))
			b[i] = np.sum(w[i]**2*dx[i]*dy[i])
			c[i] = np.sum(w[i]**2*(dy[i]**2))

			M[i] = (w[i]**2)*dx[i]
			N[i] = (w[i]**2)*dy[i]

			Px=c[i]/(a[i]*c[i]-b[i]**2)
			Qx=b[i]/(a[i]*c[i]-b[i]**2)
			Py=-b[i]/(a[i]*c[i]-b[i]**2)
			Qy=-a[i]/(a[i]*c[i]-b[i]**2)

			A[i] = Px*M[i]-Qx*N[i]
			B[i] = Py*M[i]-Qy*N[i]
		
			for j,inno in enumerate(xy_sur[str(ino)][0:-1,0]):
				if inno in In_no:
					in_pos=In_no.index(inno)
					Gx_no2no[i,in_pos] = A[i][j]
					Gy_no2no[i,in_pos] = B[i][j]
					Gx_no2no[i,i] = -np.sum(A[i])
					Gy_no2no[i,i] = -np.sum(B[i])
				else:
					ex_pos=B_no.index(inno)
					Gxq_no2no[i,ex_pos] = A[i][j]
					Gyq_no2no[i,ex_pos] = B[i][j]
					Gx_no2no[i,i] = -np.sum(A[i])
					Gy_no2no[i,i] = -np.sum(B[i])

	return Gx_no2no,Gy_no2no,Gxq_no2no,Gyq_no2no



#Obtain ncx and ncy with the same order of In_no
def Lap_biv_no2no(xy_sur,In_no,B_no,ncx,ncy):
	#Laplas Operator using BiVarPolyFit
	nin_no  = len(In_no)
	nin_bc  = len(B_no)
	Lap_biv = scysparse.lil_matrix((nin_no,nin_no),dtype=np.float64)
	BC_op = scysparse.lil_matrix((nin_no,nin_bc),dtype=np.float64)
	BC_phi = np.zeros(nin_bc)				

	for j,nc in enumerate(In_no):				

		temp = xy_sur[str(nc)]				

		for i,nsur in enumerate(temp[:,0]):
			phi_base = np.zeros(temp.shape[0])
			phi_base[i] = 1.0
			_,_,a = fit.BiVarPolyFit_X(ncx[j],ncy[j],temp[:,1],temp[:,2],phi_base)
			_,_,b = fit.BiVarPolyFit_Y(ncx[j],ncy[j],temp[:,1],temp[:,2],phi_base)
			if (nsur in In_no):
				positon=In_no.index(nsur)
				Lap_biv[j,positon] = a+b
			else:
				ex_pos=B_no.index(nsur)
				BC_op[j,ex_pos] = a+b	
	return Lap_biv, BC_op

#upwind
def Grad_no2no_upwind(xy_sur,In_no,B_no,cx,cy):

	nno_in=len(In_no)
	nno_B=len(B_no)
	if True:
		dx		 = {}
		dy 		 = {}
		nol    = {}
		Gx_no2no= scysparse.lil_matrix((nno_in,nno_in),dtype=np.float64)
		Gy_no2no= scysparse.lil_matrix((nno_in,nno_in),dtype=np.float64)
		Gxq_no2no= scysparse.lil_matrix((nno_in,nno_B),dtype=np.float64)
		Gyq_no2no= scysparse.lil_matrix((nno_in,nno_B),dtype=np.float64)
		w={}
		a=np.zeros(nno_in)
		b=np.zeros(nno_in)
		c=np.zeros(nno_in)
		M={}
		N={}
		A={}
		AA={}
		B={}
		BB={}
		#set_trace()
		#for i,ifa in enumerate(In_fa):
		for i,ino in enumerate(In_no):
			dx[i]= xy_sur[str(ino)][0:-1,1]-xy_sur[str(ino)][-1,1]
			dy[i]= xy_sur[str(ino)][0:-1,2]-xy_sur[str(ino)][-1,2]
			w[i] = 1/((dx[i]**2+dy[i]**2)**0.5)
			nol[i]= xy_sur[str(ino)][0:-1,0]
			#set_trace()
			nn=len(dx[i])
			tempaa={}
			tempbb={}
			tempcc={}		
			tempdd={}
			iii=-1
			for ii, iino in enumerate(xy_sur[str(ino)][0:-1,0]):
				direction=dx[i][ii]*cx[ino]+dy[i][ii]*cy[ino]
				if direction <=0:
					iii=iii+1
					tempaa[str(iii)]=dx[i][ii]
					tempbb[str(iii)]=dy[i][ii]
					tempcc[str(iii)]=w[i][ii]
					tempdd[str(iii)]=nol[i][ii]
			dxx=np.zeros(iii+1)
			dyy=np.zeros(iii+1)
			noll=np.zeros(iii+1)
			for iiii in range(iii+1):
				dxx[iiii]=tempaa[str(iiii)]
				dyy[iiii]=tempbb[str(iiii)]
				noll[iiii]=tempdd[str(iiii)]
			#set_trace()
			dx[i]=dxx
			dy[i]=dyy
			nol[i]=noll
			w[i]=1/((dx[i]**2+dy[i]**2)**0.5)
			nn=len(dx[i])
			#print nn

			a[i] = np.sum(w[i]**2*(dx[i]**2))
			b[i] = np.sum(w[i]**2*dx[i]*dy[i])
			c[i] = np.sum(w[i]**2*(dy[i]**2))

			M[i] = (w[i]**2)*dx[i]
			N[i] = (w[i]**2)*dy[i]

			Px=c[i]/(a[i]*c[i]-b[i]**2)
			Qx=b[i]/(a[i]*c[i]-b[i]**2)
			Py=-b[i]/(a[i]*c[i]-b[i]**2)
			Qy=-a[i]/(a[i]*c[i]-b[i]**2)

			A[i] = Px*M[i]-Qx*N[i]
			B[i] = Py*M[i]-Qy*N[i]

			for j,inno in enumerate(nol[i]):
				if inno in In_no:
					in_pos=In_no.index(inno)
					Gx_no2no[i,in_pos] = A[i][j]
					Gy_no2no[i,in_pos] = B[i][j]
					Gx_no2no[i,i] = -np.sum(A[i])
					Gy_no2no[i,i] = -np.sum(B[i])
				else:
					ex_pos=B_no.index(inno)
					Gxq_no2no[i,ex_pos] = A[i][j]
					Gyq_no2no[i,ex_pos] = B[i][j]
					Gx_no2no[i,i] = -np.sum(A[i])
					Gy_no2no[i,i] = -np.sum(B[i])
			#if ino ==137:
				
				#print ino
				#set_trace()

	return Gx_no2no,Gy_no2no,Gxq_no2no,Gyq_no2no

def Grad_for_Lap(xy_sur,In_no,B_no):

	nno_in=len(In_no)
	nno_B=len(B_no)
	nno=len(xy_sur)
	llist=range(nno)
	if True:
		dx		 = {}
		dy 		 = {}
		nol    = {}
		Gx_no2no= scysparse.lil_matrix((nno,nno),dtype=np.float64)
		Gy_no2no= scysparse.lil_matrix((nno,nno),dtype=np.float64)

		w={}
		a=np.zeros(nno)
		b=np.zeros(nno)
		c=np.zeros(nno)
		M={}
		N={}
		A={}
		AA={}
		B={}
		BB={}
		#set_trace()
		#for i,ifa in enumerate(In_fa):
		for i,ino in enumerate(llist):
			dx[i]= xy_sur[str(ino)][0:-1,1]-xy_sur[str(ino)][-1,1]
			dy[i]= xy_sur[str(ino)][0:-1,2]-xy_sur[str(ino)][-1,2]
			w[i] = 1/((dx[i]**2+dy[i]**2)**0.5)
			nol[i]= xy_sur[str(ino)][0:-1,0]
			#set_trace()
			nn=len(dx[i])

			a[i] = np.sum(w[i]**2*(dx[i]**2))
			b[i] = np.sum(w[i]**2*dx[i]*dy[i])
			c[i] = np.sum(w[i]**2*(dy[i]**2))

			M[i] = (w[i]**2)*dx[i]
			N[i] = (w[i]**2)*dy[i]

			Px=c[i]/(a[i]*c[i]-b[i]**2)
			Qx=b[i]/(a[i]*c[i]-b[i]**2)
			Py=-b[i]/(a[i]*c[i]-b[i]**2)
			Qy=-a[i]/(a[i]*c[i]-b[i]**2)

			A[i] = Px*M[i]-Qx*N[i]
			B[i] = Py*M[i]-Qy*N[i]

			for j,inno in enumerate(nol[i]):

				in_pos=llist.index(inno)
				Gx_no2no[i,in_pos] = A[i][j]
				Gy_no2no[i,in_pos] = B[i][j]
				Gx_no2no[i,i] = -np.sum(A[i])
				Gy_no2no[i,i] = -np.sum(B[i])

			#if ino ==137:
				
				#print ino
				#set_trace()
	return Gx_no2no,Gy_no2no

def Lap_grad(xy_sur,In_no,B_no):
	nno_in=len(In_no)
	nno_B=len(B_no)
	llist=range(len(xy_sur))

	Lap_G= scysparse.lil_matrix((nno_in,nno_in),dtype=np.float64)
	Lap_G_q= scysparse.lil_matrix((nno_in,nno_B),dtype=np.float64)	

	Gx_no2no,Gy_no2no=Grad_for_Lap(xy_sur,In_no,B_no)

	A=Gx_no2no.dot(Gx_no2no)
	B=Gy_no2no.dot(Gy_no2no)
	kk=A+B

#	for i,ino in enumerate(llist):
#		#print i
#		if ino in In_no:
#			for j,inno in enumerate(xy_sur[str(ino)][0:-1,0]):
#				if inno in In_no:
#					in_pos=In_no.index(inno)
#					Lap_G[i,in_pos] = kk[i,inno]
#				else:	
#					ex_pos=B_no.index(inno)
#					#set_trace()
#					Lap_G_q[i,ex_pos] = kk[i,inno]
#		if ino==110:
#			set_trace()
	for i,ino in enumerate(In_no):

		Lap_G[i,0:nno_in]=kk[i,0:nno_in]
		Lap_G_q[i,:]=kk[i,nno_in:]
		#set_trace()

	return Lap_G,Lap_G_q


def Semi_Lagr_upwind(xy_sur,In_no,B_no,cx,cy,dt):

	nno_in=len(In_no)
	nno_B=len(B_no)
	if True:
		dx		 = {}
		dy 		 = {}
		nol    = {}
		w      = {}
		AA= scysparse.lil_matrix((nno_in,nno_in),dtype=np.float64)
		BB= scysparse.lil_matrix((nno_in,nno_B),dtype=np.float64)
		new_pos_x = np.zeros(nno_in)
		new_pos_y = np.zeros(nno_in)

		#set_trace()
		#for i,ifa in enumerate(In_fa):
		for i,ino in enumerate(In_no):
			dx[i]= xy_sur[str(ino)][:,1]-xy_sur[str(ino)][-1,1]
			dy[i]= xy_sur[str(ino)][:,2]-xy_sur[str(ino)][-1,2]
			nol[i]= xy_sur[str(ino)][:,0]
			#set_trace()
			nn=len(dx[i])
			tempaa={}
			tempbb={}	
			tempdd={}
			iii=-1

			for ii, iino in enumerate(xy_sur[str(ino)][:,0]):

				direction=dx[i][ii]*cx[ino]+dy[i][ii]*cy[ino]
				new_pos_x[ino] = dx[i][-1]-cx[ino]*dt
				new_pos_y[ino] = dy[i][-1]-cy[ino]*dt

				if direction <=0:
					iii=iii+1
					tempaa[str(iii)]=dx[i][ii]
					tempbb[str(iii)]=dy[i][ii]
					tempdd[str(iii)]=nol[i][ii]
			dxx=np.zeros(iii+1)
			dyy=np.zeros(iii+1)
			noll=np.zeros(iii+1)			
			for iiii in range(iii+1):
				dxx[iiii]=tempaa[str(iiii)]
				dyy[iiii]=tempbb[str(iiii)]
				noll[iiii]=tempdd[str(iiii)]
			#set_trace()
			dx[i]=dxx
			dy[i]=dyy
			nol[i]=noll
			nn=len(dx[i])
			#griddata(np.vstack((x.flatten(),y.flatten())).T, c.flatten().T,(a,b), method='li').reshape(a.shape)

			for j,inno in enumerate(nol[i]):

				bbase=np.zeros(nol[i].shape)
				bbase[j]=1
				#set_trace()
				#w[i][j]=griddata(np.vstack((dx[i].flatten(),dy[i].flatten())).T, bbase.flatten().T,(new_pos_x[ino],new_pos_y[ino]), method='cubic').reshape(new_pos_x[ino].shape)
				a,_,_ = fit.BiVarPolyFit_X(new_pos_x[ino],new_pos_y[ino],dx[i]-new_pos_x[ino],dy[i]-new_pos_y[ino],bbase)
				#w[i][j]=a
				#set_trace()
				if inno in In_no:
					in_pos=In_no.index(inno)
					AA[i,in_pos] = a

				else:
					ex_pos=B_no.index(inno)
					BB[i,ex_pos] = a

			#if ino ==137:
				
				#print ino
				#set_trace()

	return AA,BB
