# General analyses of GBM catalog bursts 
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import ticker
plt.style.use('seaborn')
import seaborn as sns
#sns.set(style='whitegrid')
from astropy.io import fits
from astropy.time import Time
from astropy.stats import bayesian_blocks
from astropy.stats import sigma_clip, mad_std
import operator
from glob import glob
import pandas as pd
import numpy as np
import h5py
from scipy import stats
import os
import sys
from multiprocessing import Pool
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)
import rpy2.robjects as robjects
from rpy2.robjects import r
import rpy2.robjects.numpy2ri
robjects.numpy2ri.activate()
robjects.r("library(baseline)")
from xspec import *
from personal_settings import *


#databasedir='/home/lang/work/GBM/burstdownload/data/'
#databasedir='/diskb/Database/Fermi/gbm_burst/data/'
databasedir=get_databasedir()
NaI=['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb']
BGO=['b0','b1']
Det=['b0','b1','n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb']
#data in first and last three channels of BGO and NaI are used
#ignore 0,1,2,125,126,127, notice 3-124
ch1=3
ch2=124
ncore=get_ncore()
#ncore=2
##################
# SOME FUNCTIONS #
##################

# https://en.wikipedia.org/wiki/Normal_distribution
# https://en.wikipedia.org/wiki/Poisson_distribution
# cdfprob=0.997300203937 # 3 sigma
# cdfprob=0.954499736104 # 2 sigma
# cdfprob=0.682689492137 # 1 sigma
def norm_pvalue(sigma=2.0):
	p = stats.norm.cdf(sigma)-stats.norm.cdf(-sigma)
	return p


def query_fermigbrst(cdir='./'):
	fermigbrst=cdir+'/fermigbrst.txt'
	if not os.path.exists(fermigbrst):
		usersjar="/home/lang/Software/HEASARC-Xamin/users.jar"
		assert os.path.exists(usersjar), "'users.jar' is not available!"+\
			"\n download users.jar at:"+\
			"\n https://heasarc.gsfc.nasa.gov/xamin/distrib/users.jar"+\
			"\n and fix the path of usersjar."
		java_ready=os.system("java --version")
		assert not java_ready, "java not properly installed!"+\
			"\n Install Oracle Java 10 (JDK 10) in Ubuntu or Linux Mint from PPA"+\
			"\n sudo add-apt-repository ppa:linuxuprising/java"+\
			"\n sudo apt update"+\
			"\n sudo apt install oracle-java10-installer"
		fields="trigger_name,t90,t90_error,t90_start,bcat_detector_mask,"+\
			"duration_energy_low,duration_energy_high,"+\
			"back_interval_low_start,back_interval_low_stop,"+\
			"back_interval_high_start,back_interval_high_stop,"+\
			"flnc_band_epeak,flnc_band_epeak_pos_err,flnc_band_epeak_neg_err,"+\
			"flnc_band_alpha,flnc_band_alpha_pos_err,flnc_band_alpha_neg_err,"+\
			"flnc_band_beta,flnc_band_beta_pos_err,flnc_band_beta_neg_err,"+\
			"flnc_spectrum_start,flnc_spectrum_stop,scat_detector_mask,"+\
			"pflx_spectrum_start,pflx_spectrum_stop,"+\
			"pflx_band_epeak,pflx_band_epeak_pos_err,pflx_band_epeak_neg_err,"+\
			"pflx_band_alpha,pflx_band_alpha_pos_err,pflx_band_alpha_neg_err,"+\
			"pflx_band_beta,pflx_band_beta_pos_err,pflx_band_beta_neg_err"
		print('querying fermigbrst catalog using HEASARC-Xamin-users.jar (Java)...')
		query_ready=os.system("java -jar "+usersjar+" table=fermigbrst \
				fields="+fields+" sortvar=trigger_name output="+\
				cdir+"/fermigbrst.txt")
		assert not query_ready, 'failed in querying fermigbrst catalog!'
		print('successful in querying fermigbrst catalog!')
	return fermigbrst


#def format_countmap_axes(ax, title, x1, x2,ymajor_ticks,yminor_ticks):
def format_countmap_axes(ax, title, x1, x2,ymajor_ticks):
	ax.set_title(title,loc='right',fontsize=25,color='k')
	ax.set_xlim([x1,x2])
	ax.set_yscale('log')
	ax.yaxis.set_major_locator(ticker.FixedLocator(ymajor_ticks))
	#ax.yaxis.set_minor_locator(ticker.FixedLocator(yminor_ticks))
	#ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
	#ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%d'))
	ax.tick_params(axis='both',top=True,right=True,length=5,width=2,direction='out',which='both',labelsize=25)

def plot_countmap(bnname,resultdir,baseresultdir,datadir,content,tbins,viewt1,viewt2): 
	# content=['rate','base','net']
	BGOmaxcolorvalue=0.0
	NaImaxcolorvalue=0.0
	f=h5py.File(baseresultdir+'/base.h5',mode='r')
	fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=False)
	for i in range(14):
		#data in firt and last two channels of BGO and NaI are not shown
		#ignore 0,1,126,127, notice 2-125
		if content=='rate':
			C=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][0] for ch in np.arange(ch1,ch2+1) ])
		elif content=='base':
			C=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][1] for ch in np.arange(ch1,ch2+1) ])
		elif content=='net':
			C=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] for ch in np.arange(ch1,ch2+1) ])
		if i<=1:
			if BGOmaxcolorvalue < C.max():
				BGOmaxcolorvalue = C.max()
		else:
			if NaImaxcolorvalue < C.max():
				NaImaxcolorvalue = C.max()	
	for i in range(14):
		ttefile=glob(datadir+'/'+'glg_tte_'+Det[i]+'_'+bnname+'_v*.fit')
		hdu=fits.open(ttefile[0])
		ebound=hdu['EBOUNDS'].data
		emin=ebound.field(1)
		emin=emin[ch1:ch2+1]
		emax=ebound.field(2)
		emax=emax[ch1:ch2+1]				
		x = tbins
		y = np.concatenate((emin,[emax[-1]]))
		X, Y = np.meshgrid(x, y)
		if content=='rate':
			C=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][0] for ch in np.arange(ch1,ch2+1) ])
		elif content=='base':
			C=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][1] for ch in np.arange(ch1,ch2+1) ])
		elif content=='net':
			C=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] for ch in np.arange(ch1,ch2+1) ])
		C[C<1]=1
		if i<=1:
			pcmBGO = axes[i//2,i%2].pcolormesh(X, Y, C,\
				norm=colors.LogNorm(vmin=1.0, vmax=BGOmaxcolorvalue),cmap='rainbow')
			format_countmap_axes(axes[i//2,i%2],Det[i],tbins[0],tbins[-1],[1000,10000])
		else:
			pcmNaI = axes[i//2,i%2].pcolormesh(X, Y, C,\
				norm=colors.LogNorm(vmin=1.0, vmax=NaImaxcolorvalue),cmap='rainbow')
			format_countmap_axes(axes[i//2,i%2],Det[i],tbins[0],tbins[-1],[10,100])
		axes[i//2,i%2].set_xlim([viewt1,viewt2])				
	cbarBGO=fig.colorbar(pcmBGO, ax=axes[0,], orientation='vertical',fraction=0.005, aspect=100/6)
	cbarNaI=fig.colorbar(pcmNaI, ax=axes[1:,], orientation='vertical',fraction=0.005, aspect=100)
	cbarBGO.ax.tick_params(labelsize=25)
	cbarNaI.ax.tick_params(labelsize=25)
	fig.text(0.07, 0.5, 'Energy (KeV)', ha='center', va='center', rotation='vertical',fontsize=30)
	fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)
	fig.text(0.5, 0.92, bnname, ha='center', va='center',fontsize=30)
	plt.savefig(resultdir+'/'+content+'_countmap.png')
	plt.close()
	f.close()

#function for writing single spectrum (PHA I) into a fit file 
def write_phaI(spectrum_rate,bnname,det,t1,t2,outfile):
	#hdu0 / PrimaryHDU
	header0=fits.Header()
	header0.append(('creator', 'Shao', 'The name who created this PHA file'))
	header0.append(('telescop', 'Fermi', 'Name of mission/satellite'))
	header0.append(('bnname', bnname, 'Burst Name'))
	header0.append(('t1', t1, 'Start time of the PHA slice'))
	header0.append(('t2', t2, 'End time of the PHA slice'))
	hdu0=fits.PrimaryHDU(header=header0)
	#hdu1 / data unit
	a1 = np.arange(128)
	col1 = fits.Column(name='CHANNEL', format='1I', array=a1)
	col2 = fits.Column(name='COUNTS', format='1D', unit='COUNTS', array=spectrum_rate)
	#col3 = fits.Column(name='STAT_ERR', format='1D', unit='COUNTS', array=bkg_uncertainty)
	#hdu = fits.BinTableHDU.from_columns([col1, col2, col3])
	hdu1 = fits.BinTableHDU.from_columns([col1, col2])
	header=hdu1.header
	header.append(('extname', 'SPECTRUM', 'Name of this binary table extension'))
	header.append(('telescop', 'GLAST', 'Name of mission/satellite'))
	header.append(('instrume', 'GBM', 'Specific instrument used for observation'))
	header.append(('filter', 'None', 'The instrument filter in use (if any)'))
	header.append(('exposure', 1., 'Integration time in seconds'))
	header.append(('areascal', 1., 'Area scaling factor'))
	header.append(('backscal', 1., 'Background scaling factor'))
	if outfile[-3:]=='pha':
		header.append(('backfile', det+'.bkg', 'Name of corresponding background file (if any)'))
		header.append(('respfile', det+'.rsp', 'Name of corresponding RMF file (if any)'))
	else:
		header.append(('backfile', 'none', 'Name of corresponding background file (if any)'))
		header.append(('respfile', 'none', 'Name of corresponding RMF file (if any)'))
	header.append(('corrfile', 'none', 'Name of corresponding correction file (if any)'))
	header.append(('corrscal', 1., 'Correction scaling file'))
	header.append(('ancrfile', 'none', 'Name of corresponding ARF file (if any)'))
	header.append(('hduclass', 'OGIP', 'Format conforms to OGIP standard'))
	header.append(('hduclas1', 'SPECTRUM', 'PHA dataset (OGIP memo OGIP-92-007)'))
	header.append(('hduclas2', 'TOTAL', 'Indicates gross data (source + background)'))
	header.append(('hduclas3', 'COUNT', 'Indicates data stored as counts'))
	header.append(('hduvers', '1.2.1', 'Version of HDUCLAS1 format in use'))
	header.append(('poisserr', True, 'Use Poisson Errors (T) or use STAT_ERR (F)'))
	header.append(('chantype', 'PHA', 'No corrections have been applied'))
	header.append(('detchans', 128, 'Total number of channels in each rate'))
	header.append(('hduclas4', 'TYPEI', 'PHA Type I (single) or II (mulitple spectra)'))
	header.comments['TTYPE1']='Label for column 1'
	header.comments['TFORM1']='2-byte INTERGER'
	header.comments['TTYPE2']='Label for column 2'
	header.comments['TFORM2']='8-byte DOUBLE'
	header.comments['TUNIT2']='Unit for colum 2'
	#header.comments['TTYPE3']='Label for column 3'
	#header.comments['TFORM3']='8-byte DOUBLE'
	#header.comments['TUNIT3']='Unit for colum 3'
	hdul = fits.HDUList([hdu0, hdu1])
	hdul.writeto(outfile)



def copy_rspI(bnname,det,outfile):
	shortyear=bnname[2:4]
	fullyear='20'+shortyear
	datadir=databasedir+'/'+fullyear+'/'+bnname+'/'
	rspfile=glob(datadir+'/'+'glg_cspec_'+det+'_'+bnname+'_v*.rsp')
	assert len(rspfile)==1, 'response file is missing for '+'glg_cspec_'+det+'_'+bnname+'_v*.rsp'
	rspfile=rspfile[0]
	os.system('cp '+rspfile+' '+outfile)
	
	
########################	
# BEGIN base class GRB #
########################

class GRB:
	def __init__(self,bnname):
		self.bnname=bnname
		resultdir=os.getcwd()+'/results/'
		self.resultdir=resultdir+'/'+bnname+'/'
		shortyear=self.bnname[2:4]
		fullyear='20'+shortyear
		self.datadir=databasedir+'/'+fullyear+'/'+self.bnname+'/'
		self.dataready=True
		for i in range(14):
			ttefile=glob(self.datadir+'glg_tte_'+Det[i]+'_'+self.bnname+'_v*.fit')
			if not len(ttefile)==1:
				self.dataready=False
			else:
				hdu=fits.open(ttefile[0])
				event=hdu['EVENTS'].data.field(0)
				if len(event)<10:
					self.dataready=False
		if self.dataready:
			if not os.path.exists(resultdir):
				os.makedirs(resultdir)
			if not os.path.exists(self.resultdir):
				os.makedirs(self.resultdir)
			self.baseresultdir=self.resultdir+'/base/'
			self.phaIresultdir=self.resultdir+'/phaI/'

			# determine GTI1 and GTI2
			GTI_t1=np.zeros(14)
			GTI_t2=np.zeros(14)
			for i in range(14):
				ttefile=glob(self.datadir+'glg_tte_'+Det[i]+'_'+self.bnname+'_v*.fit')
				hdu=fits.open(ttefile[0])
				trigtime=hdu['Primary'].header['TRIGTIME']
				data=hdu['EVENTS'].data
				time=data.field(0)-trigtime
				GTI0_t1=time[0]
				GTI0_t2=time[-1]
				timeseq1=time[:-1]
				timeseq2=time[1:]
				deltime=timeseq2-timeseq1
				# find a gap larger than 5 second between two events 
				delindex=deltime>5 
				if len(timeseq1[delindex])>=1:
					GTItmp_t1=np.array(np.append([GTI0_t1],timeseq2[delindex]))
					GTItmp_t2=np.array(np.append(timeseq1[delindex],[GTI0_t2]))
					for kk in np.arange(len(GTItmp_t1)):
						if GTItmp_t1[kk]<=0.0 and GTItmp_t2[kk]>=0.0:
							GTI_t1[i]=GTItmp_t1[kk]
							GTI_t2[i]=GTItmp_t2[kk]
				else:
					GTI_t1[i]=GTI0_t1
					GTI_t2[i]=GTI0_t2
			self.GTI1=np.max(GTI_t1)
			self.GTI2=np.min(GTI_t2)

	def rawlc(self,viewt1=-50,viewt2=300,binwidth=0.064):		
		viewt1=np.max([self.GTI1,viewt1])
		viewt2=np.min([self.GTI2,viewt2])
		assert viewt1<viewt2, self.bnname+': Inappropriate view times for rawlc!'
		if not os.path.exists(self.resultdir+'/'+'raw_lc.png'):
			#print('plotting raw_lc.png ...')
			tbins=np.arange(viewt1,viewt2+binwidth,binwidth)
			fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=False)
			for i in range(14):
				ttefile=glob(self.datadir+'/'+'glg_tte_'+Det[i]+'_'+self.bnname+'_v*.fit')
				hdu=fits.open(ttefile[0])
				trigtime=hdu['Primary'].header['TRIGTIME']
				data=hdu['EVENTS'].data
				time=data.field(0)-trigtime
				ch=data.field(1)
				#data in firt and last two channels of BGO and NaI are not used
				#ignore 0,1,2,125,126,127, notice 3-124
				goodindex=(ch>=ch1) & (ch<=ch2)  
				time=time[goodindex]
				ebound=hdu['EBOUNDS'].data
				emin=ebound.field(1)
				emin=emin[ch1:ch2+1]
				emax=ebound.field(2)
				emax=emax[ch1:ch2+1]
				histvalue, histbin =np.histogram(time,bins=tbins)
				plotrate=histvalue/binwidth
				plotrate=np.concatenate(([plotrate[0]],plotrate))
				axes[i//2,i%2].plot(histbin,plotrate,linestyle='steps')
				axes[i//2,i%2].set_xlim([viewt1,viewt2])
				axes[i//2,i%2].tick_params(labelsize=25)
				axes[i//2,i%2].text(0.05,0.85,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
				axes[i//2,i%2].text(0.7,0.80,str(round(emin[0],1))+'-'+str(round(emax[-1],1))+' keV',\
					transform=axes[i//2,i%2].transAxes,fontsize=25)
			fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center', va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)		
			fig.text(0.5, 0.92, self.bnname, ha='center', va='center',fontsize=30)
			plt.savefig(self.resultdir+'/raw_lc.png')
			plt.close()

	def base(self,baset1=-50,baset2=300,binwidth=0.064):
		self.baset1=np.max([self.GTI1,baset1])
		self.baset2=np.min([self.GTI2,baset2])
		self.binwidth=binwidth
		self.tbins=np.arange(self.baset1,self.baset2+self.binwidth,self.binwidth)
		assert self.baset1<self.baset2, self.bnname+': Inappropriate base times!'
		if not os.path.exists(self.baseresultdir):
			#print('creating baseline in '+self.baseresultdir+' ...')
			os.makedirs(self.baseresultdir)
			f=h5py.File(self.baseresultdir+'/base.h5',mode='w')
			for i in range(14):
				grp=f.create_group(Det[i])
				ttefile=glob(self.datadir+'/'+'glg_tte_'+Det[i]+'_'+self.bnname+'_v*.fit')
				hdu=fits.open(ttefile[0])	
				trigtime=hdu['Primary'].header['TRIGTIME']
				data=hdu['EVENTS'].data
				timedata=data.field(0)-trigtime
				chdata=data.field(1)
				for ch in range(128):
					time_selected=timedata[chdata==ch]
					histvalue, histbin=np.histogram(time_selected,bins=self.tbins)
					rate=histvalue/binwidth
					r.assign('rrate',rate) 
					r("y=matrix(rrate,nrow=1)")
					#fillPeak_hwi=str(int(40/binwidth))
					#fillPeak_int=str(int(len(rate)/10))
					#fillPeak_lam=str(int(0.708711*(len(rate))**0.28228+0.27114))
					#r("rbase=baseline(y,lam = "+fillPeak_lam+", hwi="\
					#					+fillPeak_hwi+", it=10, int ="\
					#					+fillPeak_int+", method='fillPeaks')")
					fillPeak_hwi=str(int(5/binwidth))
					fillPeak_int=str(int(len(rate)/10))
					r("rbase=baseline(y,lam = 6, hwi="+fillPeak_hwi+", it=10,int ="+fillPeak_int+", method='fillPeaks')")
					r("bs=getBaseline(rbase)")
					r("cs=getCorrected(rbase)")
					bs=r('bs')[0]
					cs=r('cs')[0]
					# correct negative base to 0 and recover the net value to original rate
					corrections_index= (bs<0)
					bs[corrections_index]=0
					cs[corrections_index]=rate[corrections_index]
					f['/'+Det[i]+'/ch'+str(ch)]=np.array([rate,bs,cs])
			f.flush()
			f.close()
	
	def plotbase(self):
		self.plotbasedir=self.resultdir+'/plotbase/'
		if not os.path.exists(self.plotbasedir):
			assert os.path.exists(self.baseresultdir), 'Should have run base() before running plotbase()!'
			os.makedirs(self.plotbasedir)
			f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
			for ch in range(128):
				fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=False)
				for i in range(14):
					plotrate=f['/'+Det[i]+'/ch'+str(ch)][()][0] #rate
					plotrate=np.concatenate(([plotrate[0]],plotrate))
					axes[i//2,i%2].plot(self.tbins,plotrate,linestyle='steps',lw=3.0,color='tab:blue')
					plotbase=f['/'+Det[i]+'/ch'+str(ch)][()][1] #base
					plottime=self.tbins[:-1]+self.binwidth/2.0
					axes[i//2,i%2].plot(plottime,plotbase,linestyle='--',lw=4.0,color='tab:orange')
					axes[i//2,i%2].set_xlim([self.tbins[0],self.tbins[-1]])
					axes[i//2,i%2].tick_params(labelsize=25)
					axes[i//2,i%2].text(0.05,0.85,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
				fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center', va='center',rotation='vertical',fontsize=30)
				fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)		
				fig.text(0.5, 0.92, 'ch'+str(ch), ha='center', va='center',fontsize=30)			
				plt.savefig(self.plotbasedir+'/ch_'+str(ch)+'.png')
				plt.close()
			f.close()
		

	def check_gaussian_total_rate(self):
		if not os.path.exists(self.resultdir+'/check_gaussian_total_rate.png'):
			assert os.path.exists(self.baseresultdir), 'Should have run base() before running check_gaussian_total_rate()!'
			f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
			fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=False,sharey=False)
			for i in range(14):
				cRate=np.array([f['/'+Det[i]+'/ch'+str(ch)][()][0] for ch in np.arange(ch1,ch2+1) ])
				totalRate=np.sum(cRate,axis=0)
				median=np.median(totalRate)
				totalRate_median_part=totalRate[(totalRate>(0.1*median)) & (totalRate<(1.5*median))]
				bins=np.arange(totalRate.min(),totalRate.max(),(totalRate_median_part.max()-totalRate_median_part.min())/30)
				histvalue, histbin =np.histogram(totalRate,bins=bins)
				histvalue=np.concatenate(([histvalue[0]],histvalue))
				axes[i//2,i%2].fill_between(histbin,histvalue,step='pre',label='Observed total rate')			
				loc,scale=stats.norm.fit(totalRate_median_part)
				Y=stats.norm(loc=loc,scale=scale)
				x=np.linspace(totalRate_median_part.min(),totalRate_median_part.max(),num=100)
				axes[i//2,i%2].plot(x,Y.pdf(x)*totalRate.size*(bins[1]-bins[0]),\
								label='Gaussian Distribution',\
								linestyle='--',lw=3.0,color='tab:orange')
				axes[i//2,i%2].tick_params(labelsize=25)
				axes[i//2,i%2].text(0.5,0.8,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
				axes[i//2,i%2].axvline(totalRate_median_part.min(),ls='--',lw=1,color='k',label='Fitting region')
				axes[i//2,i%2].axvline(totalRate_median_part.max(),ls='--',lw=1,color='k')
				if i==1:
					axes[i//2,i%2].legend(fontsize=20)
			fig.text(0.07, 0.5, 'Numbers', ha='center', va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Total raw rate (s$^{-1}$; between '+str(self.baset1)+'--'+str(self.baset2)+'s)',\
			    					 ha='center', va='center',fontsize=30)		
			plt.savefig(self.resultdir+'/check_gaussian_total_rate.png')
			plt.close()
			f.close()

	def check_gaussian_net_rate(self,sigma=3):
		if not os.path.exists(self.resultdir+'/check_gaussian_net_rate.png'):
			assert os.path.exists(self.baseresultdir), 'Should have run base() before running check_gaussian_net_rate()!'
			f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
			fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=False,sharey=False)
			for i in range(14):
				cRate=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] for ch in np.arange(ch1,ch2+1) ])
				netRate=np.sum(cRate,axis=0)
				median=np.median(netRate)
				netRate_median_part=netRate[netRate<5*median]
				bins=np.arange(netRate.min(),netRate.max(),(netRate_median_part.max()-netRate_median_part.min())/30)
				histvalue, histbin =np.histogram(netRate,bins=bins)
				histvalue=np.concatenate(([histvalue[0]],histvalue))
				axes[i//2,i%2].fill_between(histbin,histvalue,step='pre',label='Observed net rate')
				loc,scale=stats.norm.fit(netRate_median_part)
				Y=stats.norm(loc=loc,scale=scale)
				x=np.linspace(netRate_median_part.min(),netRate_median_part.max(),num=100)
				axes[i//2,i%2].plot(x,Y.pdf(x)*netRate.size*(bins[1]-bins[0]),\
								label='Gaussian Distribution',\
								linestyle='--',lw=3.0,color='tab:orange')
				axes[i//2,i%2].tick_params(labelsize=25)
				axes[i//2,i%2].text(0.5,0.8,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
				gaussian_level=Y.interval(norm_pvalue(sigma))
				axes[i//2,i%2].axvline(gaussian_level[0],ls='--',lw=2,color='green',label=str(sigma)+'$\sigma$ level')
				axes[i//2,i%2].axvline(gaussian_level[1],ls='--',lw=2,color='green')
				if i==1:
					axes[i//2,i%2].legend(fontsize=20)
			fig.text(0.07, 0.5, 'Numbers', ha='center', va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Total net rate (s$^{-1}$; between '+str(self.baset1)+'--'+str(self.baset2)+'s)',\
				ha='center', va='center',fontsize=30)		
			plt.savefig(self.resultdir+'/check_gaussian_net_rate.png')
			plt.close()
			f.close()

	def plot_gaussian_level_over_net_lc(self,viewt1=-50,viewt2=300,sigma=3):
		if not os.path.exists(self.resultdir+'/gaussian_level_over_net_lc.png'):
			assert os.path.exists(self.baseresultdir), 'Should have run base() before running plot_gaussian_level_over_net_lc()!'
			viewt1=np.max([self.baset1,viewt1])
			viewt2=np.min([self.baset2,viewt2])
			f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
			fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=False)
			ylim=np.zeros((14,2))
			for i in range(14):
				cNet=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] for ch in np.arange(ch1,ch2+1) ])
				totalNet=np.sum(cNet,axis=0)
				median=np.median(totalNet)
				totalNet_median_part=totalNet[totalNet<5*median]
				loc,scale=stats.norm.fit(totalNet_median_part)
				Y=stats.norm(loc=loc,scale=scale)
				gaussian_level=Y.interval(norm_pvalue(sigma))
				totalNet=np.concatenate(([totalNet[0]],totalNet))
				axes[i//2,i%2].plot(self.tbins,totalNet,linestyle='steps',lw=3.0,color='tab:blue')
				axes[i//2,i%2].axhline(gaussian_level[1],ls='--',lw=3,\
					color='orange',label=str(sigma)+'$\sigma$ level of gaussian background')
				axes[i//2,i%2].tick_params(labelsize=25)
				axes[i//2,i%2].set_xlim([viewt1,viewt2])
				ylim[i]=axes[i//2,i%2].get_ylim()
				axes[i//2,i%2].text(0.05,0.85,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
				if i==1:
					axes[i//2,i%2].legend(fontsize=20)
			#reset the ylims to same values
			BGOymax=np.max([ylim[i,1] for i in range(2)])
			NaIymax=np.max([ylim[i+2,1] for i in range(12)])														
			for i in range(14):
				if i<=1:
					axes[i//2,i%2].set_ylim([0,BGOymax])
				else:
					axes[i//2,i%2].set_ylim([0,NaIymax])
			fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center', va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)		
			fig.text(0.5, 0.92, self.bnname, ha='center', va='center',fontsize=30)		
			plt.savefig(self.resultdir+'/gaussian_level_over_net_lc.png')
			plt.close()
			f.close()
			
			
# check SNR
	def check_snr(self,viewt1=-50,viewt2=300):
		if not os.path.exists(self.resultdir+'/check_SNR.png'):
			assert os.path.exists(self.baseresultdir), 'Should have run base() before running check_snr()!'
			viewt1=np.max([self.baset1,viewt1])
			viewt2=np.min([self.baset2,viewt2])
			f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
			fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=False)
			ylim=np.zeros((14,2))
			for i in range(14):
				cNet=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] for ch in np.arange(ch1,ch2+1) ])
				totalNet=np.sum(cNet,axis=0)
				median=np.median(totalNet)
				totalNet_median_part=totalNet[totalNet<5*median]
				loc,scale=stats.norm.fit(totalNet_median_part)
				#Y=stats.norm(loc=loc,scale=scale)
				#gaussian_level=Y.interval(norm_pvalue(sigma))
				totalNet=np.concatenate(([totalNet[0]],totalNet))
				snr=(totalNet-loc)/scale
				axes[i//2,i%2].plot(self.tbins,snr,linestyle='steps',lw=3.0,color='tab:blue')
				#axes[i//2,i%2].axhline(gaussian_level[1],ls='--',lw=3,\
				#	color='orange',label=str(sigma)+'$\sigma$ level of gaussian background')
				axes[i//2,i%2].tick_params(labelsize=25)
				axes[i//2,i%2].set_xlim([viewt1,viewt2])
				axes[i//2,i%2].text(0.05,0.85,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
				axes[i//2,i%2].axhline(y=3,color='orange',ls='--',lw=3,zorder=2,label='SNR=3')
				for t in self.tbins[snr>3]:
					axes[i//2,i%2].axvline(x=t,ymin=0.95,color='red',zorder=2)
				if i==1:
					axes[i//2,i%2].legend(fontsize=20)
			fig.text(0.07, 0.5, 'Signal-to-noise ratio', ha='center', va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)		
			fig.text(0.5, 0.92, self.bnname, ha='center', va='center',fontsize=30)			
			plt.savefig(self.resultdir+'/check_SNR.png')
			plt.close()
			f.close()			
			

# check pulse based on plot_gauss_level_over_net_lc above
	def check_pulse(self,viewt1=-50,viewt2=300,sigma=3):
		if not os.path.exists(self.resultdir+'/check_pulse.png'):
			assert os.path.exists(self.baseresultdir), 'Should have run base() before running check_pulse()!'
			viewt1=np.max([self.baset1,viewt1])
			viewt2=np.min([self.baset2,viewt2])
			f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
			fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=False)
			ylim=np.zeros((14,2))
			positiveIndex=[]
			negativeIndex=[]
			goodIndex=[]
			badIndex=[]
			#search for signals over guassian level
			for i in range(14):
				cNet=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] for ch in np.arange(ch1,ch2+1) ])
				totalNet=np.sum(cNet,axis=0)
				median=np.median(totalNet)
				totalNet_median_part=totalNet[totalNet<5*median]
				loc,scale=stats.norm.fit(totalNet_median_part)
				Y=stats.norm(loc=loc,scale=scale)
				gaussian_level=Y.interval(norm_pvalue(sigma))
				totalNet=np.concatenate(([totalNet[0]],totalNet))
				# search NaI detectors for tbins indice over and under gaussian level
				if i>=2:
					positiveIndex.extend(np.where(totalNet>gaussian_level[1])[0])
					negativeIndex.extend(np.where(totalNet<gaussian_level[0])[0])
			# search tbin where at lease 3 detectctors have the signal
			# stored in goodIndex
			positiveIndex_set=set(positiveIndex)
			for seq in positiveIndex_set:
				if positiveIndex.count(seq)>=3:
					goodIndex.extend([seq])
			# search tbins where at least 3 NaIs have an erroneous underflow
			# stored in badIndex
			negativeIndex_set=set(negativeIndex)
			for seq in negativeIndex_set:
				if negativeIndex.count(seq)>=3:
					badIndex.extend([seq])
			# remove the 0.5 second following badIndex from goodIndex
			for element in badIndex:
				for kk in range(1,round(0.5/self.binwidth)):
					if element+kk in goodIndex:
						goodIndex.remove(element+kk)
			#make plots												
			for i in range(14):
				cNet=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] for ch in np.arange(ch1,ch2+1) ])
				totalNet=np.sum(cNet,axis=0)
				median=np.median(totalNet)
				totalNet_median_part=totalNet[totalNet<5*median]
				loc,scale=stats.norm.fit(totalNet_median_part)
				Y=stats.norm(loc=loc,scale=scale)
				gaussian_level=Y.interval(norm_pvalue(sigma))
				totalNet=np.concatenate(([totalNet[0]],totalNet))
				axes[i//2,i%2].plot(self.tbins,totalNet,linestyle='steps',lw=3.0,color='tab:blue')
				axes[i//2,i%2].axhline(gaussian_level[1],ls='--',lw=3,\
					color='orange',label=str(sigma)+'$\sigma$ level of gaussian background')
				axes[i//2,i%2].tick_params(labelsize=25)
				if len(goodIndex)>=1:
					for seq in goodIndex:
						axes[i//2,i%2].axvline(x=self.tbins[seq],ymin=0.95,color='red',zorder=2)
					goodIndex_sorted=sorted(goodIndex)
					x0=self.tbins[goodIndex_sorted[0]]
					x1=self.tbins[goodIndex_sorted[-1]]
					x_width=np.max([x1-x0,2.0])
					axes[i//2,i%2].set_xlim([ np.max([x0-x_width,viewt1]),np.min([x1+x_width,viewt2]) ])
				else:
					axes[i//2,i%2].set_xlim([viewt1,viewt2])
				ylim[i]=axes[i//2,i%2].get_ylim()
				axes[i//2,i%2].text(0.05,0.85,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
			#reset the ylims to same values
			BGOymax=np.max([ylim[i,1] for i in range(2)])
			NaIymax=np.max([ylim[i+2,1] for i in range(12)])	
			for i in range(14):
				if i==1:
					axes[i//2,i%2].legend(fontsize=20)
				if i<=1:
					axes[i//2,i%2].set_ylim([0,BGOymax])
				else:
					axes[i//2,i%2].set_ylim([0,NaIymax])
			fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center', va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)
			fig.text(0.5, 0.92, self.bnname, ha='center', va='center',fontsize=30)		
			plt.savefig(self.resultdir+'/check_pulse.png')
			plt.close()
			f.close()

			
	def check_poisson_rate(self):
		if not os.path.exists(self.resultdir+'/poisson_rate/'):
			os.makedirs(self.resultdir+'/poisson_rate/')
			f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
			for ch in np.arange(ch1,ch2+1):
				fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=False,sharey=False)
				for i in range(14):
					basecount=f['/'+Det[i]+'/ch'+str(ch)][()][1]*self.binwidth
					mean=np.mean(basecount)
					Y=stats.poisson(mean)
					plotcount=f['/'+Det[i]+'/ch'+str(ch)][()][0]*self.binwidth
					plotcount=np.ceil(plotcount).astype(int)
					maxcount=np.max(plotcount)
					bins=np.arange(-0.5,maxcount+1.5)
					x_int=np.arange(0,maxcount+1)
					plothist=Y.pmf(x_int)
					plothist=np.concatenate(([plothist[0]],plothist))
					axes[i//2,i%2].plot(bins,plothist,linestyle='steps',\
									label='Baseline Poisson PMF (from base)',\
									lw=4.0,color='tab:orange')
					hist,bin_edged=np.histogram(plotcount,bins=bins)
					axes[i//2,i%2].bar(x_int,hist/np.sum(hist),\
								label='Observed count (from rate)', lw=4.0,\
								color='tab:blue')	
					axes[i//2,i%2].tick_params(labelsize=25)
					axes[i//2,i%2].text(0.05,0.85,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
					if i==1:
						axes[i//2,i%2].legend(fontsize=20)
				fig.text(0.07, 0.5, 'Normalized histogram (PMF)', ha='center', va='center',rotation='vertical',fontsize=30)
				fig.text(0.5, 0.05, 'Count in each time bin ('+str(self.binwidth)+' s) between '+\
									str(self.baset1)+'--'+str(self.baset2)+' s',\
									ha='center', va='center',fontsize=30)
				fig.text(0.5, 0.92, 'ch'+str(ch), ha='center', va='center',fontsize=30)		
				plt.savefig(self.resultdir+'/poisson_rate/'+str(ch)+'.png')
				plt.close()
			f.close()
			
	def plot_time_resolved_net_spectrum(self):
		if not os.path.exists(self.resultdir+'/time_resolved_net_spectrum/'):
			os.makedirs(self.resultdir+'/time_resolved_net_spectrum/')
			f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
			for n,t1 in enumerate(self.tbins[:-1]):
				if t1>-5 and t1<20:
					fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=True)
					for i in range(14):
						spectrum=np.zeros(ch2+1-ch1)
						for ch in np.arange(ch1,ch2+1):
							ds=f['/'+Det[i]+'/ch'+str(ch)][()][2] #net
							spectrum[ch-ch1]=np.abs(ds[n]*self.binwidth)
						axes[i//2,i%2].bar(np.arange(ch1,ch2+1),spectrum,lw=4.0,color='tab:blue')
						axes[i//2,i%2].tick_params(labelsize=25)
						axes[i//2,i%2].text(0.05,0.85,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
					fig.text(0.07, 0.5, 'Count spectrum', ha='center', va='center',rotation='vertical',fontsize=30)
					fig.text(0.5, 0.05, 'Channels',ha='center', va='center',fontsize=30)
					fig.text(0.5, 0.92, str(t1)+' s', ha='center', va='center',fontsize=30)		
					plt.savefig(self.resultdir+'/time_resolved_net_spectrum/'+str(n)+'.png')
					plt.close()
			f.close()		
			

	def check_poisson_time_resolved_net_spectrum(self):
		if not os.path.exists(self.resultdir+'/check_poisson_time_resolved_net_spectrum/'):
			os.makedirs(self.resultdir+'/check_poisson_time_resolved_net_spectrum/')
			f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
			for n,t1 in enumerate(self.tbins[:-1]):
				if t1>-5 and t1<20:
					fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=True)
					for i in range(14):
						spectrum=np.zeros(ch2+1-ch1)
						for ch in np.arange(ch1,ch2+1):
							ds=f['/'+Det[i]+'/ch'+str(ch)][()][2] #net
							spectrum[ch-ch1]=np.abs(ds[n]*self.binwidth)
						mean=np.mean(spectrum)
						Y=stats.poisson(mean)
						maxcount=np.ceil(spectrum.max())
						x_int=np.arange(0,maxcount+1)
						bins=np.arange(-0.5,maxcount+1.5)
						poissonpmf=Y.pmf(x_int)
						poissonpmf=np.concatenate(([poissonpmf[0]],poissonpmf))
						axes[i//2,i%2].plot(bins,poissonpmf,linestyle='steps',\
										label='Poisson PMF of mean net rate',\
										lw=4.0,color='tab:orange')
						hist,bin_edged=np.histogram(spectrum,bins=bins)
						axes[i//2,i%2].bar(x_int,hist/np.sum(hist),\
								label='Distribution of observed net rate',\
								lw=4.0,color='tab:blue')
						#axes[i//2,i%2].bar(np.arange(2,126),spectrum,\
						#			color='tab:blue')	
						axes[i//2,i%2].tick_params(labelsize=25)
						axes[i//2,i%2].text(0.05,0.85,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
						if i==1:
							axes[i//2,i%2].legend(fontsize=20)
					fig.text(0.07, 0.5, 'Normalized histogram (PMF)', ha='center', va='center',rotation='vertical',fontsize=30)
					fig.text(0.5, 0.05, 'Counts',ha='center', va='center',fontsize=30)
					fig.text(0.5, 0.92, str(t1)+' s', ha='center', va='center',fontsize=30)		
					plt.savefig(self.resultdir+'/check_poisson_time_resolved_net_spectrum/'+str(n)+'.png')
					plt.close()
			f.close()

	def netlc(self,viewt1=-50,viewt2=300):
		if not os.path.exists(self.resultdir+'/net_lc.png'):
			assert os.path.exists(self.baseresultdir),'Should have run base() before running netlc()!'
			#print('plotting raw_lc_with_base.png ...')
			viewt1=np.max([self.baset1,viewt1])
			viewt2=np.min([self.baset2,viewt2])
			BGOplotymax=0.0
			NaIplotymax=0.0
			# raw lc with baseline
			f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
			fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=False)
			for i in range(14):
				#data in firt and last two channels of BGO and NaI are not used
				#ignore 0,1,126,127, notice 2-125
				cNet=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] for ch in np.arange(ch1,ch2+1) ])
				totalNet=np.sum(cNet,axis=0)
				ymax=totalNet.max()
				if i<=1:
					if BGOplotymax<ymax:
						BGOplotymax=ymax
						BGOdetseq=i
				else:
					if NaIplotymax<ymax:
						NaIplotymax=ymax
						NaIdetseq=i
				cRate=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][0] for ch in np.arange(ch1,ch2+1) ])
				totalRate=np.sum(cRate,axis=0)
				totalRate=np.concatenate(([totalRate[0]],totalRate))
				axes[i//2,i%2].plot(self.tbins,totalRate,linestyle='steps',lw=3.0,color='tab:blue')
				cBase=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][1] for ch in np.arange(ch1,ch2+1) ])
				totalBase=np.sum(cBase,axis=0)
				plottime=self.tbins[:-1]+self.binwidth/2.0
				axes[i//2,i%2].plot(plottime,totalBase,linestyle='--',lw=4.0, color='tab:orange')
				axes[i//2,i%2].set_xlim([viewt1,viewt2])
				axes[i//2,i%2].tick_params(labelsize=25)
				axes[i//2,i%2].text(0.05,0.85,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
			fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center', va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)		
			fig.text(0.5, 0.92, self.bnname, ha='center', va='center',fontsize=30)			
			plt.savefig(self.resultdir+'/raw_lc_with_base.png')
			plt.close()

			# net lc
			#print('plotting net_lc.png ...')
			fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=False)
			for i in range(14):
				cNet=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] for ch in np.arange(ch1,ch2+1) ])
				totalNet=np.sum(cNet,axis=0)
				totalNet=np.concatenate(([totalNet[0]],totalNet))
				axes[i//2,i%2].plot(self.tbins,totalNet,linestyle='steps',lw=3.0,color='tab:blue')
				axes[i//2,i%2].set_xlim([viewt1,viewt2])
				if i<=1:
					axes[i//2,i%2].set_ylim([0,BGOplotymax])
				else:
					axes[i//2,i%2].set_ylim([0,NaIplotymax])
				axes[i//2,i%2].tick_params(labelsize=25)
				if i==BGOdetseq:
					axes[i//2,i%2].text(0.7,0.85,'Brightest BGO',transform=axes[i//2,i%2].transAxes,color='red',fontsize=25)
				elif i==NaIdetseq:
					axes[i//2,i%2].text(0.7,0.85,'Brightest NaI',transform=axes[i//2,i%2].transAxes,color='red',fontsize=25)
				axes[i//2,i%2].text(0.05,0.85,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
			fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center', va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)		
			fig.text(0.5, 0.92, self.bnname, ha='center', va='center',fontsize=30)			
			plt.savefig(self.resultdir+'/net_lc.png')
			plt.close()
			f.close()


	def countmap(self,viewt1=-50,viewt2=300):
		if not os.path.exists(self.resultdir+'/net_countmap.png'):
			assert os.path.exists(self.baseresultdir), 'Should have run base() before running countmap()!'
			viewt1=np.max([self.baset1,viewt1])
			viewt2=np.min([self.baset2,viewt2])
			#print('plotting rate_countmap.png ...')
			#plot_countmap(self.bnname,self.resultdir,self.baseresultdir,self.datadir,'rate',self.tbins,viewt1,viewt2)
			#print('plotting base_countmap.png ...')
			#plot_countmap(self.bnname,self.resultdir,self.baseresultdir,self.datadir,'base',self.tbins,viewt1,viewt2)
			#print('plotting net_countmap.png ...')
			plot_countmap(self.bnname,self.resultdir,self.baseresultdir,self.datadir,'net',self.tbins,viewt1,viewt2)
			#print('plotting pois_countmap.png ...')
			#plot_countmap(self.bnname,self.resultdir,self.baseresultdir,self.datadir,'pois',self.tbins,viewt1,viewt2)

	def rsp(self):
		if not os.path.exists(self.resultdir+'/response_matrix.png'):
			#print('plotting response_matrix.png ...')
			#determine max for color bar
			BGOmaxcolorvalue=0.0
			NaImaxcolorvalue=0.0
			for i in range(14):
				ttefile=glob(self.datadir+'/glg_cspec_'+Det[i]+'_'+self.bnname+'_v*.rsp')
				hdu=fits.open(ttefile[0])
				ebound=hdu['EBOUNDS'].data
				emin=ebound.field(1)
				rspdata=hdu['SPECRESP MATRIX'].data
				elo=rspdata.field(0)
				matrix=rspdata.field(5)
				filled_matrix=np.zeros((len(emin),len(elo)))
				for ii in range(len(emin)):				
					for jj in range(len(elo)):
						try:
							filled_matrix[ii][jj]=matrix[jj][ii]
						except:
							pass
				if i<=1:
					if BGOmaxcolorvalue < filled_matrix.max():
						BGOmaxcolorvalue = filled_matrix.max()
				else:
					if NaImaxcolorvalue < filled_matrix.max():
						NaImaxcolorvalue = filled_matrix.max()		
			#plot response matrix
			fig, axes= plt.subplots(7,2,figsize=(32, 40),sharex=False,sharey=False)			
			for i in range(14):
				ttefile=glob(self.datadir+'/glg_cspec_'+Det[i]+'_'+self.bnname+'_v*.rsp')
				hdu=fits.open(ttefile[0])
				ebound=hdu['EBOUNDS'].data
				emin=ebound.field(1)
				emax=ebound.field(2)
				rspdata=hdu['SPECRESP MATRIX'].data
				elo=rspdata.field(0)
				ehi=rspdata.field(1)
				matrix=rspdata.field(5)
				filled_matrix=np.zeros((len(emin),len(elo)))
				for ii in range(len(emin)):				
					for jj in range(len(elo)):
						try:
							filled_matrix[ii][jj]=matrix[jj][ii]
						except:
							pass
				x=np.concatenate((elo,[ehi[-1]]))
				y=np.concatenate((emin,[emax[-1]]))
				X, Y= np.meshgrid(x,y)
				if i<=1:
					pcmBGO=axes[i//2,i%2].pcolormesh(X,Y,filled_matrix,\
						norm=colors.LogNorm(vmin=1E-1, vmax=BGOmaxcolorvalue),cmap='rainbow')
					axes[i//2,i%2].set_xlim([200,4E4])
					axes[i//2,i%2].set_ylim([200,4E4])
				else:
					pcmNaI=axes[i//2,i%2].pcolormesh(X,Y,filled_matrix,\
						norm=colors.LogNorm(vmin=1E-2, vmax=NaImaxcolorvalue),cmap='rainbow')
					axes[i//2,i%2].set_xlim([5,1E4])
					axes[i//2,i%2].set_ylim([5,1000])
				axes[i//2,i%2].tick_params(axis='both',top=True,right=True,length=3,width=1,direction='out',which='both',labelsize=25)
				axes[i//2,i%2].text(0.1,0.7,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
				axes[i//2,i%2].set_xscale('log')
				axes[i//2,i%2].set_yscale('log')
			cbarBGO=fig.colorbar(pcmBGO, ax=axes[0,], orientation='vertical',fraction=0.005, aspect=200/6)
			cbarBGO.ax.tick_params(labelsize=25)
			cbarNaI=fig.colorbar(pcmNaI, ax=axes[1:,], orientation='vertical',fraction=0.005, aspect=200)
			cbarNaI.ax.tick_params(labelsize=25)
			fig.text(0.05, 0.5, 'Measured Energy (KeV)', ha='center',va='center', rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Incident Energy (keV)', ha='center',va='center',fontsize=30)
			plt.savefig(self.resultdir+'response_matrix.png')
			plt.close()
		
		#plot effective area (ARF)
		if not os.path.exists(self.resultdir+'/effective_area.png'):
			#print('plotting effective_area.png ...')
			fig, axes= plt.subplots(7,2,figsize=(32, 30),sharex=False,sharey=False)	
			for i in range(14):
				ttefile=glob(self.datadir+'/glg_cspec_'+Det[i]+'_'+self.bnname+'_v*.rsp')
				hdu=fits.open(ttefile[0])
				ebound=hdu['EBOUNDS'].data
				emin=ebound.field(1)
				emax=ebound.field(2)
				rspdata=hdu['SPECRESP MATRIX'].data
				elo=rspdata.field(0)
				ehi=rspdata.field(1)
				matrix=rspdata.field(5)
				filled_matrix=np.zeros((len(emin),len(elo)))
				for ii in range(len(emin)):				
					for jj in range(len(elo)):
						try:
							filled_matrix[ii][jj]=matrix[jj][ii]
						except:
							pass
				x=np.concatenate((elo,[ehi[-1]]))
				arf=np.zeros(len(elo))
				for kk in range(len(elo)):
					arf[kk]=filled_matrix[:,kk].sum()
				arf=np.concatenate(([arf[0]],arf))
				axes[i//2,i%2].plot(x,arf,linestyle='steps',lw=5)
				axes[i//2,i%2].tick_params(axis='both',top=True,right=True,length=10,width=1,direction='in',which='both',labelsize=25)
				axes[i//2,i%2].text(0.7,0.1,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
				axes[i//2,i%2].set_xscale('log')
				axes[i//2,i%2].set_yscale('log')
				if i<=1:
					axes[i//2,i%2].set_xlim([200,4E4])
					axes[i//2,i%2].set_ylim([100,300])
					axes[i//2,i%2].yaxis.set_major_locator(ticker.FixedLocator([100]))
					#axes[i//2,i%2].yaxis.set_minor_locator(ticker.FixedLocator([200,300]))
					axes[i//2,i%2].yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
					axes[i//2,i%2].yaxis.set_minor_formatter(ticker.FormatStrFormatter('%d'))	
				else:
					axes[i//2,i%2].set_xlim([5,1E4])
					axes[i//2,i%2].set_ylim([1,200])
					axes[i//2,i%2].yaxis.set_major_locator(ticker.FixedLocator([1,10,100]))
					axes[i//2,i%2].yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
					#axes[i//2,i%2].yaxis.set_minor_formatter(ticker.FormatStrFormatter('%d'))				
			fig.text(0.05, 0.5, 'Effective Area (cm$^2$)', ha='center',va='center', rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Incident Energy (keV)', ha='center',va='center',fontsize=30)
			plt.savefig(self.resultdir+'/effective_area.png')
			plt.close()

													
	def phaI(self,slicet1=0,slicet2=5):
		#print('creating a new phaI slice between',slicet1,'s and',slicet2,'s  ...')
		if not os.path.exists(self.phaIresultdir):
			os.makedirs(self.phaIresultdir)
		nslice=len(os.listdir(self.phaIresultdir))
		sliceresultdir=self.phaIresultdir+'/slice'+str(nslice)+'/'
		os.makedirs(sliceresultdir)
		fig, axes= plt.subplots(7,2,figsize=(32, 30),sharex=False,sharey=False)
		sliceindex= (self.tbins >=slicet1) & (self.tbins <=slicet2)
		valid_bins=np.sum(sliceindex)-1
		assert valid_bins>=1, self.bnname+': Inappropriate phaI slice time!'
		f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
		for i in range(14):
			total_rate=np.zeros(128)
			bkg_rate=np.zeros(128)
			total_uncertainty=np.zeros(128)
			bkg_uncertainty=np.zeros(128)
			ttefile=glob(self.datadir+'/glg_tte_'+Det[i]+'_'+self.bnname+'_v*.fit')
			hdu=fits.open(ttefile[0])
			ebound=hdu['EBOUNDS'].data
			emin=ebound.field(1)
			emax=ebound.field(2)
			energy_diff=emax-emin
			energy_bins=np.concatenate((emin,[emax[-1]]))
			for ch in np.arange(128):
				base=f['/'+Det[i]+'/ch'+str(ch)][()][1]
				rate=f['/'+Det[i]+'/ch'+str(ch)][()][0]
				bkg=base[sliceindex[:-1]][:-1]
				total=rate[sliceindex[:-1]][:-1]
				bkg_rate[ch]=bkg.mean()
				total_rate[ch]=total.mean()
				exposure=len(bkg)*self.binwidth
				bkg_uncertainty[ch]=np.sqrt(bkg_rate[ch]/exposure)
				total_uncertainty[ch]=np.sqrt(total_rate[ch]/exposure)
			#plot both rate and bkg as count/s/keV
			write_phaI(bkg_rate,self.bnname,Det[i],slicet1,slicet2,sliceresultdir+'/'+Det[i]+'.bkg')
			write_phaI(total_rate,self.bnname,Det[i],slicet1,slicet2,sliceresultdir+'/'+Det[i]+'.pha')
			copy_rspI(self.bnname,Det[i],sliceresultdir+'/'+Det[i]+'.rsp')
			bkg_diff=bkg_rate/energy_diff
			total_diff=total_rate/energy_diff
			x=np.sqrt(emax*emin)
			axes[i//2,i%2].errorbar(x,bkg_diff,yerr=bkg_uncertainty/energy_diff,linestyle='None',color='blue')
			axes[i//2,i%2].errorbar(x,total_diff,yerr=total_uncertainty/energy_diff,linestyle='None',color='red')
			bkg_diff=np.concatenate(([bkg_diff[0]],bkg_diff))
			total_diff=np.concatenate(([total_diff[0]],total_diff))
			axes[i//2,i%2].plot(energy_bins,bkg_diff,linestyle='steps',color='blue')
			axes[i//2,i%2].plot(energy_bins,total_diff,linestyle='steps',color='red')
			axes[i//2,i%2].set_xscale('log')
			axes[i//2,i%2].set_yscale('log')
			axes[i//2,i%2].tick_params(labelsize=25)
			axes[i//2,i%2].text(0.85,0.85,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
		fig.text(0.07, 0.5, 'Rate (count s$^{-1}$ keV$^{-1}$)', ha='center',va='center', rotation='vertical',fontsize=30)
		fig.text(0.5, 0.05, 'Energy (keV)', ha='center', va='center',fontsize=30)	
		plt.savefig(sliceresultdir+'/PHA_rate_bkg.png')
		plt.close()
		f.close()


	def specanalyze(self,slicename):
		slicedir=self.phaIresultdir+'/'+slicename+'/'
		os.chdir(slicedir)
		# select the most bright two NaIs (in channels 6-118) 
		# and more bright one BGO (in channels 4-124):
		BGOtotal=np.zeros(2)
		NaItotal=np.zeros(12)
		for i in range(2):
			phahdu=fits.open(slicedir+'/'+BGO[i]+'.pha')
			bkghdu=fits.open(slicedir+'/'+BGO[i]+'.bkg')
			pha=phahdu['SPECTRUM'].data.field(1)
			bkg=bkghdu['SPECTRUM'].data.field(1)
			src=pha-bkg
			plt.plot(src[4:125])
			plt.savefig(BGO[i]+'.png')
			plt.close()
			BGOtotal[i]=src[4:125].sum()
		for i in range(12):
			phahdu=fits.open(slicedir+'/'+NaI[i]+'.pha')
			bkghdu=fits.open(slicedir+'/'+NaI[i]+'.bkg')
			pha=phahdu['SPECTRUM'].data.field(1)
			bkg=bkghdu['SPECTRUM'].data.field(1)
			src=pha-bkg
			plt.plot(src[6:118])
			plt.savefig(NaI[i]+'.png')
			plt.close()
			NaItotal[i]=src[6:118].sum()
		BGOindex=np.argsort(BGOtotal)
		NaIindex=np.argsort(NaItotal)
		brightdet=[BGO[BGOindex[-1]],NaI[NaIindex[-1]],NaI[NaIindex[-2]]]
		
		# use xspec	
		alldatastr=' '.join([det+'.pha' for det in brightdet])
		AllData(alldatastr)
		AllData.ignore('1-2:**-200.0,40000.0-** 3-14:**-8.0,800.0-**')
		Model('grbm')
		Fit.nIterations=1000
		Fit.statMethod='pgstat'
		Fit.perform()
		Fit.error('3.0 3')
		Fit.perform()
		par3=AllModels(1)(3)
		print(par3.error)

		#Plot.device='/xs'
		Plot.device='/null'
		Plot.xAxis='keV'
		Plot.yLog=True
		Plot('ldata')
		for i in (1,2,3):
			energies=Plot.x(i)
			rates=Plot.y(i)
			folded=Plot.model(i)
			xErrs=Plot.xErr(i)
			yErrs=Plot.yErr(i)
			plt.errorbar(energies,rates,xerr=xErrs,yerr=yErrs,zorder=1,ls='None')
			plt.plot(energies,folded,color='black',zorder=2)
		plt.xscale('log')
		plt.yscale('log')
		plt.savefig('foldedspec.png')
		plt.close()
		Plot('eeufspec')
		for i in (1,2,3):
			energies=Plot.x(i)
			ufspec=Plot.y(i)
			folded=Plot.model(i)
			xErrs=Plot.xErr(i)
			yErrs=Plot.yErr(i)
			plt.errorbar(energies,ufspec,xerr=xErrs,yerr=yErrs,zorder=1,ls='None')
			plt.plot(energies,folded,color='black',zorder=2)
		plt.xscale('log')
		plt.yscale('log')
		plt.savefig('eeufspec.png')
		plt.close()

	def removebase(self):
		os.system('rm -rf '+self.baseresultdir)
	

#############################	
# BEGIN derived class deGRB #
#############################
# for checking snr with different binwidth in baseline
class deGRB(GRB):
		
	def de_base(self,de_baset1=-50,de_baset2=300,de_binwidth=[1.0,0.1,0.01]):
		self.de_baseresultdir=self.baseresultdir+'/de_base/'
		self.de_baset1=np.max([self.GTI1,de_baset1])
		self.de_baset2=np.min([self.GTI2,de_baset2])

		assert self.de_baset1<self.de_baset2, self.bnname+': Inappropriate base times!'
		if not os.path.exists(self.de_baseresultdir):
			os.makedirs(self.de_baseresultdir)
			for seq,binwidth in enumerate(de_binwidth):
				tbins=np.arange(self.de_baset1,self.de_baset2+binwidth,binwidth)
				f=h5py.File(self.de_baseresultdir+'/base_'+str(seq)+'.h5',mode='w')
				for i in range(14):
					grp=f.create_group(Det[i])
					ttefile=glob(self.datadir+'/'+'glg_tte_'+Det[i]+'_'+self.bnname+'_v*.fit')
					hdu=fits.open(ttefile[0])	
					trigtime=hdu['Primary'].header['TRIGTIME']
					data=hdu['EVENTS'].data
					timedata=data.field(0)-trigtime
					chdata=data.field(1)
					for ch in range(128):
						time_selected=timedata[chdata==ch]
						histvalue, histbin=np.histogram(time_selected,bins=tbins)
						rate=histvalue/binwidth
						r.assign('rrate',rate) 
						r("y=matrix(rrate,nrow=1)")
						#fillPeak_hwi=str(int(40/binwidth))
						#fillPeak_int=str(int(len(rate)/10))
						#fillPeak_lam=str(int(0.708711*(len(rate))**0.28228+0.27114))
						#r("rbase=baseline(y,lam = "+fillPeak_lam+", hwi="\
						#					+fillPeak_hwi+", it=10, int ="\
						#					+fillPeak_int+", method='fillPeaks')")
						fillPeak_hwi=str(int(5/binwidth))
						fillPeak_int=str(int(len(rate)/10))
						r("rbase=baseline(y,lam = 6, hwi="+fillPeak_hwi+", it=10,int ="+fillPeak_int+", method='fillPeaks')")
						r("bs=getBaseline(rbase)")
						r("cs=getCorrected(rbase)")
						bs=r('bs')[0]
						cs=r('cs')[0]
						# correct negative base to 0 and recover the net value to original rate
						corrections_index= (bs<0)
						bs[corrections_index]=0
						cs[corrections_index]=rate[corrections_index]
						f['/'+Det[i]+'/ch'+str(ch)]=np.array([rate,bs,cs])
				f.flush()
				f.close()
				
# check SNR
	def check_debase_snr(self,viewt1=-50,viewt2=300,de_binwidth=[1.0,0.1,0.01]):
		if not os.path.exists(self.resultdir+'/check_debase_SNR.png'):
			assert os.path.exists(self.de_baseresultdir), 'Should have run de_base() before running check_debase_snr()!'
			viewt1=np.max([self.de_baset1,viewt1])
			viewt2=np.min([self.de_baset2,viewt2])
			#f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
			fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=True)
			ylim=np.zeros((14,2))
			my_colors=['black','red','blue','green']
			for seq,binwidth in enumerate(de_binwidth):
				tbins=np.arange(self.de_baset1,self.de_baset2+binwidth,binwidth)
				f=h5py.File(self.de_baseresultdir+'/base_'+str(seq)+'.h5',mode='r')
				for i in range(14):
					cNet=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] for ch in np.arange(ch1,ch2+1) ])
					totalNet=np.sum(cNet,axis=0)
					median=np.median(totalNet)
					totalNet_median_part=totalNet[totalNet<5*median]
					loc,scale=stats.norm.fit(totalNet_median_part)
					#Y=stats.norm(loc=loc,scale=scale)
					#gaussian_level=Y.interval(norm_pvalue(sigma))
					totalNet=np.concatenate(([totalNet[0]],totalNet))
					snr=(totalNet-loc)/scale
					axes[i//2,i%2].plot(tbins,snr,linestyle='steps',lw=1.0,color=my_colors[seq],alpha=0.5,label=str(binwidth))
					#axes[i//2,i%2].axhline(gaussian_level[1],ls='--',lw=3,\
					#	color='orange',label=str(sigma)+'$\sigma$ level of gaussian background')
					axes[i//2,i%2].tick_params(labelsize=25)
					#axes[i//2,i%2].text(0.05,0.85,Det[i],transform=axes[i//2,i%2].transAxes,fontsize=25)
					#axes[i//2,i%2].axhline(y=3,color='orange',ls='--',lw=3,zorder=2,label='SNR=3')
					#for t in self.tbins[snr>3]:
					#	axes[i//2,i%2].axvline(x=t,ymin=0.95,color='red',zorder=2)
					if i==1:
						axes[i//2,i%2].legend(fontsize=20)
				f.close()
			axes[0,0].set_xlim([viewt1,viewt2])
			#axes[0,0].set_ylim([-1,4])
			fig.text(0.07, 0.5, 'Signal-to-noise ratio', ha='center', va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)		
			fig.text(0.5, 0.92, self.bnname, ha='center', va='center',fontsize=30)			
			plt.savefig(self.resultdir+'/check_debase_SNR.png')
			plt.close()
					