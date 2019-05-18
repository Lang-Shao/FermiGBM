import matplotlib
matplotlib.use('Agg')
from GBM_catalog_spectralanalysis_lib import *

# ALWAYS RUN THIS MAIN FUNCTION #
def main():
	fermigbrst=query_fermigbrst()
	df=pd.read_csv(fermigbrst,delimiter='|',header=0,skipfooter=3,engine='python')
	trigger_name=df['trigger_name'].apply(lambda x:x.strip()).values
	print(trigger_name)

	global badsampledir
	cdir=os.getcwd()
	badsampledir=cdir+'/bad_sample/'
	if not os.path.exists(badsampledir):
		os.makedirs(badsampledir)
	
	if __name__ == '__main__':
		p = Pool(ncore)
		p.map(inspect_GRB,trigger_name)	
	'''
	inspect_GRB('bn190114873')
	for bn in trigger_name:
		print(bn)
		inspect_GRB(bn)
	'''

def inspect_GRB(bnname):	
	print('Processing: '+bnname)
	grb=GRB(bnname)
	if grb.dataready:
		#currently useful
		grb.rawlc(viewt1=-50,viewt2=300,binwidth=0.064)
		grb.base(baset1=-50,baset2=300,binwidth=0.064) #MUST RUN
		grb.check_gaussian_net_rate()
		grb.plot_gaussian_level_over_net_lc()
		grb.check_pulse()
		grb.countmap()
		#remove basedir to save disk space
		#grb.removebase()
		
		#currently not useful
		#grb.plotbase()
		#grb.check_gaussian_total_rate()
		#grb.check_poisson_rate()
		#grb.plot_time_resolved_net_spectrum()
		#grb.check_poisson_time_resolved_net_spectrum()
		#grb.netlc()
		#grb.rsp()
		'''
		timebins=np.arange(-15,6,5)
		nslice=len(timebins)-1
		for i in np.arange(nslice):
			grb.phaI(slicet1=timebins[i],slicet2=timebins[i+1]) 
			grb.specanalyze('slice'+str(i))
		'''
	else:
		if not os.path.exists(badsampledir+'/'+bnname+'.txt'):
			with open(badsampledir+'/'+bnname+'.txt','w') as f:
				f.write('missing data')
		

# DO NOT REMOVE THIS PART #
if __name__ == '__main__':
	#main()
	inspect_GRB('bn190114873')
