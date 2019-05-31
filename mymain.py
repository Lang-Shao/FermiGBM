from GBM_catalog_spectralanalysis_lib import *
import matplotlib
matplotlib.use('Agg')

def main():
	fermigbrst = query_fermigbrst()
	df = pd.read_csv(fermigbrst,delimiter='|',header=0,skipfooter=3,engine='python')
	trigger_name = df['trigger_name'].apply(lambda x:x.strip()).values
	print(trigger_name)
	global badsampledir
	cdir = os.getcwd()
	badsampledir = cdir+'/bad_sample/'
	if not os.path.exists(badsampledir):
		os.makedirs(badsampledir)
	
	if __name__ == '__main__':
		p = Pool(ncore)
		p.map(inspect_GRB,trigger_name)	
	
	good_burst_bnname = []
	good_burst_t0 = []
	good_burst_t1 = []
	good_burst_duration = []
	for bnname in trigger_name:
		t0,t1,duration = read_duration(bnname)
		if duration:
			good_burst_bnname.expand(bnname)
			good_burst_t0.expand(t0)
			good_burst_t1.expand(t1)
			good_burst_duration.expand(duration)
	print(good_burst_bnname)
	print(good_burst_duration)

	'''
	inspect_GRB('bn190114873')
	for bn in trigger_name:
		print(bn)
		inspect_GRB(bn)
	'''

def inspect_GRB(bnname):	
	print('Processing: '+bnname)
	grb = GRB(bnname)
	if grb.dataready:
		#currently useful
		grb.rawlc(viewt1=-50,viewt2=300,binwidth=0.064)
		grb.base(baset1=-50,baset2=300,binwidth=0.064) #MUST RUN
		grb.check_gaussian_net_rate()
		grb.plot_gaussian_level_over_net_lc()
		grb.check_pulse()
		grb.countmap()

		#currently not useful
		#grb.check_snr()
		#grb.skymap()
		#grb.plotbase()
		#grb.check_gaussian_total_rate()
		#grb.check_poisson_rate()
		#grb.plot_time_resolved_net_spectrum()
		#grb.check_poisson_time_resolved_net_spectrum()
		#grb.netlc()
		#grb.rsp()

		#grb.multi_binwidth_base()
		#grb.check_mb_base_snr(viewt1=-1,viewt2=25)
		#grb.check_mb_base_gaussian_net_rate()	
		
		'''
		timebins = np.arange(-15,6,5)
		nslice = len(timebins)-1
		for i in np.arange(nslice):
			grb.phaI(slicet1=timebins[i],slicet2=timebins[i+1]) 
			grb.specanalyze('slice'+str(i))
		'''
		
		#remove basedir to save disk space
		#grb.removebase()
	else:
		if not os.path.exists(badsampledir+'/'+bnname+'.txt'):
			with open(badsampledir+'/'+bnname+'.txt','w') as f:
				f.write('missing data')
	#print(read_duration(bnname))	

def read_duration(bnname):
	duration_result = os.getcwd()+'/results/'+bnname+'/t0_t1_duration.txt'
	t0,t1,duration = None,None,None	
	if os.path.exists(duration_result):
		with open(duration_result) as f:
			t0,t1,duration = np.array(f.readline().split()).astype(np.float)
			#print(t0,t1,duration)
	return t0,t1,duration

# Always run this Main part #
if __name__ == '__main__':
	main()
	#inspect_GRB('bn190114873')
