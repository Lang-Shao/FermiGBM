from GBM_catalog_spectralanalysis_lib import *
import matplotlib
matplotlib.use('Agg')

def main():
	fermigbrst = query_fermigbrst()
	df = pd.read_csv(fermigbrst,delimiter='|',header=0,skipfooter=3,engine='python')
	trigger_name = df['trigger_name'].apply(lambda x:x.strip()).values
	#t90 = df['t90'].apply(lambda x:x.strip()).values
	print('cataloged_trigger_name= ',trigger_name)
	cdir = os.getcwd()
	badsampledir = cdir+'/bad_sample/'
	resultdir = cdir+'/results/'
	if not os.path.exists(badsampledir):
		os.makedirs(badsampledir)
	if not os.path.exists(resultdir):
		os.makedirs(resultdir)
	'''
	@timer
	def multiprocessing_GRB():
		if __name__ == '__main__':
			p = Pool(ncore)
			total_num = len(trigger_name)
			p.map(inspect_GRB,zip(trigger_name,
					[total_num]*total_num,
					[resultdir]*total_num,
					[badsampledir]*total_num))	
	multiprocessing_GRB()
	'''
	good_burst_bnname = []
	good_burst_t0 = []
	good_burst_t1 = []
	good_burst_duration = []
	for bnname in trigger_name:
		t0,t1,duration = read_duration(bnname,resultdir)
		if duration:
			good_burst_bnname.append(bnname)
			good_burst_t0.append(t0)
			good_burst_t1.append(t1)
			good_burst_duration.append(duration)
	#print(good_burst_bnname)
	#print(good_burst_duration)
	if not os.path.exists('./duration_hist.png'):
		duration_bins = np.logspace(-2,3,101)
		histvalue, histbin = np.histogram(good_burst_duration,bins=duration_bins)
		histvalue = np.concatenate(([histvalue[0]],histvalue))
		plt.plot(histbin,histvalue,ls='steps')
		
		plt.xscale('log')
		plt.yscale('log')
		plt.savefig('./duration_hist.png')
	
	#catalog_burst_duration = df[df.columns[1]].values
	#print(catalog_burst_duration)
	#print(float(df.loc[1,df.columns[1]]))
	
	
	df_good_burst=pd.DataFrame(np.array([good_burst_bnname,good_burst_duration,good_burst_t0,good_burst_t1]).T,
											columns=['bnname','duration','t0','t1'])
	df_good_burst.to_csv('./good_burst.csv',index=False)
	
		
	print(df[df.columns[0]].values)
	
	index = trigger_name == good_burst_bnname[0]
	print(good_burst_bnname[0],good_burst_duration[0],df[df.columns[1]].values[index])
	
	
	'''
	if not os.path.exists('./catalog_duration_hist.png'):
		duration_bins = np.logspace(-2,3,101)
		histvalue, histbin = np.histogram(catalog_burst_duration,bins=duration_bins)
		histvalue = np.concatenate(([histvalue[0]],histvalue))
		plt.plot(histbin,histvalue,ls='steps')
		
		plt.xscale('log')
		plt.yscale('log')
		plt.savefig('./catalog_duration_hist.png')
	'''
	'''
	inspect_GRB('bn190114873')
	for bn in trigger_name:
		print(bn)
		inspect_GRB(bn)
	'''

####################
## Some functions ##
####################

def inspect_GRB(pars):
	bnname, total_num, resultdir, badsampledir = pars
	index_GRB = len(os.listdir(resultdir))+1
	print('[Processing: '+bnname+' ] '+str(index_GRB)+'/'+str(total_num)
			+' ('+str(round(index_GRB/total_num*100,1))+'%)',end='\r')
	grb = GRB(bnname, resultdir)
	if grb.dataready:
		#currently useful
		grb.rawlc(viewt1=-50,viewt2=300,binwidth=0.064)
		grb.base(baset1=-50,baset2=300,binwidth=0.064) #MUST RUN
		grb.check_gaussian_net_rate()
		grb.plot_gaussian_level_over_net_lc()
		grb.check_pulse()
		grb.countmap()
		#remove basedir to save disk space
		grb.removebase()
		
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
	else:
		if not os.path.exists(badsampledir+'/'+bnname+'.txt'):
			with open(badsampledir+'/'+bnname+'.txt','w') as f:
				f.write('missing data')	


def read_duration(bnname,resultdir):
	duration_result = resultdir+'/'+bnname+'/t0_t1_duration.txt'
	t0,t1,duration = None,None,None	
	if os.path.exists(duration_result):
		with open(duration_result) as f:
			t0,t1,duration = np.array(f.readline().split()).astype(np.float)
	return t0,t1,duration

############
# RUN MAIN #
############
if __name__ == '__main__':
	main()
	#inspect_GRB('bn190114873')
