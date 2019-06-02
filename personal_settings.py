# modify this file for personal settings
# the changes of which should be ignored during each commit

def get_usersjar():
	usersjar = "/home/lang/Software/HEASARC-Xamin/users.jar"
	return usersjar

def get_databasedir():
	databasedir = '/diskb/Database/Fermi/gbm_burst/data/'
	#databasedir = '/home/lang/work/GBM/burstdownload/data/'
	return databasedir

def get_dailydatabasedir():
	dailydatabasedir = '/diska/Fermi_GBM_daily/data/'
	#dailydatabasedir = '/home/lang/work/GBM/daily/data/'
	return dailydatabasedir

def get_ncore():
	#ncore = 3
	ncore = 82
	return ncore
