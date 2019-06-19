# modify this file for personal settings
# the changes of which should be ignored during each commit

def get_usersjar():
	usersjar = "/home/lang/Software/HEASARC-Xamin/users.jar"
	return usersjar

def get_databasedir():
	databasedir = '/diskb/Database/Fermi/gbm_burst/data/'
	#databasedir = '/home/lang/work/GBM/burstdownload/data/'
	return databasedir

def get_dailydatabasedirs():
	dailydatabasedir_a = '/diska/Fermi_GBM_daily/data/'
	dailydatabasedir_b = '/diskb/Database/Fermi/Fermi_GBM_daily/data/'
	#dailydatabasedir = '/home/lang/work/GBM/daily/data/'
	return dailydatabasedir_a, dailydatabasedir_b

def get_ncore():
	#ncore = 3
	ncore = 82
	return ncore
