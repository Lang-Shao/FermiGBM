# General analyses of GBM catalog bursts 
import warnings
import os
import sys
import time
import operator
import itertools
import functools
from multiprocessing import Pool
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import ticker
plt.style.use('seaborn')
from mpl_toolkits.basemap import Basemap
import numpy as np
from glob import glob
import h5py
import pandas as pd
import seaborn as sns
#sns.set(style='whitegrid')
from scipy import stats
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.stats import bayesian_blocks
from astropy.stats import sigma_clip, mad_std
from astropy.coordinates import get_body_barycentric, cartesian_to_spherical
from astropy.coordinates import get_sun, SkyCoord, cartesian_to_spherical
from spherical_geometry.polygon import SphericalPolygon
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)
import rpy2.robjects as robjects
from rpy2.robjects import r
import rpy2.robjects.numpy2ri
robjects.numpy2ri.activate()
robjects.r("library(baseline)")
from xspec import *
from personal_settings import *

databasedir = get_databasedir()
NaI = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb']
BGO = ['b0','b1']
Det = ['b0','b1','n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb']
#data in first and last three channels of BGO and NaI are used
#ignore 0,1,2,125,126,127, notice 3-124
ch1 = 3
ch2 = 124
ncore = get_ncore()

########################
# Classes for skymap #
########################

# ****the followings come from zjh_gbmgeometry.py****
class GBM_detector:
	def __init__(self,name,quaternion):
		self.name = name
		self.quaternion = quaternion
		self.position = self.get_position()
		p_lon = cartesian_to_spherical(self.position[0],self.position[1],
											self.position[2])[2].deg
		p_lat = cartesian_to_spherical(self.position[0],self.position[1],
											self.position[2])[1].deg
		self.center = SkyCoord(ra=p_lon, dec=p_lat, frame='icrs',
													unit='deg')

	def get_position(self):
		X = np.mat(self.gbm_xyz).T
		mat0 = self.get_mat(self.quaternion[0],self.quaternion[1],
							self.quaternion[2],self.quaternion[3])
		X1 = mat0*X
		x = np.array([X1[0],X1[1],X1[2]])
		x = np.array([x[0][0][0],x[1][0][0],x[2][0][0]])
		return x

	def get_mat(self,p1,p2,p3,p0):
		mat = np.mat(np.zeros((3, 3)))
		mat[0, 0] = p0 ** 2 + p1 ** 2 - p2 ** 2 - p3 ** 2
		mat[0, 1] = 2 * (p1 * p2 - p0 * p3)
		mat[0, 2] = 2 * (p0 * p2 + p1 * p3)
		mat[1, 0] = 2 * (p3 * p0 + p2 * p1)
		mat[1, 1] = p0 ** 2 + p2 ** 2 - p3 ** 2 - p1 ** 2
		mat[1, 2] = 2 * (p2 * p3 - p1 * p0)
		mat[2, 0] = 2 * (p1 * p3 - p0 * p2)
		mat[2, 1] = 2 * (p0 * p1 + p3 * p2)
		mat[2, 2] = p0 ** 2 + p3 ** 2 - p1 ** 2 - p2 ** 2
		return mat

	def get_fov(self,radius):
		if radius >= 60:
			steps = 5000 ## could be modified to speed up the plotting
		elif radius >= 30:
			steps = 400 ## could be modified to speed up the plotting
		else:
			steps = 100 ## could be modified to speed up the plotting
		j2000 = self.center.icrs
		poly = SphericalPolygon.from_cone(j2000.ra.value,j2000.dec.value,
												radius,steps = steps)
		re =  [p for p in poly.to_radec()][0]
		return re

	def contains_point(self,point):
		steps = 300
		j2000 = self.center.icrs
		poly = SphericalPolygon.from_cone(j2000.ra.value,j2000.dec.value,
											self.radius,steps = steps)
		return poly.contains_point(point.cartesian.xyz.value)

class NaI0(GBM_detector):
	def __init__(self,quaternion,point = None):
		self.az = 45.89
		self.zen = 90 - 20.58
		self.radius = 60.0
		self.gbm_xyz = np.array([0.2446677589,0.2523893824,0.9361823057])
		super(NaI0, self).__init__('n0',quaternion)

class NaI1(GBM_detector):
	def __init__(self, quaternion, point=None):
		self.az = 45.11
		self.zen = 90 - 45.31
		self.radius = 60.0
		self.gbm_xyz = np.array([0.5017318971,0.5036621127,0.7032706462])
		super(NaI1, self).__init__('n1', quaternion)

class NaI2(GBM_detector):
	def __init__(self, quaternion, point=None):
		self.az = 58.44
		self.zen = 90 - 90.21
		self.radius = 60.0
		self.gbm_xyz = np.array([0.5233876659,0.8520868147,-0.0036651682])
		super(NaI2, self).__init__('n2', quaternion)

class NaI3(GBM_detector):
	def __init__(self, quaternion, point=None):
		self.az = 314.87
		self.zen = 90 - 45.24
		self.radius = 60.0
		self.gbm_xyz = np.array([0.5009495177,-0.5032279093,0.7041386753])
		super(NaI3, self).__init__('n3', quaternion)

class NaI4(GBM_detector):
	def __init__(self, quaternion, point=None):
		self.az = 303.15
		self.zen = 90. - 90.27
		self.radius = 60.0
		self.gbm_xyz = np.array([ 0.5468267487,-0.8372325378,-0.0047123847])
		super(NaI4, self).__init__('n4', quaternion)

class NaI5(GBM_detector):
	def __init__(self, quaternion, point=None):
		self.az = 3.35
		self.zen = 90 - 89.97
		self.radius = 60.0
		self.gbm_xyz = np.array([0.9982910766,0.0584352143,0.0005236008])
		super(NaI5, self).__init__('n5', quaternion)

class NaI6(GBM_detector):
	def __init__(self, quaternion, point=None):
		self.az = 224.93
		self.zen = 90 - 20.43
		self.radius = 60.0
		self.gbm_xyz = np.array([-0.2471260191,-0.2465229020,0.9370993606])
		super(NaI6, self).__init__('n6', quaternion)

class NaI7(GBM_detector):
	def __init__(self, quaternion, point=None):
		self.az = 224.62
		self.zen = 90 - 46.18
		self.radius = 60.0
		self.gbm_xyz = np.array([-0.5135631636,-0.5067957667,0.6923950822])
		super(NaI7, self).__init__('n7', quaternion)

class NaI8(GBM_detector):
	def __init__(self, quaternion, point=None):
		self.az =  236.61
		self.zen = 90 - 89.97
		self.radius = 60.0
		self.gbm_xyz = np.array([-0.5503349679,-0.8349438131,0.0005235846])
		super(NaI8, self).__init__('n8', quaternion)

class NaI9(GBM_detector):
	def __init__(self, quaternion, point=None):
		self.az = 135.19
		self.zen = 90 - 45.55
		self.radius = 60.0
		self.gbm_xyz = np.array([-0.5064476761,0.5030998708,0.7002865795])
		super(NaI9, self).__init__('n9', quaternion)

class NaIA(GBM_detector):
	def __init__(self, quaternion, point=None):
		self.az = 123.73
		self.zen = 90 - 90.42
		self.radius = 60.0
		self.gbm_xyz = np.array([-0.5552650628,0.8316411478,-0.0073303046])
		super(NaIA, self).__init__('na', quaternion)

class NaIB(GBM_detector):
	def __init__(self, quaternion, point=None):
		self.az = 183.74
		self.zen = 90 - 90.32
		self.radius = 60.0
		self.gbm_xyz = np.array([-0.9978547710,-0.0652279514,-0.0055850266])
		super(NaIB, self).__init__('nb', quaternion)

class BGO0(GBM_detector):
	def __init__(self,quaternion,point = None):
		self.az = 0.0
		self.zen = 0.0
		self.radius = 90.0
		self.gbm_xyz = np.array([1.0,0.0,0.0])
		super(BGO0, self).__init__('b0',quaternion)

class BGO1(GBM_detector):
	def __init__(self, quaternion, point=None):
		self.az = 180.0
		self.zen = 0.0
		self.radius = 90.0
		self.gbm_xyz = np.array([-1.0,0.0,0.0])
		super(BGO1, self).__init__('b1', quaternion)


class GBMtime:
	def __init__(self):
		self.utc_start_time = '2008-08-07T03:35:44.0'
		self.mjd_start_time = 51910

	@classmethod
	def met_to_utc(self,met):
		if (met <= 54832.00000000):
			utc_tt_diff = 65.184
		elif (met <= 56109.00000000):
			utc_tt_diff = 66.184
		elif (met <= 57204.00000000):
			utc_tt_diff = 67.184
		elif (met <=  57754.00000000):
			utc_tt_diff = 68.184
		else:
			utc_tt_diff = 69.184
		mjdutc = ((met - utc_tt_diff) / 86400.0) + 51910 + 0.0007428703703
		met1 = Time(mjdutc,scale= 'utc',format = 'mjd')
		return met1

	@classmethod
	def utc_to_met(self,utc0):
		tt_time = Time(utc0, format='fits', scale='utc').mjd
		mmt = (tt_time - 0.0007428703703 - 51910) * 86400.0
		if mmt <= (252460801.000 - 65.184):
			dt = 65.184
		elif mmt <= (362793602.000 - 66.184):
			dt = 66.184
		elif mmt <= (457401603.000 - 67.184):
			dt = 67.184
		elif mmt <= (504921604.000 - 68.184):
			dt = 68.184
		else:
			dt = 69.184
		met = mmt + dt
		return met

	@classmethod
	def utc_time(self,utc0):
		tt = Time(utc0,format = 'fits',scale = 'utc')
		return tt

class GBM:
	def __init__(self,quaternion,sc_pos = None,gbm_time = None):
		if gbm_time is not None:
			if (isinstance(gbm_time, str)):
				self.time = GBMtime.utc_time(gbm_time)
			else:
				self.time = GBMtime.met_to_utc(gbm_time)
		else:
			self.time = None
		self.n0 = NaI0(quaternion)
		self.n1 = NaI1(quaternion)
		self.n2 = NaI2(quaternion)
		self.n3 = NaI3(quaternion)
		self.n4 = NaI4(quaternion)
		self.n5 = NaI5(quaternion)
		self.n6 = NaI6(quaternion)
		self.n7 = NaI7(quaternion)
		self.n8 = NaI8(quaternion)
		self.n9 = NaI9(quaternion)
		self.na = NaIA(quaternion)
		self.nb = NaIB(quaternion)
		self.b0 = BGO0(quaternion)
		self.b1 = BGO1(quaternion)
		self.detectors = OrderedDict(n0=self.n0,n1=self.n1,n2=self.n2,
				n3=self.n3,n4=self.n4,n5=self.n5,n6=self.n6,n7=self.n7,
				n8=self.n8,n9=self.n9,na=self.na,nb=self.nb,
				b0=self.b0,b1=self.b1)
		self.NaI_detectors = OrderedDict(n0=self.n0,n1=self.n1,n2=self.n2,
				n3=self.n3,n4=self.n4,n5=self.n5,n6=self.n6,n7=self.n7,
				n8=self.n8,n9=self.n9,na=self.na,nb=self.nb)
		self.BGO_detectors = OrderedDict(b0=self.b0,b1=self.b1)
		self.quaternion = quaternion
		self.sc_pos = sc_pos

	def get_centers(self, NaI=True, BGO=True):
		centers = {}
		if(NaI):
			for key in self.NaI_detectors.keys():
				centers[self.NaI_detectors[key].name] = self.NaI_detectors[key].center
		if(BGO):
			for key in self.BGO_detectors.keys():
				centers[self.BGO_detectors[key].name] = self.BGO_detectors[key].center
		return centers

	def get_good_centers(self, point=None, NaI=True, BGO=True):
		if point is not None:
			centers = {}
			if(NaI):
				for key in self.NaI_detectors.keys():
					if(self.NaI_detectors[key].contains_point(point)):
						centers[self.NaI_detectors[key].name] = self.NaI_detectors[key].center
			if(BGO):
				for key in self.BGO_detectors.keys():
					if(self.BGO_detectors[key].contains_point(point)):
						centers[self.BGO_detectors[key].name] = self.BGO_detectors[key].center
			return centers
		else:
			sys.exit('No source position sepcified!')
			return False

	def get_fov(self,radius,NaI=True,BGO=True):
		polys = {}
		detector_list = []
		if(NaI):
			for key in self.NaI_detectors.keys():
				polys[self.NaI_detectors[key].name] = self.NaI_detectors[key].get_fov(radius)
				detector_list.append(key)
		if(BGO):
			for key in self.BGO_detectors.keys():
				polys[self.BGO_detectors[key].name] = self.BGO_detectors[key].get_fov(radius)
				detector_list.append(key)
		return detector_list,polys

	def get_good_fov(self,radius,point=None,NaI=True,BGO=True):
		if point is not None:
			polys = {}
			detector_list = []
			if(NaI):
				for key in self.NaI_detectors.keys():
					if(self.NaI_detectors[key].contains_point(point)):
						polys[self.NaI_detectors[key].name] = self.NaI_detectors[key].get_fov(radius)
						detector_list.append(key)
			if(BGO):
				for key in self.BGO_detectors.keys():
					if(self.BGO_detectors[key].contains_point(point)):
						polys[self.BGO_detectors[key].name] = self.BGO_detectors[key].get_fov(radius)
						detector_list.append(key)
			return detector_list,polys
		else:
			sys.exit('No source position sepcified!')
			return False

	def get_separation(self,source=None,NaI=True,BGO=True):
		tab = Table(names=["Detector", "Separation"], dtype=["|S2", np.float64])
		if source is not None:
			if(NaI):
				for key in self.NaI_detectors.keys():
					sep = self.NaI_detectors[key].center.separation(source)
					tab.add_row([key, sep])
			if(BGO):
				for key in self.BGO_detectors.keys():
					sep = self.BGO_detectors[key].center.separation(source)
					tab.add_row([key, sep])
			tab['Separation'].unit = u.degree
			tab.sort("Separation")
			return tab
		else:
			sys.exit('No source position sepcified!')

	def get_earth_point(self):
		if self.sc_pos is not None:
			self.calc_earth_points()
			return self.earth_points
		else:
			sys.exit('No SC position!')

	def calc_earth_points(self):
		xyz_position = SkyCoord(x=self.sc_pos[0],
								y=self.sc_pos[1],
								z=self.sc_pos[2],
								frame='icrs',representation_type='cartesian')
		earth_radius = 6371. * u.km
		fermi_radius = np.sqrt((self.sc_pos ** 2).sum())
		horizon_angle = 90 - np.rad2deg(np.arccos((earth_radius / fermi_radius).to(u.dimensionless_unscaled)).value)
		horizon_angle = (180 - horizon_angle) * u.degree
		num_points = 3000
		ra_grid_tmp = np.linspace(0, 360, num_points)
		dec_range = [-90, 90]
		cosdec_min = np.cos(np.deg2rad(90.0 + dec_range[0]))
		cosdec_max = np.cos(np.deg2rad(90.0 + dec_range[1]))
		v = np.linspace(cosdec_min, cosdec_max, num_points)
		v = np.arccos(v)
		v = np.rad2deg(v)
		v -= 90.
		dec_grid_tmp = v
		ra_grid = np.zeros(num_points ** 2)
		dec_grid = np.zeros(num_points ** 2)
		itr = 0
		for ra in ra_grid_tmp:
			for dec in dec_grid_tmp:
				ra_grid[itr] = ra
				dec_grid[itr] = dec
				itr += 1
		all_sky = SkyCoord(ra=ra_grid, dec=dec_grid, frame='icrs', unit='deg')
		condition = all_sky.separation(xyz_position) > horizon_angle
		self.earth_points = all_sky[condition]

	def detector_plot(self,radius=60.0,point=None,good=False,projection='moll',
							lat_0=0,lon_0=180,map=None, 
							NaI=True,BGO=True,show_bodies=False):
		map_flag = False
		if map is None:
			fig = plt.figure(figsize=(20,20))
			ax = fig.add_subplot(111)
			map = Basemap(projection=projection,lat_0=lat_0,lon_0=lon_0,
					resolution='l',area_thresh=1000.0,celestial=True,ax=ax)
		else:
			map_flag = True
		if good and point:
			detector_list,fovs = self.get_good_fov(radius=radius,
											point=point,NaI=NaI,BGO=BGO)
		else:
			detector_list,fovs = self.get_fov(radius,NaI=NaI,BGO=BGO)
		if point:
			ra,dec = map(point.ra.value,point.dec.value)
			map.plot(ra,dec , '*', color='#f36c21' , markersize=20.)
		for key in detector_list:
			ra,dec = fovs[self.detectors[key].name]
			ra,dec = map(ra,dec)
			map.plot(ra,dec,'.',color = '#74787c',markersize = 3)
			x,y = map(self.detectors[key].center.icrs.ra.value,
						self.detectors[key].center.icrs.dec.value)
			plt.text(x-200000, y-200000,self.detectors[key].name,
										 color='#74787c', size=22)
		if show_bodies and self.sc_pos is not None:
			earth_points = self.get_earth_point()
			lon, lat = earth_points.ra.value, earth_points.dec.value
			lon,lat = map(lon,lat)
			map.plot(lon, lat, ',', color="#0C81F9", alpha=0.1, markersize=4.5)
			if self.time is not None:
				earth_r = get_body_barycentric('earth',self.time)
				moon_r = get_body_barycentric('moon',self.time)
				r_e_m = moon_r - earth_r
				r = self.sc_pos -np.array([r_e_m.x.value,r_e_m.y.value,r_e_m.z.value])*u.km
				moon_point_d = cartesian_to_spherical(-r[0],-r[1],-r[2])
				moon_ra,moon_dec = moon_point_d[2].deg,moon_point_d[1].deg
				moon_point = SkyCoord(moon_ra,moon_dec,frame='icrs', unit='deg')
				moon_ra,moon_dec = map(moon_point.ra.deg,moon_point.dec.deg)
				map.plot(moon_ra,moon_dec,'o',color = '#72777b',markersize = 20)
				plt.text(moon_ra,moon_dec,'  moon',size = 20)
			if show_bodies and self.time is not None:
				tmp_sun = get_sun(self.time)
				sun_position = SkyCoord(tmp_sun.ra.deg,tmp_sun.dec.deg,
											unit='deg', frame='icrs')
				sun_ra,sun_dec = map(sun_position.ra.value,sun_position.dec.value)
				map.plot(sun_ra,sun_dec ,'o',color='#ffd400',markersize=40)
				plt.text(sun_ra-550000,sun_dec-200000,'sun',size=20)
		if not map_flag:
			if projection == 'moll':
				az1 = np.arange(0, 360, 30)
				zen1 = np.zeros(az1.size) + 2
				azname = []
				for i in az1:
					azname.append(r'${\/%s\/^{\circ}}$' % str(i))
				x1, y1 = map(az1, zen1)
				for index, value in enumerate(az1):
					plt.text(x1[index], y1[index], azname[index], size=20)
			_ = map.drawmeridians(np.arange(0, 360, 30),dashes=[1,0],color='#d9d6c3')
			_ = map.drawparallels(np.arange(-90, 90, 15),dashes=[1,0],
								labels=[1,0,0,1], color='#d9d6c3',size=20)
			map.drawmapboundary(fill_color='#f6f5ec')



##################
# SOME FUNCTIONS #
##################

def timer(func):
	"""Print the runtime of the decorated function"""
	@functools.wraps(func)
	def wrapper_timer(*args,**kwargs):
		print(f"Running {func.__name__!r} ...")
		start_time=time.perf_counter()
		value=func(*args,**kwargs)
		end_time=time.perf_counter()
		run_time=end_time-start_time
		print(f"Finished {func.__name__!r} in {run_time:.4f} sec")
		return value
	return wrapper_timer

def open_fit(file_link):
	f = fits.open(file_link)
	time = f[1].data.field(0)
	qsj1 = f[1].data.field(1)
	qsj2 = f[1].data.field(2)
	qsj3 = f[1].data.field(3)
	qsj4 = f[1].data.field(4)
	pos_x = f[1].data.field(8)
	pos_y = f[1].data.field(9)
	pos_z = f[1].data.field(10)
	return time,qsj1,qsj2,qsj3,qsj4,pos_x,pos_y,pos_z

def find_right_list(file_link, met):
	time,qsj1,qsj2,qsj3,qsj4,pos_x,pos_y,pos_z = open_fit(file_link)
	dt = (time - met)**2
	dt = np.array(dt)
	dtmin=dt.min()
	if dtmin >= 1: 
		qsj, pos = None, None
	else:
		index = np.where(dt == dtmin)
		qsj = np.array([qsj1[index][0],qsj2[index][0],qsj3[index][0],qsj4[index][0]])
		pos = np.array([pos_x[index][0],pos_y[index][0],pos_z[index][0]])
	return qsj, pos

def met2utc_shao(myMET):
	UTC0 = Time('2001-01-01',format='iso',scale='utc')
	if isinstance(myMET,(list,tuple,np.ndarray)):
		myMETsize = len(myMET)
		utc_tt_diff = np.zeros(myMETsize)
		#from Fermi MET to UTC
		# 4 leap seconds after 2007:
		#'2008-12-31 23:59:60' MET=252460801.000000
		#'2012-06-30 23:59:60' MET=362793602.000000
		#'2015-06-30 23:59:60' MET=457401603.000000
		#'2016-12-31 23:59:60' MET=504921604.000000
		for i in range(myMETsize):
			if myMET[i] < 237693816.736: # valid data start at 2008-07-14 02:03:35.737
				print('**** ERROR: value Met must be larger than 237693816.736!!! ****')
			elif myMET[i] <= 252460801.000:
				utc_tt_diff[i] = 33.0
			elif myMET[i] <= 362793602.000:
				utc_tt_diff[i] = 34.0
			elif myMET[i] <= 457401603.000:
				utc_tt_diff[i] = 35.0
			elif myMET[i] <= 504921604.000:
				utc_tt_diff[i] = 36.0
			else:
				utc_tt_diff[i] = 37.0
		myTimeGPS = Time(np.array(myMET)+UTC0.gps-utc_tt_diff,format='gps')
		return myTimeGPS.iso
	elif np.isscalar(myMET):
		if myMET < 237693816.736: # # valid data start at 2008-07-14 02:03:35.737
			print('**** ERROR: value Met must be larger than 237693816.736!!! ****')
		elif myMET <= 252460801.000:
			utc_tt_diff = 33.0
		elif myMET <= 362793602.000:
			utc_tt_diff = 34.0
		elif myMET <= 457401603.000:
			utc_tt_diff = 35.0
		elif myMET <= 504921604.000:
			utc_tt_diff = 36.0
		else:
			utc_tt_diff = 37.0
		myTimeGPS = Time(myMET+UTC0.gps-utc_tt_diff,format='gps')
		return myTimeGPS.iso
	else:
		print('Check your input format!')
		return None

Det_pointing=[SkyCoord(45.89, 90-20.58, unit='deg'),
			SkyCoord(45.11, 90-45.31, unit='deg'),
			SkyCoord(58.44, 90-90.21, unit='deg'),
			SkyCoord(314.87, 90-45.24, unit='deg'),
			SkyCoord(303.15, 90-90.27, unit='deg'),
			SkyCoord(3.35, 90-89.97, unit='deg'),
			SkyCoord(224.93, 90-20.43, unit='deg'),
			SkyCoord(224.62, 90-46.18, unit='deg'),
			SkyCoord(236.61, 90-89.97, unit='deg'),
			SkyCoord(135.19, 90-45.55, unit='deg'),
			SkyCoord(123.73, 90-90.42, unit='deg'),
			SkyCoord(183.74, 90-90.32, unit='deg')]

def if_closeDet(det_list):
	"""Check at least three detectors are closely related with each other
	"""
	test = False
	for det3_1, det3_2, det3_3 in itertools.combinations(det_list,3):
		inner_test = True		
		for det1, det2 in itertools.combinations([det3_1, det3_2, det3_3],2):
			if Det_pointing[det1].separation(Det_pointing[det2]).deg > 60.0:
				inner_test = False
				break			
		if inner_test:
			test = True			
			break
	return test

# https://en.wikipedia.org/wiki/Normal_distribution
# https://en.wikipedia.org/wiki/Poisson_distribution
# cdfprob = 0.997300203937 # 3 sigma
# cdfprob = 0.954499736104 # 2 sigma
# cdfprob = 0.682689492137 # 1 sigma
def norm_pvalue(sigma=2.0):
	p = stats.norm.cdf(sigma)-stats.norm.cdf(-sigma)
	return p


def query_fermigbrst(cdir='./'):
	fermigbrst = cdir+'/fermigbrst.txt'
	if not os.path.exists(fermigbrst):
		usersjar = get_usersjar()
		assert os.path.exists(usersjar), """'users.jar' is not available! 
			download users.jar at:
			https://heasarc.gsfc.nasa.gov/xamin/distrib/users.jar
			and update the path of usersjar in 'personal_settings.py'."""
		java_ready = os.system("java --version")
		assert not java_ready, """java not properly installed!
			Install Oracle Java 10 (JDK 10) in Ubuntu or Linux Mint from PPA
			$ sudo add-apt-repository ppa:linuxuprising/java
			$ sudo apt update
			$ sudo apt install oracle-java10-installer"""
		fields = ("trigger_name,t90,t90_error,t90_start,"
			"ra,dec,Error_Radius,lii,bii,"
			"fluence,fluence_error,"
			"flux_64,flux_64_error,flux_64_time")
		print('querying fermigbrst catalog using HEASARC-Xamin-users.jar ...')
		query_ready = os.system("java -jar "+usersjar+" table=fermigbrst fields="
				+fields+" sortvar=trigger_name output="+cdir+"/fermigbrst.txt")
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
	# content = ['rate','base','net']
	BGOmaxcolorvalue = 0.0
	NaImaxcolorvalue = 0.0
	f = h5py.File(baseresultdir+'/base.h5',mode='r')
	fig, axes = plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=False)
	for i in range(14):
		#data in firt and last two channels of BGO and NaI are not shown
		#ignore 0,1,126,127, notice 2-125
		if content == 'rate':
			C = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][0] for ch in np.arange(ch1,ch2+1) ])
		elif content == 'base':
			C = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][1] for ch in np.arange(ch1,ch2+1) ])
		elif content == 'net':
			C = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] for ch in np.arange(ch1,ch2+1) ])
		if i <= 1:
			if BGOmaxcolorvalue < C.max():
				BGOmaxcolorvalue = C.max()
		else:
			if NaImaxcolorvalue < C.max():
				NaImaxcolorvalue = C.max()	
	for i in range(14):
		ttefile = glob(datadir+'/'+'glg_tte_'+Det[i]+'_'+bnname+'_v*.fit')
		hdu = fits.open(ttefile[0])
		ebound = hdu['EBOUNDS'].data
		emin = ebound.field(1)
		emin = emin[ch1:ch2+1]
		emax = ebound.field(2)
		emax = emax[ch1:ch2+1]				
		x = tbins
		y = np.concatenate((emin,[emax[-1]]))
		X, Y = np.meshgrid(x, y)
		if content == 'rate':
			C = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][0] 
							for ch in np.arange(ch1,ch2+1) ])
		elif content == 'base':
			C = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][1] 
							for ch in np.arange(ch1,ch2+1) ])
		elif content == 'net':
			C=np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] 
							for ch in np.arange(ch1,ch2+1) ])
		C[C < 1] = 1
		if i <= 1:
			pcmBGO = axes[i//2,i%2].pcolormesh(X, Y, C,
				norm = colors.LogNorm(vmin=1.0, vmax=BGOmaxcolorvalue),
				cmap='rainbow')
			format_countmap_axes(axes[i//2,i%2],Det[i],
						tbins[0],tbins[-1],[1000,10000])
		else:
			pcmNaI = axes[i//2,i%2].pcolormesh(X, Y, C,
				norm = colors.LogNorm(vmin=1.0, vmax=NaImaxcolorvalue),
				cmap='rainbow')
			format_countmap_axes(axes[i//2,i%2],Det[i],
							tbins[0],tbins[-1],[10,100])
		axes[i//2,i%2].set_xlim([viewt1,viewt2])				
	cbarBGO = fig.colorbar(pcmBGO, ax=axes[0,], orientation='vertical',
							fraction=0.005, aspect=100/6)
	cbarNaI = fig.colorbar(pcmNaI, ax=axes[1:,], orientation='vertical',
							fraction=0.005, aspect=100)
	cbarBGO.ax.tick_params(labelsize=25)
	cbarNaI.ax.tick_params(labelsize=25)
	fig.text(0.07, 0.5, 'Energy (KeV)', ha='center', va='center',
			rotation='vertical',fontsize=30)
	fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)
	fig.text(0.5, 0.92, bnname, ha='center', va='center',fontsize=30)
	plt.savefig(resultdir+'/'+content+'_countmap.png')
	plt.close()
	f.close()

#function for writing single spectrum (PHA I) into a fit file 
def write_phaI(spectrum_rate,bnname,det,t1,t2,outfile):
	#hdu0 / PrimaryHDU
	header0 = fits.Header()
	header0.append(('creator', 'Shao', 'The name who created this PHA file'))
	header0.append(('telescop', 'Fermi', 'Name of mission/satellite'))
	header0.append(('bnname', bnname, 'Burst Name'))
	header0.append(('t1', t1, 'Start time of the PHA slice'))
	header0.append(('t2', t2, 'End time of the PHA slice'))
	hdu0 = fits.PrimaryHDU(header=header0)
	#hdu1 / data unit
	a1 = np.arange(128)
	col1 = fits.Column(name='CHANNEL', format='1I', array=a1)
	col2 = fits.Column(name='COUNTS', format='1D', unit='COUNTS', 
						array=spectrum_rate)
	#col3 = fits.Column(name='STAT_ERR', format='1D', unit='COUNTS', array=bkg_uncertainty)
	#hdu = fits.BinTableHDU.from_columns([col1, col2, col3])
	hdu1 = fits.BinTableHDU.from_columns([col1, col2])
	header = hdu1.header
	header.append(('extname', 'SPECTRUM', 'Name of this binary table extension'))
	header.append(('telescop', 'GLAST', 'Name of mission/satellite'))
	header.append(('instrume', 'GBM', 'Specific instrument used for observation'))
	header.append(('filter', 'None', 'The instrument filter in use (if any)'))
	header.append(('exposure', 1., 'Integration time in seconds'))
	header.append(('areascal', 1., 'Area scaling factor'))
	header.append(('backscal', 1., 'Background scaling factor'))
	if outfile[-3:] == 'pha':
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
	header.comments['TTYPE1'] = 'Label for column 1'
	header.comments['TFORM1'] = '2-byte INTERGER'
	header.comments['TTYPE2'] = 'Label for column 2'
	header.comments['TFORM2'] = '8-byte DOUBLE'
	header.comments['TUNIT2'] = 'Unit for colum 2'
	#header.comments['TTYPE3']='Label for column 3'
	#header.comments['TFORM3']='8-byte DOUBLE'
	#header.comments['TUNIT3']='Unit for colum 3'
	hdul = fits.HDUList([hdu0, hdu1])
	hdul.writeto(outfile)



def copy_rspI(bnname,det,outfile):
	shortyear = bnname[2:4]
	fullyear = '20'+shortyear
	datadir = databasedir+'/'+fullyear+'/'+bnname+'/'
	rspfile = glob(datadir+'/'+'glg_cspec_'+det+'_'+bnname+'_v*.rsp')
	assert len(rspfile) == 1, ('response file is missing for '
							'glg_cspec_'+det+'_'+bnname+'_v*.rsp')
	rspfile = rspfile[0]
	os.system('cp '+rspfile+' '+outfile)
	
	
###################	
# BEGIN class GRB #
###################

class GRB:
	def __init__(self, bnname, resultdir):
		self.bnname = bnname
		shortyear = self.bnname[2:4]
		fullyear = '20'+shortyear
		self.datadir = databasedir+'/'+fullyear+'/'+self.bnname+'/'
		self.dataready = True
		for i in range(14):
			ttefile = glob(self.datadir+'glg_tte_'
							+Det[i]+'_'+self.bnname+'_v*.fit')
			if not len(ttefile) == 1:
				self.dataready = False
			else:
				hdu = fits.open(ttefile[0])
				event = hdu['EVENTS'].data.field(0)
				if len(event) < 10:
					self.dataready = False
		if self.dataready:	
			self.baset1 = None
			self.baset2 = None
			self.binwidth = None
			self.tbins = None
			self.baseresultdir = None
			self.phaIresultdir = None
			self.GTI1 = None
			self.GTI2 = None
			#resultdir = os.getcwd()+'/results/'
			self.resultdir = resultdir+'/'+bnname+'/'
			if not os.path.exists(self.resultdir):
				os.makedirs(self.resultdir)
			self.baseresultdir = self.resultdir+'/base/'
			self.phaIresultdir = self.resultdir+'/phaI/'

			# determine GTI1 and GTI2
			GTI_t1 = np.zeros(14)
			GTI_t2 = np.zeros(14)
			for i in range(14):
				ttefile = glob(self.datadir+'glg_tte_'+Det[i]
								+'_'+self.bnname+'_v*.fit')
				hdu = fits.open(ttefile[0])
				trigtime = hdu['Primary'].header['TRIGTIME']
				data = hdu['EVENTS'].data
				time = data.field(0)-trigtime
				GTI0_t1 = time[0]
				GTI0_t2 = time[-1]
				timeseq1 = time[:-1]
				timeseq2 = time[1:]
				deltime = timeseq2-timeseq1
				# find a gap larger than 5 second between two events 
				delindex = deltime > 5 
				if len(timeseq1[delindex]) >= 1:
					GTItmp_t1 = np.array(np.append([GTI0_t1],timeseq2[delindex]))
					GTItmp_t2 = np.array(np.append(timeseq1[delindex],[GTI0_t2]))
					for kk in np.arange(len(GTItmp_t1)):
						if GTItmp_t1[kk] <= 0.0 and GTItmp_t2[kk] >= 0.0:
							GTI_t1[i] = GTItmp_t1[kk]
							GTI_t2[i] = GTItmp_t2[kk]
				else:
					GTI_t1[i] = GTI0_t1
					GTI_t2[i] = GTI0_t2
			self.GTI1 = np.max(GTI_t1)
			self.GTI2 = np.min(GTI_t2)

	def rawlc(self,viewt1=-50,viewt2=300,binwidth=0.064):		
		viewt1 = np.max([self.GTI1,viewt1])
		viewt2 = np.min([self.GTI2,viewt2])
		assert viewt1 < viewt2, self.bnname+': Inappropriate view times for rawlc!'
		if not os.path.exists(self.resultdir+'/'+'raw_lc.png'):
			#print('plotting raw_lc.png ...')
			tbins = np.arange(viewt1,viewt2+binwidth,binwidth)
			fig, axes = plt.subplots(7,2,figsize=(32, 20),
								sharex=True,sharey=False)
			for i in range(14):
				ttefile = glob(self.datadir+'/'+'glg_tte_'
							+Det[i]+'_'+self.bnname+'_v*.fit')
				hdu = fits.open(ttefile[0])
				trigtime = hdu['Primary'].header['TRIGTIME']
				data = hdu['EVENTS'].data
				time = data.field(0)-trigtime
				ch = data.field(1)
				#data in firt and last two channels of BGO and NaI are not used
				#ignore 0,1,2,125,126,127, notice 3-124
				goodindex = (ch >= ch1) & (ch <= ch2)  
				time = time[goodindex]
				ebound = hdu['EBOUNDS'].data
				emin = ebound.field(1)
				emin = emin[ch1:ch2+1]
				emax = ebound.field(2)
				emax = emax[ch1:ch2+1]
				histvalue, histbin = np.histogram(time,bins=tbins)
				plotrate = histvalue/binwidth
				plotrate = np.concatenate(([plotrate[0]],plotrate))
				axes[i//2,i%2].plot(histbin,plotrate,drawstyle='steps')
				axes[i//2,i%2].set_xlim([viewt1,viewt2])
				axes[i//2,i%2].tick_params(labelsize=25)
				axes[i//2,i%2].text(0.05,0.85,Det[i],fontsize=25,
									transform=axes[i//2,i%2].transAxes)
				axes[i//2,i%2].text(0.7,0.80,(str(round(emin[0],1))
							+'-'+str(round(emax[-1],1))+' keV'),
							transform=axes[i//2,i%2].transAxes,fontsize=25)
			fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center',
						va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center',
								va='center',fontsize=30)		
			fig.text(0.5, 0.92, self.bnname, ha='center',
								va='center',fontsize=30)
			plt.savefig(self.resultdir+'/raw_lc.png')
			plt.close()
			
	def skymap(self, catalog_ra=None, catalog_dec=None):
		if not os.path.exists(self.resultdir+"/skymap_animated.gif"):
			trigdatfile = glob(self.datadir+'/glg_trigdat_all_'+self.bnname+'_v*.fit')
			hdu = fits.open(trigdatfile[0])
			trigtime_met = hdu['Primary'].header['TRIGTIME']
			timelist_met = np.arange(trigtime_met-10, trigtime_met+40, 10)
			ra_obj = hdu['Primary'].header['RA_OBJ']
			dec_obj = hdu['Primary'].header['DEC_OBJ']
			grb = SkyCoord(ra_obj, dec_obj, unit='deg',frame = 'icrs')
			for seq, t_met in enumerate(timelist_met):
				timestr = met2utc_shao(t_met)
				t = Time(timestr,format='iso',scale='utc')
				timestrisot = t.isot
				year = timestr[:4]
				yearshort = timestr[2:4]
				month = timestr[5:7]
				day = timestr[8:10]
				dailydatabasedir_ab = get_dailydatabasedirs()
				if int(year) >= 2013:
					dailydatabasedir = dailydatabasedir_ab[0]
				else:
					dailydatabasedir = dailydatabasedir_ab[1]
				localdir = dailydatabasedir+'/'+year+'/'+month+'/'+day+'/'
				filef = 'glg_poshist_all_'+yearshort+month+day
				filelist = glob(localdir+filef+'*')
				if len(filelist) != 1:
					#print('***ERROR:  check if '+filef+' is available***')
					break
				filelink = filelist[0]
				qsj, pos = find_right_list(filelink, t_met)
				if type(qsj) == type(None):
					break
				myGBM = GBM(qsj,pos*u.m,timestrisot)		
				fig = plt.figure(figsize=(20,10))
				ax = fig.add_subplot(111)
				map = Basemap(projection='moll', lat_0=0, lon_0=180, resolution='l',
									area_thresh=1000.0, celestial=True, ax=ax)
				myGBM.detector_plot(radius=10, lat_0=0, lon_0=90, point=grb,
										show_bodies=True, BGO=False, map=map)
				x,y=map(grb.ra.value,grb.dec.value)
				label='  '+self.bnname
				plt.text(x, y, label, fontsize=10)
				az1 = np.arange(0,360,30)
				zen1 = np.zeros(az1.size)+2
				azname = []
				for i in az1:
					azname.append(r'${\/%s\/^{\circ}}$'%str(i))
				x1,y1 = map(az1,zen1)
				for index,value in enumerate(az1):
					plt.text(x1[index],y1[index],azname[index],size = 20)
				_ = map.drawmeridians(np.arange(0, 360, 30), dashes=[1,0],color='#d9d6c3')
				_ = map.drawparallels(np.arange(-90, 90, 15), dashes=[1,0],
									labels=[1,0,0,1], color='#d9d6c3',size=20)
				map.drawmapboundary(fill_color='#f6f5ec')
				if catalog_ra:
					ra = "{}h{}m{}s".format(catalog_ra[0],catalog_ra[1],catalog_ra[2])
					dec = "{}d{}m{}s".format(catalog_dec[0],catalog_dec[1],catalog_dec[2])
					mysource = SkyCoord(ra, dec, frame = 'icrs')
					x, y = map(mysource.ra.deg,mysource.dec.deg)
					map.plot(x, y, color='r',marker='o',markersize=20,ls='None')
					plt.text(x, y, '    GBM catalog', fontsize=10)
				plt.title(timestr+' (T0+'+str((seq*10-10))+' s)',fontsize=25)
				plt.savefig(self.resultdir+'/skymap_'+str(seq)+'.png')
				plt.close()
			if os.path.exists(self.resultdir+'/skymap_0.png'):
				os.system("convert -delay 40 -resize 800x600 -loop 0 "+self.resultdir+"/skymap_*.png "+self.resultdir+"/skymap_animated.gif")

		
	def base(self,baset1=-50,baset2=300,binwidth=0.064):
		self.baset1 = np.max([self.GTI1,baset1])
		self.baset2 = np.min([self.GTI2,baset2])
		self.binwidth = binwidth
		self.tbins = np.arange(self.baset1,self.baset2+self.binwidth,
											self.binwidth)
		assert self.baset1 < self.baset2, self.bnname+': Inappropriate base times!'
		if not os.path.exists(self.baseresultdir):
			#print('creating baseline in '+self.baseresultdir+' ...')
			os.makedirs(self.baseresultdir)
			f = h5py.File(self.baseresultdir+'/base.h5',mode='w')
			for i in range(14):
				grp = f.create_group(Det[i])
				ttefile = glob(self.datadir+'/'+'glg_tte_'+Det[i]
										+'_'+self.bnname+'_v*.fit')
				hdu = fits.open(ttefile[0])	
				trigtime = hdu['Primary'].header['TRIGTIME']
				data = hdu['EVENTS'].data
				timedata = data.field(0)-trigtime
				chdata = data.field(1)
				for ch in range(128):
					time_selected = timedata[chdata==ch]
					histvalue, histbin = np.histogram(time_selected,
													bins=self.tbins)
					rate = histvalue/binwidth
					r.assign('rrate',rate) 
					r("y=matrix(rrate,nrow=1)")
					fillPeak_hwi = str(int(5/binwidth))
					fillPeak_int = str(int(len(rate)/10))
					r("rbase=baseline(y,lam=6,hwi="+fillPeak_hwi
						+",it=10,int="+fillPeak_int+",method='fillPeaks')")
					r("bs=getBaseline(rbase)")
					r("cs=getCorrected(rbase)")
					bs = r('bs')[0]
					cs = r('cs')[0]
					# correct negative base to 0 and recover the net value to original rate
					corrections_index = (bs < 0)
					bs[corrections_index] = 0
					cs[corrections_index] = rate[corrections_index]
					f['/'+Det[i]+'/ch'+str(ch)] = np.array([rate,bs,cs])
			f.flush()
			f.close()
	
	def plotbase(self):
		self.plotbasedir = self.resultdir+'/plotbase/'
		if not os.path.exists(self.plotbasedir):
			assert os.path.exists(self.baseresultdir), ("Should have run base() "
				"before running plotbase()!")
			os.makedirs(self.plotbasedir)
			f = h5py.File(self.baseresultdir+'/base.h5',mode='r')
			for ch in range(128):
				fig, axes = plt.subplots(7,2,figsize=(32, 20),
									sharex=True,sharey=False)
				for i in range(14):
					plotrate = f['/'+Det[i]+'/ch'+str(ch)][()][0] #rate
					plotrate = np.concatenate(([plotrate[0]],plotrate))
					axes[i//2,i%2].plot(self.tbins,plotrate,drawstyle='steps',
										lw=3.0,color='tab:blue')
					plotbase = f['/'+Det[i]+'/ch'+str(ch)][()][1] #base
					plottime = self.tbins[:-1]+self.binwidth/2.0
					axes[i//2,i%2].plot(plottime,plotbase,linestyle='--',
										lw=4.0,color='tab:orange')
					axes[i//2,i%2].set_xlim([self.tbins[0],self.tbins[-1]])
					axes[i//2,i%2].tick_params(labelsize=25)
					axes[i//2,i%2].text(0.05,0.85,Det[i],
						transform=axes[i//2,i%2].transAxes,fontsize=25)
				fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center',
						va='center',rotation='vertical',fontsize=30)
				fig.text(0.5, 0.05, 'Time (s)', ha='center',
									va='center',fontsize=30)		
				fig.text(0.5, 0.92, 'ch'+str(ch), ha='center',
									va='center',fontsize=30)			
				plt.savefig(self.plotbasedir+'/ch_'+str(ch)+'.png')
				plt.close()
			f.close()
		

	def check_gaussian_total_rate(self):
		if not os.path.exists(self.resultdir+'/check_gaussian_total_rate.png'):
			assert os.path.exists(self.baseresultdir), ('Should have run base() '
				'before running check_gaussian_total_rate()!')
			f = h5py.File(self.baseresultdir+'/base.h5',mode='r')
			fig, axes = plt.subplots(7,2,figsize=(32, 20),
									sharex=False,sharey=False)
			for i in range(14):
				cRate = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][0] 
								for ch in np.arange(ch1,ch2+1) ])
				totalRate = np.sum(cRate,axis=0)
				median = np.median(totalRate)
				totalRate_median_part = totalRate[(totalRate > (0.1*median)) 
											& (totalRate < (1.5*median))]
				bins = np.arange(totalRate.min(),totalRate.max(),
				(totalRate_median_part.max()-totalRate_median_part.min())/30)
				histvalue, histbin = np.histogram(totalRate,bins=bins)
				histvalue = np.concatenate(([histvalue[0]],histvalue))
				axes[i//2,i%2].fill_between(histbin,histvalue,step='pre',
											label='Observed total rate')			
				loc,scale = stats.norm.fit(totalRate_median_part)
				Y = stats.norm(loc=loc,scale=scale)
				x = np.linspace(totalRate_median_part.min(),
							totalRate_median_part.max(),num=100)
				axes[i//2,i%2].plot(x,Y.pdf(x)*totalRate.size*(bins[1]-bins[0]),
								label='Gaussian Distribution',
								linestyle='--',lw=3.0,color='tab:orange')
				axes[i//2,i%2].tick_params(labelsize=25)
				axes[i//2,i%2].text(0.5,0.8,Det[i],fontsize=25,
							transform=axes[i//2,i%2].transAxes)
				axes[i//2,i%2].axvline(totalRate_median_part.min(),
							ls='--',lw=1,color='k',label='Fitting region')
				axes[i//2,i%2].axvline(totalRate_median_part.max(),
										ls='--',lw=1,color='k')
				if i==1:
					axes[i//2,i%2].legend(fontsize=20)
			fig.text(0.07, 0.5, 'Numbers', ha='center', va='center',
								rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Total raw rate (s$^{-1}$; between'
				 				+str(self.baset1)+'--'+str(self.baset2)+'s)',
			    				ha='center', va='center',fontsize=30)		
			plt.savefig(self.resultdir+'/check_gaussian_total_rate.png')
			plt.close()
			f.close()

	def check_gaussian_net_rate(self,sigma=3):
		if not os.path.exists(self.resultdir+'/check_gaussian_net_rate.png'):
			assert os.path.exists(self.baseresultdir), ('Should have run base() '
							'before running check_gaussian_net_rate()!')
			f = h5py.File(self.baseresultdir+'/base.h5',mode='r')
			fig, axes = plt.subplots(7,2,figsize=(32, 20),
									sharex=False,sharey=False)
			for i in range(14):
				cNet = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] 
									for ch in np.arange(ch1,ch2+1) ])
				totalNet = np.sum(cNet,axis=0)
				#clip the outliers and fit the central median part
				mask = sigma_clip(totalNet,sigma=5,maxiters=5,stdfunc=mad_std).mask
				myfilter = list(map(operator.not_, mask))
				totalNet_median_part = totalNet[myfilter]
				bins = np.arange(totalNet.min(),totalNet.max(),
					(totalNet_median_part.max()-totalNet_median_part.min())/25)
				histvalue, histbin = np.histogram(totalNet,bins=bins)
				histvalue = np.concatenate(([histvalue[0]],histvalue))
				axes[i//2,i%2].fill_between(histbin,histvalue,step='pre',
												label='Observed net rate')
				loc,scale = stats.norm.fit(totalNet_median_part)
				Y = stats.norm(loc=loc,scale=scale)
				x = np.linspace(totalNet_median_part.min(),
								totalNet_median_part.max(),
								num=100)
				axes[i//2,i%2].plot(x,Y.pdf(x)*totalNet.size*(bins[1]-bins[0]),
							label='Gaussian Distribution within clipped region',
							linestyle='--',lw=3.0,color='tab:orange')
				axes[i//2,i%2].tick_params(labelsize=25)
				axes[i//2,i%2].text(0.5,0.8,Det[i],fontsize=25,
								transform=axes[i//2,i%2].transAxes)
				gaussian_level = Y.interval(norm_pvalue(sigma))
				axes[i//2,i%2].axvline(gaussian_level[0],ls='--',lw=2,
							color='green',label=str(sigma)+'$\sigma$ level')
				axes[i//2,i%2].axvline(gaussian_level[1],ls='--',lw=2,
							color='green')
				if i == 1:
					axes[i//2,i%2].legend(fontsize=20)
			fig.text(0.07, 0.5, 'Numbers', ha='center', va='center',
									rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Total net rate (s$^{-1}$; between'
						+str(self.baset1)+'--'+str(self.baset2)+'s)',
						ha='center', va='center',fontsize=30)		
			plt.savefig(self.resultdir+'/check_gaussian_net_rate.png')
			plt.close()
			f.close()

	def plot_gaussian_level_over_net_lc(self,viewt1=-50,viewt2=300,sigma=3):
		if not os.path.exists(self.resultdir+'/gaussian_level_over_net_lc.png'):
			assert os.path.exists(self.baseresultdir), ('Should have run base() '
					'before running plot_gaussian_level_over_net_lc()!')
			viewt1 = np.max([self.baset1,viewt1])
			viewt2 = np.min([self.baset2,viewt2])
			f = h5py.File(self.baseresultdir+'/base.h5',mode='r')
			fig, axes = plt.subplots(7,2,figsize=(32, 20),
									sharex=True,sharey=False)
			ylim = np.zeros((14,2))
			for i in range(14):
				cNet = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] 
									for ch in np.arange(ch1,ch2+1) ])
				totalNet = np.sum(cNet,axis=0)
				#clip the outliers and fit the central median part
				mask = sigma_clip(totalNet,sigma=3,maxiters=20,stdfunc=mad_std).mask
				myfilter = list(map(operator.not_, mask))
				totalNet_median_part = totalNet[myfilter]
				loc,scale = stats.norm.fit(totalNet_median_part)
				Y = stats.norm(loc=loc,scale=scale)
				gaussian_level = Y.interval(norm_pvalue(sigma))
				totalNet = np.concatenate(([totalNet[0]],totalNet))
				axes[i//2,i%2].plot(self.tbins,totalNet,drawstyle='steps',
									lw=3.0,color='tab:blue')
				axes[i//2,i%2].axhline(gaussian_level[1],
					ls='--',lw=3,color='orange',
					label=str(sigma)+'$\sigma$ level of gaussian background')
				axes[i//2,i%2].tick_params(labelsize=25)
				axes[i//2,i%2].set_xlim([viewt1,viewt2])
				ylim[i] = axes[i//2,i%2].get_ylim()
				axes[i//2,i%2].text(0.05,0.85,Det[i],fontsize=25,
								transform=axes[i//2,i%2].transAxes)
				if i == 1:
					axes[i//2,i%2].legend(fontsize=20)
			#reset the ylims to same values
			BGOymax = np.max([ylim[i,1] for i in range(2)])
			NaIymax = np.max([ylim[i+2,1] for i in range(12)])
			for i in range(14):
				if i <= 1:
					axes[i//2,i%2].set_ylim([0,BGOymax])
				else:
					axes[i//2,i%2].set_ylim([0,NaIymax])
			fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center',
							va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center', 
								va='center',fontsize=30)		
			fig.text(0.5, 0.92, self.bnname, ha='center',
								va='center',fontsize=30)		
			plt.savefig(self.resultdir+'/gaussian_level_over_net_lc.png')
			plt.close()
			f.close()
			
			
# check SNR
	def check_snr(self,viewt1=-50,viewt2=300):
		if not os.path.exists(self.resultdir+'/check_SNR.png'):
			assert os.path.exists(self.baseresultdir), ('Should have run base() '
					'before running check_snr()!')
			viewt1 = np.max([self.baset1,viewt1])
			viewt2 = np.min([self.baset2,viewt2])
			f = h5py.File(self.baseresultdir+'/base.h5',mode='r')
			fig, axes = plt.subplots(7,2,figsize=(32, 20),
								sharex=True,sharey=False)
			ylim = np.zeros((14,2))
			for i in range(14):
				cNet = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] 
								for ch in np.arange(ch1,ch2+1) ])
				totalNet = np.sum(cNet,axis=0)
				median = np.median(totalNet)
				totalNet_median_part = totalNet[totalNet<5*median]
				loc,scale = stats.norm.fit(totalNet_median_part)
				totalNet = np.concatenate(([totalNet[0]],totalNet))
				snr = (totalNet-loc)/scale
				axes[i//2,i%2].plot(self.tbins,snr,drawstyle='steps',
										lw=3.0,color='tab:blue')
				axes[i//2,i%2].tick_params(labelsize=25)
				axes[i//2,i%2].set_xlim([viewt1,viewt2])
				axes[i//2,i%2].text(0.05,0.85,Det[i],fontsize=25,
							transform=axes[i//2,i%2].transAxes)
				axes[i//2,i%2].axhline(y=3,color='orange',ls='--',
								lw=3,zorder=2,label='SNR=3')
				for t in self.tbins[snr > 3]:
					axes[i//2,i%2].axvline(x=t,ymin=0.95,color='red',zorder=2)
				if i == 1:
					axes[i//2,i%2].legend(fontsize=20)
			fig.text(0.07, 0.5, 'Signal-to-noise ratio', ha='center',
					va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center',
								va='center',fontsize=30)		
			fig.text(0.5, 0.92, self.bnname, ha='center',
								va='center',fontsize=30)			
			plt.savefig(self.resultdir+'/check_SNR.png')
			plt.close()
			f.close()			
			

# check pulse based on plot_gauss_level_over_net_lc above
	def check_pulse(self,viewt1=-50,viewt2=300,sigma=3):
		if not os.path.exists(self.resultdir+'/check_pulse.png'):
			assert os.path.exists(self.baseresultdir), ('Should have run base() '
						'before running check_pulse()!')
			viewt1 = np.max([self.baset1,viewt1])
			viewt2 = np.min([self.baset2,viewt2])
			f = h5py.File(self.baseresultdir+'/base.h5',mode='r')
			fig, axes = plt.subplots(7,2,figsize=(32, 20),
									sharex=True,sharey=False)
			ylim = np.zeros((14,2))
			positiveIndex_2d = [0]*12
			negativeIndex = []
			goodIndex = []
			badIndex = []
			#search for signals over guassian level
			for i in range(14):
				cNet = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] 
									for ch in np.arange(ch1,ch2+1) ])
				totalNet = np.sum(cNet,axis=0)
				#clip the outliers and fit the central median part
				mask = sigma_clip(totalNet,sigma=3,maxiters=20,stdfunc=mad_std).mask
				myfilter = list(map(operator.not_, mask))
				totalNet_median_part = totalNet[myfilter]
				loc,scale = stats.norm.fit(totalNet_median_part)
				Y = stats.norm(loc=loc,scale=scale)
				gaussian_level = Y.interval(norm_pvalue(sigma))
				totalNet = np.concatenate(([totalNet[0]],totalNet))
				# search NaI detectors for tbins indice over and under gaussian level
				if i >= 2:
					#positiveIndex.extend(np.where(totalNet>gaussian_level[1])[0])
					positiveIndex_2d[i-2] = np.where(totalNet>gaussian_level[1])[0]
					negativeIndex.extend(np.where(totalNet<gaussian_level[0])[0])
			# search tbin where at lease 3 detectctors have the signal
			# stored in goodIndex
			positiveIndex=list(np.concatenate(positiveIndex_2d,axis=0).flatten())
			positiveIndex_set = set(positiveIndex)
			for seq in positiveIndex_set:
				if positiveIndex.count(seq) >= 3:
					goodIndex.extend([seq])
			# check at least 2 detectors are closely related
			goodIndex_2d = [0]*len(goodIndex)
			for seq, index in enumerate(goodIndex):
				goodIndex_2d[seq] = []
				for i in range(12):
					if index in positiveIndex_2d[i]:
						goodIndex_2d[seq].extend([i])
			for seq, index in enumerate(goodIndex):
				if not if_closeDet(goodIndex_2d[seq]):
					goodIndex.pop(seq)
			# search tbins where at least 3 NaIs have an erroneous underflow
			# stored in badIndex
			negativeIndex_set = set(negativeIndex)
			for seq in negativeIndex_set:
				if negativeIndex.count(seq) >= 3:
					badIndex.extend([seq])
			# remove the 0.5 second following badIndex from goodIndex
			for element in badIndex:
				for kk in range(1,round(0.5/self.binwidth)):
					if element+kk in goodIndex:
						goodIndex.remove(element+kk)
			#make plots	and save t0_t1_duration
			if goodIndex:
				goodIndex_sorted = sorted(goodIndex)
				x0 = self.tbins[goodIndex_sorted[0]]
				x1 = self.tbins[goodIndex_sorted[-1]]
				x_width = np.max([x1-x0,2.0])
				with open(self.resultdir+'/t0_t1_duration.txt','w') as f_tmp:
					f_tmp.write(str(round(self.tbins[goodIndex_sorted[0]]-self.binwidth,5))+' '
						+str(round(self.tbins[goodIndex_sorted[-1]],5))+' '
						+str(round(self.tbins[goodIndex_sorted[-1]]-self.tbins[goodIndex_sorted[0]]+self.binwidth,5)))										
			for i in range(14):
				cNet = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] 
							for ch in np.arange(ch1,ch2+1) ])
				totalNet = np.sum(cNet,axis=0)
				#clip the outliers and fit the central median part
				mask = sigma_clip(totalNet,sigma=5,maxiters=5,stdfunc=mad_std).mask
				myfilter = list(map(operator.not_, mask))
				totalNet_median_part = totalNet[myfilter]
				loc,scale = stats.norm.fit(totalNet_median_part)
				Y = stats.norm(loc=loc,scale=scale)
				gaussian_level = Y.interval(norm_pvalue(sigma))
				totalNet = np.concatenate(([totalNet[0]],totalNet))
				axes[i//2,i%2].plot(self.tbins,totalNet,drawstyle='steps',
									lw=3.0,color='tab:blue')
				axes[i//2,i%2].axhline(gaussian_level[1],
					ls='--',lw=3,color='orange',
					label=str(sigma)+'$\sigma$ level of gaussian background')
				axes[i//2,i%2].tick_params(labelsize=25)
				if goodIndex:
					for seq in goodIndex:
						axes[i//2,i%2].axvline(x=self.tbins[seq],
									ymin=0.95,color='red',zorder=2)
					axes[i//2,i%2].set_xlim([ np.max([x0-x_width,viewt1]),
											np.min([x1+x_width,viewt2]) ])
				else:
					axes[i//2,i%2].set_xlim([viewt1,viewt2])
				ylim[i] = axes[i//2,i%2].get_ylim()
				axes[i//2,i%2].text(0.05,0.85,Det[i],fontsize=25,
								transform=axes[i//2,i%2].transAxes)
			#reset the ylims to same values
			BGOymax = np.max([ylim[i,1] for i in range(2)])
			NaIymax = np.max([ylim[i+2,1] for i in range(12)])
			for i in range(14):
				if i == 1:
					axes[i//2,i%2].legend(fontsize=20)
				if i <= 1:
					axes[i//2,i%2].set_ylim([0,BGOymax])
				else:
					axes[i//2,i%2].set_ylim([0,NaIymax])
			fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center',
					va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center',
							va='center',fontsize=30)
			fig.text(0.5, 0.92, self.bnname, ha='center',
							va='center',fontsize=30)		
			plt.savefig(self.resultdir+'/check_pulse.png')
			plt.close()
			f.close()

			
	def check_poisson_rate(self):
		if not os.path.exists(self.resultdir+'/poisson_rate/'):
			os.makedirs(self.resultdir+'/poisson_rate/')
			f = h5py.File(self.baseresultdir+'/base.h5',mode='r')
			for ch in np.arange(ch1,ch2+1):
				fig, axes = plt.subplots(7,2,figsize=(32, 20),
								sharex=False,sharey=False)
				for i in range(14):
					basecount = f['/'+Det[i]+'/ch'+str(ch)][()][1]*self.binwidth
					mean = np.mean(basecount)
					Y = stats.poisson(mean)
					plotcount = f['/'+Det[i]+'/ch'+str(ch)][()][0]*self.binwidth
					plotcount = np.ceil(plotcount).astype(int)
					maxcount = np.max(plotcount)
					bins = np.arange(-0.5,maxcount+1.5)
					x_int = np.arange(0,maxcount+1)
					plothist = Y.pmf(x_int)
					plothist = np.concatenate(([plothist[0]],plothist))
					axes[i//2,i%2].plot(bins,plothist,drawstyle='steps',
									label='Baseline Poisson PMF (from base)',
									lw=4.0,color='tab:orange')
					hist,bin_edged = np.histogram(plotcount,bins=bins)
					axes[i//2,i%2].bar(x_int,hist/np.sum(hist),
								label='Observed count (from rate)', lw=4.0,
								color='tab:blue')	
					axes[i//2,i%2].tick_params(labelsize=25)
					axes[i//2,i%2].text(0.05,0.85,Det[i],fontsize=25,
							transform=axes[i//2,i%2].transAxes)
					if i == 1:
						axes[i//2,i%2].legend(fontsize=20)
				fig.text(0.07, 0.5, 'Normalized histogram (PMF)', ha='center',
						va='center',rotation='vertical',fontsize=30)
				fig.text(0.5, 0.05, 'Count in each time bin ('
						+str(self.binwidth)+' s) between '+str(self.baset1)
						+'--'+str(self.baset2)+' s',
						ha='center', va='center',fontsize=30)
				fig.text(0.5, 0.92, 'ch'+str(ch), ha='center',
									va='center',fontsize=30)		
				plt.savefig(self.resultdir+'/poisson_rate/'+str(ch)+'.png')
				plt.close()
			f.close()
			
	def plot_time_resolved_net_spectrum(self):
		if not os.path.exists(self.resultdir+'/time_resolved_net_spectrum/'):
			os.makedirs(self.resultdir+'/time_resolved_net_spectrum/')
			f = h5py.File(self.baseresultdir+'/base.h5',mode='r')
			for n,t1 in enumerate(self.tbins[:-1]):
				if t1 >- 5 and t1 < 20:
					fig, axes = plt.subplots(7,2,figsize=(32, 20),
									sharex=True,sharey=True)
					for i in range(14):
						spectrum = np.zeros(ch2+1-ch1)
						for ch in np.arange(ch1,ch2+1):
							ds = f['/'+Det[i]+'/ch'+str(ch)][()][2] #net
							spectrum[ch-ch1] = np.abs(ds[n]*self.binwidth)
						axes[i//2,i%2].bar(np.arange(ch1,ch2+1),spectrum,
											lw=4.0,color='tab:blue')
						axes[i//2,i%2].tick_params(labelsize=25)
						axes[i//2,i%2].text(0.05,0.85,Det[i],fontsize=25,
								transform=axes[i//2,i%2].transAxes)
					fig.text(0.07, 0.5, 'Count spectrum', ha='center',
							va='center',rotation='vertical',fontsize=30)
					fig.text(0.5, 0.05, 'Channels',ha='center',
												va='center',fontsize=30)
					fig.text(0.5, 0.92, str(t1)+' s', ha='center',
												va='center',fontsize=30)		
					plt.savefig(self.resultdir+'/time_resolved_net_spectrum/'
												+str(n)+'.png')
					plt.close()
			f.close()		
			

	def check_poisson_time_resolved_net_spectrum(self):
		if not os.path.exists(self.resultdir+'/check_poisson_time_resolved_net_spectrum/'):
			os.makedirs(self.resultdir+'/check_poisson_time_resolved_net_spectrum/')
			f = h5py.File(self.baseresultdir+'/base.h5',mode='r')
			for n,t1 in enumerate(self.tbins[:-1]):
				if t1 >- 5 and t1 < 20:
					fig, axes = plt.subplots(7,2,figsize=(32, 20),
										sharex=True,sharey=True)
					for i in range(14):
						spectrum = np.zeros(ch2+1-ch1)
						for ch in np.arange(ch1,ch2+1):
							ds = f['/'+Det[i]+'/ch'+str(ch)][()][2] #net
							spectrum[ch-ch1] = np.abs(ds[n]*self.binwidth)
						mean = np.mean(spectrum)
						Y = stats.poisson(mean)
						maxcount = np.ceil(spectrum.max())
						x_int = np.arange(0,maxcount+1)
						bins = np.arange(-0.5,maxcount+1.5)
						poissonpmf = Y.pmf(x_int)
						poissonpmf = np.concatenate(([poissonpmf[0]],poissonpmf))
						axes[i//2,i%2].plot(bins,poissonpmf,drawstyle='steps',
										label='Poisson PMF of mean net rate',
										lw=4.0,color='tab:orange')
						hist,bin_edged = np.histogram(spectrum,bins=bins)
						axes[i//2,i%2].bar(x_int,hist/np.sum(hist),
								label='Distribution of observed net rate',
								lw=4.0,color='tab:blue')
						#axes[i//2,i%2].bar(np.arange(2,126),spectrum,
						#			color='tab:blue')	
						axes[i//2,i%2].tick_params(labelsize=25)
						axes[i//2,i%2].text(0.05,0.85,Det[i],fontsize=25,
									transform=axes[i//2,i%2].transAxes)
						if i == 1:
							axes[i//2,i%2].legend(fontsize=20)
					fig.text(0.07, 0.5, 'Normalized histogram (PMF)', 
								ha='center', va='center',
								rotation='vertical',fontsize=30)
					fig.text(0.5, 0.05, 'Counts',ha='center',
											va='center',fontsize=30)
					fig.text(0.5, 0.92, str(t1)+' s', ha='center',
											va='center',fontsize=30)		
					plt.savefig(self.resultdir
							+'/check_poisson_time_resolved_net_spectrum/'
							+str(n)+'.png')
					plt.close()
			f.close()

# mb=multi binwidth
	def multi_binwidth_base(self,mb_baset1=-50,mb_baset2=300,mb_binwidth=[1.0,0.1,0.01]):
		self.mb_baseresultdir = self.baseresultdir+'/mb_base/'
		self.mb_baset1 = np.max([self.GTI1,mb_baset1])
		self.mb_baset2 = np.min([self.GTI2,mb_baset2])

		assert self.mb_baset1<self.mb_baset2, self.bnname+': Inappropriate base times!'
		if not os.path.exists(self.mb_baseresultdir):
			os.makedirs(self.mb_baseresultdir)
			for seq,binwidth in enumerate(mb_binwidth):
				tbins = np.arange(self.mb_baset1,self.mb_baset2+binwidth,binwidth)
				f = h5py.File(self.mb_baseresultdir+'/base_'+str(seq)+'.h5',mode='w')
				for i in range(14):
					grp = f.create_group(Det[i])
					ttefile = glob(self.datadir+'/'+'glg_tte_'+Det[i]
										+'_'+self.bnname+'_v*.fit')
					hdu = fits.open(ttefile[0])	
					trigtime = hdu['Primary'].header['TRIGTIME']
					data = hdu['EVENTS'].data
					timedata = data.field(0)-trigtime
					chdata = data.field(1)
					for ch in range(128):
						time_selected = timedata[chdata==ch]
						histvalue, histbin = np.histogram(time_selected,bins=tbins)
						rate = histvalue/binwidth
						r.assign('rrate',rate) 
						r("y=matrix(rrate,nrow=1)")
						fillPeak_hwi = str(int(5/binwidth))
						fillPeak_int = str(int(len(rate)/10))
						r("rbase=baseline(y,lam = 6, hwi="+fillPeak_hwi
								+", it=10,int ="+fillPeak_int
								+", method='fillPeaks')")
						r("bs=getBaseline(rbase)")
						r("cs=getCorrected(rbase)")
						bs = r('bs')[0]
						cs = r('cs')[0]
						# correct negative base to 0 and recover the net value to original rate
						corrections_index = bs<0
						bs[corrections_index] = 0
						cs[corrections_index] = rate[corrections_index]
						f['/'+Det[i]+'/ch'+str(ch)] = np.array([rate,bs,cs])
				f.flush()
				f.close()
				
				
# check gaussian distribution for different bin sizes; mb=multi binwidth
	def check_mb_base_gaussian_net_rate(self,sigma=3,mb_binwidth=[1.0,0.1,0.01]):
		if not os.path.exists(self.resultdir+'/check_mb_base_gaussian_net_rate_0.png'):
			assert os.path.exists(self.mb_baseresultdir), ('Should have run '
				'multi_binwidth_base() before running check_mb_base_gaussian_net_rate()!')
			for seq,binwidth in enumerate(mb_binwidth):
				tbins = np.arange(self.mb_baset1,self.mb_baset2+binwidth,binwidth)
				f = h5py.File(self.mb_baseresultdir+'/base_'+str(seq)+'.h5',mode='r')
				fig, axes = plt.subplots(7,2,figsize=(32, 20),
										sharex=False,sharey=False)
				for i in range(14):
					cNet = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] 
								for ch in np.arange(ch1,ch2+1) ])
					totalNet = np.sum(cNet,axis=0)
					#print('before clip: binwidth',binwidth,'det_',i,stats.shapiro(totalNet))
					#clip the outliers and fit the central median part
					mask = sigma_clip(totalNet,sigma=5,maxiters=5,stdfunc=mad_std).mask
					myfilter = list(map(operator.not_, mask))
					totalNet_median_part = totalNet[myfilter]
					#print('after clip: binwidth',binwidth,'det_',i,stats.shapiro(totalNet_median_part))
					bins=np.arange(totalNet.min(),totalNet.max(),
					(totalNet_median_part.max()-totalNet_median_part.min())/25)
					histvalue, histbin = np.histogram(totalNet,bins=bins)
					histvalue = np.concatenate(([histvalue[0]],histvalue))
					axes[i//2,i%2].fill_between(histbin,histvalue,step='pre',
						label = 'Observed net rate, binwidth='+str(binwidth))
					loc,scale = stats.norm.fit(totalNet_median_part)
					Y = stats.norm(loc=loc,scale=scale)
					x = np.linspace(totalNet_median_part.min(),
									totalNet_median_part.max(),num=100)
					axes[i//2,i%2].plot(x,Y.pdf(x)*totalNet.size*(bins[1]-bins[0]),
								label='Gaussian Distribution',
								linestyle='--',lw=3.0,color='tab:orange')
					axes[i//2,i%2].tick_params(labelsize=25)
					axes[i//2,i%2].text(0.05,0.85,Det[i],fontsize=25,
									transform=axes[i//2,i%2].transAxes)
					gaussian_level = Y.interval(norm_pvalue(sigma))
					axes[i//2,i%2].axvline(totalNet_median_part.max(),ls='--',
						lw=2,color='green',label='region for gaussing fitting')
					axes[i//2,i%2].axvline(totalNet_median_part.min(),
											ls='--',lw=2,color='green')
					if i ==1 :
						axes[i//2,i%2].legend(fontsize=20)
				f.close()
				fig.text(0.07, 0.5, 'Numbers', ha='center', va='center',
										rotation='vertical',fontsize=30)
				fig.text(0.5, 0.05, 'Total net rate (s$^{-1}$; between '
					+str(self.mb_baset1)+'--'+str(self.mb_baset2)+'s)',
					ha='center', va='center',fontsize=30)					
				plt.savefig(self.resultdir+'/check_mb_base_gaussian_net_rate_'
													+str(seq)+'.png')
				plt.close()
		
				
# check SNR with different bin sizes; mb=multi binwidth
	def check_mb_base_snr(self,viewt1=-50,viewt2=300,mb_binwidth=[1.0,0.1,0.01]):
		if not os.path.exists(self.resultdir+'/check_mb_base_SNR.png'):
			assert os.path.exists(self.mb_baseresultdir), ('Should have run '
						'multi_binwidth_base() before running check_debase_snr()!')
			viewt1 = np.max([self.mb_baset1,viewt1])
			viewt2 = np.min([self.mb_baset2,viewt2])
			fig, axes = plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=True)
			ylim = np.zeros((14,2))
			my_colors = ['black','red','blue']
			for seq,binwidth in enumerate(mb_binwidth):
				tbins = np.arange(self.mb_baset1,self.mb_baset2+binwidth,binwidth)
				f = h5py.File(self.mb_baseresultdir+'/base_'+str(seq)+'.h5',mode='r')
				for i in range(14):
					cNet = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] 
									for ch in np.arange(ch1,ch2+1) ])
					totalNet = np.sum(cNet,axis=0)
					#clip the outliers and fit the central median part
					mask = sigma_clip(totalNet,sigma=5,maxiters=5,stdfunc=mad_std).mask
					myfilter = list(map(operator.not_, mask))
					totalNet_median_part = totalNet[myfilter]
					loc,scale = stats.norm.fit(totalNet_median_part)
					totalNet = np.concatenate(([totalNet[0]],totalNet))
					snr = (totalNet-loc)/scale
					print('binwidth=',binwidth,'loc&scale: Det_'+str(i),':',loc,scale)
					#axes[i//2,i%2].plot(tbins,snr,drawstyle='steps',lw=1.0,
					#	color=my_colors[seq],alpha=0.5,label=str(binwidth))
					axes[i//2,i%2].plot(tbins,totalNet,drawstyle='steps',lw=1.0,
						color=my_colors[seq],alpha=0.5,label=str(binwidth))
					axes[i//2,i%2].tick_params(labelsize=25)
					if i == 1:
						axes[i//2,i%2].legend(fontsize=20)
				f.close()
			axes[0,0].set_xlim([viewt1,viewt2])
			#axes[0,0].set_ylim([-1,4])
			fig.text(0.07, 0.5, 'Signal-to-noise ratio', ha='center',
							va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center', 
							va='center',fontsize=30)		
			fig.text(0.5, 0.92, self.bnname, ha='center',
							va='center',fontsize=30)			
			plt.savefig(self.resultdir+'/check_mb_base_SNR.png')
			plt.close()

	def netlc(self,viewt1=-50,viewt2=300):
		if not os.path.exists(self.resultdir+'/net_lc.png'):
			assert os.path.exists(self.baseresultdir), ('Should have run base() '
						'before running netlc()!')
			#print('plotting raw_lc_with_base.png ...')
			viewt1 = np.max([self.baset1,viewt1])
			viewt2 = np.min([self.baset2,viewt2])
			BGOplotymax = 0.0
			NaIplotymax = 0.0
			# raw lc with baseline
			f = h5py.File(self.baseresultdir+'/base.h5',mode='r')
			fig, axes = plt.subplots(7,2,figsize=(32, 20),
									sharex=True,sharey=False)
			for i in range(14):
				#data in firt and last two channels of BGO and NaI are not used
				#ignore 0,1,126,127, notice 2-125
				cNet = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] 
								for ch in np.arange(ch1,ch2+1) ])
				totalNet = np.sum(cNet,axis=0)
				ymax = totalNet.max()
				if i <= 1:
					if BGOplotymax < ymax:
						BGOplotymax = ymax
						BGOdetseq = i
				else:
					if NaIplotymax < ymax:
						NaIplotymax = ymax
						NaIdetseq = i
				cRate = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][0] 
									for ch in np.arange(ch1,ch2+1) ])
				totalRate = np.sum(cRate,axis=0)
				totalRate = np.concatenate(([totalRate[0]],totalRate))
				axes[i//2,i%2].plot(self.tbins,totalRate,drawstyle='steps',
												lw=3.0,color='tab:blue')
				cBase = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][1] 
										for ch in np.arange(ch1,ch2+1) ])
				totalBase = np.sum(cBase,axis=0)
				plottime = self.tbins[:-1]+self.binwidth/2.0
				axes[i//2,i%2].plot(plottime,totalBase,linestyle='--',
									lw=4.0, color='tab:orange')
				axes[i//2,i%2].set_xlim([viewt1,viewt2])
				axes[i//2,i%2].tick_params(labelsize=25)
				axes[i//2,i%2].text(0.05,0.85,Det[i],fontsize=25,
								transform=axes[i//2,i%2].transAxes)
			fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center',
						va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center', 
									va='center',fontsize=30)		
			fig.text(0.5, 0.92, self.bnname, ha='center',
									va='center',fontsize=30)			
			plt.savefig(self.resultdir+'/raw_lc_with_base.png')
			plt.close()

			# net lc
			#print('plotting net_lc.png ...')
			fig, axes = plt.subplots(7,2,figsize=(32, 20),
									sharex=True,sharey=False)
			for i in range(14):
				cNet = np.array([ f['/'+Det[i]+'/ch'+str(ch)][()][2] 
								for ch in np.arange(ch1,ch2+1) ])
				totalNet = np.sum(cNet,axis=0)
				totalNet = np.concatenate(([totalNet[0]],totalNet))
				axes[i//2,i%2].plot(self.tbins,totalNet,drawstyle='steps',
									lw=3.0,color='tab:blue')
				axes[i//2,i%2].set_xlim([viewt1,viewt2])
				if i <= 1:
					axes[i//2,i%2].set_ylim([0,BGOplotymax])
				else:
					axes[i//2,i%2].set_ylim([0,NaIplotymax])
				axes[i//2,i%2].tick_params(labelsize=25)
				if i == BGOdetseq:
					axes[i//2,i%2].text(0.7,0.85,'Brightest BGO',
							transform=axes[i//2,i%2].transAxes,
							color='red',fontsize=25)
				elif i == NaIdetseq:
					axes[i//2,i%2].text(0.7,0.85,'Brightest NaI',
							transform=axes[i//2,i%2].transAxes,
							color='red',fontsize=25)
				axes[i//2,i%2].text(0.05,0.85,Det[i],fontsize=25,
								transform=axes[i//2,i%2].transAxes)
			fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center',
						va='center',rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center', 
									va='center',fontsize=30)		
			fig.text(0.5, 0.92, self.bnname, ha='center',
									va='center',fontsize=30)			
			plt.savefig(self.resultdir+'/net_lc.png')
			plt.close()
			f.close()


	def countmap(self,viewt1=-50,viewt2=300):
		if not os.path.exists(self.resultdir+'/net_countmap.png'):
			assert os.path.exists(self.baseresultdir), ('Should have run base() '
						'before running countmap()!')
			viewt1 = np.max([self.baset1,viewt1])
			viewt2 = np.min([self.baset2,viewt2])
			#print('plotting rate_countmap.png ...')
			#plot_countmap(self.bnname,self.resultdir,self.baseresultdir,
			#		self.datadir,'rate',self.tbins,viewt1,viewt2)
			#print('plotting base_countmap.png ...')
			#plot_countmap(self.bnname,self.resultdir,self.baseresultdir,
			#		self.datadir,'base',self.tbins,viewt1,viewt2)
			#print('plotting net_countmap.png ...')
			plot_countmap(self.bnname,self.resultdir,self.baseresultdir,
						self.datadir,'net',self.tbins,viewt1,viewt2)
			#print('plotting pois_countmap.png ...')
			#plot_countmap(self.bnname,self.resultdir,self.baseresultdir,
			#		self.datadir,'pois',self.tbins,viewt1,viewt2)

	def rsp(self):
		if not os.path.exists(self.resultdir+'/response_matrix.png'):
			#print('plotting response_matrix.png ...')
			#determine max for color bar
			BGOmaxcolorvalue = 0.0
			NaImaxcolorvalue = 0.0
			for i in range(14):
				ttefile = glob(self.datadir+'/glg_cspec_'+Det[i]
										+'_'+self.bnname+'_v*.rsp')
				hdu = fits.open(ttefile[0])
				ebound = hdu['EBOUNDS'].data
				emin = ebound.field(1)
				rspdata = hdu['SPECRESP MATRIX'].data
				elo = rspdata.field(0)
				matrix = rspdata.field(5)
				filled_matrix = np.zeros((len(emin),len(elo)))
				for ii in range(len(emin)):				
					for jj in range(len(elo)):
						try:
							filled_matrix[ii][jj] = matrix[jj][ii]
						except:
							pass
				if i <= 1:
					if BGOmaxcolorvalue < filled_matrix.max():
						BGOmaxcolorvalue = filled_matrix.max()
				else:
					if NaImaxcolorvalue < filled_matrix.max():
						NaImaxcolorvalue = filled_matrix.max()		
			#plot response matrix
			fig, axes = plt.subplots(7,2,figsize=(32, 40),
								sharex=False,sharey=False)			
			for i in range(14):
				ttefile = glob(self.datadir+'/glg_cspec_'+Det[i]
									+'_'+self.bnname+'_v*.rsp')
				hdu = fits.open(ttefile[0])
				ebound = hdu['EBOUNDS'].data
				emin = ebound.field(1)
				emax = ebound.field(2)
				rspdata = hdu['SPECRESP MATRIX'].data
				elo = rspdata.field(0)
				ehi = rspdata.field(1)
				matrix = rspdata.field(5)
				filled_matrix = np.zeros((len(emin),len(elo)))
				for ii in range(len(emin)):				
					for jj in range(len(elo)):
						try:
							filled_matrix[ii][jj] = matrix[jj][ii]
						except:
							pass
				x = np.concatenate((elo,[ehi[-1]]))
				y = np.concatenate((emin,[emax[-1]]))
				X, Y = np.meshgrid(x,y)
				if i <= 1:
					pcmBGO = axes[i//2,i%2].pcolormesh(X,Y,filled_matrix,
						norm=colors.LogNorm(vmin=1E-1, vmax=BGOmaxcolorvalue),
						cmap='rainbow')
					axes[i//2,i%2].set_xlim([200,4E4])
					axes[i//2,i%2].set_ylim([200,4E4])
				else:
					pcmNaI = axes[i//2,i%2].pcolormesh(X,Y,filled_matrix,
						norm=colors.LogNorm(vmin=1E-2, vmax=NaImaxcolorvalue),
						cmap='rainbow')
					axes[i//2,i%2].set_xlim([5,1E4])
					axes[i//2,i%2].set_ylim([5,1000])
				axes[i//2,i%2].tick_params(axis='both',top=True,
								right=True,length=3,width=1,
								direction='out',which='both',labelsize=25)
				axes[i//2,i%2].text(0.1,0.7,Det[i],fontsize=25,
								transform=axes[i//2,i%2].transAxes)
				axes[i//2,i%2].set_xscale('log')
				axes[i//2,i%2].set_yscale('log')
			cbarBGO = fig.colorbar(pcmBGO, ax=axes[0,],
					orientation='vertical',fraction=0.005, aspect=200/6)
			cbarBGO.ax.tick_params(labelsize=25)
			cbarNaI = fig.colorbar(pcmNaI, ax=axes[1:,],
					orientation='vertical',fraction=0.005, aspect=200)
			cbarNaI.ax.tick_params(labelsize=25)
			fig.text(0.05, 0.5, 'Measured Energy (KeV)',
					ha='center',va='center', rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Incident Energy (keV)', ha='center',
									va='center',fontsize=30)
			plt.savefig(self.resultdir+'response_matrix.png')
			plt.close()
		
		#plot effective area (ARF)
		if not os.path.exists(self.resultdir+'/effective_area.png'):
			#print('plotting effective_area.png ...')
			fig, axes = plt.subplots(7,2,figsize=(32, 30),
								sharex=False,sharey=False)	
			for i in range(14):
				ttefile = glob(self.datadir+'/glg_cspec_'+Det[i]
									+'_'+self.bnname+'_v*.rsp')
				hdu = fits.open(ttefile[0])
				ebound = hdu['EBOUNDS'].data
				emin = ebound.field(1)
				emax = ebound.field(2)
				rspdata = hdu['SPECRESP MATRIX'].data
				elo = rspdata.field(0)
				ehi = rspdata.field(1)
				matrix = rspdata.field(5)
				filled_matrix = np.zeros((len(emin),len(elo)))
				for ii in range(len(emin)):				
					for jj in range(len(elo)):
						try:
							filled_matrix[ii][jj] = matrix[jj][ii]
						except:
							pass
				x = np.concatenate((elo,[ehi[-1]]))
				arf = np.zeros(len(elo))
				for kk in range(len(elo)):
					arf[kk] = filled_matrix[:,kk].sum()
				arf = np.concatenate(([arf[0]],arf))
				axes[i//2,i%2].plot(x,arf,drawstyle='steps',lw=5)
				axes[i//2,i%2].tick_params(axis='both',top=True,
								right=True,length=10,width=1,
								direction='in',which='both',labelsize=25)
				axes[i//2,i%2].text(0.7,0.1,Det[i],fontsize=25,
								transform=axes[i//2,i%2].transAxes)
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
			fig.text(0.05, 0.5, 'Effective Area (cm$^2$)', ha='center',
						va='center', rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Incident Energy (keV)', ha='center',
						va='center',fontsize=30)
			plt.savefig(self.resultdir+'/effective_area.png')
			plt.close()

													
	def phaI(self,slicet1=0,slicet2=5):
		#print('creating a new phaI slice between',slicet1,'s and',slicet2,'s  ...')
		if not os.path.exists(self.phaIresultdir):
			os.makedirs(self.phaIresultdir)
		nslice = len(os.listdir(self.phaIresultdir))
		sliceresultdir = self.phaIresultdir+'/slice'+str(nslice)+'/'
		os.makedirs(sliceresultdir)
		fig, axes = plt.subplots(7,2,figsize=(32, 30),sharex=False,sharey=False)
		sliceindex = (self.tbins >= slicet1) & (self.tbins <= slicet2)
		valid_bins = np.sum(sliceindex)-1
		assert valid_bins >= 1, self.bnname+': Inappropriate phaI slice time!'
		f = h5py.File(self.baseresultdir+'/base.h5',mode='r')
		for i in range(14):
			total_rate = np.zeros(128)
			bkg_rate = np.zeros(128)
			total_uncertainty = np.zeros(128)
			bkg_uncertainty = np.zeros(128)
			ttefile = glob(self.datadir+'/glg_tte_'+Det[i]
									+'_'+self.bnname+'_v*.fit')
			hdu = fits.open(ttefile[0])
			ebound = hdu['EBOUNDS'].data
			emin = ebound.field(1)
			emax = ebound.field(2)
			energy_diff = emax-emin
			energy_bins = np.concatenate((emin,[emax[-1]]))
			for ch in np.arange(128):
				base = f['/'+Det[i]+'/ch'+str(ch)][()][1]
				rate = f['/'+Det[i]+'/ch'+str(ch)][()][0]
				bkg = base[sliceindex[:-1]][:-1]
				total = rate[sliceindex[:-1]][:-1]
				bkg_rate[ch] = bkg.mean()
				total_rate[ch] = total.mean()
				exposure = len(bkg)*self.binwidth
				bkg_uncertainty[ch] = np.sqrt(bkg_rate[ch]/exposure)
				total_uncertainty[ch] = np.sqrt(total_rate[ch]/exposure)
			#plot both rate and bkg as count/s/keV
			write_phaI(bkg_rate,self.bnname,Det[i],slicet1,slicet2,
								sliceresultdir+'/'+Det[i]+'.bkg')
			write_phaI(total_rate,self.bnname,Det[i],slicet1,slicet2,
								sliceresultdir+'/'+Det[i]+'.pha')
			copy_rspI(self.bnname,Det[i],sliceresultdir+'/'+Det[i]+'.rsp')
			bkg_diff = bkg_rate/energy_diff
			total_diff = total_rate/energy_diff
			x = np.sqrt(emax*emin)
			axes[i//2,i%2].errorbar(x,bkg_diff,
						yerr=bkg_uncertainty/energy_diff,
						linestyle='None',color='blue')
			axes[i//2,i%2].errorbar(x,total_diff,
						yerr=total_uncertainty/energy_diff,
						linestyle='None',color='red')
			bkg_diff = np.concatenate(([bkg_diff[0]],bkg_diff))
			total_diff = np.concatenate(([total_diff[0]],total_diff))
			axes[i//2,i%2].plot(energy_bins,bkg_diff,
								drawstyle='steps',color='blue')
			axes[i//2,i%2].plot(energy_bins,total_diff,
								drawstyle='steps',color='red')
			axes[i//2,i%2].set_xscale('log')
			axes[i//2,i%2].set_yscale('log')
			axes[i//2,i%2].tick_params(labelsize=25)
			axes[i//2,i%2].text(0.85,0.85,Det[i],fontsize=25,
								transform=axes[i//2,i%2].transAxes)
		fig.text(0.07, 0.5, 'Rate (count s$^{-1}$ keV$^{-1}$)', 
					ha='center',va='center', rotation='vertical',fontsize=30)
		fig.text(0.5, 0.05, 'Energy (keV)', ha='center', 
							va='center',fontsize=30)	
		plt.savefig(sliceresultdir+'/PHA_rate_bkg.png')
		plt.close()
		f.close()


	def specanalyze(self,slicename):
		slicedir = self.phaIresultdir+'/'+slicename+'/'
		os.chdir(slicedir)
		# select the most bright two NaIs (in channels 6-118) 
		# and more bright one BGO (in channels 4-124):
		BGOtotal = np.zeros(2)
		NaItotal = np.zeros(12)
		for i in range(2):
			phahdu = fits.open(slicedir+'/'+BGO[i]+'.pha')
			bkghdu = fits.open(slicedir+'/'+BGO[i]+'.bkg')
			pha = phahdu['SPECTRUM'].data.field(1)
			bkg = bkghdu['SPECTRUM'].data.field(1)
			src = pha-bkg
			plt.plot(src[4:125])
			plt.savefig(BGO[i]+'.png')
			plt.close()
			BGOtotal[i] = src[4:125].sum()
		for i in range(12):
			phahdu = fits.open(slicedir+'/'+NaI[i]+'.pha')
			bkghdu = fits.open(slicedir+'/'+NaI[i]+'.bkg')
			pha = phahdu['SPECTRUM'].data.field(1)
			bkg = bkghdu['SPECTRUM'].data.field(1)
			src = pha-bkg
			plt.plot(src[6:118])
			plt.savefig(NaI[i]+'.png')
			plt.close()
			NaItotal[i] = src[6:118].sum()
		BGOindex = np.argsort(BGOtotal)
		NaIindex = np.argsort(NaItotal)
		brightdet = [BGO[BGOindex[-1]],NaI[NaIindex[-1]],NaI[NaIindex[-2]]]
		
		# use xspec	
		alldatastr = ' '.join([det+'.pha' for det in brightdet])
		AllData(alldatastr)
		AllData.ignore('1-2:**-200.0,40000.0-** 3-14:**-8.0,800.0-**')
		Model('grbm')
		Fit.nIterations = 1000
		Fit.statMethod = 'pgstat'
		Fit.perform()
		Fit.error('3.0 3')
		Fit.perform()
		par3 = AllModels(1)(3)
		print(par3.error)

		#Plot.device='/xs'
		Plot.device = '/null'
		Plot.xAxis = 'keV'
		Plot.yLog = True
		Plot('ldata')
		for i in (1,2,3):
			energies = Plot.x(i)
			rates = Plot.y(i)
			folded = Plot.model(i)
			xErrs = Plot.xErr(i)
			yErrs = Plot.yErr(i)
			plt.errorbar(energies,rates,xerr=xErrs,yerr=yErrs,
										zorder=1,ls='None')
			plt.plot(energies,folded,color='black',zorder=2)
		plt.xscale('log')
		plt.yscale('log')
		plt.savefig('foldedspec.png')
		plt.close()
		Plot('eeufspec')
		for i in (1,2,3):
			energies = Plot.x(i)
			ufspec = Plot.y(i)
			folded = Plot.model(i)
			xErrs = Plot.xErr(i)
			yErrs = Plot.yErr(i)
			plt.errorbar(energies,ufspec,xerr=xErrs,yerr=yErrs,
										zorder=1,ls='None')
			plt.plot(energies,folded,color='black',zorder=2)
		plt.xscale('log')
		plt.yscale('log')
		plt.savefig('eeufspec.png')
		plt.close()

	def removebase(self):
		os.system('rm -rf '+self.baseresultdir)
					
