### Gust Front Case Identification From METAR Data ###
### Created 2024/10/10 1602Z ###
### J. W. Thiesing 2024 ###

import datetime as dt
from datetime import datetime
import numpy as np
import pandas as pd

RADAR_GIS_PATH = "./RADARS.txt"
EARTH_RADIUS = 6371.0
CASE_TIMEWINDOW_HALF = 30 # Minutes before and after the case time that will be downloaded

def calc_latlon_dist(lat0, lon0, lat1, lon1):
	return np.arccos(np.sin(np.radians(lat0))*np.sin(np.radians(lat1))+np.cos(np.radians(lat0))*np.cos(np.radians(lat1))*np.cos(np.radians(lon1-lon0)))*EARTH_RADIUS

def nearestmetar(radarsite):
	import json
	from urllib.request import urlopen
	#BBCONST = 0.5

	radar_gis = pd.read_csv(RADAR_GIS_PATH, delimiter='|', index_col=0, names=['operator','lat','lon','elev','n','state','name'])
	radar_line = radar_gis.loc[radar_gis.index == radarsite]
	radar_lat, radar_lon = radar_line['lat'].item(), radar_line['lon'].item()
	#bb_bot = radar_lat-BBCONST
	#bb_top = radar_lat+BBCONST
	#bb_left = radar_lon-BBCONST
	#bb_right = radar_lon+BBCONST
	if radar_line.empty:
		sys.exit('Error: radar not found in RADARS.txt')

	#download_url = f'https://aviationweather.gov/api/data/stationinfo?bbox={bb_bot}%2C{bb_left}%2C{bb_top}%2C{bb_right}&format=json'
	#with urlopen(download_url) as url:
	#	urldec = url.read().decode('utf-8')
	#jsondecode = json.loads(urldec)
	#if len(jsondecode) < 1:
	#	sys.exit(f'No METAR stations found within {BBCONST}º of radar site {radarsite.upper()}')

	distances = []
	#for station in jsondecode:
	stations = pd.read_csv('./METAR.csv')
	for ii, station in stations.iterrows():
		stlat, stlon = station['lat'], station['lon']
		distances.append(calc_latlon_dist(radar_lat, radar_lon, stlat, stlon))
	distances = np.array(distances)
	mindistii = np.where(distances == np.nanmin(distances))[0][0]

	#return jsondecode[mindistii] # Closest METAR/ASOS/AWOS site
	#print(distances[mindistii])
	return stations.loc[mindistii]

def downloadmetar(site, startdate, enddate):
	from urllib.request import urlopen
	import io

	MAINLINK = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?data=tmpf,dwpf,relh,drct,sknt,p01i,alti,mslp,gust,peak_wind_gust,peak_wind_drct,peak_wind_time'

	year1, month1, day1, hour1, minute1 = startdate.year, startdate.month, startdate.day, startdate.hour, startdate.minute
	year2, month2, day2, hour2, minute2 = enddate.year, enddate.month, enddate.day, enddate.hour, enddate.minute
	station = site['stid']

	sublink = f'&station={station}&year1={year1}&month1={month1}&day1={day1}&hour1={hour1}&minute1={minute1}&year2={year2}&month2={month2}&day2={day2}&hour2={hour2}&minute2={minute2}'
	dllink = MAINLINK + sublink

	with urlopen(dllink) as url:
		urldec = url.read().decode('utf-8')
	data = pd.read_csv(io.StringIO(urldec), na_values=['M'])

	return data

def calc_dirchange(deg1, deg2):
	if deg1 >= deg2:
		larger = deg1
		smaller = deg2
	else:
		larger = deg2
		smaller = deg1
	delta = larger-smaller
	if delta > 180:
		delta = 360-delta
	return delta

def findcases(metardata):
	dirchangethreshold = 90/60 # 90 degrees / hour is the threshold

	rowpairs = []
	for ii, row in metardata.iterrows():
		if ii > 0:
			prevrow = metardata.loc[ii-1]
			dirchange = calc_dirchange(float(row['drct']), float(prevrow['drct']))
			prevtime = datetime.strptime(prevrow['valid'],'%Y-%m-%d %H:%M')
			currtime = datetime.strptime(row['valid'],'%Y-%m-%d %H:%M')
			timechangemin = (currtime-prevtime).total_seconds()/60.0
			gusts = row['gust']
			dirchangerate = dirchange/timechangemin
			if np.isnan(dirchangerate) or np.isnan(row['sknt']):
				continue
			if ((dirchangerate >= dirchangethreshold) and (gusts >= 15 or row['sknt'] >= 15)):
				rowpairs.append((prevrow, row, prevtime - dt.timedelta(seconds=(CASE_TIMEWINDOW_HALF+30)*60), currtime + dt.timedelta(seconds=CASE_TIMEWINDOW_HALF*60)))
				#print(dirchange, timechangemin)

	#print(rowpairs)
	return rowpairs

def mergecases(cases):
	newcases = []
	for ii, case in enumerate(cases):
		if ii > 0:
			prevcase = cases[ii-1]
		else:
			continue
		prevend = prevcase[3]
		currbeg = case[2]
		if prevend-currbeg > dt.timedelta(seconds=0):
			cases[ii] = (case[0], case[1], prevcase[2], case[3])
		else:
			newcases.append(prevcase)
	return newcases

def downloadnexrad(radarsite, cases):
	from nexrad_vwp_s3 import downloadrange

	parentdir = './Data/NEXRAD/'
	
	radarobjs = []
	for case in cases:
		starttime = case[2]
		endtime = case[3]
		datistr = dt.datetime.strftime(starttime, '%Y%m%d_%H')
		childdir = f'{radarsite.upper()}{datistr}/'
		downloaddir = parentdir + childdir
		rangeobjs = downloadrange(starttime, endtime, radarsite, downloaddir)
		for obj in rangeobjs: radarobjs.append(obj)

	return radarobjs

def loadnexrad(downdir='./Data/NEXRAD/', ii=0):
	import os, pyart
	import os.path

	radarobjs = []
	d = []
	for c in os.walk(downdir):
		if c[0].replace(downdir, '') == '': continue
		if '.DS_Store' in c[0]: continue
		if '_labels' in c[0]: continue
		d.append(c)
	d.sort()
	parent = d[ii][0]
	files = d[ii][2]
	files.sort()
	for file in files:
		path = parent+'/'+file
		if '.DS_Store' in path or '.mat' in path: continue
		if os.path.isfile(parent+'_labels/'+file+'.mat'): continue
		radarobjs.append((pyart.io.read(path), path))
	return radarobjs

def plotradarobj(radarobj, stlat, stlon, savepath='./'):
	import matplotlib.pyplot as plt
	from nexrad_plot import plot_nexrad
	from find_params import find_params
	import cartopy.crs as ccrs
	from mpl_point_clicker import clicker
	import matplotlib.path as path
	from scipy.io import savemat
	import os
	
	plt.rcParams.update({'font.size': 14})

	savepathsplit = savepath.split('/')
	savepath = '/'.join(savepathsplit[:-1])+'_labels/'+savepathsplit[-1]
	print(savepath)
	if not os.path.exists('/'.join(savepath.split('/')[:-1])): os.makedirs('/'.join(savepath.split('/')[:-1]))

	global ax0
	global ax1
	fig, ax = plt.subplots(1,2, gridspec_kw={'wspace': 0.2, 'hspace': 0.1, 'left': 0.05, 'right': 0.95, 'bottom': 0.05, 'top': 0.95})
	ax0, ax1 = ax

	start_time = dt.datetime.strptime(radarobj.time['units'][14:], '%Y-%m-%dT%H:%M:%SZ')
	end_time = start_time + dt.timedelta(seconds=radarobj.time['data'][-1])

	plot_nexrad(ax0, radarobj, 'reflectivity', find_params('reflectivity', 15, 100, stlat, stlon, 0, 0., start_time, end_time, ('reflectivity')))
	plot_nexrad(ax1, radarobj, 'velocity', find_params('velocity', 15, 100, stlat, stlon, 0, 0., start_time, end_time, ('velocity')))
	plt.tight_layout()
	plt.show()
	print("How many gust fronts would you like to identify in this sweep?:")
	casecount = int(input())
	if casecount == 0:
		XX, YY = np.meshgrid(np.arange(-100,100.5,0.5), np.arange(-100,100.5,0.5))
		evalbox = np.zeros((401,401))
		savemat(savepath+'.mat', {'evalbox':evalbox,'xi2':XX,'yi2':YY})
		return

	fig, ax = plt.subplots(1,2, gridspec_kw={'wspace': 0.2, 'hspace': 0.1, 'left': 0.05, 'right': 0.95, 'bottom': 0.05, 'top': 0.95})
	ax0, ax1 = ax
	plot_nexrad(ax0, radarobj, 'reflectivity', find_params('reflectivity', 15, 100, stlat, stlon, 0, 0., start_time, end_time, ('reflectivity')))
	plot_nexrad(ax1, radarobj, 'velocity', find_params('velocity', 15, 100, stlat, stlon, 0, 0., start_time, end_time, ('velocity')))

	markerlabels = []
	for ii in range(0, casecount):
		cstr = 'GF'+str(ii+1)
		markerlabels.append(cstr)
	markers = ["o", "X", "*", "s", "P"]
	colors = ['fuchsia']*casecount

	klicker = clicker(
		ax0,
		markerlabels,
		markers=markers[:casecount],
		colors=colors
	)

	plt.tight_layout()
	plt.show()

	coordsall = klicker.get_positions()

	pathsall = []
	for key in coordsall.keys():
		arr = coordsall[key].tolist()
		#arr.append([0,0])
		arr = np.array(arr)
		codes = [np.uint8(1)]
		for ii in range(0, len(arr)-1): codes.append(np.uint8(2))
		#codes.append(np.uint8(79))
		#print(codes)

		pathsall.append(path.Path(arr, codes=codes))

	# Path plotting code, optional
	plotpaths_debug = False
	if plotpaths_debug:
		import matplotlib.patches as patches
		fig, ax = plt.subplots(1,2, gridspec_kw={'wspace': 0.2, 'hspace': 0.1, 'left': 0.05, 'right': 0.95, 'bottom': 0.05, 'top': 0.95})
		ax0, ax1 = ax
		plot_nexrad(ax0, radarobj, 'reflectivity', find_params('reflectivity', 15, 100, stlat, stlon, 0, 0., start_time, end_time, ('reflectivity')))
		plot_nexrad(ax1, radarobj, 'velocity', find_params('velocity', 15, 100, stlat, stlon, 0, 0., start_time, end_time, ('velocity')))
		for path in pathsall:
			patch = patches.PathPatch(path, facecolor='fuchsia', lw=6)
			ax0.add_patch(patch)
		plt.show()

	XX, YY = np.meshgrid(np.arange(-100,100.5,0.5), np.arange(-100,100.5,0.5))
	XX, YY = XX*1e3, YY*1e3
	evalbox = np.zeros((401,401))
	allpoints = []
	for x in range(0,401):
			for y in range(0,401):
				allpoints.append((XX[x,y], YY[x,y]))
	for path in pathsall:
		inside = path.contains_points(allpoints)
		inside = np.reshape(inside,(401,401))
		for x in range(0,401):
			for y in range(0,401):
				if inside[x,y]: evalbox[x,y] = 1

	savemat(savepath+'.mat', {'evalbox':evalbox,'xi2':XX/1e3,'yi2':YY/1e3})

	return

def save_asos_cases(metardata, radarsite, cases):
	parentdir = './Data/ASOS/'
	
	metardata['valid'] = pd.to_datetime(metardata['valid'], format='%Y-%m-%d %H:%M')
	#radarobjs = []
	for case in cases:
		starttime = case[2]
		endtime = case[3]
		datistr = dt.datetime.strftime(starttime, '%Y%m%d_%H')
		fname = parentdir + f'{radarsite.upper()}{datistr}.csv'
		#downloaddir = parentdir + childdir
		#rangeobjs = downloadrange(starttime, endtime, radarsite, downloaddir)
		#for obj in rangeobjs: radarobjs.append(obj)
		delta0 = dt.timedelta(seconds=0)
		metardata[(metardata['valid'] - starttime >= delta0) & (metardata['valid'] - endtime < delta0)].to_csv(fname)

	return

if __name__ == "__main__":
	import argparse, sys

	# Parse arguments!!
	parser = argparse.ArgumentParser(
						prog='caseident.py',
						description='Automatically identify gust front cases for a given NEXRAD site in a given date range using METAR and NEXRAD data',
						epilog='')
	parser.add_argument('radar')
	parser.add_argument('startdate')
	parser.add_argument('enddate')
	args = parser.parse_args(sys.argv[1:])
	startdate = datetime.strptime(args.startdate, '%Y%m%d')
	enddate = datetime.strptime(args.enddate, '%Y%m%d')

	# SEGMENT FOR WORK TO SAVE ASOS DATA, 2025/02/27:
	nearestsite = nearestmetar(args.radar.lower())
	metardata = downloadmetar(nearestsite, startdate, enddate)
	cases = findcases(metardata)
	cases = mergecases(cases)
	save_asos_cases(metardata, args.radar.upper(), cases)

	# SEGMENT FOR DOWNLOADING NEW DATA
	# nearestsite = nearestmetar(args.radar.lower())
	# metardata = downloadmetar(nearestsite, startdate, enddate)
	# cases = findcases(metardata)
	# cases = mergecases(cases)
	# radarobjs = downloadnexrad(args.radar.upper(), cases)

	# SEGMENT FOR LABELLING DOWNLOADED DATA
	# ii = 16
	# # testdir = './Data/Test/'
	# downdir = './Data/NEXRAD/'
	# radarobjs = loadnexrad(downdir=downdir, ii=ii)
	# #print(radarobjs)
	# for radarobj in radarobjs:
	# 	#print(radarobj)
	# 	radarobj, path = radarobj
	# 	plotradarobj(radarobj, 0,0, savepath=path)