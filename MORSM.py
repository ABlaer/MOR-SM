#!/usr/bin/env python


"""
Copyright (C) by Almog Blaer 

   __  __  ____  _____        _____ __  __ 
  |  \/  |/ __ \|  __ \      / ____|  \/  |
  | \  / | |  | | |__) |____| (___ | \  / |
  | |\/| | |  | |  _  /______\___ \| |\/| |
  | |  | | |__| | | \ \      ____) | |  | |
  |_|  |_|\____/|_|  \_\    |_____/|_|  |_|
                                      
                                      
Get ready and fasten your seat belts, we are going to launch a synthetic earthquake at any location you wish.
This code can depict a finite segment that is aimed to be planted in SW4 software.
All you need is to tell us  about your computetional domain and  about the segment's kinematic.

the general code steps are:
 1. Defining the the segment's dimensions (width,length and slip by Goda (2016) equations for desired magnitude)
 2. Fitting a location parameters for the slip and time functions based on  disired diractivity.
 3. Computing the sliding time (Tr) from stage I and stage II for velocity I and velocity II respectively 
 4. The features above are distributed by the time and slip function on each pixel on the segment
 5. You can generate  Source Time Function (STF) for the earthquake you have just set.
 6. Save your file as SW4 input
 
parameters list: 

dh: set the grid spacing in your computational domain
Xc: set north-south Cartesian location of the segment's center
Yc: set the location east-west Cartesian location of the segment's center
Zc: set the depth hypocenter in Cartesian location of the segment's center
dip: segment's dip
strike: segment's strike
rake: segment's angle from the horizon
Ga: Shear modulus 
Vr_2:  set the second velocity of stage II
Vr_1: set the first velocity  of stage I
sec_stage1: set how long  the first stage will be with Vr_1
EveMag: set the desired magnitude.

aD: set the max slip location (down-up direction)
bD: set the max slip location (south-north or east-west direction)
aH: set the nucliation location  (down-up direction)
bH: set the nucliation location  (south-north or east-west direction).

bH, aH > 0 segment with northern (+X) diractivity and downwards (+Z)
bH, aH < 0 segment with southern (-X) diractivity and upwnwards (-Z)
bH < 0, aH > 0 segment with southern (-X) diractivity and downwards (+Z)
bH > 0, aH < 0  segment with northern (+X) diractivity and upwnwards (-Z)
aH = bH Simetric segment

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob
import math
import logging
from mpl_toolkits.mplot3d.axes3d import get_test_data

_LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
loglevel = 'DEBUG'  # any one of: CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET
formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d | %(name)s | %(levelname)s | %(message)s',
                              datefmt='%Y-%m-%dT%H:%M:%S')

log = logging.getLogger('MOR-SM')
log.setLevel(loglevel)


params = {'dh': 171, # set the grid spacing in your computational domain
    'Xc': 175.489469137e3, # set north-south Cartesian location of the segment's center
    'Yc': 127683.51375307699, # set the location east-west Cartesian location of the segment's center
    'Zc': 10000, # set the depth hypocenter in Cartesian location of the segment's center
    'dip': 90, # segment's dip
    'strike': 0, # segment's strike
    'rake': 0, # # segment's angle from the horizon
    'Ga': 30000000000.0, # Shear modulus 
    'Vr_2': 2008.9999999999998, # set the second velocity od the segment of stage II
    'Vr_1': 500, # set the first velocity od the segment  of stage I
    'aH': 0.0, # set the max slip location (down-up direction)
    'bH': 0.2, # set the max slip location (south-north or east-west direction)
    'aD': -0.3, # set the first pixel location time to operate in the segment (down-up direction)
    'bD': -0.4, # set the first pixel location time to operate in the segment (south-north or east-west direction)
    'sec_stage1': 10, # set how long  the first stage will be with Vr_1
    'EveMag': 6.5 # set the desired magnitude
 }


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='''MOR-SM - Moment Rate Oriented - Slip Model''',
    epilog='''Created by Almog Blaer (blaer@post.bgu.ac.il), 2021 @ GSI/BGU''')
parser.version = '1.0'
parser.add_argument('-v', '--verbose', help='verbose - print log messages to screen?', action='store_true', default=False)
parser.add_argument('-l', '--log_level', choices=_LOG_LEVEL_STRINGS, default=loglevel,
                    help=f"Log level (Default: {loglevel}). see Python's Logging module for more details")
parser.add_argument('--logfile', metavar='log file name', help='log to file', default=None)
parser.add_argument('-p','--paramfile', metavar='parameter-file', help='Parameter file.', default=None)
parser.add_argument('-o', '--outfile', metavar='output-file', help='Output SW4 source commands file (see Chapter 11.2 in SW4 manual)',default='morsm.txt')



class MORSM(object):
    def __init__(self, args):
        if args is not None:
            self.log = log
            self.set_logger(args.verbose, args.log_level, args.logfile)
            log.debug(args)
            if args.paramfile is not None:
                get_parmeters_from_file(args.paramfile)
            self.outfile = args.outfile
            log.debug(f'Output to {self.outfile}\nParameters:\n{params}')

    def set_logger(self, verbose, log_level, logfile):
        if verbose:
            # create console handler
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            ch.setFormatter(formatter)
            if logging.StreamHandler not in [h.__class__ for h in self.log.handlers]:
                self.log.addHandler(ch)
            else:
                self.log.warning('log Stream handler already applied.')
        if logfile:
            # create file handler
            fh = TimedRotatingFileHandler(logfile,
                                          when='midnight',
                                          utc=True)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            if TimedRotatingFileHandler not in [h.__class__ for h in self.log.handlers]:
                self.log.addHandler(fh)
            else:
                self.log.warning('Log file handler already applied.')

    def run(self):
        fault, slip, time = getslipmodel()
        saveslipmodel_sw4(fault, slip, time, self.outfile)
        saveslipmodel_data(fault, slip, time, self.outfile + '.dat')


def get_parmeters_from_file(filename):
    log.info('Loading paramters from file: {}'.format(filename))
    if not os.path.isfile(filename):
        log.warning('{} not found'.format(filename))
        return
    try:
        loc = {}
        with open(filename, "rb") as source_file:
            code = compile(source_file.read(), filename, "exec")
            exec(code, loc)
    except Exception as ex:
        log.warning('Failed in adding parameters: {}'.format(ex))
    # update default params from file
    for k in params.keys():
        if loc.get(k) is not None:
            params[k] = loc[k]


def mag2fault_goda(EveMag):

    """
    This function define the segment kinematics by Goda (2016) using empirical equations magnitude-features.

    """
    max_slip = 10**(-3.7393 + 0.615 * EveMag) + 0.2249   # Segment's max slip
    mean_slip = 10**(-4.3611 + 0.6238 * EveMag) + 0.2502 # Segment's average slip
    W = (10**(-0.6892 + 0.2893 * EveMag) + 0.1464) * 1000 # Segment's width
    Y = (10**(-2.1621 + 0.5493 * EveMag) + 0.1717) * 1000 # Segment's length
    return max_slip, mean_slip, W, Y


def getslipmodel(params=params):
    """Generating the slip model"""
    dh = params['dh']
    Xc = params['Xc']
    Yc = params['Yc']
    Zc = params['Zc']
    dip = params['dip']
    strike = params['strike']
    rake = params['rake']
    Ga = params['Ga']
    Vr_2 = params['Vr_2']
    Vr_1 = params['Vr_1']
    aH = params['aH']
    bH = params['bH']
    aD = params['aD']
    bD = params['bD']
    sec_stage1 = params['sec_stage1']
    EveMag = params['EveMag']
    max_slip, mean_slip, W, Y = mag2fault_goda(EveMag)
    fault_center = np.array([[Xc], [Yc], [Zc]])
    fault_area = (W * Y)
    Ts_1 = 2 * np.sqrt(fault_area) / (3 * Vr_1) # compute the first stage long time with Vr_1
    Ts_2 = 2 * np.sqrt(fault_area) / (3 * Vr_2)  # compute the second stage long time with Vr_2
    w = np.arange(-(W / 2), (W / 2), dh)
    w = w.reshape(len(w), 1)
    y = np.arange(-(Y / 2), (Y / 2), dh)
    y = y.reshape(len(y), 1)
    fault = np.zeros((len(w) * len(y), 3))
    slip = np.zeros((len(w) * len(y), 1))
    time = np.zeros((len(w) * len(y), 1))
    vec1 = np.array([[np.cos(strike * np.pi / 180)], [np.sin(strike * np.pi / 180)], [np.sin(rake * np.pi / 180)]])
    vec2 = np.array([[np.sin(rake * np.pi / 180)], [np.cos(dip * np.pi / 180)], [np.sin(dip * np.pi / 180)]])
    count_slip = np.zeros((1000000, 1))
    count = 0
    for i in w:
        for j in y:
            vector1 = j * vec1
            vector2 = i * vec2
            fault[count, :] = (fault_center + vector1 + vector2).T
            slip[count, 0] = max_slip * np.exp(-((i) / (W / 2) + aD)**2 - ((j) / (Y / 2) + bD)**2)
            time_INV_1 = (Ts_1 / (2**0.5)) * ((((i) / (W / 2)) + aH)**2 + (((j) / (Y / 2)) + bH)**2)**0.5
            time_INV_2 = (Ts_2 / (2**0.5)) * ((((i) / (W / 2)) + aH)**2 + (((j) / (Y / 2)) + bH)**2)**0.5
            if time_INV_1 <= sec_stage1:
                time[count, 0] = time_INV_1
            elif time_INV_1 > sec_stage1:
                time[count, 0] = (Vr_2 * sec_stage1 - (Vr_1 * sec_stage1)) / Vr_2 + time_INV_2
            count = count + 1
    return fault, slip, time

   

def saveslipmodel_sw4(outfile):
    fault, slip, time = getslipmodel()
    data = np.hstack((fault, slip, time))
    data = pd.DataFrame(data)
    # get some values
    max_slip, _, W, Y = mag2fault_goda(params['EveMag'])
    sum_slip = slip[:, 0].sum() 
    mean_slip = slip[:, 0].mean()
    M0 = sum_slip * params['Ga'] * params['dh']**2
    Mw = (2 / 3) * (np.log10(M0) - 9.1)
    fault_area = (W * Y) / (1000**2)
    # add parameters to file header
    header = f'''#Vr_1 [m/sec]={params["Vr_1"]}
#Vr_2 [m/sec]={params["Vr_2"]}
#dh [m]={params["dh"]}
#Xc={params["Xc"]}
#Yc={params["Yc"]}
#Zc={params["Zc"]}
#dip [deg]={params["dip"]}
#strike [deg]={params["strike"]}
#rake [deg]={params["rake"]}
#Ga [Pa]={params["Ga"]}
#aH={params["aH"]}
#bH={params["bH"]}
#aD={params["aD"]}
#bD={params["bD"]}


#Fault area [km^2] {fault_area:.2f}
#Magnitude [Mw] {Mw:.2f}
#Seismic moment [Nm] {M0:.2f}
#Total Slip [m] {sum_slip:.2f}
#length [m] {Y:.2f}
#width [m] {W:.2f}
#The mean Slip [m] {mean_slip:.2f}
'''
    # set prefix to values
    data[0] = 'x=' + data[0].astype(str)
    data[1] = 'y=' + data[1].astype(str)
    data[2] = 'z=' + data[2].astype(str)
    data[3] = 'm0=' + (data[3] * params['Ga'] * params['dh']**2).astype(str)
    data[4] = 't0=' + (data[4] + 2).astype(str)
    data[5] = "strike=0"
    data[6] = "rake=0"
    data[7] = "dip=90"
    data[8] = "freq=6.2831"
    data[9] = "type=Gaussian"
    data[10] = 'source'
    data = data[[10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    # save to file
    with open(outfile, 'w') as f:
        f.write(header)
        data.to_csv(f, sep=" ", header=None, index=None)
    
    print(header)

        
def createfig(first_y_ticks=None,
              seconed_y_ticks=None,
              x_ticks=None, **kwargs):

    fault, slip, time = getslipmodel()
    moment_rise = np.hstack((fault, slip, time))
    moment_rise = pd.DataFrame(moment_rise, columns=['x','y','z','m0','t'])
   
   
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator)
                                   
    plt.rcParams.update({'font.size': 12})
    fig,(ax3,ax4) = plt.subplots(2,1,sharex=True,figsize=(7,15))

   
    moment_rise['m0']= params['Ga']*moment_rise['m0']* params['dh']**2
    moment_rise=moment_rise[['m0','t']]
    moment_rise=moment_rise.sort_values(['t'])
    moment_rise=moment_rise.to_numpy()

    moment=np.ones((len(moment_rise),2))
    for i in range(len(moment_rise)):
        moment[i,0]= moment_rise[i,1]
        moment[i,1]= moment_rise[:i,0].sum()

    x=moment[:,0]
    y=moment[:,1]
    s=np.linspace(moment[:,0].min(),moment[:,0].max(),50)
    a=np.interp(s,x,y)
    ax3.plot(s,a,'r.-')
    ax3.yaxis.set_major_locator(MultipleLocator(first_y_ticks))
    ax4.yaxis.set_major_locator(MultipleLocator(seconed_y_ticks))
    ax4.xaxis.set_major_locator(MultipleLocator(x_ticks))
    ax3.set_ylabel('cumulative seismic moment [Nm]')
    bb=np.gradient(a,s[1]-s[0])
    ax4.plot(s,bb,label='STF')
    ax4.legend()
    ax4.set_xlabel('Time since OT [sec]')
    ax4.set_ylabel('Moment rate [Nm/sec]')                       
    plt.show()
    
    return  



def saveslipmodel_data(outfile=None):

    plt.rcParams.update({'font.size': 16})
    fault, slip, time = getslipmodel()
    data = np.hstack((fault, slip, time))
    data = pd.DataFrame(data)
    # save to file
    data.to_csv(outfile, sep=" ", header=None, index=None)
    
    header = saveslipmodel_sw4(outfile)
    print(header)
       
    return data


def Slip_and_time_distribution():
    
    _, _, W, Y = mag2fault_goda(params['EveMag'])
    w = np.arange(-(W / 2), (W / 2), params['dh'])
    w = w.reshape(len(w), 1)
    y = np.arange(-(Y / 2), (Y / 2), params['dh'])
    y = y.reshape(len(y), 1)
    
    ones = np.ones((len(w), len(y))) 
    X = w * ones 
    Y = np.tile(y.T , (len(w), 1)) 
    _, slip, time = getslipmodel()
    
    Z_slip = slip.reshape((len(w), len(y)))
 
    Z_time = time.reshape((len(w), len(y)))
 
    
    fig, ax2 = plt.subplots(figsize=(10,5))
    cm = plt.cm.get_cmap('jet')
    sc = ax2.contourf(Y, X , Z_slip, vmin=slip.min(), vmax=slip.max(), cmap=cm)
    cbar = fig.colorbar(sc,  ax=ax2, shrink=0.9)
    cbar.set_label(r'Slip, m')
   
    
    contours =  plt.contour(Y, X, Z_time, 30 ,colors='black')
    plt.clabel(contours, inline=True, fontsize=8,fmt='%1.1f')
    ax2.set_xlabel('Length, km')
    ax2.set_ylabel('Width, km')
    plt.show()
    

def Slip_and_time_distribution_3D():
    
    _, _, W, Y = mag2fault_goda(params['EveMag'])
    w = np.arange(-(W / 2), (W / 2), params['dh'])
    w = w.reshape(len(w), 1)
    y = np.arange(-(Y / 2), (Y / 2), params['dh'])
    y = y.reshape(len(y), 1)
    
    ones = np.ones((len(w), len(y)))
    X = w * ones
    Y = np.tile(y.T , (len(w), 1))
    _, slip, time = getslipmodel()
    
    Z_slip = slip.reshape((len(w), len(y)))
 
    Z_time = time.reshape((len(w), len(y)))
  
    
    fig, ax1 = plt.subplots(figsize=(20,20))

    ax1 = plt.axes(projection='3d')
    ax1.contour3D(Y, X, Z_time, 30, cmap='binary',zorder=2)

   

    surf = ax1.plot_surface(Y, X, Z_slip, rstride=1, cstride=1, cmap='jet', vmin=slip.min(), vmax=slip.max(),
                       linewidth=0, antialiased=False,zorder=1,alpha=0.5)
    cbar= fig.colorbar(surf, shrink=0.5, aspect=10)
    cbar.set_label('Slip, m')
    ax1.set_xlabel('Length, km',labelpad=7)
    ax1.set_ylabel('Width, km',labelpad=7)
    ax1.set_zlabel('Time ,sec',labelpad=7)
    plt.show()    
    
    
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    morsm = MORSM(args)
    morsm.run()
    