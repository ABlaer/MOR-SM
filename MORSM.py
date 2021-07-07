#!/usr/bin/env python


"""
Copyright (C) by Almog Blaer 



  __  __  ____  _____        _____ __  __ 
 |  \/  |/ __ \|  __ \      / ____|  \/  |
 | \  / | |  | | |__) |____| (___ | \  / |
 | |\/| | |  | |  _  /______\___ \| |\/| |
 | |  | | |__| | | \ \      ____) | |  | |
 |_|  |_|\____/|_|  \_\    |_____/|_|  |_|
                                      
                                      
Get ready and tighten belts, we going to launch an synthetic earthquake in any location you wish.
This code can depict a finite segment aim to be panted in SW4 software.
You only need to tell us about your computertion domain and  about the segment's kinematic.

the general code steps are:
1. Defining the the segment's dimensions (width,length and slip by Goda (2016) equations for desired magnitude
2. Fitting a location parameters for the slip and time functions
3. Computing the "sliding time" Tm from stage I and stage II for velocity I and velocity II respectively 
4. The features above are distributed by the time and slip function on each pixel on the segment
5. You can generate  STF to the earthquake model you have just set. 

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
    'alpha': 0, # # segment's angle frim the horizon
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

##### Parser for command line

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='''MOR-SM - Moment Rate Oriented - Slip Model''',
    epilog='''Created by Almog Blaer (blaer@post.bgu.ac.il), 2021 @ GSI/BGU''')
parser.add_argument('-v', '--verbose', help='verbose - print log messages to screen?', action='store_true', default=False)
parser.add_argument('-l', '--log_level', choices=_LOG_LEVEL_STRINGS, default=loglevel,
                    help=f"Log level (Default: {loglevel}). see Python's Logging module for more details")
parser.add_argument('--logfile', metavar='log file name', help='log to file', default=None)
parser.add_argument('-p','--paramfile', metavar='parameter-file', help='Parameter file.', default=None)
parser.add_argument('-o', '--outfile', metavar='output-file', help='Output SW4 source commands file (see Chapter 11.2 in SW4 manual)',default='morsm.txt')

### Parameters in param file

#dh=171  # what is this?
#Xc=175.489469137e3
#Yc= 127.68351375307698e3
#Zc=10000
#dip=90
#strike=0
#alpha=0
#Ga=30e9
#Vr_2=2870*0.7
#Vr_1=500
#aH=0.0
#bH=0.2
#aD=-0.3
#bD=-0.4
#sec_stage1=10
#EveMag=6.5 

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
    alpha = params['alpha']
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
    vec1 = np.array([[np.cos(strike * np.pi / 180)], [np.sin(strike * np.pi / 180)], [np.sin(alpha * np.pi / 180)]])
    vec2 = np.array([[np.sin(alpha * np.pi / 180)], [np.cos(dip * np.pi / 180)], [np.sin(dip * np.pi / 180)]])
    count_slip = np.zeros((1000000, 1))
    count = 0
    for i in w:
        for j in y:
            vector1 = j * vec1
            vector2 = i * vec2
            fault[count, :] = (fault_center + vector1 + vector2).T
            slip[count, 0] = max_slip * np.exp(-((i) / (W / 2) - aH)**2 - ((j) / (Y / 2) - bH)**2)
            time_INV_1 = (Ts_1 / (2**0.5)) * ((((i) / (W / 2)) - aD)**2 + (((j) / (Y / 2)) - bD)**2)**0.5
            time_INV_2 = (Ts_2 / (2**0.5)) * ((((i) / (W / 2)) - aD)**2 + (((j) / (Y / 2)) - bD)**2)**0.5
            if time_INV_1 <= sec_stage1:
                time[count, 0] = time_INV_1
            elif time_INV_1 > sec_stage1:
                time[count, 0] = (Vr_2 * sec_stage1 - (Vr_1 * sec_stage1)) / Vr_2 + time_INV_2
            count = count + 1
    return fault, slip, time
    
def cratefig(fault, slip, time, outfile):
    data = np.hstack((fault, slip, time))
    data = pd.DataFrame(data)
    # save to file
    data.to_csv("time.txt", sep=" ", header=None, index=None)
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator)
                                   
    plt.rcParams.update({'font.size': 12})
    fig,(ax3,ax4) = plt.subplots(2,1,sharex=True,figsize=(7,15))


    time=pd.read_csv('time.txt',sep=" ",header=None, names=['x','y','z','m0','t'])
    time['m0']= params['Ga']*time['m0']* params['dh']**2
    time=time[['m0','t']]
    time=time.sort_values(['t'])
    time=time.to_numpy()

    moment=np.ones((len(time),2))
    for i in range(len(time)):
        moment[i,0]=time[i,1]
        moment[i,1]=time[:i,0].sum()

    x=moment[:,0]
    y=moment[:,1]
    s=np.linspace(moment[:,0].min(),moment[:,0].max(),50)
    a=np.interp(s,x,y)
    ax3.plot(s,a,'r.-')

    # ax3.yaxis.set_major_locator(MultipleLocator(2e17))
    # ax3.yaxis.set_minor_locator(MultipleLocator(1e17))
    # ax3.xaxis.set_major_locator(MultipleLocator(1))
    # ax3.xaxis.set_minor_locator(MultipleLocator(0.1)) 
    ax3.set_ylabel('cumulative seismic moment [Nm]')

    bb=np.gradient(a,s[1]-s[0])
    ax4.plot(s,bb,label='STF')
    ax4.legend()
    # ax4.yaxis.set_major_locator(MultipleLocator(1e17))
    # ax4.yaxis.set_minor_locator(MultipleLocator(1e16))
    # ax4.xaxis.set_major_locator(MultipleLocator(1))
    # ax4.xaxis.set_minor_locator(MultipleLocator(0.1)) 
    ax4.set_xlabel('time since OT [sec]')
    ax4.set_ylabel('moment rate [Nm/sec]')
    fig.savefig('r.png')



def saveslipmodel_sw4(fault, slip, time, outfile):
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
#aplha [deg]={params["alpha"]}
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


def saveslipmodel_data(fault, slip, time, outfile):
    data = np.hstack((fault, slip, time))
    data = pd.DataFrame(data)
    # save to file
    data.to_csv(outfile, sep=" ", header=None, index=None)






if __name__ == '__main__':
    args = parser.parse_args()
    morsm = MORSM(args)
    morsm.run()