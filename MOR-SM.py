"""
  __  __  ____  _____        _____ __  __ 
 |  \/  |/ __ \|  __ \      / ____|  \/  |
 | \  / | |  | | |__) |____| (___ | \  / |
 | |\/| | |  | |  _  /______\___ \| |\/| |
 | |  | | |__| | | \ \      ____) | |  | |
 |_|  |_|\____/|_|  \_\    |_____/|_|  |_|
                                      



                                              """

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
import glob
import math
np.seterr(all='warn')


import os
def prepend_line(file_name, line):
    """ Insert given string as a new line at the beginning of a file """
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)

list_of_lines = ['Another line to prepend', 'Second Line to prepend',  'Third Line to prepend']


##############   Variables Setting          ##################
   

dh=171
Xc=175.489469137e3
Yc= 127.68351375307698e3
Zc=10000
dip=90
strike=0
alpha=0
Ga=30e9
Vr_2=2870*0.7
Vr_1=500
aH=0.0
bH=0.2
aD=-0.3
bD=-0.4
sec_stage1=10


EveMag=input('Enter the Event Magnitude (SW4): ')  
EveMag=np.float(EveMag)
print("The Event Magnitude is [Goda 2016]", EveMag)


max_slip=10**(-3.7393+0.615*EveMag)+0.2249 
mean_slip=10**(-4.3611+0.6238*EveMag)+0.2502
W=(10**(-0.6892+0.2893*EveMag)+0.1464)*1000
Y=(10**(-2.1621+0.5493*EveMag)+0.1717)*1000
print("Max slip is [Goda 2016]", round(max_slip,2))
print("Mean slip is [Goda 2016]", round(mean_slip,2))
print("Fault width is [Goda 2016]", round(W,2))
print("Fault length is [Goda 2016]", round(Y,2))

##############################################################

fault_center=np.array([[Xc],[Yc],[Zc]])

w=np.arange(-(W/2),(W/2),dh)
y=np.arange(-(Y/2),(Y/2),dh)

w=w.reshape(len(w),1)
y=y.reshape(len(y),1)


fault=np.zeros((len(w)*len(y),3))
slip=np.zeros((len(w)*len(y),1))
time=np.zeros((len(w)*len(y),1))
count_slip=np.zeros((1000000,1))

count=0
for i in w: 
    for j in y:
      
        vector1=j*np.array([[np.cos(strike*np.pi/180)],[np.sin(strike*np.pi/180)],[np.sin(alpha*np.pi/180)]])
        vector2=i*np.array([[np.sin(alpha*np.pi/180)],[np.cos(dip*np.pi/180)],[np.sin(dip*np.pi/180)]])
        surface=fault_center+vector1+vector2
        fault[count,:]=surface.T
        slip_pixel=max_slip*np.exp(-((i)/(W/2)-aH)**2-((j)/(Y/2)-bH)**2)
        slip[count,0]=slip_pixel
        fault_area=(W*Y)
        
        
     
        Ts_1=2*np.sqrt(fault_area)/(3*Vr_1)
        Ts_2=2*np.sqrt(fault_area)/(3*Vr_2)
        time_INV_1=(Ts_1/(2**0.5))*((((i)/(W/2))-aD)**2+(((j)/(Y/2))-bD)**2)**0.5  
        time_INV_2=(Ts_2/(2**0.5))*((((i)/(W/2))-aD)**2+(((j)/(Y/2))-bD)**2)**0.5 
        if time_INV_1<=sec_stage1:
           time[count,0]=time_INV_1
        elif time_INV_1>sec_stage1:
           time[count,0]=(Vr_2*sec_stage1-(Vr_1*sec_stage1))/Vr_2+time_INV_2
         
        
        count=count+1
        
       
file=np.hstack((fault,slip,time))
file=pd.DataFrame(file) 
file.to_csv('file.txt',sep=" ",header=None,index=None)

file[0] = 'x=' + file[0].astype(str)
file[1] = 'y=' + file[1].astype(str)
file[2] = 'z=' + file[2].astype(str)
file[3] = 'm0='+ (file[3]*Ga*dh**2).astype(str)
file[4] = 't0=' + (file[4]+2).astype(str)

file[5] ="strike=0" 
file[6] ="rake=0"
file[7] ="dip=90"
file[8] ="freq=6.2831"
file[9] ="type=Gaussian"


file[10]= 'source'
file=file[[10,0,1,2,3,4,5,6,7,8,9]]
file.to_csv('MOR-SM.txt',sep=" ",header=None,index=None)





print(" ")
print("########Creating segment by the given parameters...########")
print(" ")
print(" ")
sum_slip= slip[:,0].sum()
mean_slip= slip[:,0].mean()
M0=sum_slip*Ga*dh**2
Mw=(2/3)*(np.log10(M0)-9.1)
fault_area=(W*Y)/(1000**2)
print("Fault area [km^2] ",round(fault_area,2))
print("Magnitude [Mw]", round(Mw,2))
print("Seismic moment [Nm]", round(M0,2))
print("Total Slip [m] ", round(sum_slip,2))
print("The mean Slip [m] ", round(mean_slip,2))

list_of_lines=['#Vr_1 [m/sec]='+str(Vr_1),'#Vr_2 [m/sec]='+str(Vr_2),'#dh [m]='+str(dh),'#Xc='+str(Xc),'#Yc='+str(Yc),'#Zc='+str(Zc),'#dip [deg]='+str(dip),'#strike [deg]='+str(strike),'#aplha [deg]='+str(alpha),'#Ga [Pa]='+str(Ga),'#aH='+str(aH),'#bH='+str(bH),'#aD='+str(aD),'#bD='+str(bD),"  ","  ","#Fault area [km^2]"+str(round(fault_area,2)),"#Magnitude [Mw] "+str(round(Mw,2)),"#Seismic moment [Nm] "+str(round(M0,2)),"#Total Slip [m] "+str(round(sum_slip,2)),"#length [m] "+str(round(Y,2)),"#width [m] "+str(round(W,2)),"#The mean Slip [m] "+str(round(mean_slip,2))]
[prepend_line("MOR-SM.txt", line) for line in list_of_lines ]



#os.system('gmt info file.txt')

###########################################################
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
                               
plt.rcParams.update({'font.size': 12})
fig,(ax3,ax4) = plt.subplots(2,1,sharex=True,figsize=(7,15))



time=pd.read_csv('file.txt',sep=" ",header=None, names=['x','y','z','m0','t'])
time['m0']=Ga*time['m0']*dh**2
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
