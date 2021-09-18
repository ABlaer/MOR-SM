```python
import MORSM
from IPython.display import Image
%matplotlib widget
from mpl_toolkits import mplot3d
```
Copyright (C) by Almog Blaer 

                                      
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

# MOR-SM parameters setting
dh: set the grid spacing in your computational domain
Xc: set north-south Cartesian location of the segment's center
Yc: set the location east-west Cartesian location of the segment's center
Zc: set the depth hypocenter in Cartesian location of the segment's center
dip: segment's dip
strike: segment's strike
alpha: segment's angle frim the horizon
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


```python
Image('Computational domain.png')
```




    

    




```python
MORSM.params['Vr_1']=2500
MORSM.params['Vr_2']=2500

MORSM.params['aH']=0.3
MORSM.params['bH']=0.6

MORSM.params['aD']=0.0
MORSM.params['bD']=0.0

MORSM.params['EveMag']=6.5
MORSM.params['sec_stage1']=0
MORSM.params
```




    {'dh': 171,
     'Xc': 175489.469137,
     'Yc': 127683.51375307699,
     'Zc': 10000,
     'dip': 90,
     'strike': 0,
     'alpha': 0,
     'Ga': 30000000000.0,
     'Vr_2': 2500,
     'Vr_1': 2500,
     'aH': 0.3,
     'bH': 0.6,
     'aD': 0.0,
     'bD': 0.0,
     'sec_stage1': 0,
     'EveMag': 6.5}



# Generate  Source Time Function (STF)


```python
MORSM.createfig(first_y_ticks=5e18,seconed_y_ticks=5e17,x_ticks=1)
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …
    ![png](Comulative_moment_and_STF.png)

# Create slip and time distribution


```python
MORSM.Slip_and_time_distribution()
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …
    ![png](Slip_Time_functions_2D.png)


```python
MORSM.Slip_and_time_distribution_3D()
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …
    ![png](Slip_Time_functions_3D.png)

# Save file


```python
MORSM.saveslipmodel_sw4('Simulation_name.txt')
```

    #Vr_1 [m/sec]=2500
    #Vr_2 [m/sec]=2500
    #dh [m]=171
    #Xc=175489.469137
    #Yc=127683.51375307699
    #Zc=10000
    #dip [deg]=90
    #strike [deg]=0
    #aplha [deg]=0
    #Ga [Pa]=30000000000.0
    #aH=0.3
    #bH=0.6
    #aD=0.0
    #bD=0.0
    
    
    #Fault area [km^2] 404.18
    #Magnitude [Mw] 6.69
    #Seismic moment [Nm] 13810436054444288000.00
    #Total Slip [m] 15743.23
    #length [m] 25778.19
    #width [m] 15679.21
    #The mean Slip [m] 1.13
    



```python

```