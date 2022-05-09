
*Copyright (C) by Almog Blaer*
```
         __  __  ____  _____        _____ __  __ 
	|  \/  |/ __ \|  __ \      / ____|  \/  |
	| \  / | |  | | |__) |____| (___ | \  / |
	| |\/| | |  | |  _  /______\___ \| |\/| |
	| |  | | |__| | | \ \      ____) | |  | |
	|_|  |_|\____/|_|  \_\    |_____/|_|  |_|

``` 
# MOR-SM
:aquarius: Moment-rate ORriented Slip Model, enables control on earthquakes moment rate timing, as seen from inversions word-wide :part_alternation_mark: 

### What is MOR-SM?

MOR-SM is moment-rate oriented slip model (kinematic approach), for seismic wave propagation simulation software - SW4 [^1].
The code contains a collection of a few command-line tools for manipulating slip function, time function and 
source time function, using source physical properties.

General code steps:

 1. Defining the segment's dimensions (width, length and slip) by Goda et al. (2016) [^2] equations for the desired magnitude.
 2. Fitting location parameters for the slip and time functions based on the desired diractivity.
 3. Computing the sliding time (Tr) from stage I and stage II for velocity I and velocity II respectively. 
 4. The features above are distributed by the time and slip function on each pixel on the segment.
 5. Generating Source Time Function (STF) for the earthquake you have just set.
 6. Compare your earthquake model to world-wide :earth_africa: inversions, SCARDEC database (Valle et al. 2011) [^3].
 7. Save your file as SW4 input.


[^1]:
    Petersson, N. A., and B. Sjögreen, 2017, SW4 v2.0.

[^2]:
    Goda, K., Y. Tomohiro, Nobuhito Mori, and Takuma Maruyama, 2016, 
    New Scaling Relationships of Earthquake Source Parameters for Stochastic Tsunami Simulation, 
    Coastal Engineering Journal, 58, no. 3, 1–40, doi: 10.1142/S0578563416500108.
[^3]:
    Vall´ee, M. & Douet, V., 2016. A new database of source time functions
    (STFs) extracted from the SCARDEC method, Phys. Earth planet. Inter.,
    257, 149–157.

### Installation

     pip install MORSM

or 

     git clone https://github.com/ABlaer/MOR-SM.git

### Depndencies

- argparse
- sys
- os
- pandas
- numpy
- matplotlib
- glob
- logging
- mpl_toolkits.mplot3d.axes3d

### Usage

     $ python MORSM.py [-v] [-l] [--logfile] [-p] [-o]

optional arguments:

     -h, --help            show this help message and exit
     -v, --verbose         verbose - print log messages to screen?
     -l {CRITICAL,ERROR,WARNING,INFO,DEBUG}, --log_level {CRITICAL,ERROR,WARNING,INFO,DEBUG}  Log level (Default: DEBUG). see Python's Logging module for more details
     --logfile log file name log to file
     -p parameter-file, --paramfile parameter-file Parameter file.
     -o output-file, --outfile output-file Output SW4 source commands file (see Chapter 11.2 in SW4 manual)

### Credits

MOR-SM relies on research with Ben-Gurion University of the Negev and Geological Survey of Israel. My thanks to Dr. Ran Nof and Professor Michael Tsesarsky for participating in this process.

### License

*Copyright (c) 2021 by Almog Blaer.*
MOR-SM is released under the GNU Lesser General Public License version 3 or any later version. See LICENSE.TXT for full details.

