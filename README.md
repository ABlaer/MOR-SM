	 __  __  ____  _____        _____ __  __ 
	|  \/  |/ __ \|  __ \      / ____|  \/  |
	| \  / | |  | | |__) |____| (___ | \  / |
	| |\/| | |  | |  _  /______\___ \| |\/| |
	| |  | | |__| | | \ \      ____) | |  | |
	|_|  |_|\____/|_|  \_\    |_____/|_|  |_|
 
Copyright (C) by Almog Blaer 
 
# MOR-SM
### moment-rate oriented slip model

=======


## What is MOR-SM?

MOR-SM is moment-rate oriented slip model for seismic wave propagation simulation software - SW4.
The code contains collection of a few command-line tools for manipulating slip function, time function and 
source time function for depicting the source physical properties.


## Installation


     pip install MOR-SM

or 

     git clone https://github.com/ABlaer/MOR-SM.git


## Depndencies

### Python modules

* argparse
* sys
* os
* pandas
* numpy
* matplotlib
* glob
* logging
* mpl_toolkits.mplot3d.axes3d

### Usage

     $ python MORSM.py [-v] [-l] [--logfile] [-p] [-o]

optional arguments:

     -h, --help            show this help message and exit
     -v, --verbose         verbose - print log messages to screen?
     -l {CRITICAL,ERROR,WARNING,INFO,DEBUG}, --log_level {CRITICAL,ERROR,WARNING,INFO,DEBUG}  Log level (Default: DEBUG). see Python's Logging module for more details
     --logfile log file name log to file
     -p parameter-file, --paramfile parameter-file Parameter file.
     -o output-file, --outfile output-file Output SW4 source commands file (see Chapter 11.2 in SW4 manual)

===============================================




## License

Copyright (c) 2021 by Almog Blaer.

MOR-SM is released under the GNU Lesser General Public License version 3 or any later version. See LICENSE.TXT for full details.


## Acknowledgment

MOR-SM relies on research with  Ben-Gurion University of the Negev and Geological Survey of Israel.
I gratefully acknowledge Dr. Ran Nof and Prof. Michael Tsesarsky for taking part in this process. 
