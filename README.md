	 __  __  ____  _____        _____ __  __ 
	|  \/  |/ __ \|  __ \      / ____|  \/  |
	| \  / | |  | | |__) |____| (___ | \  / |
	| |\/| | |  | |  _  /______\___ \| |\/| |
	| |  | | |__| | | \ \      ____) | |  | |
	|_|  |_|\____/|_|  \_\    |_____/|_|  |_|
 
Copyright (C) by Almog Blaer 
 
# MOR-SM

## moment-rate oriented slip model

<<<<<<< HEAD


=======
"""
   __  __  ____  _____        _____ __  __ 
  |  \/  |/ __ \|  __ \      / ____|  \/  |
  | \  / | |  | | |__) |____| (___ | \  / |
  | |\/| | |  | |  _  /______\___ \| |\/| |
  | |  | | |__| | | \ \      ____) | |  | |
  |_|  |_|\____/|_|  \_\    |_____/|_|  |_|
  
Copyright (C) by Almog Blaer 
  
"""

 
>>>>>>> 96e4c92107356f17c44ec6cc0f50037724c15b77


## What is MOR-SM?

MOR-SM is moment-rate oriented slip model for seismic wave propagation simulation software - SW4.
The code collection a few command-line tools for manipulating slip function, time function and 
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

 1. -h, --help            show this help message and exit
 2. -v, --verbose         verbose - print log messages to screen?
 3. -l {CRITICAL,ERROR,WARNING,INFO,DEBUG}, --log_level {CRITICAL,ERROR,WARNING,INFO,DEBUG}  Log level (Default: DEBUG). see Python's Logging module for more details
 4. --logfile log file name log to file
 5. -p parameter-file, --paramfile parameter-file Parameter file.
 6. -o output-file, --outfile output-file Output SW4 source commands file (see Chapter 11.2 in SW4 manual)

===============================================





## License

Copyright (c) 2021 by Almog Blaer.

MOR-SM is released under the GNU Lesser General Public License version 3 or any later version. See LICENSE.TXT for full details.


## Acknowledgment

MOR-SM is relies on research with both Ben-Gurion University of the Negev and Geological Survey of Israel.
I gratefully acknowledge for Dr. Ran Nof and Prof. Michael Tsesarsky for taking part of this code. 
