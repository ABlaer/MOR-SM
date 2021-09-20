import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
install_requires = [
    'pandas',
    'numpy',
    'matplotlib.pyplot',
    'os',
    'sys',
    'argparse',
    'glob'
    'math'
    'logging',
    'mpl_toolkits.mplot3d.axes3d'
    ]
    
    
setuptools.setup(
    name="MOR-SM",
    version="1.0.0",
    author="almog_blaer",
    author_email="blaer@post.bgu.ac.il",
    description="moment-rate oriented slip distribution for SW4 Seismic Waves simulation",
    long_description=long_description,
    long_description_content_type="This code can depict a finite segment that is aimed to be planted in SW4- Seismic Waves simulation",
    keywords=["MOR-SM", "seismology", "SW4","EEW", "GMM"],
    url="https://github.com/ABlaer/MOR-SM",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved ::  GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
)
