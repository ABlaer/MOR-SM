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
    keywords=["MOR-SM", "seismology", "SW4","EEW", "GMM"],0
    url="https://github.com/ABlaer/MOR-SM",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        ],
 
)
