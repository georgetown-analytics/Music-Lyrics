'''
a Minimal setup.py file
'''

from setuptools import setup
config = {
	"name": "foo",
	"version": "0.1",
	"description": "example of a project layout",
	"url": "https://github.com/aww195/project-layout",
	"author": "Flying Circus",
	"author_email": "aww45@georgetown.edu",
	"license": "MIT",
	"packages": [
		"foo",
	],
	"zip_safe": "False",
}

setup(**config)

'''
opt/anaconda3/lib/python3.8/site-packages/setuptools/__init__.py in setup(**attrs)
    151     # Make sure we have any requirements needed to interpret 'attrs'.
    152     _install_setup_requires(attrs)
--> 153     return distutils.core.setup(**attrs)
    154 
    155 

/opt/anaconda3/lib/python3.8/distutils/core.py in setup(**attrs)
    111             raise SystemExit("error in setup command: %s" % msg)
    112         else:
--> 113             raise SystemExit("error in %s setup command: %s" % \
    114                   (attrs['name'], msg))
    115 

SystemExit: error in foo setup command: 'zip_safe' must be a boolean value (got 'False')
(base) Gretzky@GretzkyMacPro music_analysis % 
'''
print("Super-fun")
