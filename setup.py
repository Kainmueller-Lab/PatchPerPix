from setuptools import setup

setup(
        name='PatchPerPix',
        version='0.1',
        description='Dense patch prediction per pixel for instance segmentation.',
        url='',
        author='Peter Hirsch, Lisa Mais, Dagmar Kainmueller',
        author_email='kainmuellerlab@mdc-berlin.de',
        license='MIT',
        install_requires=[
            'colorcet',
            'Cython',
            'dask',
            'h5py',
            'joblib',
            'mahotas',
            'malis @ git+https://github.com/TuragaLab/malis.git@master#egg=malis',
            'matplotlib',
            'neurolight @ git+https://github.com/maisli/neurolight.git@master#egg=neurolight',
            'networkx',
            'numcodecs',
            'numpy',
            'pycuda',
            'pynrrd',
            'python-louvain',
            'scikit-image',
            'scipy',
            'tifffile',
            'toml',
            'toolz',
            'zarr'
        ],
        packages=[
                'PatchPerPix',
                'PatchPerPix.vote_instances',
                'PatchPerPix.visualize',
                'PatchPerPix.evaluate',
                'PatchPerPix.models',
                'PatchPerPix.util',
        ]
)
