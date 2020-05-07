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
            'zarr',
            'dask',
            'scikit-image',
            'pycuda',
            'pynrrd',
            'python-louvain',
            'toolz',
            'tifffile',
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
