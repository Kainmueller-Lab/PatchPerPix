from setuptools import setup

setup(
        name='PatchPerPix',
        version='0.9',
        description='Dense patch prediction per pixel for instance segmentation.',
        url='',
        author='Peter Hirsch, Lisa Mais, Dagmar Kainmueller',
        author_email='firstname.lastname@mdc-berlin.de',
        license='MIT',
        install_requires=[
            'colorcet',
            'Cython',
            'h5py',
            'joblib',
            'matplotlib',
            'natsort',
            'networkx',
            'numcodecs',
            'numpy',
            'pycuda',
            'scikit-image',
            'scipy',
            'tifffile',
            'toml',
            'tqdm',
            'zarr',
            'evaluate-instance-segmentation @ git+https://github.com/Kainmueller-Lab/evaluate-instance-segmentation.git@master#egg=evaluate-instance-segmentation',
            'gunpowder @ git+https://github.com/Kainmueller-Lab/gunpowder.git@6babc4119d6ed6049b5b9def322dfd6c155579e7#egg=gunpowder',
            'neurolight @ git+https://github.com/maisli/neurolight.git@master#egg=neurolight',
            'funlib.learn.torch @ git+https://github.com/Kainmueller-Lab/funlib.learn.torch@ppp#egg=funlib.learn.torch',
        ],
        packages=[
                'PatchPerPix',
                'PatchPerPix.vote_instances',
                'PatchPerPix.visualize',
                'PatchPerPix.evaluate',
                'PatchPerPix.util',
        ]
)
