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
            'mahotas',
            'matplotlib',
            'natsort',
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
            'tqdm',
            'zarr',
            'augment @ git+https://github.com/funkey/augment.git@master#egg=augment',
            'evaluate-instance-segmentation @ git+https://github.com/Kainmueller-Lab/evaluate-instance-segmentation.git@master#egg=evaluate-instance-segmentation',
            'gunpowder @ git+https://github.com/Kainmueller-Lab/gunpowder.git@6babc4119d6ed6049b5b9def322dfd6c155579e7#egg=gunpowder'
            'malis @ git+https://github.com/TuragaLab/malis.git@master#egg=malis',
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
