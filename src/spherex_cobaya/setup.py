from setuptools import setup

setup(
    name='spherex_cobaya',
    version='0.1.0',
    description='SPHEREx simulated likelihoods',
    url='https://github.com/chenheinrich/spherex_cobaya',
    author='Chen Heinrich',
    author_email='chenhe@caltech.edu',
    license='BSD 2-clause',
    packages=['spherex_cobaya'],
    #install_requires=['camb==1.3.2',
    #                  'cobaya==3.0.3'],

    # TODO double check license
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
