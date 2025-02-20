from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(name="serrec_validator",
        version="0.0.4",
        url='https://github.com/AyoubMKDM/SerRec-Validator',
        description="A framework for service recommendation evaluation using the WS-DREAM dataset",
        long_description_content_type="text/markdown",
        long_description=open('README.md').read(),
        packages=["serrec_validator"],
        license='BSD',
        author="Ayoub Mokeddem",
        author_email="ayoubmkdm3@gmail.com",
        install_requires=requirements,
        extras_requires={
            "dev": ["pytest>=7.0","twine>=4.0.2"],
        },
        include_package_data=True,
        package_data={'serrec_validator':['wsdream/*.txt']},
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Information Technology',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: System :: Benchmark',
            'License :: OSI Approved :: BSD License',
        ],
        python_requires=">=3.8",
        )