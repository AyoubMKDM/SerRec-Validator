from setuptools import setup

#TODO add classifiers
setup(name="wsdream_utility",
        version="0.1",
        description="Dataset helper",
        packages=["wsdream_helper"],
        author="Ayoub Mokeddem",
        author_email="ayoubmkdm3@gmail.com",
        install_requires=["numpy",
                            "pandas"],
        include_package_data=True,
        package_data={'':['wsdream/*']})