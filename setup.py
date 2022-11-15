from setuptools import setup

setup(name="wsdream_utility",
        version="0.1",
        description="Dataset helper",
        packages=["wsdream_helper"],
        author="Ayoub Mokeddem",
        author_email="ayoubmkdm3@gmail.com",
        install_requires=["numpy>=1.18.5",
                            "pandas >= 1.4.2"])