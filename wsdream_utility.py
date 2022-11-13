import urllib.request as urlopener
import os
from zipfile import ZipFile
import shutil
import pandas as pd
import numpy as np

def dataset_downloader(dir=None, url="https://zenodo.org/record/1133476/files/wsdream_dataset1.zip?download=1"):
    """
    Retrieve and download all the WS-DREAM dataset as a zip file, then extract the content from the zip file and delete the latter.
    This will add to the current directory four files: rtMatrix.txt, tpMatrix.txt, userlist.txt, wslist.txt.
    Parameters:
        dir - String for the folder name specifying where the dataset will be stored default value is the current directory
        url - String of the URL of the downloadable zip file with default value 'https://zenodo.org/record/1133476/files/wsdream_dataset1.zip?download=1'
    """
    # TODO Handle http and url exceptions .. Use https://docs.python.org/3/library/urllib.error.html#urllib.error.URLError 
    file_name = url.split('/')[-1]
    page = urlopener.urlopen(url)
    meta = page.info()
    file_size = int(meta.get("Content-Length")[0])
    print ("Downloading: %s (%s bytes)" % (file_name, file_size))
    urlopener.urlretrieve(url, file_name)
    print ('Unzip data files...')
    with ZipFile(file_name, 'r') as archive:
        for name in archive.namelist():
            filename = os.path.basename(name)
            # skip directories
            if not filename:
                continue
            # copy file (taken from zipfile's extract)
            print (filename)
            source = archive.open(name)
            if(dir is None):
                target = open(filename, "wb")
            else:
                extraction_path = os.path.join(os.getcwd(),dir)
                if not os.path.exists(extraction_path):
                    print(f"Creating '{dir}/' folder ...")
                    os.mkdir(dir)
                target = open(os.path.join(extraction_path,filename),'wb')
            with source, target:
                shutil.copyfileobj(source, target)

    os.remove(file_name)
    print('==============================================')
    print('Downloading data done!\n')
    pass


def dataframe_fromtxt(file):
    """
    The original dataset provided is in .txt files which is not a convenient way to work with the dataset files.
    This method reads the txt file containing the list of users or services, and translate the content to a pandas DataFrame.
    Parameters:
        file - String contain a file_name.txt of the.
    return:
        pandas.DataFrame object of the content of the file passed as a parameter
    """
    data_file = open(file, 'r', encoding='utf-8', errors='replace')
    indices = data_file.readline().strip().split('\t')
    indices = [index.strip('[]') for index in indices] 
    # Creating the DataFrame of users/services with the title line first
    df = pd.DataFrame(columns=indices)
    data_file.readline()
    for line in data_file:
        df.loc[len(df)] = line.strip().split('\t')
    df.replace(to_replace="null",value=None,inplace=True)
    data_file.close()
    return df


class dataset:
    """
    Singleton class contains an instance of the dataset, with the different tables i.e. usersList, servicesList, responseTimeMatrix, throughputMatrix.
    To simplify the interaction with the dataset.
    
    Attributes:
    usersList (DataFrame) contains information on 339 service users. with the indices: User ID, IP Address, Country, Continent, AS, Latitude, Longitude, Region, City.
    servicesList (DataFrame) contains information on the 5,825 Web services. with the indices: Service ID, WSDL Address, Service Provider, IP Address, Country, Continent, AS, Latitude, Longitude, Region, City,
    responseTimeMatrix (ndarray) 339 * 5825 user-item matrix of response-time.
    throughputMatrix (ndarray) 339 * 5825 user-item matrix for throughput.

    Methods:
    save_lists_tocsv (None) saves the usersList, and servicesList DataFrames into a CSV file when needed.
    """
    # TODO create a final static url var
    __USERS_LIST_FILE_NAME="userlist.txt"
    __SERVICES_LIST_FILE_NAME="wslist.txt"
    __RESPONSE_TIME_MATRIX_FILE_NAME = "rtMatrix.txt"
    __THROUGHPUT_MATRIX_FILE_NAME = "tpMatrix.txt"
    __dir = None
    __instance = None

    def __new__(cls, dir=None, url="https://zenodo.org/record/1133476/files/wsdream_dataset1.zip?download=1"):
        if (cls.__instance is None):
            print('Creating the dataset')
            cls.__dir = dir
            full_path = os.getcwd()
            # print(full_path)
            if dir is not None:
                full_path = os.path.join(full_path,cls.__dir)
                # print(full_path)
                
            # Check if the data in the specified directory exists
            if os.path.exists(full_path):
                ls = os.listdir(full_path)
                # print(ls)
                # If the folder exist check if all the files exist
                if (cls.__USERS_LIST_FILE_NAME not in ls) or (cls.__SERVICES_LIST_FILE_NAME not in ls) \
                    or (cls.__RESPONSE_TIME_MATRIX_FILE_NAME not in ls) or (cls.__THROUGHPUT_MATRIX_FILE_NAME not in ls):
                        dataset_downloader(url=url,dir=cls.__dir)
            else:
                dataset_downloader(url=url, dir=dir)

            # Initialize the class atributes 
            cls.__initialize()
            cls.__instance = super(dataset, cls).__new__(cls)
        return cls.__instance 

    def __initialize(cls):
        if cls.__dir is None:
            cls.usersList = dataframe_fromtxt(file=cls.__USERS_LIST_FILE_NAME)
            cls.servicesList = dataframe_fromtxt(file=cls.__SERVICES_LIST_FILE_NAME)
            cls.responseTimeMatrix = np.loadtxt(cls.__RESPONSE_TIME_MATRIX_FILE_NAME)
            cls.throughputMatrix = np.loadtxt(cls.__THROUGHPUT_MATRIX_FILE_NAME)
        else:
            cls.usersList = dataframe_fromtxt(file=os.path.join(cls.__dir, cls.__USERS_LIST_FILE_NAME))
            cls.servicesList = dataframe_fromtxt(file=os.path.join(cls.__dir, cls.__SERVICES_LIST_FILE_NAME))
            cls.responseTimeMatrix = np.loadtxt(os.path.join(cls.__dir, cls.__RESPONSE_TIME_MATRIX_FILE_NAME))
            cls.throughputMatrix = np.loadtxt(os.path.join(cls.__dir, cls.__THROUGHPUT_MATRIX_FILE_NAME))     
        
        cls.servicesList['IP No.'].replace("0",value=None,inplace=True)
        pass


    def save_lists_tocsv(self):
        """
        Create a CSV file from usersList and servicesList DataFrames. and store it in the current directory.
        Parameters:
            None
        Return:
            None
        """
        self.usersList.to_csv("usersList.csv")
        self.servicesList.to_csv("servicesList.csv")
        pass