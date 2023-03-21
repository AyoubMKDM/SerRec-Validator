import urllib.request as urlopener
import os
from zipfile import ZipFile
import shutil
import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from .utility import Normalization, DatasetFactory



def dataset_downloader(dir=None, url="https://zenodo.org/record/1133476/files/wsdream_dataset1.zip?download=1"):
    """
    Retrieve and download the WS-DREAM dataset as a zip file from the specified URL, 
    then extract the content from the zip file and delete the latter. This function 
    will add four files (rtMatrix.txt, tpMatrix.txt, userlist.txt, wslist.txt) to the 
    specified directory or the current directory if no directory is specified.

    Parameters:
    dir (str): The directory where the dataset will be stored. Default is the current directory.
    url (str): The URL of the downloadable zip file. The URL can be modified to download the 
    WS-DREAM first dataset in case of a broken link or changing in the repository.
    The default is 'https://zenodo.org/record/1133476/files/wsdream_dataset1.zip?download=1'.

    Returns:
    None

    Example:
    >>> dataset_downloader('my_data_folder')
    """
    # TODO make this return the file name
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
    Reads a .txt file containing a list of users or services and returns a pandas DataFrame object.

    Parameters:
        file (str): The name of the .txt file to be read.

    Returns:
        pd.DataFrame: A DataFrame containing the content of the .txt file.

    Example:
        >>> dataframe_fromtxt('userlist.txt')
            User ID 	IP Address  	Country 	    IP No.  	AS  	                                Latitude	Longitude
        0   0	        12.108.127.138	United States	208437130	AS7018 AT&T Services, Inc.	            38	        -97
        1   1	        12.46.129.15	United States	204374287	AS7018 AT&T Services, Inc.	            38.0464	    -122.23
        2   2	        122.1.115.91	Japan	        2046915419	AS4713 NTT Communications Corporation	35.685	    139.7514
            ..  ..
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


class WsdreamReader:
    """
    The Wsdream class is a singleton class that contains an instance of the WS-DREAM dataset, 
    with the different tables i.e. usersList, servicesList, responseTimeMatrix, throughputMatrix. 
    This class is designed to simplify the interaction with the dataset.

    Attributes:

    usersList (pandas.DataFrame): contains information on 339 service users, 
    including User ID, IP Address, Country, Continent, AS, Latitude, Longitude, Region, City.
    servicesList (pandas.DataFrame): contains information on the 5,825 Web services, 
    including Service ID, WSDL Address, Service Provider, IP Address, Country, Continent, AS, Latitude, Longitude, Region, City.
    responseTimeMatrix (numpy.ndarray): 339 x 5825 user-item matrix of response-time.
    throughputMatrix (numpy.ndarray): 339 x 5825 user-item matrix for throughput.
    Methods:

    save_lists_tocsv(): saves the usersList and servicesList DataFrames into a CSV file when needed.
    """
    # TODO create a final static url var interactions_full_df
    __USERS_LIST_FILE_NAME="userlist.txt"
    __SERVICES_LIST_FILE_NAME="wslist.txt"
    __RESPONSE_TIME_MATRIX_FILE_NAME = "rtMatrix.txt"
    __THROUGHPUT_MATRIX_FILE_NAME = "tpMatrix.txt"
    __READER = Reader()
    __dir = None
    __instance = None

    # TODO this verification is slow and computationaly expensive check it or add a way to skip later
    def __new__(cls, dir=None, url="https://zenodo.org/record/1133476/files/wsdream_dataset1.zip?download=1"):
        # TODO lose the ifs for verifying the exictance of the files on the system
        if (cls.__instance is None):
            print('Creating the dataset object ...')
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
            cls.df_responseTime = cls._df_from_matrix(cls, cls.responseTimeMatrix)
            cls.df_throughput = cls._df_from_matrix(cls, cls.throughputMatrix)
            # Creating the class
            cls.__instance = super(WsdreamReader, cls).__new__(cls)
            print("\t\t** DONE ** \n The dataset is accessible")
        return cls.__instance 

    # TODO add normalisation attribute
    def _df_from_matrix(self, matrix):
        # Converting matrix to list
        list_dataset = self._list_from_matrix(self, matrix)
        # Converting list to Pandas DataFrame
        pd_list = pd.DataFrame(list_dataset,columns=['UsersID', 'ServicesID', 'Rating'])
        return pd_list

    def _list_from_matrix(self, matrix):
        data_list = [[i,j,matrix[i][j]] for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if matrix[i][j] != -1]
        return data_list

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

class WsdreamDataset(DatasetFactory):
    def __init__(self, wsdream, normalization_strategy) -> None:
        self.wsdream = wsdream
        self.normalization_strategy = normalization_strategy

    def get_responseTime(self, density=100, random_state=6):
        """
        Returns ResponseTime in Surprise.Dataset object with whatever density percentage you want from 0 to 100.
        Parameters:
            density - int, optional for the density of the data to work with, by default it's 100.
            random_state - int, optional for the data randomization, used for randomizing lower density data and obtaining consisting results.
        """
        # TODO raise an exception
        frac = density/100
        copy = self.wsdream.df_responseTime.sample(frac=frac, random_state=random_state, ignore_index=True)
        # Converting Dataframe to surprise Dataset object
        min = int(self.wsdream.df_responseTime.Rating.min()) - 1
        max = int(self.wsdream.df_responseTime.Rating.max()) + 1
        reader = Reader(rating_scale=(min, max))
        data = Dataset.load_from_df(copy, reader)
        # #Data Normalization get_surprise_dataset
        # pd_list = normalization(pd_list)
        return data
        
    def get_throughput(self, density=100, random_state=6):
        """
        Returns Throughput in Surprise.Dataset object with whatever density percentage from 0 to 100.
        Parameters:
            density - int, optional for the density of the data to work with, by default it's 100.
            random_state - int, optional for the data randomization, used for randomizing lower density data and obtaining consisting results.
        """
        # TODO raise an exception
        frac = density/100
        copy = self.wsdream.df_throughput.sample(frac=frac, random_state=random_state, ignore_index=True)
        # Converting Dataframe to surprise Dataset object
        min = int(self.wsdream.df_throughput.Rating.min()) - 1
        max = int(self.wsdream.df_throughput.Rating.max()) + 1
        reader = Reader(rating_scale=(min, max))
        data = Dataset.load_from_df(copy, reader)
        # #Data Normalization get_surprise_dataset
        # pd_list = normalization(pd_list)
        return data
    
    def get_users(self) -> pd.DataFrame:
        return self.wsdream.usersList
    
    def get_services(self) -> pd.DataFrame:
        return self.wsdream.servicesList
    
