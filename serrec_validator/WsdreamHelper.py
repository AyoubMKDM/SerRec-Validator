import os
import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from .utility import DatasetFactory,NormalizationStrategy
from .Normalization import basic
from importlib.resources import files
import errno


class WsdreamReader:
    """
    Singleton class that reads the dataset with the different tables from the hard drive and makes 
    them available as Python data structures that are easy to manipulate.

    Attributes:
    usersList (pandas.DataFrame): contains information on 339 service users, 
        including User ID, IP Address, Country, Continent, AS, Latitude, Longitude, Region, City.
    servicesList (pandas.DataFrame): contains information on the 5,825 Web services, 
        including Service ID, WSDL Address, Service Provider, IP Address, Country, Continent, AS, Latitude, Longitude, Region, City.
    responseTimeMatrix (numpy.ndarray): 339 x 5825 user-item matrix of response-time.
    throughputMatrix (numpy.ndarray): 339 x 5825 user-item matrix for throughput.

    Methods:
    #TODO change this
    save_lists_tocsv(): saves the usersList and servicesList DataFrames into a CSV file when needed.

    Parameters:
    dir : str
        The directory where the dataset files are located. If not provided, the current working directory is used.
    """
    __USERS_LIST_FILE_NAME="userlist.txt"
    __SERVICES_LIST_FILE_NAME="wslist.txt"
    __RESPONSE_TIME_MATRIX_FILE_NAME = "rtMatrix.txt"
    __THROUGHPUT_MATRIX_FILE_NAME = "tpMatrix.txt"
    __dir = None
    __instance = None

    def __new__(cls, dir : str =None):
        if (cls.__instance is None):
            print('Creating the dataset object ...')
            cls.__dir = ""
            if dir is not None:
                cls.__dir = dir
            else:
                #TODO change this when changing the package name
                # dir = files('wsdream_helper.wsdream')
                pass

            print('Checking the availability of all files in the dataset ...')
            cls._files_checker()

            # Initialize the class atributes 
            print('Reading files from disk ...')
            cls._files_reader()
            cls.df_responseTime = cls._df_from_matrix(cls, cls.response_time_matrix)
            cls.df_throughput = cls._df_from_matrix(cls, cls.throughput_matrix)
            # Creating the class
            cls.__instance = super(WsdreamReader, cls).__new__(cls)
            print("\t\t** DONE ** \n The dataset is accessible")
        return cls.__instance 

    @classmethod
    def _files_reader(cls):
        cls.users_df = cls.dataframe_fromtxt(path=os.path.join(cls.__dir, cls.__USERS_LIST_FILE_NAME))
        cls.services_df = cls.dataframe_fromtxt(path=os.path.join(cls.__dir, cls.__SERVICES_LIST_FILE_NAME))
        cls.response_time_matrix = np.loadtxt(os.path.join(cls.__dir, cls.__RESPONSE_TIME_MATRIX_FILE_NAME))
        cls.throughput_matrix = np.loadtxt(os.path.join(cls.__dir, cls.__THROUGHPUT_MATRIX_FILE_NAME))  
        cls.services_df['IP No.'].replace("0",value=None,inplace=True) 
    
    @classmethod 
    def _files_checker(cls):
        full_path = os.path.join(os.getcwd(),cls.__dir)
        # Check if the data in the specified directory exists
        if not os.path.exists(full_path):
            cls._raise_FileNotFoundError(full_path)
        else:
            ls = os.listdir(full_path)
            # If the folder exist check if all the files exist
            if (cls.__USERS_LIST_FILE_NAME not in ls):
                cls._raise_FileNotFoundError(os.path.join(full_path, cls.__USERS_LIST_FILE_NAME))
            elif (cls.__SERVICES_LIST_FILE_NAME not in ls):
                cls._raise_FileNotFoundError(os.path.join(full_path, cls.__SERVICES_LIST_FILE_NAME))
            elif (cls.__RESPONSE_TIME_MATRIX_FILE_NAME not in ls):
                cls._raise_FileNotFoundError(os.path.join(full_path, cls.__RESPONSE_TIME_MATRIX_FILE_NAME))
            elif (cls.__THROUGHPUT_MATRIX_FILE_NAME not in ls):
                cls._raise_FileNotFoundError(os.path.join(full_path, cls.__THROUGHPUT_MATRIX_FILE_NAME))

    @classmethod
    def _raise_FileNotFoundError(cls, dir:str):
        print('')
        raise FileNotFoundError(errno.ENOENT, 
                                f"File not found \
                                \nYou need to download the dataset first: \n\t>>> path = WsdreamDataset1Downloader.download(\'<The folder where to save the dataset>\')\
                                \n{os.strerror(errno.ENOENT)}", dir)
            

    def _df_from_matrix(self, matrix):
        # Converting matrix to list
        list_dataset = self._list_from_matrix(self, matrix)
        # Converting list to Pandas DataFrame
        pd_list = pd.DataFrame(list_dataset,columns=['UsersID', 'ServicesID', 'Rating'])
        return pd_list

    def _list_from_matrix(self, matrix):
        data_list = [[i,j,matrix[i][j]] for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if matrix[i][j] != -1]
        return data_list

    def save_list_tocsv(self, listName: str):
        """
        Create a CSV file from users_df or services_df DataFrames. and store it in the current directory under the name usersList or servicesList respectively.
        Parameters:
            listName (str): The name of the list to be saved, acceptable values 'users_df' or 'services_df'
        Return:
            None
        """
        if listName != 'users_df' and listName != 'services_df':
            print(f'No such attibute with the name "{listName}."')
        elif listName == 'users_df':
            self.users_df.to_csv(listName[:-3] + "List.csv",index=False)
        elif listName == 'services_df':
            self.services_df.to_csv(listName[:-3] + "List.csv",index=False)
        print(f'"{listName}" is saved to the file "{listName[:-3]}List.csv".')
        

    @staticmethod
    def dataframe_fromtxt(path : str) -> pd.DataFrame:
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
        with open(path, 'r', encoding='utf-8', errors='replace') as data_file:
            indices = data_file.readline().strip().split('\t')
            indices = [index.strip('[]') for index in indices] 
            # Creating the DataFrame of users/services with the title line first
            df = pd.DataFrame(columns=indices)
            data_file.readline()
            for line in data_file:
                df.loc[len(df)] = line.strip().split('\t')
            df.replace(to_replace="null",value=None,inplace=True)
            
        return df

class WsdreamDataset(DatasetFactory):
    def __init__(self, wsdream: WsdreamReader, normalization_strategy: NormalizationStrategy = basic) -> None:
        self.wsdream = wsdream
        self.normalization_strategy = normalization_strategy
        self._responseTime =  self.normalization_strategy.normalize(self.wsdream.df_responseTime)
        self._throughput = self.normalization_strategy.normalize(self.wsdream.df_throughput)

    def get_responseTime(self, density=100, random_state=6):
        """
        Returns ResponseTime in Surprise.Dataset object with whatever density percentage you want from 0 to 100.
        Parameters:
            density - int, optional for the density of the data to work with, by default it's 100.
            random_state - int, optional for the data randomization, used for randomizing lower density data and obtaining consisting results.
        """
        if density <= 0 or density > 100:
            raise ValueError("Density must be a percentage value between 1 and 100.")
        frac = density/100
        copy = pd.DataFrame.sample(self._responseTime, frac=frac, random_state=random_state, ignore_index=True)
        # Converting Dataframe to surprise Dataset object
        min = int(self._responseTime.Rating.min()) - 1
        max = int(self._responseTime.Rating.max()) + 1
        reader = Reader(rating_scale=(min, max))
        data = Dataset.load_from_df(copy, reader)
        return data
        
    def get_throughput(self, density=100, random_state=6):
        """
        Returns Throughput in Surprise.Dataset object with whatever density percentage from 0 to 100.
        Parameters:
            density - int, optional for the density of the data to work with, by default it's 100.
            random_state - int, optional for the data randomization, used for randomizing lower density data and obtaining consisting results.
        """
        if density <= 0 or density > 100:
            raise ValueError("Density must be a percentage value between 1 and 100.")
        frac = density/100
        copy = pd.DataFrame.sample(self._throughput, frac=frac, random_state=random_state, ignore_index=True)
        # Converting Dataframe to surprise Dataset object
        min = int(self._throughput.Rating.min()) - 1
        max = int(self._throughput.Rating.max()) + 1
        reader = Reader(rating_scale=(min, max))
        data = Dataset.load_from_df(copy, reader)
        return data
    
    def get_users(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame containing the list of users in the Wsdream dataset.

        Returns:
            A pandas DataFrame with columns 'User ID', 'IP Address, 'Country', 'IP No.', 'AS',
                'Latitude', and 'Longitude' containing information about the users in the Wsdream dataset1.
        """
        return self.wsdream.users_df
    
    def get_services(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame containing the list of services in the Wsdream dataset.

        Returns:
            A pandas DataFrame with columns 'Service ID', 'WSDL Address', 'Service Provider', 'IP Address', 
                'Country', 'IP No.', 'AS', 'Latitude', 'Longitude' containing information about the services in the Wsdream dataset1.
        """

        return self.wsdream.services_df
    
