import os
import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from .utility import DatasetFactory,NormalizationStrategy
from .Normalization import Reverse
from importlib.resources import files
import errno
import pkg_resources
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WsdreamLoader:
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
    FILES = {
        'USERS_LIST': "userlist.txt",
        'SERVICES_LIST': "wslist.txt",
        'RESPONSE_TIME_MATRIX': "rtMatrix.txt",
        'THROUGHPUT_MATRIX': "tpMatrix.txt"
    }
    __dir = None
    __instance = None

    def __new__(cls, dir : str =None, builtin: bool =False):
        if (cls.__instance is None):
            print('Creating the dataset object ...')
            cls.__dir = ""
            if dir is not None:
                cls.__dir = dir
            
            if not(builtin):
                print('Checking the availability of all files in the dataset ...')
                cls._files_checker()

            # Initialize the class atributes 
            print('Reading files from storage ...')
            cls._files_reader()
            
            # Creating the class
            cls.__instance = super(WsdreamLoader, cls).__new__(cls)
            print("\t\t** DONE ** \n The dataset is accessible")
        return cls.__instance 
    
    @classmethod
    def _files_reader(cls):
        # Use pkg_resources to access the dataset files from the package or resource folder
        cls._users_df = cls.dataframe_fromtxt(cls._get_resource_path(cls.FILES['USERS_LIST']))
        cls._services_df = cls.dataframe_fromtxt(cls._get_resource_path(cls.FILES['SERVICES_LIST']))
        cls.response_time_matrix = np.loadtxt(cls._get_resource_path(cls.FILES['RESPONSE_TIME_MATRIX']))
        cls.throughput_matrix = np.loadtxt(cls._get_resource_path(cls.FILES['THROUGHPUT_MATRIX']))
        # defining missing values
        cls._services_df['IP No.'].replace("0", value=pd.NA, inplace=True)
        cls.response_time_matrix[cls.response_time_matrix == -1.] = np.nan
        cls.throughput_matrix[cls.throughput_matrix == -1.] = np.nan

    @classmethod
    def _files_checker(cls):
        # Check if the data files exist in the package
        try:
            cls._get_resource_path(cls.FILES['USERS_LIST'])
            cls._get_resource_path(cls.FILES['SERVICES_LIST'])
            cls._get_resource_path(cls.FILES['RESPONSE_TIME_MATRIX'])
            cls._get_resource_path(cls.FILES['THROUGHPUT_MATRIX'])
        except FileNotFoundError as e:
            cls._raise_FileNotFoundError(e)

    @staticmethod
    def _get_resource_path(filename):
        """Fetch the resource path for files bundled in the package."""
        try:
            # pkg_resources.resource_filename gives the full path to the resource
            resource_path = pkg_resources.resource_filename(
                __name__, f'wsdream/{filename}'  # Adjust the subfolder as necessary
            )
            if not os.path.exists(resource_path):
                raise FileNotFoundError(f"Resource {filename} not found.")
            return resource_path
        except Exception as e:
            raise FileNotFoundError(f"Error accessing resource {filename}: {str(e)}")

    @classmethod
    def _raise_FileNotFoundError(cls, error):
        print(f"Error: {str(error)}")
        raise error

    def df_from_matrix(self, matrix):
        """
        Converts the given matrix into a DataFrame and skips rows with invalid values.
        
        Parameters:
            matrix (numpy.ndarray): The matrix to convert.
        
        Returns:
            pd.DataFrame: The resulting DataFrame with valid rows.
        """
        list_dataset = list_dataset = [
            [i, j, matrix[i][j]]
            for i in range(matrix.shape[0])
            for j in range(matrix.shape[1])
        ]
        # Converting list to Pandas DataFrame
        pd_list = pd.DataFrame(list_dataset,columns=['User ID', 'Service ID', 'Rating'])
        
        return pd_list

    def save_list_tocsv(self, listName: str):
        """
        Saves the specified DataFrame to a CSV file. The DataFrame is either `users_df` or `services_df`.
        
        Parameters:
            listName (str): The name of the DataFrame to save. Must be either 'users_df' or 'services_df'.
        
        Returns:
            None
        
        Example:
            >>> save_list_tocsv('users_df')
            "users_df" is saved to the file "usersList.csv".
        """
        if listName != 'users_df' and listName != 'services_df':
            print(f'No such attibute with the name "{listName}."')
        elif listName == 'users_df':
            self._users_df.to_csv(listName[:-3] + "List.csv",index=False)
        elif listName == 'services_df':
            self._services_df.to_csv(listName[:-3] + "List.csv",index=False)
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
    def __init__(self, wsdream: WsdreamLoader, normalization_strategy: NormalizationStrategy = None) -> None:
        self.wsdream = wsdream
        self.normalization_strategy = normalization_strategy
        # normalization_strategy = Reverse.normalize
        self._responseTime = self.wsdream.response_time_matrix
        self._throughput = self.wsdream.throughput_matrix
        self._responseTime = Reverse.normalize(self._responseTime) 
        if normalization_strategy is not None:
            self._responseTime = self.normalization_strategy.normalize(self.wsdream.throughput_matrix)
            self._throughput = self.normalization_strategy.normalize(self.wsdream.throughput_matrix)
        self._responseTime = wsdream.df_from_matrix(self._responseTime)
        self._throughput = wsdream.df_from_matrix(self._throughput)

    def get_responseTime(self, density=100, random_state=6):
        """
        Returns ResponseTime in Surprise.Dataset object with specified density.
        Parameters:
            density : int
                Density of the data to work with, default value is 100.
            random_state : int
                Used for randomizing lower density data and obtaining consistent results.
        Returns:
            surprise.dataset.DatasetAutoFolds
        """
        try:
            frac = density/100
            # Dropping the na raws TODO add it as an argument either drop em or option to replace them with
            rt = self._responseTime.dropna(subset=['Rating'])
            #definig the Reader object
            max_rating = int(rt['Rating'].max() + 1)
            min_rating = int(rt['Rating'].min() - 1)
            sample = rt.sample(frac=frac, random_state=random_state, ignore_index=True)
            reader =  Reader(rating_scale=(min_rating, max_rating))
            # Convert DataFrame to surprise Dataset object
            data = Dataset.load_from_df(sample, reader)
            return data
        except Exception as e:
            logger.error('Failed to get response time. Error: %s', e)
            raise
        
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
        users = self.wsdream._users_df.copy()
        # Casting dictionary
        convert_dict = {'User ID': int,
                        'IP No.': int,
                        'Latitude': float,
                        'Longitude': float
                        }
        users = users.astype(convert_dict)

        return users
    
    def get_services(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame containing the list of services in the Wsdream dataset.

        Returns:
            A pandas DataFrame with columns 'Service ID', 'WSDL Address', 'Service Provider', 'IP Address', 
                'Country', 'IP No.', 'AS', 'Latitude', 'Longitude' containing information about the services in the Wsdream dataset1.
        """
        # Casting dictionary
        services = self.wsdream._services_df.copy()
        convert_dict = {'Service ID': int,
                        'IP No.': pd.Int64Dtype(),
                        'Latitude': float,
                        'Longitude': float
                        }
        services = services.astype(convert_dict)
        return services
    
def load_wsdream():
    """TODO add docstring
    """
    dir = pkg_resources.resource_filename(__name__, 'wsdream/')        
    return WsdreamDataset(WsdreamLoader(dir,builtin=True))
    

    
