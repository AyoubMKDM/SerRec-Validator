from surprise.model_selection import train_test_split,LeaveOneOut
from abc import ABC, abstractmethod, abstractstaticmethod
import pandas as pd
import numpy as np
from surprise import Dataset, Trainset

class NormalizationStrategy(ABC):
    """
    This is an abstract class that provides a template for the normalization classes.

    Methods:
    normalize(data_df: pd.DataFrame) -> pd.DataFrame:
        This method should be overridden by concrete normalization classes to normalize the input data.
        
    revert_normalization(data_df: pd.DataFrame) -> pd.DataFrame:
        This method should be overridden by concrete normalization classes to revert normalization of the input data.
    """
    @abstractstaticmethod
    def normlize(data: np.ndarray) -> np.ndarray:
        pass

    @abstractstaticmethod
    def revert_normalization(data_df: pd.DataFrame) -> pd.DataFrame:
        pass

class DatasetFactory(ABC):
    """
    This is an abstract class representing a dataset factory for service datasets.
    Subclasses must implement the following methods:

    - get_responseTime() -> Surprise.Dataset: Returns the response time dataset in Surprise Dataset format.
    - get_throughput() -> Surprise.Dataset: Returns the throughput dataset in Surprise Dataset format.
    - get_users() -> pd.DataFrame: Returns the users DataFrame.
    - get_services() -> pd.DataFrame: Returns the services DataFrame.
    """
    @abstractmethod
    def get_responseTime(self) -> Dataset:
        pass

    @abstractmethod
    def get_throughput(self) -> Dataset:
        pass

    @abstractmethod
    def get_users(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_services(self) -> pd.DataFrame:
        pass

def getPopularityRanks(data_df, max_value):
  service_popularity_df = data_df.groupby('Service ID')['Rating'].sum().sort_values(ascending=False).reset_index()
  rankings = service_popularity_df.set_index('Service ID') / max_value

  return rankings.to_dict()['Rating']

class Sets:
    """
    The Sets class is used for creating various sets of data for service recommendation evaluation. It has the following attributes:

    Attributes:
    full_trainSet: 
        A Trainset object representing the full training dataset.
    anti_testset_from_full_data: 
        A list of tuple objects representing the negative samples (i.e., services that the user did not interact with) from the full training dataset.
    accuracy_splits: 
        A tuple of Trainset and a list representing the training and testing datasets for evaluating the accuracy of the recommendation system.
    hit_splits: 
        A tuple of Trainset and a list representing the training and testing datasets for evaluating the hit rate, cumulative hit rate, 
        rating hit rate, average reciprocal hit rank of the recommendation system.
    anti_testSet_for_hits: 
        A list of tuple objects representing the negative samples for evaluating the hit rate of the recommendation system.
        
    Parameters:
    dataset:
        A DatasetFactory object for the data to be splitted
    random_state:
        An integer is used to set the random seed for splitting the dataset.
    """
    full_trainSet: Trainset
    anti_testset_from_full_data: list
    accuracy_splits: tuple #(trainSet_for_accuracy: Trainset, testSet_for_accuracy: list)
    hit_splits: tuple #(trainSet_for_hits: Trainset, testSet_for_hits: list)
    anti_testSet_for_hits: list 

    def __init__(self, dataset : DatasetFactory, random_state=6) -> None:
        self.full_trainSet = dataset.build_full_trainset()

        self.anti_testset_from_full_data = self.full_trainSet.build_anti_testset()
        
        self.accuracy_splits = train_test_split(dataset)

        LOOCV = LeaveOneOut(n_splits=1,random_state=random_state)
        for trainset, testset in LOOCV.split(dataset):
            self.hit_splits = (trainset, testset)

        self.anti_testSet_for_hits = self.hit_splits[0].build_anti_testset()        
    

class DataSplitter:
    """
    A class for splitting datasets into train/test sets for evaluation of machine learning models.

    Parameters:
    dataset : WsdreamDataset
        A dataset object containing response time and throughput data.
    density : int, optional
        The percentage of the data to use for evaluation, from 1 to 100. Default is 100.
    random_state : int, optional
        Seed for the random number generator for splitting the data into train/test sets. Default is 6.

    Attributes:
    response_time : Sets
        An object containing the train/test splits for response time data.
    throughput : Sets
        An object containing the train/test splits for throughput data.
    """    
    # TODO add polymorphism on __init__() to work with data
    def __init__(self, dataset, density=100, random_state=6) -> None:
        response_time = dataset.get_responseTime(density, random_state)
        throughput = dataset.get_throughput(density, random_state)

        self.response_time = Sets(response_time, random_state)
        self.throughput = Sets(throughput, random_state)



