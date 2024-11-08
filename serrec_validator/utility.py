from surprise.model_selection import train_test_split,LeaveOneOut
from abc import ABC, abstractmethod, abstractstaticmethod
import pandas as pd
import numpy as np
from surprise import Dataset, Trainset
from geopy.distance import geodesic
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import cdist
import os
from typing import Tuple, Dict


def __services_data_preprocessing(services_df):
    df = __imputing_location_data(services_df)
    # Step 2: One-hot encode the 'Service Provider' and 'WSDL Address'
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(services_df[['Service Provider', 'WSDL Address']])

    return df, encoded_features

def __imputing_location_data(services_df: pd.DataFrame) -> pd.DataFrame:
    df = services_df.copy()
    
    # Pre-compute mean latitude and longitude per country
    country_means = df.groupby('Country')[['Latitude', 'Longitude']].transform('mean')
    
    # Fill missing values within each country
    df['Latitude'].fillna(country_means['Latitude'], inplace=True)
    df['Longitude'].fillna(country_means['Longitude'], inplace=True)
    
    # Global mean for remaining missing values
    global_lat_mean = df['Latitude'].mean()
    global_long_mean = df['Longitude'].mean()
    df['Latitude'].fillna(global_lat_mean, inplace=True)
    df['Longitude'].fillna(global_long_mean, inplace=True)
    
    return df



def load_similarity_matrices(services_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads or computes similarity matrices based on geographical and encoded feature similarities. 
    Saves matrices to disk if not previously saved.

    Args:
        services_df (pd.DataFrame): DataFrame with service data for computing similarity matrices.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Geographical similarity matrix.
            - np.ndarray: Encoded feature similarity matrix.
    """
    if os.path.exists('geo_similarity_matrix.pkl') and os.path.exists('encoded_similarity_matrix.pkl'):
        with open('geo_similarity_matrix.pkl', 'rb') as f:
            geo_similarity_matrix = pickle.load(f)
        with open('encoded_similarity_matrix.pkl', 'rb') as f:
            encoded_similarity_matrix = pickle.load(f)
        print("Loaded precomputed similarity matrices.")
        return geo_similarity_matrix, encoded_similarity_matrix

    print("Similarity matrices not found .. Computing similarity matrices... \nThis may take a while")
    location_df, encoded_features = __services_data_preprocessing(services_df)
    
    # Step 3: Calculate Geographical Similarity
    coords = location_df[['Latitude', 'Longitude']].values
    distances = cdist(coords, coords, metric=lambda u, v: geodesic(u, v).kilometers)
    geo_similarity_matrix = 1 / (1 + distances)

    # Step 4: Calculate Similarity Using Encoded Features
    encoded_similarity_matrix = cosine_similarity(encoded_features.toarray())

    # Save matrices
    with open('geo_similarity_matrix.pkl', 'wb') as f:
        pickle.dump(geo_similarity_matrix, f)
    with open('encoded_similarity_matrix.pkl', 'wb') as f:
        pickle.dump(encoded_similarity_matrix, f)

    return geo_similarity_matrix, encoded_similarity_matrix




class NormalizationStrategy(ABC):
    """
    Abstract base class providing a template for normalization strategies.
    
    Methods:
        normalize(data: np.ndarray) -> np.ndarray:
            Abstract method for normalizing data.
            
        revert_normalization(data_df: pd.DataFrame) -> pd.DataFrame:
            Abstract method for reverting the normalization process.
    """
    @abstractstaticmethod
    def normlize(data: np.ndarray) -> np.ndarray:
        pass

    @abstractstaticmethod
    def revert_normalization(data_df: pd.DataFrame) -> pd.DataFrame:
        pass

class DatasetFactory(ABC):
    """
    Abstract base class representing a dataset factory for service datasets.

    Methods:
        get_responseTime() -> Dataset:
            Returns the response time dataset in Surprise Dataset format.
            
        get_throughput() -> Dataset:
            Returns the throughput dataset in Surprise Dataset format.
            
        get_users() -> pd.DataFrame:
            Returns the users DataFrame.
            
        get_services() -> pd.DataFrame:
            Returns the services DataFrame.
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

def getPopularityRanks(data_df: pd.DataFrame, max_value: float) -> Dict[int, float]:
    """
    Calculates service popularity ranks based on summed ratings, normalized by the maximum rating value.

    Args:
        data_df (pd.DataFrame): DataFrame with columns 'Service ID' and 'Rating'.
        max_value (float): Maximum rating value used for normalization.

    Returns:
        dict: A dictionary with 'Service ID' as keys and normalized popularity ranks as values.
    """
    service_popularity_series = data_df.groupby('Service ID')['Rating'].sum().div(max_value)
    return service_popularity_series.to_dict()


class Sets:
    """
    Class for creating various data splits for service recommendation evaluation.

    Attributes:
        full_trainSet (Trainset): Full training dataset.
        anti_testset_from_full_data (list): Negative samples from the full training dataset.
        accuracy_splits (tuple): Tuple of Trainset and list for accuracy evaluation.
        hit_splits (tuple): Tuple of Trainset and list for hit rate evaluation.
        anti_testSet_for_hits (list): Negative samples for evaluating hit rate.
        
    Parameters:
        dataset (DatasetFactory): DatasetFactory object for the data to be split.
        random_state (int): Seed for random operations to ensure reproducibility.
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
    Splits datasets into training and testing sets for evaluation of machine learning models.

    Parameters:
        dataset (DatasetFactory): Dataset object containing response time and throughput data.
        density (int, optional): Percentage of data to use for evaluation, from 1 to 100. Default is 100.
        random_state (int, optional): Seed for random operations to ensure reproducibility. Default is 6.

    Attributes:
        response_time (Sets): Train/test splits for response time data.
        throughput (Sets): Train/test splits for throughput data.
    """
    # TODO add polymorphism on __init__() to work with data
    def __init__(self, dataset, density=100, random_state=6) -> None:
        response_time = dataset.get_responseTime(density, random_state)
        throughput = dataset.get_throughput(density, random_state)

        self.response_time = Sets(response_time, random_state)
        self.throughput = Sets(throughput, random_state)



