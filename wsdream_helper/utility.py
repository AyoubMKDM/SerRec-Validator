from surprise.model_selection import train_test_split,LeaveOneOut
from abc import ABC, abstractmethod, abstractstaticmethod
import pandas as pd
from surprise import Dataset, Trainset

class NormalizationStrategy(ABC):
    @abstractstaticmethod
    def normlize(data_df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractstaticmethod
    def revert_normalization(data_df: pd.DataFrame) -> pd.DataFrame:
        pass


def getPopularityRanks(data_df, max_value):
  service_popularity_df = data_df.groupby('ServicesID')['Rating'].sum().sort_values(ascending=False).reset_index()
  rankings = service_popularity_df.set_index('ServicesID') / max_value

  return rankings.to_dict()['Rating']

class Sets:
    full_trainSet: Trainset
    anti_testset_from_full_data: list
    accuracy_splits: tuple #(trainSet_for_accuracy: Trainset, testSet_for_accuracy: list)
    hit_splits: tuple #(trainSet_for_hits: Trainset, testSet_for_hits: list)
    anti_testSet_for_hits: list 

    def __init__(self, dataset, random_state=6) -> None:
        self.full_trainSet = dataset.build_full_trainset()

        self.anti_testset_from_full_data = self.full_trainSet.build_anti_testset()
        
        self.accuracy_splits = train_test_split(dataset)

        LOOCV = LeaveOneOut(n_splits=1,random_state=random_state)
        for trainset, testset in LOOCV.split(dataset):
            self.hit_splits = (trainset, testset)

        self.anti_testSet_for_hits = self.hit_splits[0].build_anti_testset()        
    

class DataSplitter:
    """
    This class will alow us to have a consistance in the evaluation by having the same sets for training and testing models.
    """
    # TODO add polymorphism on __init__() to work with data
    def __init__(self, dataset, density=100, random_state=6) -> None:
        response_time = dataset.get_responseTime(density, random_state)
        throughput = dataset.get_throughput(density, random_state)

        self.response_time_splits = Sets(response_time, random_state)
        self.throughput_splits = Sets(throughput, random_state)


class DatasetFactory(ABC):

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