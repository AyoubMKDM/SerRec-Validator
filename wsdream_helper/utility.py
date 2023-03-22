from surprise.model_selection import train_test_split,LeaveOneOut
from abc import ABC, abstractmethod, abstractstaticmethod
import pandas as pd
from surprise import Dataset

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

class DataSplitter:
    """
    This class will alow us to have a consistance in the evaluation by having the same sets for training and testing models.
    """
    # TODO add polymorphism on __init__() to work with data
    def __init__(self, dataset, density=100, random_state=6) -> None:
        response_time = dataset.get_responseTime(density, random_state)
        throughput = dataset.get_throughput(density, random_state)

        self.trainset_from_full_data = {"response_time" : response_time.build_full_trainset(),
                                         "through_put" : throughput.build_full_trainset()}
        self.anti_testset_from_full_data = {"response_time" : self.trainset_from_full_data['response_time'].build_anti_testset(),
                                         "through_put" : self.trainset_from_full_data['through_put'].build_anti_testset()}

        response_time_trainset, response_time_testset = train_test_split(response_time)
        throughput_trainset, throughput_testset = train_test_split(throughput)
        
        self.splitset_for_accuracy = {"response_time" : (response_time_trainset, response_time_testset),
                                        "through_put" : (throughput_trainset, throughput_testset)}

        LOOCV = LeaveOneOut(n_splits=1,random_state=random_state)
        for response_time_trainset, response_time_testset in LOOCV.split(response_time):
            self.splitset_for_hit_rate = {"response_time" : (response_time_trainset, response_time_testset)}
            
        for throughput_trainset, throughput_testset in LOOCV.split(throughput):
            self.splitset_for_hit_rate['through_put'] = (throughput_trainset, throughput_testset)
        
        self.anti_testset_for_hit_rate = {"response_time" : response_time_trainset.build_anti_testset(),
                            "through_put" : throughput_trainset.build_anti_testset()}

        


        
    # def get_trainsets_for_accuracy():
    #     pass

    # def get_testsets_for_accuracy():
    #     pass

    # def get_trainsets_for_hit_rate():
    #     pass

    # def get_testsets_for_hit_rate():
    #     pass

    # def get_trainsets_from_full_data():
    #     pass

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

from surprise import Trainset
from dataclasses import dataclass

@dataclass
class Sets:
    full_trainSet: Trainset
    trainSet_for_accuracy: Trainset
    testSet_for_accuracy: list
    trainSet_for_hits: Trainset
    testSet_for_hits: list
    anti_testSet_for_hits: list
