from .utility import NormalizationStrategy
from pandas import DataFrame

class NormalizationBasic(NormalizationStrategy):
    @staticmethod
    def normalize(data_df: DataFrame) -> DataFrame:
        max = data_df['Rating'].max()
        data_df['Rating'] = max - data_df['Rating']
        return data_df
    
    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        # TODO implement this method to revert the normalization on recommendation results
        pass

class NormalizationScalingToRange(NormalizationStrategy):
    @staticmethod
    def normalize(data_df: DataFrame) -> DataFrame:
        data_df = NormalizationBasic.normalize(data_df)
        min = data_df['Rating'].min()
        max = data_df['Rating'].max()
        data_df['Rating'] = (data_df['Rating'] - min) / (max - min)
        return data_df
    
    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        # TODO implement this method to revert the normalization on recommendation results
        pass
    
class NormalizationZScore(NormalizationStrategy):
    @staticmethod
    def normalize(data_df: DataFrame) -> DataFrame:
        mean = data_df['Rating'].mean()
        std = data_df['Rating'].std()
        data_df['Rating'] = (data_df['Rating'] - mean)/std
        return NormalizationBasic.normalize(data_df)

    
    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        # TODO implement this method to revert the normalization on recommendation results
        pass

class NormalizationClipping(NormalizationStrategy):
    # TODO implement this class
    @staticmethod
    def normalize(data_df: DataFrame) -> DataFrame:
        pass

    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        pass

class NormalizationLogScaling(NormalizationStrategy):
    # TODO implement this class
    @staticmethod
    def normalize(data_df: DataFrame) -> DataFrame:
        pass

    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        pass