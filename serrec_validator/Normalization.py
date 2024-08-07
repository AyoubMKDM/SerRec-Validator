from .utility import NormalizationStrategy
from pandas import DataFrame
import numpy as np

class reverse(NormalizationStrategy):
    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        max = data.max()
        data = max - data
        return data
    
    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        # TODO implement this method to revert the normalization on recommendation results
        pass

class scalingToRange(NormalizationStrategy):
    """
    This class implements a normalization strategy for a Pandas DataFrame with a 'Rating' column. The method 
    scales each rating value in the 'Rating' column to the range [0,1], also known as min-max normalization.

    Methods:
        normalize(data_df: DataFrame) -> DataFrame:
            Normalize the 'Rating' column of the input DataFrame using the scaling to range normalization.

        revert_normalization(data_df: DataFrame) -> None:
            This method reverts the normalization on recommendation results, to get the real rating values.

    """
    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        min = np.min(data, axis=1, where=data>=0, initial=2)
        min = min.reshape(min.shape[0],1)
        max = np.max(data, axis=1)
        max = max.reshape(max.shape[0],1)
        data = (data - min) / (max - min)
        return data
    
    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        # TODO implement this method to revert the normalization on recommendation results
        pass
    
class zScore(NormalizationStrategy):
    """
    This class implements a normalization strategy for a Pandas DataFrame with a 'Rating' column. The method 
    scales each rating value in the 'Rating' column to a z-score.

    Methods:
        normalize(data_df: DataFrame) -> DataFrame:
            Normalize the 'Rating' column of the input DataFrame using the z-score normalization.

        revert_normalization(data_df: DataFrame) -> None:
            This method reverts the normalization on recommendation results, to get the real rating values.

    """
    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        data = (data - data.mean())/data.std()
        return data

    
    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        # TODO implement this method to revert the normalization on recommendation results
        pass

class modified_zScore(NormalizationStrategy):
    """
    This class implements a normalization strategy for a Pandas DataFrame with a 'Rating' column. The method 
    scales each rating value in the 'Rating' column to a z-score.

    Methods:
        normalize(data_df: DataFrame) -> DataFrame:
            Normalize the 'Rating' column of the input DataFrame using the z-score normalization.

        revert_normalization(data_df: DataFrame) -> None:
            This method reverts the normalization on recommendation results, to get the real rating values.

    """
    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        mean = np.mean(data, axis=1, where=data>=0)
        mean = mean.reshape(mean.shape[0],1)
        std = np.std(data, axis=1, where=data>=0)
        std = std.reshape(std.shape[0],1)
        data = (data - mean)/std
        return data

    
    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        # TODO implement this method to revert the normalization on recommendation results
        pass

class clipping(NormalizationStrategy):
    # TODO implement this class
    @staticmethod
    def normalize(data_df: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        pass

class logScaling(NormalizationStrategy):
    # TODO implement this class
    @staticmethod
    def normalize(data_df: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        pass