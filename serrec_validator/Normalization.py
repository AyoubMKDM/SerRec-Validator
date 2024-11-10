from .utility import NormalizationStrategy
from pandas import DataFrame
import numpy as np

class Reverse(NormalizationStrategy):
    """TODO fix it// Reverse normalization strategy that subtracts each value from the maximum value.

    Methods:
        normalize(data: np.ndarray) -> np.ndarray:
            Apply reverse normalization to the data.

        revert_normalization(data_df: DataFrame) -> DataFrame:
            Reverts the reverse normalization applied to the data.
    """
    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        """Normalize data by subtracting from the maximum value."""
        max = data.max()
        data = max - data
        return data
    
    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        # TODO implement this method to revert the normalization on recommendation results
        pass

class scalingToRange(NormalizationStrategy):
    """Min-max normalization to scale data to the range [0, 1].

    Methods:
        normalize(data: np.ndarray) -> np.ndarray:
            Normalize the data to the range [0, 1].

        revert_normalization(data_df: DataFrame) -> DataFrame:
            Reverts the min-max normalization applied to the data.
    """
    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        """Apply min-max scaling to the data."""
        min = np.min(data, where=data>=0)
        # min = min.reshape(min.shape[0],1)
        max = np.max(data)
        # max = max.reshape(max.shape[0],1)
        data = (data - min) / (max - min)
        return data
    
    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        # TODO implement this method to revert the normalization on recommendation results
        pass
    
class zScore(NormalizationStrategy):
    """Z-score normalization to scale data based on mean and standard deviation.

    Methods:
        normalize(data: np.ndarray) -> np.ndarray:
            Normalize the data using Z-score normalization.

        revert_normalization(data_df: DataFrame) -> DataFrame:
            Reverts the Z-score normalization applied to the data.
    """
    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        """Normalize data to Z-scores."""
        data = (data - data.mean())/data.std()
        return data

    
    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        # TODO implement this method to revert the normalization on recommendation results
        pass

class modified_zScore(NormalizationStrategy):
    """Modified Z-score normalization that scales based on row-wise mean and standard deviation.

    Methods:
        normalize(data: np.ndarray) -> np.ndarray:
            Normalize the data using modified Z-score normalization.

        revert_normalization(data_df: DataFrame) -> DataFrame:
            Reverts the modified Z-score normalization applied to the data.
    """
    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        """Apply modified Z-score normalization (row-wise)."""
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

class LogScaling(NormalizationStrategy):
    """Logarithmic scaling normalization for compressing the range of values.

    Methods:
        normalize(data: np.ndarray) -> np.ndarray:
            Apply logarithmic scaling to the data.

        revert_normalization(data_df: DataFrame) -> DataFrame:
            Reverts the logarithmic scaling normalization applied to the data.
    """
    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        """Apply log scaling to the data."""
        # Applying log scaling (ensure no negative values before log)
        data[data <= 0] = 1e-6  # Avoid log(0) by replacing non-positive values with a small number
        return np.log(data)

    @staticmethod
    def revert_normalization(data_df: DataFrame) -> DataFrame:
        """Revert log scaling normalization."""
        # Revert log scaling by applying the exponential function
        return np.exp(data_df)  # Placeholder, adjust based on actual data format.