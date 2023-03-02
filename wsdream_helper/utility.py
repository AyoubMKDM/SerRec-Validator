class Normalization:
    @staticmethod
    def scaling_to_range(data_df):
        min = data_df['Rating'].min()
        max = data_df['Rating'].max()
        data_df['Rating'] = 1 - (data_df['Rating'] - min) / (max - min)
        return data_df

    @staticmethod
    def z_score(data_df):
        mean = data_df['Rating'].mean()
        std = data_df['Rating'].std()
        # min = (data_df['Rating'].min() - mean)/std
        max = (data_df['Rating'].max() - mean)/std
        data_df['Rating'] = max - (data_df['Rating'] - mean)/std
        return data_df

    @staticmethod
    def clipping(data_df):
        pass

    @staticmethod
    def log_scaling(data_df):
        pass

class RevertNormalization:
    # TODO implement static methods for reverting the normalization on recommendation results
    @staticmethod
    def scaling_to_range(data_df):
        pass

    @staticmethod
    def z_score(data_df):
        pass

    @staticmethod
    def clipping(data_df):
        pass

    @staticmethod
    def log_scaling(data_df):
        pass


def getPopularityRanks(data_df, max_value):
  service_popularity_df = data_df.groupby('ServicesID')['Rating'].sum().sort_values(ascending=False).reset_index()
  rankings = service_popularity_df.set_index('ServicesID') / max_value

  return rankings.to_dict()['Rating']