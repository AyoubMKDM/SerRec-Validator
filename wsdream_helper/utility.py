from surprise.model_selection import train_test_split,LeaveOneOut

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

        LOOCV = LeaveOneOut(n_splits=1,random_state=randome_state)
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