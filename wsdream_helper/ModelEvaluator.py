from . import EvaluationMetrics
from .utility import DataSplitter
from surprise import AlgoBase
from functools import singledispatch

#TODO implement the verbose functionality on all the methods
class ModelEvaluator:
    #TODO move the metrics list from the constructor to the evaluate method
    def __init__(self, metrics=['RMSE','MAE', 'HR', 'ARHR', 'CHR', 'Coverage', 'Diversity', 'Novelty']):
        self.metrics = metrics

    # TODO implement the verbose action
    def evaluate(self, algo, splits, verbose=False):
        evaluation_dict = dict()
        trainSet, testSet = splits.accuracy_splits
        algo.fit(trainSet)
        predictions = algo.test(testSet)
        for metric in self.metrics:
            if metric.lower() == 'rmse':
                evaluation_dict[metric] = EvaluationMetrics.RMSE(predictions,verbose=False)
            elif metric.lower() ==  'mae':
                evaluation_dict[metric] = EvaluationMetrics.MAE(predictions,verbose=False)
            elif metric.lower() ==  'mse':
                evaluation_dict[metric] = EvaluationMetrics.MSE(predictions,verbose=False)
        # Hit rate splitting and computation take some time it would be helpful to skip it if not necessary
        if ('hr' in x for x in self.metrics):
            dic = self.hit_rate_evaluation(algo, splits)
            evaluation_dict.update(dic)

        return evaluation_dict

    # TODO implement the verbose action
    def hit_rate_evaluation(self, algo, splits, verbose=False):
        evaluation_dict = dict()
        trainSet, testSet = splits.hit_splits
        bigTestSet = splits.anti_testSet_for_hits
        algo.fit(trainSet)
        predictions = algo.test(testSet)
        allPredictions = algo.test(bigTestSet)

        topNPredicted = EvaluationMetrics.get_top_n(allPredictions, n=10)
        for metric in self.metrics:
            if metric.lower() == 'hr':
                evaluation_dict[metric] = EvaluationMetrics.hit_rate(topNPredicted,predictions,verbose=False)
            elif metric.lower() ==  'arhr':
                evaluation_dict[metric] = EvaluationMetrics.average_reciprocal_hit_rank(topNPredicted,predictions,verbose=False)
            elif metric.lower() ==  'chr':
                evaluation_dict[metric] = EvaluationMetrics.cumulative_hit_rate(topNPredicted,predictions,verbose=False)

        return evaluation_dict
    
    def evaluation_automator(self, algos, dataset, random_state=6, densities:list=[10,20,30], metrics:list[str]=['RMSE','MAE', 'HR', 'ARHR', 'CHR'], verbose:bool=True):
        results = dict()
        # creating the different data splits
        splits = [DataSplitter(dataset, item, random_state) for item in densities]
        #evaluate the models
        for data, density in zip(splits, densities):
            results[f"response time {density}%"] = self.compare(algos,data.response_time, metrics, verbose=False)
            results[f"throughput {density}%"] = self.compare(algos,data.throughput, metrics, verbose=False)
        return results
    

    @singledispatch
    def compare(self, algos, data: DataSplitter, metrics:list[str]=['RMSE','MAE', 'HR', 'ARHR', 'CHR'], verbose:bool=True) -> dict:
        #TODO write and detialed error message
        raise NotImplementedError("ERROR")

    @compare.register(list)
    def compare(self, algos, data: DataSplitter, metrics:list[str]=['RMSE','MAE', 'HR', 'ARHR', 'CHR'], verbose:bool=True) -> dict:
        results = dict()
        #evaluate the models
        for model in algos:
            last_index = 0
            model_name = self.__get_model_name(model)
            if model_name in results.keys():
                for key in results.keys():
                    #Check if the model name already available if so add a number to the model name
                    if key.find(model_name) > -1 :
                        if key.find('_') >-1:
                            if (last_index < int(key.split("_")[1])):
                                last_index = int(key.split("_")[1])
                model_name += "_" + str(last_index+1)
                
            results[model_name] = self.evaluate(algo=model, splits=data)
        return results

    #TODO implement the next method
    # @compare.register(AlgoBase)
    # def compare(self, algos, data: DataSplitter, metrics:list[str]=['RMSE','MAE', 'HR', 'ARHR', 'CHR'], verbose:bool=True) -> dict:
    #     # classicAlgos = [UIPCC, PMF, NMF, PMF, NTF, algos]
    #     # return self.compare(classicAlgos, data, metrics, verbose)
    #     pass
    
    #This private method take a model as a parameter and returns its name
    def __get_model_name(self, algo):
        name = str(algo).split('object')[0]
        name = name.split('.')[-1]
        return name
        

    #TODO implement the display_results method
    def display_results(results:dict) -> None:
        pass