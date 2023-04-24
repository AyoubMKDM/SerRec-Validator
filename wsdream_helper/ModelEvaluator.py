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
        trainSet, testSet = splits.splitset_for_accuracy['response_time']
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
        trainSet, testSet = splits.splitset_for_hit_rate['response_time']
        bigTestSet = splits.anti_testset_for_hit_rate['response_time']
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
    
    @singledispatch
    def compare(self, algos, data: DataSplitter, metrics:list[str], verbose:bool=True) -> dict:
        #TODO write and detialed error message
        raise NotImplementedError("ERROR")

    @compare.register(list[AlgoBase])
    def compare(self, algos, data: DataSplitter, metrics:list[str], verbose:bool=True) -> dict:
        results = dict()
        for model in algos:
            results[model] = self.evaluate(algo=model, splits=data)
        return results

    @compare.register(AlgoBase)
    def compare(self, algos, data: DataSplitter, metrics:list[str], verbose:bool=True) -> dict:
        classicAlgos = [UPICC, PMF, NMF, PMF, NTF, algos]
        return self.compare(algos, data, metrics, verbose)
        

    #TODO implement the display_results method
    def display_results(results:dict) -> None:
        pass