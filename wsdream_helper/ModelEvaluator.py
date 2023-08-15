from . import EvaluationMetrics
from .utility import DataSplitter, DatasetFactory
from surprise import AlgoBase
from functools import singledispatch
from tabulate import tabulate
import pandas as pd

# TODO implement the verbose functionality on all the methods
# TODO add docstrings 
def evaluate(algo: AlgoBase, splits: DataSplitter,
             metrics: list = ['RMSE', 'MAE', 'HR', 'ARHR', 'CHR', 'Coverage', 'Diversity', 'Novelty'],
             verbose: bool = True,
             k:int = 10):
    evaluation_dict = dict()
    trainSet, testSet = splits.accuracy_splits
    if verbose:
        print('Training the model ...')
    algo.fit(trainSet)
    predictions = algo.test(testSet)
    if verbose:
        print('Evaluating accuracy ...')
    for metric in metrics:
        if metric.lower() == 'rmse':
            evaluation_dict[metric] = EvaluationMetrics.RMSE(predictions, verbose=False)
        elif metric.lower() == 'mae':
            evaluation_dict[metric] = EvaluationMetrics.MAE(predictions, verbose=False)
        elif metric.lower() == 'mse':
            evaluation_dict[metric] = EvaluationMetrics.MSE(predictions, verbose=False)
    # Hit rate splitting and computation take some time it would be helpful to
    # skip it if not necessary
    if verbose:
        print('Evaluating Hits ...')
    if ('hr' in x for x in metrics):
        dic = _hit_rate_evaluation(algo, splits, k=k)
        evaluation_dict.update(dic)
    if verbose:
        print('Results:')
        for key in evaluation_dict.keys():
            print(f'{key} \t {evaluation_dict[key]:.4f}')
    return evaluation_dict

def _hit_rate_evaluation(algo: AlgoBase, splits: DataSplitter, k: int,
                         metrics: list = ['RMSE', 'MAE', 'HR', 'ARHR', 'CHR', 'Coverage', 'Diversity', 'Novelty']):
    evaluation_dict = dict()
    trainSet, testSet = splits.hit_splits
    bigTestSet = splits.anti_testSet_for_hits
    algo.fit(trainSet)
    predictions = algo.test(testSet)
    allPredictions = algo.test(bigTestSet)
    topNPredicted = EvaluationMetrics.get_top_n(allPredictions, n=k)
    for metric in metrics:
        if metric.lower() == 'hr':
            evaluation_dict[metric] = EvaluationMetrics.hit_rate(topNPredicted, predictions, verbose=False)
        elif metric.lower() == 'arhr':
            evaluation_dict[metric] = EvaluationMetrics.average_reciprocal_hit_rank(topNPredicted, predictions,
                                                                                     verbose=False)
        elif metric.lower() == 'chr':
            evaluation_dict[metric] = EvaluationMetrics.cumulative_hit_rate(topNPredicted, predictions, verbose=False)
    return evaluation_dict

 # TODO if densities is empty
def benchmark(algos: list, dataset: DatasetFactory, k : int = 10,
                         ignore_response_time: bool = False, ignore_throuput: bool = False, random_state: int = 6,
                         densities: list = [10, 20, 30],
                         metrics: list[str] = ['RMSE', 'MAE', 'HR', 'ARHR', 'CHR'], verbose: bool = True):
    results = dict()
    # creating the different data splits
    splits = [DataSplitter(dataset, item, random_state) for item in densities]
    # evaluate the models
    for data, density in zip(splits, densities):
        if not ignore_response_time:
            if verbose:
                print(f'Training the different models on the response time data with the density {density}%')
            results[f"response time {density}%"] = compare(algos, data.response_time, k, metrics, verbose=verbose)
        if not ignore_throuput:
            if verbose:
                print(f'Training the different models on the throughput data with the density {density}%')
            results[f"throughput {density}%"] = compare(algos, data.throughput, k, metrics, verbose=verbose)
    return results

@singledispatch
def compare(algos: any, data: DataSplitter, k: int = 10, metrics: list[str] = ['RMSE', 'MAE', 'HR', 'ARHR', 'CHR'],
            verbose: bool = True) -> dict:
    # TODO write and detialed error message
    raise NotImplementedError("Unsupported type")

@compare.register(list)
def _(algos: list, data: DataSplitter, k: int = 10, metrics: list[str] = ['RMSE', 'MAE', 'HR', 'ARHR', 'CHR'],
      verbose: bool = True) -> dict:
    algos_dictionary = dict()
    if not isinstance(algos[0], AlgoBase):
        raise TypeError("Unsupported type: you should pass a dictionary with AlgoBase models")
    # Naming the models
    for model in algos:
        last_index = 0
        model_name = __get_model_name(model)
        if model_name in algos_dictionary.keys():
            for key in algos_dictionary.keys():
                # Check if the model name already available if so add a number to
                # the model name
                if key.find(model_name) > -1:
                    if key.find('_') > -1:
                        if (last_index < int(key.split("_")[1])):
                            last_index = int(key.split("_")[1])
            model_name += "_" + str(last_index + 1)
        algos_dictionary[model_name] = model
    results = compare(algos_dictionary, data=data, metrics=metrics, verbose=verbose)
    return results

@compare.register(dict)
def _(algos: dict, data: DataSplitter, k: int = 10, metrics: list[str] = ['RMSE', 'MAE', 'HR', 'ARHR', 'CHR'],
      verbose: bool = True) -> dict:
    results = dict()
    # evaluate the models
    for model_name in algos.keys():
        model = algos[model_name]
        if isinstance(model, AlgoBase):
            if verbose:
                print(f'Evaluating {model_name}')
            results[model_name] = evaluate(algo=model, k=k, splits=data, metrics=metrics, verbose=verbose)
        else:
            raise TypeError("Unsupported type: you should pass a dictionary with AlgoBase models")
    if verbose:
        display_results(results=results, metrics=metrics)
    return results

# This private method take a model as a parameter and returns its name
def __get_model_name(algo: AlgoBase):
    name = str(algo).split('object')[0]
    name = name.split('.')[-1]
    return name

def display_results(results: dict, metrics: list) -> None:
    content = {'model_name': []}
    for metric in metrics:
        content[metric] = []
    for model_name in results.keys():
        content['model_name'].append(model_name)
        for metric in metrics:
            content[metric].append(results[model_name][metric])
    df = pd.DataFrame(data=content)
    print('Results summary:')
    print(tabulate(df, headers='keys', tablefmt='fancy_grid'))