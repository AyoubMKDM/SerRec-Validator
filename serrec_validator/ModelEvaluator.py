from . import EvaluationMetrics
from .utility import DataSplitter, DatasetFactory
from surprise import AlgoBase
from functools import singledispatch
from tabulate import tabulate
import pandas as pd
from typing import List, Tuple, Dict


def evaluate(algo: AlgoBase, splits: DataSplitter,
             metrics: list = ['RMSE', 'MAE', 'HR', 'ARHR', 'CHR', 'Coverage', 'Diversity', 'Novelty'],
             verbose: bool = True,
             k:int = 10) -> Dict[str, float]:
    """Evaluates a model on given metrics.

    Args:
        algo (AlgoBase): The recommendation algorithm to evaluate.
        splits (DataSplitter): The DataSplitter object providing train and test sets.
        metrics (List[str], optional): List of metrics for evaluation. Defaults to common metrics.
        verbose (bool, optional): If True, print progress and results. Defaults to True.
        k (int, optional): Number of top predictions for hit rate metrics. Defaults to 10.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics and their computed values.
    """
    if metrics is None:
        metrics = ['RMSE', 'MAE', 'HR', 'ARHR', 'CHR', 'Coverage', 'Diversity', 'Novelty']
    
    evaluation_dict = {}
    train_set, test_set = splits.accuracy_splits

    if verbose:
        print('Training the model...')
    algo.fit(train_set)
    
    if verbose:
        print('Generating predictions...')
    predictions = algo.test(test_set)

    if verbose:
        print('Evaluating accuracy metrics...')
    for metric in metrics:
        metric = metric.lower()
        if metric == 'rmse':
            evaluation_dict['RMSE'] = EvaluationMetrics.RMSE(predictions, verbose=verbose)
        elif metric == 'mae':
            evaluation_dict['MAE'] = EvaluationMetrics.MAE(predictions, verbose=verbose)
        elif metric == 'mse':
            evaluation_dict['MSE'] = EvaluationMetrics.MSE(predictions, verbose=verbose)
    
    if any(m.lower() in ['hr', 'arhr', 'chr'] for m in metrics):
        if verbose:
            print('Evaluating hit rate metrics...')
        evaluation_dict.update(_hit_rate_evaluation(algo, splits, k, metrics, verbose= verbose))

    if verbose:
        print('Evaluation Results:')
        for key, value in evaluation_dict.items():
            print(f'{key}: {value:.4f}')
    return evaluation_dict


def _hit_rate_evaluation(algo: AlgoBase, splits: DataSplitter, k: int,
                         metrics: list = ['RMSE', 'MAE', 'HR', 'ARHR', 'CHR', 'Coverage', 'Diversity', 'Novelty'], 
                         verbose: bool = False) -> Dict[str, float]:
    """Calculates hit rate-related metrics for the model.

    Args:
        algo (AlgoBase): The algorithm for recommendation.
        splits (DataSplitter): DataSplitter object providing train and test sets.
        k (int): Number of top predictions for hit rate metrics.
        metrics (List[str]): List of metrics to calculate.

    Returns:
        Dict[str, float]: Dictionary of hit rate metrics and their computed values.
    """
    evaluation_dict = {}
    train_set, test_set = splits.hit_splits
    big_test_set = splits.anti_testSet_for_hits
    
    algo.fit(train_set)
    predictions = algo.test(test_set)
    all_predictions = algo.test(big_test_set)
    top_n_predictions = EvaluationMetrics.get_top_n(all_predictions, n=k)
    
    for metric in metrics:
        metric = metric.lower()
        if metric == 'hr':
            evaluation_dict['HR'] = EvaluationMetrics.hit_rate(top_n_predictions, predictions, verbose= verbose)
        elif metric == 'arhr':
            evaluation_dict['ARHR'] = EvaluationMetrics.average_reciprocal_hit_rank(top_n_predictions, predictions, verbose= verbose)
        elif metric == 'chr':
            evaluation_dict['CHR'] = EvaluationMetrics.cumulative_hit_rate(top_n_predictions, predictions, verbose= verbose)
    return evaluation_dict

def benchmark(
    algos: List[AlgoBase],
    dataset: DatasetFactory,
    k: int = 10,
    ignore_response_time: bool = False,
    ignore_throughput: bool = False,
    random_state: int = 6,
    densities: List[int] = None,
    metrics: List[str] = None,
    verbose: bool = True
) -> Dict[str, dict]:
    """Benchmarks multiple models on a dataset across different densities.

    Args:
        algos (List[AlgoBase]): List of algorithms to benchmark.
        dataset (DatasetFactory): The dataset to use for benchmarking.
        k (int, optional): Number of top predictions for hit rate metrics. Defaults to 10.
        ignore_response_time (bool, optional): If True, ignore response time evaluation. Defaults to False.
        ignore_throughput (bool, optional): If True, ignore throughput evaluation. Defaults to False.
        random_state (int, optional): Random state for reproducibility. Defaults to 6.
        densities (List[int], optional): List of densities to evaluate. Defaults to [10, 20, 30].
        metrics (List[str], optional): Metrics to evaluate. Defaults to common metrics.
        verbose (bool, optional): If True, print progress and results. Defaults to True.

    Returns:
        Dict[str, dict]: Dictionary of results for each model and density.
    """
    if densities is None:
        densities = [10, 20, 30]
    if metrics is None:
        metrics = ['RMSE', 'MAE', 'HR', 'ARHR', 'CHR']

    results = {}
    splits = [DataSplitter(dataset, density, random_state) for density in densities]

    for data, density in zip(splits, densities):
        if not ignore_response_time:
            if verbose:
                print(f'Training models on response time data with {density}% density')
            results[f"Response Time {density}%"] = compare(algos, data.response_time, k, metrics, verbose=verbose)
        
        if not ignore_throughput:
            if verbose:
                print(f'Training models on throughput data with {density}% density')
            results[f"Throughput {density}%"] = compare(algos, data.throughput, k, metrics, verbose=verbose)

    return results

@singledispatch
def compare(algos: any, data: DataSplitter, k: int = 10, metrics: list[str] = ['RMSE', 'MAE', 'HR', 'ARHR', 'CHR'],
            verbose: bool = True) -> dict:
        """Compares multiple algorithms on a dataset split.

        Args:
            algos (Union[list, dict]): List or dictionary of algorithms to compare.
            data (DataSplitter): DataSplitter object with training and testing splits.
            k (int, optional): Number of top predictions for hit rate metrics. Defaults to 10.
            metrics (List[str], optional): Metrics to evaluate. Defaults to common metrics.
            verbose (bool, optional): If True, print progress and results. Defaults to True.

        Raises:
            NotImplementedError: If the algos parameter is neither a list nor a dictionary.
        """
        raise NotImplementedError("The provided algorithms must be a list or dictionary.")

@compare.register(list)
def _(algos: list, data: DataSplitter, k: int = 10, metrics: List[str] = None, verbose: bool = True) -> dict:
    if metrics is None:
        metrics = ['RMSE', 'MAE', 'HR', 'ARHR', 'CHR']
    
    algos_dict = {}
    for model in algos:
        if not isinstance(model, AlgoBase):
            raise TypeError("The algorithms list must contain only AlgoBase instances.")
        model_name = __get_model_name(model)
        
        # Ensure unique naming for duplicate models
        suffix = 1
        while model_name in algos_dict:
            model_name = f"{model_name}_{suffix}"
            suffix += 1
        
        algos_dict[model_name] = model

    return compare(algos_dict, data=data, k=k, metrics=metrics, verbose=verbose)


@compare.register(dict)
def _(algos: dict, data: DataSplitter, k: int = 10, metrics: List[str] = None, verbose: bool = True) -> dict:
    if metrics is None:
        metrics = ['RMSE', 'MAE', 'HR', 'ARHR', 'CHR']
    
    results = {}
    for model_name, model in algos.items():
        if not isinstance(model, AlgoBase):
            raise TypeError("All models must be instances of AlgoBase.")
        if verbose:
            print(f'Evaluating {model_name}...')
        results[model_name] = evaluate(algo=model, splits=data, metrics=metrics, verbose=verbose, k=k)

    if verbose:
        display_results(results, metrics)
    return results


# This private method take a model as a parameter and returns its name
def __get_model_name(algo: AlgoBase):
    # name = str(algo).split('object')[0]
    # name = name.split('.')[-1]
    # return name
    return algo.__class__.__name__

def display_results(results: Dict[str, dict], metrics: List[str]) -> None:
    """Displays the results of model evaluations in a table format.

    Args:
        results (Dict[str, dict]): Dictionary containing model names and their respective metric results.
        metrics (List[str]): List of metrics for which results are available.
    """
    headers = ["Model"] + metrics
    table_data = [[model] + [round(results[model][metric], 4) for metric in metrics] for model in results.keys()]
    print(tabulate(table_data, headers=headers, tablefmt="pipe"))