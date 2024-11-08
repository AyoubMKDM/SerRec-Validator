from surprise import accuracy
from collections import defaultdict
import itertools
from .utility import getPopularityRanks
from typing import List, Tuple, Dict
from surprise.prediction_algorithms.predictions import Prediction
import pandas as pd


def MAE(predictions: List[Prediction], verbose: bool = False) -> float:
    """
    Calculates the Mean Absolute Error (MAE) from a list of predictions.

    Args:
        predictions (list[surprise.prediction_algorithms.predictions.Prediction]): 
            List of predictions containing user ID, service ID, rating, estimation, and details.
        verbose (bool, optional): If True, prints the computed value. Defaults to False.

    Returns:
        float: The Mean Absolute Error of predictions.

    Raises:
        ValueError: If `predictions` is empty.
    """
    if not predictions:
        raise ValueError("Predictions list is empty.")
    return accuracy.mae(predictions, verbose)


def RMSE(predictions: List[Prediction], verbose: bool = False) -> float:
    """
    Calculates the Root Mean Square Error (RMSE) from a list of predictions.

    Args:
        predictions (list[surprise.prediction_algorithms.predictions.Prediction]): 
            List of predictions containing user ID, service ID, rating, estimation, and details.
        verbose (bool, optional): If True, prints the computed value. Defaults to False.

    Returns:
        float: The Root Mean Square Error of predictions.

    Raises:
        ValueError: If `predictions` is empty.
    """
    if not predictions:
        raise ValueError("Predictions list is empty.")
    return accuracy.rmse(predictions, verbose)

def MSE(predictions: List[Prediction], verbose: bool = False) -> float:
    """
    Calculates the Mean Square Error (MSE) from a list of predictions.

    Args:
        predictions (list[surprise.prediction_algorithms.predictions.Prediction]): 
            List of predictions containing user ID, service ID, rating, estimation, and details.
        verbose (bool, optional): If True, prints the computed value. Defaults to False.

    Returns:
        float: The Mean Square Error of predictions.

    Raises:
        ValueError: If `predictions` is empty.
    """
    if not predictions:
        raise ValueError("Predictions list is empty.")
    return accuracy.mse(predictions, verbose)

def get_top_n(predictions: List[Prediction], n: int = 10, minimumRating: float = 0.0, verbose: bool = False) -> defaultdict:
    """
    Generates top-N recommendations for each user.

    Args:
        predictions (list[surprise.prediction_algorithms.predictions.Prediction]): 
            List of predictions containing user ID, service ID, rating, estimation, and details.
        n (int, optional): Number of recommendations to return for each user. Defaults to 10.
        minimumRating (float, optional): Minimum rating to include in recommendations. Defaults to 0.0.
        verbose (bool, optional): If True, prints the computed recommendations. Defaults to False.

    Returns:
        defaultdict(int, list): Dictionary with user IDs as keys and lists of recommended service IDs and predicted ratings as values.

    Raises:
        ValueError: If `predictions` is empty.
    """
    if not predictions:
        raise ValueError("Predictions list is empty.")
    topN = defaultdict(list)
    for userID, serviceID, _, estimatedRating, _ in predictions:
        if estimatedRating >= minimumRating:
            topN[int(userID)].append((int(serviceID), estimatedRating))

    for userID, ratings in topN.items():
        ratings.sort(key=lambda x: x[1], reverse=True)
        topN[userID] = ratings[:n]
        if verbose:
            print(f'Top {n} recommendations for user {userID}:', *ratings[:n], sep="\n\t")

    return topN

def hit_rate(top_n_predicted: defaultdict, left_out_predictions: List[Prediction], verbose: bool = True) -> float:
    """
    Computes the hit rate of a set of top-N recommendations.

    Args:
        top_n_predicted (defaultdict): Top-N predictions for each user.
        left_out_predictions (list[surprise.prediction_algorithms.predictions.Prediction]): 
            Left-out predictions containing user ID, service ID, rating, estimation, and details.
        verbose (bool, optional): If True, prints the computed value. Defaults to True.

    Returns:
        float: Hit rate calculated as the ratio of hits to total left-out predictions.

    Raises:
        ValueError: If `left_out_predictions` is empty.
    """
    if not left_out_predictions:
        raise ValueError("Left-out predictions list is empty.")
    hits, total = 0, 0

    for left_out_item in left_out_predictions:
        userID, leftOutServiceID = left_out_item[:2]
        hit = any(serviceID == leftOutServiceID for serviceID, _ in top_n_predicted[int(userID)])
        hits += hit
        total += 1
    if verbose:
        print(f'Hit Rate = {hits / total:.4f}')
    return hits / total

def cumulative_hit_rate(top_n_predicted: defaultdict, left_out_predictions: List[Prediction], ratingCutoff: float = 0.0, verbose: bool = True) -> float:
    """
    Computes the cumulative hit rate, considering only ratings above a threshold.

    Args:
        top_n_predicted (defaultdict): Top-N predictions for each user.
        left_out_predictions (list[surprise.prediction_algorithms.predictions.Prediction]): 
            Left-out predictions containing user ID, service ID, rating, estimation, and details.
        ratingCutoff (float, optional): Minimum rating threshold. Defaults to 0.0.
        verbose (bool, optional): If True, prints the computed value. Defaults to True.

    Returns:
        float: Cumulative hit rate for the left-out predictions.

    Raises:
        ValueError: If `left_out_predictions` is empty.
    """
    if not left_out_predictions:
        raise ValueError("Left-out predictions list is empty.")
    hits, total = 0, 0

    for userID, leftOutServiceID, actualRating, _, _ in left_out_predictions:
        if actualRating >= ratingCutoff:
            hit = any(serviceID == leftOutServiceID for serviceID, _ in top_n_predicted[int(userID)])
            hits += hit
            total += 1
    if verbose:
        print(f'Cumulative Hit Rate = {hits / total:.4f}')
    return hits / total

def rating_hit_rate(top_n_predicted: defaultdict, left_out_predictions: List[Prediction], verbose: bool = True) -> List[Tuple[float, float]]:
    """
    Computes the rating hit rate, categorized by rating values.

    Args:
        top_n_predicted (defaultdict): Top-N predictions for each user.
        left_out_predictions (list[surprise.prediction_algorithms.predictions.Prediction]): 
            Left-out predictions containing user ID, service ID, rating, estimation, and details.
        verbose (bool, optional): If True, prints the computed values. Defaults to True.

    Returns:
        list of tuples: Rating and hit rate for each rating level.

    Raises:
        ValueError: If `left_out_predictions` is empty.
    """
    if not left_out_predictions:
        raise ValueError("Left-out predictions list is empty.")
    hits = defaultdict(float)
    total = defaultdict(float)
    rHR_list = []

    for userID, leftOutServiceID, actualRating, _, _ in left_out_predictions:
        hit = any(serviceID == leftOutServiceID for serviceID, _ in top_n_predicted[int(userID)])
        hits[actualRating] += hit
        total[actualRating] += 1

    for rating in sorted(hits.keys()):
        rHR_list.append((rating, hits[rating] / total[rating]))
    if verbose:
        print('Rating Hit Rate:', *rHR_list, sep='\n\t')
    return rHR_list

def average_reciprocal_hit_rank(top_n_predicted: defaultdict, left_out_predictions: List[Prediction], verbose: bool = True) -> float:
    """
    Computes the Average Reciprocal Hit Rank (ARHR).

    Args:
        top_n_predicted (defaultdict): Top-N predictions for each user.
        left_out_predictions (list[surprise.prediction_algorithms.predictions.Prediction]): 
            Left-out predictions containing user ID, service ID, rating, estimation, and details.
        verbose (bool, optional): If True, prints the computed value. Defaults to True.

    Returns:
        float: The ARHR for the predictions.

    Raises:
        ValueError: If `left_out_predictions` is empty.
    """
    if not left_out_predictions:
        raise ValueError("Left-out predictions list is empty.")
    summation, total = 0, 0

    for userID, leftOutServiceID, _, _, _ in left_out_predictions:
        hitRank = next((rank for rank, (serviceID, _) in enumerate(top_n_predicted[int(userID)], start=1)
                        if serviceID == leftOutServiceID), 0)
        if hitRank:
            summation += 1.0 / hitRank
        total += 1
    if verbose:
        print(f'Average Reciprocal Hit Rank = {summation / total:.4f}')
    return summation / total

def coverage(top_n_predicted: defaultdict, total_services: int, verbose: bool = True) -> float:
    """
    Calculates the coverage of the recommendations.

    Args:
        top_n_predicted (defaultdict): Top-N predictions for each user.
        total_services (int): Total number of available services.
        verbose (bool, optional): If True, prints the computed value. Defaults to True.

    Returns:
        float: The coverage ratio of unique recommended services.

    """
    recommended_services = {serviceID for ratings in top_n_predicted.values() for serviceID, _ in ratings}
    coverage = len(recommended_services) / total_services if total_services != 0 else 0
    if verbose:
        print(f'Service Coverage = {coverage:.4f}')
    return coverage


def diversity(top_n_predicted: defaultdict, services_df: pd.DataFrame, similarity_matrix: List[List[float]]) -> float:
    """
    Calculates the diversity of the top-N recommendations for each user.

    Diversity measures how different recommended items are from each other, based on item features.

    Args:
        top_n_predicted (defaultdict): Top-N predictions for each user, where keys are user IDs, 
            and values are lists of tuples (service ID, predicted rating).
        service_features (dict): Dictionary where keys are service IDs and values are lists of 
            feature vectors representing each service's characteristics.
        verbose (bool, optional): If True, prints the computed diversity value. Defaults to True.

    Returns:
        float: The average diversity score across all users.

    Raises:
        ValueError: If `top_n_predicted` or `service_features` is empty.
    """
    total, n = 0, 0

    for ratings in top_n_predicted.values():
        for service1, service2 in itertools.combinations(ratings, 2):
            idx1 = services_df.index[services_df['Service ID'] == service1[0]].tolist()[0]
            idx2 = services_df.index[services_df['Service ID'] == service2[0]].tolist()[0]
            total += similarity_matrix[idx1][idx2]
            n += 1
    return 1 - (total / n if n else 0)


def novelty(top_n_predicted: defaultdict, data_df: pd.DataFrame, max_value: int, verbose: bool = True) -> float:
    """
    Calculates the novelty of the top-N recommendations for each user.

    Novelty measures how uncommon recommended items are, based on their popularity.

    Args:
        top_n_predicted (defaultdict): Top-N predictions for each user, where keys are user IDs, 
            and values are lists of tuples (service ID, predicted rating).
        popularity_ranks (dict): Dictionary where keys are service IDs and values are integers 
            representing the popularity rank of each service (lower rank indicates higher popularity).
        verbose (bool, optional): If True, prints the computed novelty value. Defaults to True.

    Returns:
        float: The average novelty score across all users.

    Raises:
        ValueError: If `top_n_predicted` or `popularity_ranks` is empty.
    """
    rankings = getPopularityRanks(data_df, max_value)
    total, n = sum(rankings[rating[0]] for ratings in top_n_predicted.values() for rating in ratings), sum(len(ratings) for ratings in top_n_predicted.values())
    if verbose:
        print(f'Novelty = {total / n:.4f}')
    return total / n if n else 0