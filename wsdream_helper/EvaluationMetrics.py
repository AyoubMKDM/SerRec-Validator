from surprise import accuracy

#The functionality of both dictionaries and defaultdict are almost same except for the fact 
#that defaultdict never raises a KeyError. It provides a default value for the key that does not exists.
from collections import defaultdict
import itertools

def MAE(predictions, verbose=True):
    return accuracy.mae(predictions, verbose)

def RMSE(predictions, verbose=True):
    return accuracy.rmse(predictions, verbose)

def MSE(predictions, verbose=True):
    return accuracy.mse(predictions, verbose)

def get_top_n(predictions, n=10, minimumRating=0.0, verbose=False):
    topN = defaultdict(list)
    for userID, serviceID, actualRating, estimatedRating, _ in predictions:
        if (estimatedRating >= minimumRating):
            topN[int(userID)].append((int(serviceID), estimatedRating))

    for userID, ratings in topN.items():
        ratings.sort(key=lambda x: x[1], reverse=True)
        topN[int(userID)] = ratings[:n]
        if verbose:
            print(f'Top {n} recommendations for user with id "{userID}" are:')
            print('\t\t', *ratings[:n],sep="\n\t\t")

    return topN

def hit_rate(top_n_predicted, left_out_predictions, verbose=True):
    hits = 0
    total = 0

    # For each left-out rating
    for left_out_item in left_out_predictions:
        userID = left_out_item[0]
        leftOutServiceID = left_out_item[1]
        # Is it in the predicted top n for this user?
        hit = False
        for serviceID, predictedRating in top_n_predicted[int(userID)]:
            if (int(leftOutServiceID) == int(serviceID)):
                hit = True
                break
        if (hit) :
            hits += 1

        total += 1
    if verbose:
        print(f'Hit rate = \t {hits/total:.4f}')
    # Compute overall precision
    return hits/total

def cumulative_hit_rate(top_n_predicted, left_out_predictions, ratingCutoff=0, verbose=True):
    hits = 0
    total = 0

    # For each left-out rating
    for userID, leftOutServiceID, actualRating, estimatedRating, _ in left_out_predictions:
        # Only look at ability to recommend things the users actually liked...
        if (actualRating >= ratingCutoff):
            # Is it in the predicted top 10 for this user?
            hit = False
            for serviceID, predictedRating in top_n_predicted[int(userID)]:
                if (int(leftOutServiceID) == serviceID):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

    if verbose:
        print(f'cHR = \t {hits/total:.4f}')
    # Compute overall precision
    return hits/total

def rating_hit_rate(top_n_predicted, left_out_predictions, verbose=True):
    hits = defaultdict(float)
    total = defaultdict(float)
    rHR_list = []

    # For each left-out rating
    for userID, leftOutServiceID, actualRating, estimatedRating, _ in left_out_predictions:
        # Is it in the predicted top N for this user?
        hit = False
        for serviceID, predictedRating in top_n_predicted[int(userID)]:
            if (int(leftOutServiceID) == serviceID):
                hit = True
                break
        if (hit) :
            hits[actualRating] += 1
        total[actualRating] += 1

    # Compute overall precision
    for rating in sorted(hits.keys()):
        rHR_list.append((rating, hits[rating] / total[rating]))
    if verbose:
        print('rHr = ', *rHR_list, sep='\n\t')
    return rHR_list

def average_reciprocal_hit_rank(top_n_predicted, left_out_predictions, verbose=True):
    summation = 0
    total = 0
    # For each left-out rating
    for userID, leftOutServiceID, actualRating, estimatedRating, _ in left_out_predictions:
        # Is it in the predicted top N for this user?
        hitRank = 0
        rank = 0
        for serviceID, predictedRating in top_n_predicted[int(userID)]:
            rank = rank + 1
            if (int(leftOutServiceID) == serviceID):
                hitRank = rank
                break
        if (hitRank > 0) :
            summation += 1.0 / hitRank

        total += 1
    if verbose:
        print(f'ARHR = \t{summation / total:.4}')
    return summation / total

# What percentage of users have at least one "good" recommendation
def user_coverage(top_n_predicted, number_users, ratingThreshold=0, verbose=True):
    hits = 0
    for userID in top_n_predicted.keys():
        hit = False
        for serviceID, predictedRating in top_n_predicted[userID]:
            if (predictedRating >= ratingThreshold):
                hit = True
                break
        if (hit):
            hits += 1

    if verbose:
        print(f'Coverage = \t{hits/number_users:.4}')
    return hits / number_users

def Diversity(top_n_predicted, sims_algo):
    n = 0
    total = 0
    sims_matrix = sims_algo.compute_similarities()
    for user_id in top_n_predicted.keys():
        pairs = itertools.combinations(top_n_predicted[user_id], 2)
        for pair in pairs:
            service1 = pair[0][0]
            service2 = pair[1][0]
            innerID1 = sims_algo.trainset.to_inner_iid(str(service1))
            innerID2 = sims_algo.trainset.to_inner_iid(str(service2))
            similarity = sims_matrix[innerID1][innerID2]
            total += similarity
            n += 1

    S = total / n
    return (1-S)

def novelty(top_n_predicted, rankings, verbose=True):
    n = 0
    total = 0
    for userID in top_n_predicted.keys():
        for rating in top_n_predicted[userID]:
            serviceID = rating[0]
            rank = rankings[serviceID]
            total += rank
            n += 1

    if verbose:
        print(f'Novelty = \t{total/n:.4}')
    return total / n




    
