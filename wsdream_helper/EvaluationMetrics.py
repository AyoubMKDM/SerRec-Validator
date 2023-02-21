from surprise import accuracy

#The functionality of both dictionaries and defaultdict are almost same except for the fact 
#that defaultdict never raises a KeyError. It provides a default value for the key that does not exists.
from collections import defaultdict
import itertools

class Metrics:

    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    def get_top_n(predictions,n=10,minimum_throuput=0.):
        data_list = list[3]
        top_n = defaultdict(list)
        # top_n = defaultdict([userID, serviceID, actualThroughput, estimatedThroughput 
        for user_id, service_id, actualThroughput, estimatedThroughput, _ in predictions:
            if estimatedThroughput >= minimum_throuput:
                top_n[int(user_id)].append(int(service_id),estimatedThroughput)

        for user_id, throuputs in top_n.items():
            throuputs.sort(key=lambda x: x[1], reverse=True)

            top_n[int(user_id)] = throuputs[:n]

        return top_n

    def HitRate(top_n_predicted, left_out_predictions):
        hits = 0
        total = 0

        # For each left-out rating
        for left_out in left_out_predictions:
            user_id = left_out[0]
            left_out_service_id = left_out[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for service_id, predicted_throuput in top_n_predicted[int(user_id)]:
                if (int(left_out_service_id) == int(service_id)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total

    def CumulativeHitRate(top_n_prediction, left_out_predictions, throuput_cut_off=0):
        hits = 0
        total = 0

        # For each left-out rating
        for user_id, left_out_service_id, actual_throuput, estimated_throuput, _ in left_out_predictions:
            # Only look at ability to recommend things the users actually liked...
            if (actual_throuput >= throuput_cut_off):
                # Is it in the predicted top 10 for this user?
                hit = False
                for service_id, predicted_throuput in top_n_prediction[int(user_id)]:
                    if (int(left_out_service_id) == service_id):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1

        # Compute overall precision
        return hits/total

    def RatingHitRate(top_n_predicted, left_out_predictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for user_id, left_out_service_id, actual_throuput, estimated_throuput, _ in left_out_predictions:
            # Is it in the predicted top N for this user?
            hit = False
            for service_id, predicted_throuput in top_n_predicted[int(user_id)]:
                if (int(left_out_service_id) == service_id):
                    hit = True
                    break
            if (hit) :
                hits[actual_throuput] += 1

            total[actual_throuput] += 1

        # Compute overall precision
        for throuput in sorted(hits.keys()):
            print (throuput, hits[throuput] / total[throuput])

    def AverageReciprocalHitRank(top_n_predicted, left_out_predictions):
        summation = 0
        total = 0
        # For each left-out rating
        for user_id, left_out_service_id, actual_throuput, estimated_throuput, _ in left_out_predictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for service_id, predicted_throuput in top_n_predicted[int(user_id)]:
                rank = rank + 1
                if (int(left_out_service_id) == service_id):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def UserCoverage(top_n_predicted, num_users, throuput_threshold=0):
        hits = 0
        for user_id in top_n_predicted.keys():
            hit = False
            for service_id, predicted_throuput in top_n_predicted[user_id]:
                if (predicted_throuput >= throuput_threshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / num_users

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

    def Novelty(top_n_predicted, rankings):
        n = 0
        total = 0
        for user_id in top_n_predicted.keys():
            for throuput in top_n_predicted[user_id]:
                service_id = throuput[0]
                rank = rankings[service_id]
                total += rank
                n += 1
        return total / n




    
