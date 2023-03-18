from wsdream_helper import wsdream_utility
from surprise import SVD
from wsdream_helper import EvaluationMetrics
from surprise import KNNBaseline
from wsdream_helper.utility import DataSplitter

dataset = wsdream_utility.Wsdream()

print("Loading response time data ...")
data = dataset.get_responseTime(density=5)
splits = DataSplitter(dataset, 5, 6)


print("\nComputing item similarities so we can measure diversity later...")
# fullTrainSet = data.build_full_trainset()
fullTrainSet = splits.trainset_from_full_data['response_time']
sim_options = {'name': 'pearson_baseline', 'user_based': False}
simsAlgo = KNNBaseline(sim_options=sim_options)
simsAlgo.fit(fullTrainSet)

print("\nBuilding recommendation model...")
trainSet, testSet = splits.splitset_for_accuracy['response_time']

algo = SVD(random_state=10)
algo.fit(trainSet)

print("\nComputing recommendations...")
predictions = algo.test(testSet)

print("\nEvaluating accuracy of model...")
print("RMSE: ", EvaluationMetrics.RMSE(predictions))
print("MAE: ", EvaluationMetrics.MAE(predictions))

print("\nEvaluating top-10 recommendations...")

trainSet, testSet = splits.splitset_for_hit_rate['response_time']

# Train model without left-out ratings
algo.fit(trainSet)

# Predicts ratings for left-out ratings only
print("Predict ratings for left-out set...")
leftOutPredictions = algo.test(testSet)

# Build predictions for all ratings not in the training set
print("Predict all missing ratings...")
allPredictions = algo.test(splits.anti_testset_for_hit_rate["response_time"])

# Compute top 10 recs for each user
print("Compute top 10 recs per user...")
topNPredicted = EvaluationMetrics.get_top_n(allPredictions, n=10)

# See how often we recommended a movie the user actually rated
print("\nHit Rate: ", EvaluationMetrics.hit_rate(topNPredicted, leftOutPredictions))

# Break down hit rate by rating value
print("\nrHR (Hit Rate by Rating value): ")
EvaluationMetrics.rating_hit_rate(topNPredicted, leftOutPredictions)

# See how often we recommended a movie the user actually liked
print("\ncHR (Cumulative Hit Rate, rating >= 4): ", EvaluationMetrics.cumulative_hit_rate(topNPredicted, leftOutPredictions))

# Compute ARHR
print("\nARHR (Average Reciprocal Hit Rank): ", EvaluationMetrics.average_reciprocal_hit_rank(topNPredicted, leftOutPredictions))

print("\nComputing complete recommendations, no hold outs...")
algo.fit(fullTrainSet)
# bigTestSet = fullTrainSet.build_anti_testset()
allPredictions = algo.test(splits.anti_testset_from_full_data['response_time'])
topNPredicted = EvaluationMetrics.get_top_n(allPredictions, n=10)

# Print user coverage with a minimum predicted rating of 4.0:
print("\nUser coverage: ", EvaluationMetrics.user_coverage(topNPredicted, fullTrainSet.n_users))

# # Measure diversity of recommendations:
# print("\nDiversity: ", EvaluationMetrics.Diversity(topNPredicted, simsAlgo))

# Measure novelty (average popularity rank of recommendations):
print("\nNovelty (average popularity rank): ", EvaluationMetrics.novelty(topNPredicted, data.df, wsdream_utility.Wsdream.usersList.shape[0]))

