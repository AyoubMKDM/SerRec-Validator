[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
# SerRec-Validator
SerRec-Validator is a Python package developed to address the challenges of benchmarking and evaluating service recommendations. This framework offers a comprehensive set of evaluation metrics and tools, facilitating the development and assessment of recommender models in various domains.

The inspiration behind SerRec-Validator came from the need to enhance the evaluation of service recommendation systems. Traditional metrics like RMSE and MSE often fail to reflect real-world performance, making it challenging to assess the effectiveness of these systems. SerRec-Validator strives to overcome these limitations and provide a user-friendly platform for researchers, developers, and practitioners to evaluate their service recommendation models accurately.

## Features

SerRec-Validator comes with a variety of features to streamline the evaluation process:

- **Dataset Preparation**: The framework allows users to download and preprocess the WS-DREAM dataset, which contains Quality of Service (QoS) attributes for web services. Additionally, SerRec-Validator offers the flexibility to work with custom datasets, empowering users to import and process their data for evaluation.

- **Normalization**: Data normalization is crucial for accurate evaluation. SerRec-Validator provides built-in normalization functions and supports various normalization strategies to transform data to a common scale and eliminate biases.

- **Integration with Surprise Library**: The package is built on top of the Surprise library, a powerful toolkit for recommender systems. This integration allows users to easily implement and evaluate various recommender models using Surprise's pre-implemented algorithms or by creating their own models.

- **Evaluation Metrics**: SerRec-Validator offers a wide range of evaluation metrics, including Hit Rate, Average Reciprocal Hit Rate (ARHR), Cumulative Hit Rate (CHR), Rating Hit Rate (RHR), Accuracy metrics (RMSE, MAE, MSE), Novelty, Diversity, and Coverage. These metrics provide comprehensive insights into the performance of recommender systems.

- **Use Cases and Comparisons**: Users can evaluate and compare multiple recommender models based on various performance criteria using the framework's capabilities. The documentation includes detailed use cases to guide users through different scenarios.

## Installation

To install SerRec-Validator, use the following command:

```bash
$ pip install serrec-validator
```

## Usage
### Loading the Dataset
To load the WS-DREAM dataset into the framework, use the following Python code:
```python
from serrec_validator import Wsdream, WsdreamDataset1Downloader

WsdreamDataset1Downloader.download(dir='dataset')

wsdream_reader = Wsdream.WsdreamReader(dir='dataset')
wsdream_dataset = Wsdream.WsdreamDataset(wsdream_reader)
```

### Implementing Recommender Models
You can implement recommender models using Surprise algorithms. Here's an example using the SVD algorithm:
```python
from surprise import SVD
from serrec_validator.utility import DataSplitter

splits = DataSplitter(wsdream_z_score, density=20, random_state=6)
train_set, test_set = splits.response_time.accuracy_splits

algo = SVD(random_state=6)
algo.fit(train_set)
```

### Evaluating Model Performance
To evaluate model performance, use the evaluation metrics provided by SerRec-Validator:
```python
from serrec_validator import EvaluationMetrics

left_out_predictions = algo.test(test_set)
all_predictions = algo.test(splits.anti_test_set_for_hits)
top_n_predicted = EvaluationMetrics.get_top_n(all_predictions, n=10)

# Compute Hit Rate
hit_rate = EvaluationMetrics.hit_rate(top_n_predicted, left_out_predictions)
print("Hit Rate:", hit_rate)

# Compute Accuracy Metrics
rmse = EvaluationMetrics.rmse(left_out_predictions)
mae = EvaluationMetrics.mae(left_out_predictions)
mse = EvaluationMetrics.mse(left_out_predictions)

print("RMSE:", rmse)
print("MAE:", mae)
print("MSE:", mse)

# Compute Novelty and Diversity
novelty = EvaluationMetrics.novelty(top_n_predicted)
diversity = EvaluationMetrics.diversity(top_n_predicted)

print("Novelty:", novelty)
print("Diversity:", diversity)
```

We welcome contributions and feedback from the community to further improve and expand the functionality of SerRec-Validator. Together, we can enhance the evaluation of service recommendations and contribute to the advancement of this vital research field.











