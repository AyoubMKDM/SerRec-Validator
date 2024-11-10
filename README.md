![GitHub version](https://img.shields.io/badge/version-0.1-8A2BE2)
![python versions](https://img.shields.io/badge/python-3.8%2B-green)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
# SerRec-Validator
**SerRec-Validator** is a Python package developed to benchmark and evaluate service recommendation systems. It offers a comprehensive set of evaluation metrics, data normalization strategies, and tools to support the development and assessment of recommender models in various domains, with a specific focus on web service recommendations.

The framework was inspired by the limitations of traditional evaluation metrics (e.g., RMSE and MSE) that often fail to reflect real-world performance. **SerRec-Validator** addresses these challenges by providing a more holistic evaluation approach, ensuring that service recommendation systems can be assessed accurately and effectively.

This package integrates with the **WS-DREAM** dataset and the **Surprise library**, enabling users to train, evaluate, and compare recommender models with ease.


## Features

- **WS-DREAM Dataset Integration**: The package includes built-in support for the WS-DREAM dataset, which contains Quality of Service (QoS) attributes for web services. Users can also import and preprocess their custom datasets for evaluation.
  
- **Normalization**: SerRec-Validator provides built-in data normalization functions to transform recommendation matrices to a common scale, improving the reliability of model evaluations.
  
- **Surprise Library Integration**: Built on top of the **Surprise** library, this framework allows easy integration with various collaborative filtering models, including built-in algorithms like SVD, KNNBasic, and user-defined models.
  
- **Comprehensive Evaluation Metrics**: The package offers a variety of metrics for assessing recommender systems, such as:
  - Accuracy: RMSE, MAE, MSE
  - Diversity, Coverage and Novelty
  - Hit Rate, Cumulative Hit Rate, and Average Reciprocal Hit Rate (ARHR)
  
- **Flexible Use Cases and Comparisons**: SerRec-Validator supports the evaluation and comparison of multiple models based on different performance criteria, with detailed documentation and example use cases.


## Installation
To install **SerRec-Validator**, you can use `pip`:

```bash
$ pip install -i https://test.pypi.org/simple/serrec-validator
```

Alternatively, follow these steps to install from the source:
1. Clone the repository:
   ```bash
   git clone https://github.com/AyoubMKDM/serrec-validator.git
   cd cd SerRec-Validator/
   ```
2. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. Install the package:
   ```bash
   python setup.py install
    ```


## Quick Start Guide

For a hands-on introduction to SerRec-Validator, we’ve prepared a Quick Start Guide as an interactive [Google Colab Notebook](https://colab.research.google.com/drive/10qspRCXTODU5_MGa2w1p3E_5mWPdao4Q?usp=sharing). This notebook covers:

1. **Installation**: Step-by-step instructions to set up the framework.
2. **Dataset Preparation**: Guidance on loading the WS-DREAM dataset or custom datasets.
3. **Model Training and Evaluation**: Examples demonstrating the integration with the Surprise library for training and evaluating collaborative filtering models.
4. **Metrics Calculation**: An overview of available evaluation metrics, with code examples.
5. **Normalization Techniques**: Sample code for normalizing data before feeding it into models.

Click [here](https://colab.research.google.com/drive/10qspRCXTODU5_MGa2w1p3E_5mWPdao4Q?usp=sharing) to get started and follow along with practical examples.


Let me know if you’d like any adjustments!

## Usage

### Download the WS-DREAM Dataset
The framework comes with a built-in version of the WS-DREAM dataset. However, if you prefer to download the dataset manually, you can run the following script:

```bash
  python -m serrec_validator.WsdreamDownloader --url <dataset_url> --dir <directory_to_save>
  ```
By default, the dataset will be downloaded from:

```bash
  https://zenodo.org/record/1133476/files/wsdream_dataset1.zip?download=1
  ```

### Loading the Dataset

You can easily load the WS-DREAM dataset using the `WsdreamLoader` class. Here's how:

```python
from serrec_validator.WsdreamLoader import WsdreamLoader

# Load dataset (use the built-in version or specify a custom path)
loader = WsdreamLoader(built_in=True)  # Automatically loads bundled data
# Alternatively, use a custom path:
# loader = WsdreamLoader(dir="<path_to_dataset>")

# Accessing the loaded data
users_df = loader.users_df
services_df = loader.services_df
response_time_matrix = loader.response_time_matrix
throughput_matrix = loader.throughput_matrix
```

### Integrating with Surprise
The SerRec-Validator framework integrates seamlessly with the Surprise library, allowing you to apply collaborative filtering models such as SVD or KNNBasic. Here’s an example:

  ```python
  from surprise import Dataset, Reader
  from serrec_validator.WsdreamDataset import WsdreamDataset
  
  # Load the dataset using WsdreamDataset
  dataset = WsdreamDataset(wsdream_loader_instance)
  
  # Apply a collaborative filtering algorithm (e.g., SVD)
  from surprise import SVD
  from surprise import accuracy
  from serrec_validator.utility import DataSplitter
  
  # Split dataset into training and test sets
  splits = DataSplitter(wsdream_loader_instance, density=20, random_state=6)
  train_set, test_set = splits.response_time.accuracy_splits
  
  # Train the model
  algo = SVD(random_state=6)
  algo.fit(train_set)
  
  # Evaluate the model
  predictions = algo.test(test_set)
  accuracy.rmse(predictions)
  ```


### Perform Model Evaluation
Use the **ModelEvaluator** class to evaluate a recommender model:

  ```python
  from serrec_validator.ModelEvaluator import ModelEvaluator
  
  # Evaluate the model (e.g., SVD) based on a specific density and dataset
  results = ModelEvaluator.evaluate(algo=svd, density=5, dataset=wsdream)
  print(results)
  ```

### Evaluation Metrics
You can compute various evaluation metrics using the `EvaluationMetrics` class. For example:

```python
from serrec_validator import EvaluationMetrics

# Make predictions and compute metrics
left_out_predictions = algo.test(test_set)
all_predictions = algo.test(splits.anti_test_set_for_hits)
top_n_predicted = EvaluationMetrics.get_top_n(all_predictions, n=10)

# Calculate Hit Rate
hit_rate = EvaluationMetrics.hit_rate(top_n_predicted, left_out_predictions)
print("Hit Rate:", hit_rate)

# Calculate accuracy metrics (RMSE, MAE, MSE)
rmse = EvaluationMetrics.rmse(left_out_predictions)
mae = EvaluationMetrics.mae(left_out_predictions)
mse = EvaluationMetrics.mse(left_out_predictions)
print("RMSE:", rmse)
print("MAE:", mae)
print("MSE:", mse)

# Calculate Novelty and Diversity
novelty = EvaluationMetrics.novelty(top_n_predicted)
diversity = EvaluationMetrics.diversity(top_n_predicted)
print("Novelty:", novelty)
print("Diversity:", diversity)
```


### Normalization Strategies
SerRec-Validator provides several normalization strategies to ensure that your data is prepared for optimal model training:


  ```python
  from serrec_validator.Normalization import zScore
  
  # Normalize response time matrix using z-score normalization
  normalized_data = zScore.normalize(response_time_matrix)
  ```

### Key Modules

  * `WsdreamDownloader`: Downloads and extracts the WS-DREAM dataset.
  * `WsdreamLoader`: Loads the WS-DREAM dataset and makes it available for use in the framework.
  * `ModelEvaluator`: Evaluates different recommendation models using various metrics.
  * `Normalization`: Contains methods to normalize the dataset and the response times for better model performance.
  * `EvaluationMetrics`: Computes various metrics like novelty, diversity, and accuracy for evaluating the recommendation models.

  
## Results

The following tables summarize the performance of different recommendation models on the WS-DREAM dataset across three data density levels: 10%, 20%, and 30%. Key evaluation metrics include RMSE, MAE, Diversity, Novelty, and Coverage.

### 10% Data Density

| Model         | RMSE   | MAE    | Diversity | Novelty   | Coverage  |
|---------------|--------|--------|-----------|-----------|-----------|
| SlopeOne      | 1.5126 | 0.6898 | 0.3336    | 0.1507    | 0.0033    |
| SVD           | **1.4227** | 0.5626 | 0.3505    | 0.0950    | **0.0167**    |
| KNNBaseline   | 1.5127 | **0.5603** | 0.3511    | **0.1538**    | 0.0156    |
| BaselineOnly  | 1.5391 | 0.6812 | 0.2509    | 0.1225    | 0.0027    |
| KNNBasic      | 1.9632 | 0.9745 | **0.9903**    | 0.0391    | 0.0062    |

### 20% Data Density

| Model         | RMSE   | MAE    | Diversity | Novelty   | Coverage  |
|---------------|--------|--------|-----------|-----------|-----------|
| SlopeOne      | 1.4574 | 0.6663 | 0.3659    | **0.1953**    | 0.0034    |
| SVD           | **1.2839** | 0.5045 | 0.3693    | 0.1382    | 0.0172    |
| KNNBaseline   | 1.3347 | **0.4832** | 0.3418    | 0.1644    | 0.0155    |
| BaselineOnly  | 1.4614 | 0.6621 | 0.3664    | 0.1662    | 0.0034    |
| KNNBasic      | 1.5121 | 0.5040 | **0.6756**    | 0.1694    | **0.0312**    |

### 30% Data Density

| Model         | RMSE   | MAE    | Diversity | Novelty   | Coverage  |
|---------------|--------|--------|-----------|-----------|-----------|
| SlopeOne      | 1.4915 | 0.6679 | 0.3526    | 0.2794    | 0.0043    |
| SVD           | **1.2763** | 0.4860 | 0.3941    | 0.2970    | 0.0240    |
| KNNBaseline   | 1.3387 | **0.4617** | 0.3224    | 0.3419    | 0.0167    |
| BaselineOnly  | 1.4934 | 0.6684 | 0.3526    | 0.2631    | 0.0041    |
| KNNBasic      | 1.4897 | 0.4896 | **0.6524**    | **0.3585**    | **0.0357**    |

These results demonstrate the effectiveness of various recommendation algorithms under different data densities. The optimized models reveal improvements in several metrics, such as RMSE and MAE, while also showing distinct characteristics in novelty, diversity, and coverage. For more details on the optimization process and the implications of these findings, refer to the accompanying research paper.

## Acknowledgements
The dataset used in this framework is sourced from the WS-DREAM project, which provides a real-world dataset for web service recommendations. Thanks to the authors of WS-DREAM for making this dataset available.


## License

This project is licensed under the [BSD3-Clause](https://opensource.org/licenses/BSD-3-Clause) license, so it can be used for pretty much everything, including commercial applications.


### Contributing
We welcome contributions to this project! If you’d like to improve the framework, feel free to fork the repository, submit pull requests, or open issues. Here are some ways you can help:

  * Add more evaluation metrics.
  * Improve data loading and normalization methods.
  * Enhance the integration with other recommendation libraries.
  * Suggest additional datasets for benchmarking.

I'd love to know how SerRec-Validator is useful to you. Please don't hesitate to open an issue and describe how you use it!
