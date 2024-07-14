# Bike-Sharing-Demand-Prediction

## Project Overview

This project aims to predict bike-sharing demand using machine learning models. We use data from the UCI Machine Learning Repository, which includes daily and hourly counts of rental bikes between 2011 and 2012 in the Capital Bikeshare system with corresponding weather and seasonal information.

## Directory Structure


## Getting Started

### Prerequisites

Ensure you have Python 3.x installed. Install the required packages using:

```sh
pip install -r requirements.txt
```
### Data

Place the __day.csv__ and __hour.csv__ files in the __data/__ directory.

### Running the Project

- #### Data Loading and Preprocessing

  Load and preprocess the data using the __preprocess.py__ script in the __utils__ directory.

- #### Model Training

  Train the models using the __train.py__ script in the __model__ directory. This script trains multiple models and evaluates their     performance using cross-validation.

  ```sh
  python model/train.py
  ```

- #### Hyperparameter Tuning
 
  Tune the hyperparameters of the models using the __tune.py__ script in the __model__ directory.

  ```sh
  python model/tune.py
  ```

- #### Visualization

  Generate various visualizations to understand the data and model performance using the __visualization.py__ script.

  ```sh
  python eda/visualization.py
  ```

## Unit Tests

Unit tests are provided for each module. You can run the tests using:

```sh
python -m unittest discover -s tests
```

## Scripts Description

### `data/`

- `day.csv` and `hour.csv`: The dataset files.

### `model/`

- `train.py`: Contains functions for training machine learning models.
- `tune.py`: Contains functions for hyperparameter tuning of the models.
- `results.py`: Contains the final results of the trained and tuned models.

### `notebooks/`

- `exploratory_analysis.ipynb`: Jupyter notebook for exploratory data analysis.

### `tests/`

- `test_data_loader.py`: Unit tests for the data loading functions.
- `test_preprocess.py`: Unit tests for the data preprocessing functions.
- `test_model.py`: Unit tests for the model training functions.
- `test_tune.py`: Unit tests for the hyperparameter tuning functions.

### `utils/`

- `preprocess.py`: Contains functions for data preprocessing.

### `visualization.py`

Generates various visualizations for data analysis and model performance evaluation.
