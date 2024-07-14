from data.data_loader import load_data
from eda.exploratory_analysis import summarize_dataset
from eda.visualization import visualize_all, summarize_outliers
from utils.preprocess import preprocess_data
from model.train import train_models
from model.tune import tune_hyperparameters
from utils.evaluate import evaluate_tune_models_train_set, evaluate_tune_models_test_set
from model.results import plot_results

def main():
    # Load data
    day_df, hour_df = load_data('data/day.csv', 'data/hour.csv')

    hour_df = hour_df.reset_index(drop=True).set_index('dteday')
    
    # EDA
    summary = summarize_dataset(hour_df)
    print(summary)

    # Changing the index to 'instant' as it is unique for all the rows
    hour_df = hour_df.reset_index().set_index('instant')

    # Adding additional time variables for better visualization and insights
    hour_df['day'] = hour_df['dteday'].dt.day
    hour_df['year'] = hour_df['yr'].apply(lambda x: 2011 if x==0 else 2012)

    visualize_all(hour_df)

    # Checking the outlier distribution across various features
    outlier_summary = summarize_outliers(hour_df)
    
    # Preprocess Data
    X_train, X_test, y_train, y_test = preprocess_data(hour_df)
    
    # Train models
    results = train_models(X_train, y_train)
    print(results)
    
    # Hyperparameter tuning
    top_models = ['RandomForest', 'XGBoost']
    best_models = tune_hyperparameters(top_models, X_train, y_train)
    print(best_models)

    # Evaluate tuned models on the training set using cross-validation
    comparison_df = evaluate_tune_models_train_set(best_models, X_train, y_train, results, top_models)
    print(comparison_df)

    # Evaluate the final models on the test set
    test_results = evaluate_tune_models_test_set(best_models, X_test, y_test)
    print(test_results)

    # PLot the final results
    plot_results(best_models, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()