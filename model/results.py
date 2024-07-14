# Function to plot learning curves
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve


def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training score')
    plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
    plt.title(f'Learning Curve: {title}')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.show()


# Function to plot residuals
def plot_residuals(model, X_train, y_train, X_test, y_test, title):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_train_pred, label='Train', alpha=0.7)
    plt.scatter(y_test, y_test_pred, label='Test', alpha=0.7, color='red')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Residuals Plot: {title}')
    plt.legend()
    plt.grid()
    plt.show()

def plot_results(best_models, X_train, y_train, X_test, y_test):
    # Plot learning curves for the tuned models
    for model_name, model in best_models.items():
        plot_learning_curve(model, X_train, y_train.values.ravel(), model_name)

    # Plot residuals for the tuned models
    for model_name, model in best_models.items():
        plot_residuals(model, X_train, y_train.values.ravel(), X_test, y_test.values.ravel(), model_name)



