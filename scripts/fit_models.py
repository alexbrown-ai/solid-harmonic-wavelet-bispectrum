import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from scattering.coefficients import (
    calculate_coefficients,
    calculate_coefficients_morlet,
)


def find_lowest_factor(num, threshold):
    for i in range(threshold, num + 1):
        if num % i == 0:
            return i
    return None


def get_folds(data):
    num_folds = find_lowest_factor(data.shape[0], threshold=3)
    P = np.random.permutation(data.shape[0]).reshape((num_folds, -1))
    cross_val_folds = []

    for i_fold in range(num_folds):
        fold = (np.concatenate(P[np.arange(num_folds) != i_fold], axis=0), P[i_fold])
        cross_val_folds.append(fold)

    return cross_val_folds


def fit_regressor(pipeline, param_grid, X, target, M, N, scattering_config, higher_config, output_name, hos=None,
                  morlet=False):
    if morlet:
        L = scattering_config["L"]
        J = scattering_config["J"]
        scattering_coef = calculate_coefficients_morlet(X, M, N, L, J, output_name)
        scattering_coef = scattering_coef.reshape(scattering_coef.shape[0], -1)
    else:
        scattering_coef, target = calculate_coefficients(X, target, M, N, scattering_config, higher_config, output_name,
                                                         hos)

    X_train, X_test, y_train, y_test = train_test_split(scattering_coef, target, test_size=0.2)
    cross_val_folds = get_folds(X_train)

    grid_search = GridSearchCV(pipeline, param_grid, cv=cross_val_folds, n_jobs=4, verbose=True,
                               scoring="neg_mean_absolute_error")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    target_prediction = best_model.predict(X_test)

    mae_per_target, rmse_per_target = [], []
    if len(target.shape) > 1:
        for i in range(target.shape[1]):
            mae = mean_absolute_error(y_test[:, i], target_prediction[:, i])
            rmse = root_mean_squared_error(y_test[:, i], target_prediction[:, i])
            mae_per_target.append(mae)
            rmse_per_target.append(rmse)
    else:
        mae = mean_absolute_error(y_test, target_prediction)
        rmse = root_mean_squared_error(y_test, target_prediction)
        mae_per_target.append(mae)
        rmse_per_target.append(rmse)

    for i, (mae, rmse) in enumerate(zip(mae_per_target, rmse_per_target)):
        print(f"Target {i}: MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    return target_prediction, y_test


def fit_classifier(pipeline, param_grid, X, target, M, N, scattering_config, higher_config, output_name, hos=None,
                   morlet=False):
    if morlet:
        L = scattering_config["L"]
        J = scattering_config["J"]
        scattering_coef = calculate_coefficients_morlet(X, M, N, L, J, output_name)
        scattering_coef = scattering_coef.reshape(scattering_coef.shape[0], -1)
    else:
        scattering_coef, _ = calculate_coefficients(X, target, M, N, scattering_config, higher_config, output_name, hos)
    print(scattering_coef.shape)

    X_train, X_test, y_train, y_test = train_test_split(scattering_coef, target, test_size=0.1, random_state=42)
    cross_val_folds = get_folds(X_train)

    grid_search = GridSearchCV(pipeline, param_grid, scoring="f1", cv=cross_val_folds, n_jobs=4, verbose=True)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    target_prediction = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, target_prediction)
    precision = precision_score(y_test, target_prediction)
    recall = recall_score(y_test, target_prediction)
    f1 = f1_score(y_test, target_prediction)

    try:
        best_params = grid_search.best_params_
        print(
            f"Best params: {best_params}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

    except Exception:
        print(
            f"Best params: unknown, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
