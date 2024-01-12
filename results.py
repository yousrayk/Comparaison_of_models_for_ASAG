import numpy as np
import pandas as pd
from Feature_Extraction import *
from Training import *
from tabulate import tabulate

df = pd.read_csv('answers_similarity_scores.csv')


def train_test_split(data, percentage):
    msk = np.random.rand(len(data)) < (percentage / 100)
    data_train = df[msk]
    data_test = df[~msk]

    return data_train, data_test


def calculate_average(given_list):
    return sum(given_list) / len(given_list)


def calculate_metrics(model_name):
    train_data, test_data = train_test_split(df, 70)
    model_name_score = 'normalized_' + model_name.lower() + '_score'
    train_data_x = train_data[model_name_score]
    train_data_y = train_data['score_avg']

    test_data_x = test_data[model_name_score]
    test_data_y = test_data['score_avg'].to_list()

    regression = RegressionAnalyzer(train_data_x, train_data_y, test_data_x)

    test_y_pred_lin = [float(x) for x in regression.linear_regression()]
    test_y_pred_rid = [float(x) for x in regression.ridge_regression()]
    test_y_pred_iso = [float(x) for x in regression.isotonic_regression()]

    metrics_lin = EvaluationMetrics(test_data_y, test_y_pred_lin)
    metrics_rid = EvaluationMetrics(test_data_y, test_y_pred_rid)
    metrics_iso = EvaluationMetrics(test_data_y, test_y_pred_iso)


    return metrics_iso.root_mean_squared_error(), metrics_iso.pearson_correlation(), metrics_lin.root_mean_squared_error(), metrics_lin.pearson_correlation(), metrics_rid.root_mean_squared_error(), metrics_rid.pearson_correlation()


if __name__ == '__main__':
    model_name = str(input('Choose one model (bert, elmo, gpt2) to find its RMSE and Person Correlation: '))

    lin_rmse = []
    lin_pearson = []

    rid_rmse = []
    rid_pearson = []

    iso_rmse = []
    iso_pearson = []

    for i in range(0, 1000):
        iso_rmse_score, iso_pc_score, lin_rmse_score, lin_pc_score, rid_rmse_score, rid_pc_score = calculate_metrics(model_name)
        
        lin_rmse.append(lin_rmse_score)
        lin_pearson.append(lin_pc_score)

        rid_rmse.append(rid_rmse_score)
        rid_pearson.append(rid_pc_score)

        iso_rmse.append(iso_rmse_score)
        iso_pearson.append(iso_pc_score)

    table = [
        ["Metric", "Isotonic Regression", "Linear Regression", "Ridge Regression"],
        ["RMSE", round(calculate_average(iso_rmse), 3), round(calculate_average(lin_rmse), 3), round(calculate_average(rid_rmse), 3)],
        ["Pearson Correlation", round(calculate_average(iso_pearson), 3), round(calculate_average(lin_pearson), 3), round(calculate_average(rid_pearson), 3)]
    ]

    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))