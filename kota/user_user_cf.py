from builtins import print

import pandas as pd
import numpy as np


def read_source_data(data_file_path: str):
    source_df = pd.read_excel(data_file_path)
    return source_df.head()


def get_top_neighbours(uu_corr, target_user, n=5):
    target_corr = uu_corr.loc[target_user]
    top_neighbours = target_corr.nlargest(n + 1).iloc[1 :]
    print("target user's top neighbours:",top_neighbours)
    return top_neighbours


def get_target_user_movie_score(target_movie, target_user, top_neighbours):
    """ Calculate the user-movie score for a target user. """
    ratings_sum = 0
    weight_sum = 0
    for target_user, each_weight in zip(top_neighbours.index, top_neighbours.values):
        # Test element-wise for NaN and return result as a boolean array.
        if np.isnan(target_movie[target_user]):
            continue
        ratings_sum += target_movie[target_user] * each_weight
        weight_sum += each_weight
    if weight_sum == 0:
        return 0
    return ratings_sum/weight_sum


def get_user_movie_score_normalized(target_movie, target_user) :
    """ Calculate the ucer-movie score for a target user with normalization. """

    top_neighbours = get_top_neighbours(uu_corr, target_user)

    ratings_sum = 0
    weight_sum = 0

    user_rating_mean = source_data_frame.loc[:, target_user].mean()

    for target_user, each_weight in zip(
            top_neighbours.index,
            top_neighbours.values,
    ) :
        if np.isnan(target_movie[target_user]) :
            continue

        movie_user_mean = source_data_frame.loc[:, target_user].mean()
        ratings_sum += (target_movie[target_user] - movie_user_mean) * each_weight
        weight_sum += each_weight

    if weight_sum == 0 :
        return 0

    return user_rating_mean + ratings_sum / weight_sum

def print_prediction_results(target_user_id, top_neighbours, source_data_frame, calculation_function) :
    """ Print results. """
    predict_result = source_data_frame.apply(
        calculation_function,
        axis=1,
        args=(target_user_id, top_neighbours,),
    )
    print ("******************")
    predict_result = get_target_user_movie_score()
    final_result = predict_result.sort_values(ascending=False)[:3]
    print (f"For tagert user {target_user_id}, the predict results as below: {final_result}")


if __name__ == '__main__':
    source_data_frame = read_source_data('UUCF Assignment Spreadsheet.xls')
    uu_corr = source_data_frame.corr()
    top_neighbours = get_top_neighbours(uu_corr, 3867)
    print_prediction_results(3867, top_neighbours,source_data_frame, get_target_user_movie_score)
    print_prediction_results(89, source_data_frame, get_target_user_movie_score)
    print_prediction_results(3867, source_data_frame, get_user_movie_score_normalized)
    print_prediction_results(89, source_data_frame, get_user_movie_score_normalized)
