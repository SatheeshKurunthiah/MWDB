from __future__ import division
import pandas as pd
import os

# Get current working directory and set input, output folder
__current_directory = os.getcwd()
__input_folder = __current_directory + '/../Input'
__output_folder = __current_directory + '/../Output'

# Read input files
genome_tag = pd.read_csv(__input_folder + '/genome-tags.csv')
actor_info = pd.read_csv(__input_folder + '/imdb-actor-info.csv')
ml_movies = pd.read_csv(__input_folder + '/mlmovies.csv')
ml_ratings = pd.read_csv(__input_folder + '/mlratings.csv')
ml_tags = pd.read_csv(__input_folder + '/mltags.csv')
ml_users = pd.read_csv(__input_folder + '/mlusers.csv')
movie_actors = pd.read_csv(__input_folder + '/movie-actor.csv')

# Private members (variables)
__count = 0
__sum = 0
__a = 1
__b = 2


# Get tag weight by keeping oldest timestamp in DB as reference and get delta value in seconds for each row
def __get_tag_weight__(row):
    base_time = pd.to_datetime(oldest_timestamp)
    curr_time = pd.to_datetime(row['timestamp'])
    difference = curr_time - base_time
    return (difference.days * 24 * 60 * 60) + difference.seconds


# Normalize tag weight (1-2) based on below formula
# Formula ((val - A)*(b-a))/(B-A) + a
# [A, B] --> [a, b]
# A = 1, B = latest_timestamp delta in seconds, a = 0, b = 1
def __normalize_tag_weight__(row):
    global latest_timestamp
    return ((row['weight'] - 0) * (__b - __a)) / (latest_timestamp - 0) + __a


# Get actor weight by normalizing to a range of 1 - 2
def __get_actor_weight__(row):
    inverse_weight = highest_rank / row['actor_movie_rank']
    return (((inverse_weight - 1) * (__b - __a)) / highest_rank) + __a


# Save any data frame to a CSV file
def save_df(df, file_name):
    df.to_csv(__output_folder + '/' + file_name + '.csv', index=False, encoding='utf-8')


# Pre process ML Tags DB for convenience - Get normalized Tag Weight
ml_tags = ml_tags.sort_values(['timestamp'], ascending=False)
oldest_timestamp = ml_tags.iloc[-1].timestamp
ml_tags['weight'] = ml_tags.apply(lambda row: __get_tag_weight__(row), axis=1)
latest_timestamp = ml_tags.iloc[0].weight
ml_tags['weight'] = ml_tags.apply(lambda row: __normalize_tag_weight__(row), axis=1)


# Pre process Movie Actor DB for convenience - Get normalized Actor rank
highest_rank = movie_actors.sort_values('actor_movie_rank').iloc[-1].actor_movie_rank
movie_actors['actorweight'] = movie_actors.apply(lambda row: __get_actor_weight__(row), axis=1)
