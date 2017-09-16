from __future__ import division
import LoadData as Data
import math

# Task 3: Get all movies watched by a user
# Parameters of combined ml-movies, ml-tags and genome-tags
__user_movies_parameters = ['userid', 'movieid', 'tagid', 'moviename', 'timestamp', 'weight']
__user_tag_parameters = __user_movies_parameters + ['tag']

__user_movie_table = Data.pd.merge(Data.ml_movies, Data.ml_tags, on='movieid')[__user_movies_parameters]
__user_tag_table = Data.pd.merge(__user_movie_table, Data.genome_tag, on='tagid')[__user_tag_parameters]


# Returns all movies watched by a user (Considers both rating and review and sends combined list)
def __get_movies_watch_by_user__(user_id):
    tags_record = __user_movie_table[__user_movie_table['userid'] == user_id].sort_values(['timestamp'], ascending=False)
    list1 = []
    for index, row in tags_record.iterrows():
        if row['movieid'] not in list1:
            list1.append(row['movieid'])

    rating_record = Data.ml_ratings[Data.ml_ratings['userid'] == user_id]
    list2 = []
    for index, row in rating_record.iterrows():
        if row['movieid'] not in list1:
            list2.append(row['movieid'])

    return list(set(list1) | set(list2))


# count of tags (based of tag timestamp) in given movie / total count of tags in given movie
def __compute_tf_tag_weight__(movie_id):
    records = __user_tag_table[__user_tag_table['movieid'] == movie_id].sort_values(['timestamp'], ascending=False)
    tag_list = {}
    for index, row in records.iterrows():
        if row['tag'] not in tag_list.keys():
            tag_list[row['tag']] = row.weight
        else:
            tag_list[row['tag']] += row.weight
    total_tag_weight = sum(tag_list.values())
    for tag, weight in tag_list.iteritems():
        tf_weight = weight / total_tag_weight
        tag_list[tag] = tf_weight / len(records)

    return tag_list


# Get all movies watched by user and aggregates tags in all movies and print TF value in descending order
def __get_tf_info__(user_id):
    movies_list = __get_movies_watch_by_user__(user_id)
    tag_list = {}
    all_tag_list = []
    for movie in movies_list:
        tf_list = __compute_tf_tag_weight__(movie)
        for key, value in tf_list.items():
            if key in tag_list:
                tag_list[key] += value
            else:
                tag_list[key] = value

    for key, value in sorted(tag_list.items(), key=lambda x: x[1], reverse=True):
        all_tag_list.append({'userid': user_id, 'tag': str(key), 'tfweight': str(value)})
        print('  <' + str(key) + ', ' + str(value) + '>')

    return all_tag_list


# Get all movies watched by user and aggregates tags in all movies and print TF IDF value in descending order
# count of movies in DB / count of tags (based of tag timestamp) of movies in DB
def __get_tfidf_info__(user_id):
    total_movies_count = len(Data.ml_movies)
    movies = __get_movies_watch_by_user__(user_id)
    tag_list = {}
    all_tag_list = []

    for movie in movies:
        tf_value = __compute_tf_tag_weight__(movie)
        for tag, weight in tf_value.iteritems():
            tag_id = __user_tag_table[__user_tag_table['tag'] == tag].iloc[0].tagid
            total_tag_count_in_all_movies = len(Data.ml_tags[Data.ml_tags['tagid'] == tag_id].groupby('movieid').apply(list))
            idf_value = math.log(total_movies_count/total_tag_count_in_all_movies, 2) * weight
            if tag in tag_list:
                tag_list[tag] += idf_value
            else:
                tag_list[tag] = idf_value

    for key, value in sorted(tag_list.items(), key=lambda x: x[1], reverse=True):
        all_tag_list.append({'userid': user_id, 'tag': key, 'tfidfweight': value})
        print('  <' + str(key) + ', ' + str(value) + '>')

    return all_tag_list


# Calls appropriate method based on model
def get_movies_by_user_id(user_id, model):
    if model == 'tf':
        __get_tf_info__(user_id)
    if model == 'tf-idf':
        __get_tfidf_info__(user_id)


# Save user model in output folder
def process_user_model():
    tf_list = []
    tf_idf_list = []
    for user_id in Data.ml_ratings['userid'].unique():
        tf_data = __get_tf_info__(user_id)
        for entry in tf_data:
            entry_dict = {}
            for key, value in entry.iteritems():
                entry_dict[key] = value
            tf_list.append(entry_dict)

        tf_idf_data = __get_tfidf_info__(user_id)
        for entry in tf_idf_data:
            entry_dict = {}
            for key, value in entry.iteritems():
                entry_dict[key] = value
            tf_idf_list.append(entry_dict)

    tf_data_frame = Data.pd.DataFrame(tf_list)
    if_idf_data_frame = Data.pd.DataFrame(tf_idf_list)
    tf_data_frame['tfidfweight'] = tf_data_frame.apply(
        lambda new_row: if_idf_data_frame[if_idf_data_frame['tag'] == new_row['tag']].iloc[0].tfidfweight, axis=1)

    Data.save_df(tf_data_frame, 'User-Model.csv')
