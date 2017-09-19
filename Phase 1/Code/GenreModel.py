from __future__ import division
import LoadData as Data
import math

# Task 2: Get all movies under given genre
# Parameters of combined tables of Ml-tags, ml-movies and genome-tags
__genreTableParameters = ['movieid', 'tagid', 'moviename', 'genres', 'timestamp', 'weight']
__genreTagTableParameters = __genreTableParameters + ['tag']
__genreTable = Data.pd.merge(Data.ml_tags, Data.ml_movies, on='movieid')[__genreTableParameters]
__genreTagTable = Data.pd.merge(Data.genome_tag, __genreTable, on='tagid')[__genreTagTableParameters]


# Returns all tags for a given movie set and sum of their weights
def __get_all_tags_in_genre__(movie_set):
    tags_list = []
    tags_weight = 0
    for index, row in movie_set.iterrows():
        records = __genreTagTable[__genreTagTable['movieid'] == row['movieid']]
        tags_weight += records['weight'].sum()
        tags_list = list(set(tags_list) | set(records.tag.unique().tolist()))

    return {'tags': tags_list, 'weight': tags_weight}


# Returns all genres in DB
def __get_all_genre_list__():
    return Data.pd.DataFrame(Data.ml_movies.genres.str.split('|').tolist()).stack().unique().tolist()


# Returns list of unique tags (time stamp weight added for multiple entries) in given movie set
# count of tags (based of tag timestamp) in movies of given genre / total count of tags in given genre
def __get_all_tag_weight(movies, genre):
    tags = __get_all_tags_in_genre__(movies)
    tag_list = {}
    for tag in tags['tags']:
        rec = __genreTagTable[(__genreTagTable['tag'] == tag) & (__genreTagTable['genres'].str.contains(genre))]
        for index, row in rec.iterrows():
            time_weight = round(row.weight / tags['weight'], 6) * 1000
            tag_count = len(tags['tags'])
            if tag_count > 0:
                if tag in tag_list:
                    tag_list[tag] += round(time_weight / tag_count, 6)
                else:
                    tag_list[tag] = round(time_weight / tag_count, 6)

    return tag_list


# Computes TF Values for each tag in movie and aggregates results of all movies under given genre and prints the result
def __get_tf_for_genre__(genre):
    movies = Data.ml_movies[Data.ml_movies['genres'].map(lambda genres: genre in genres)]
    all_tag_list = []
    tf_value = __get_all_tag_weight(movies, genre)

    for key, value in sorted(tf_value.items(), key=lambda x: x[1], reverse=True):
        all_tag_list.append({'tag': str(key), 'tfweight': str(round(value, 8))})
        print('  ' + str(key) + ', ' + str(round(value, 8)) + '')

    return all_tag_list


# Computes TF IDF values
# count of genres in DB / total genres count having given tag in DB
def __get_tfidf_for_genre__(genre):
    movies = Data.ml_movies[Data.ml_movies['genres'].map(lambda genres: genre in genres)]
    genre_count = len(__get_all_genre_list__())
    tf_value = __get_all_tag_weight(movies, genre)
    tag_list = {}
    all_tag_list = []
    for tag, weight in tf_value.iteritems():
        total_genres_with_tag = len(Data.pd.DataFrame(__genreTagTable[(__genreTagTable['tag'] == tag)]['genres'].str.split('|').tolist()).stack().unique())
        if total_genres_with_tag > 0:
            log_value = round(math.log((genre_count / total_genres_with_tag), 2), 8)
            tfidf_value = weight * log_value
            if tag in tag_list.keys():
                tag_list[tag] += tfidf_value
            else:
                tag_list[tag] = tfidf_value

    for key, value in sorted(tag_list.items(), key=lambda x: x[1], reverse=True):
        all_tag_list.append({'tag': key, 'tfidfweight': round(value, 8)})
        print('  ' + str(key) + ', ' + str(round(value, 8)) + '')

    return all_tag_list


# Calls appropriate method based on model
def get_movies_by_genre(genre, model):
    if model == 'tf':
        __get_tf_for_genre__(genre)
    if model == 'tf-idf':
        __get_tfidf_for_genre__(genre)


# Saves genre model in output folder
def process_all_genre():
    tf_list = []
    tf_idf_list = []
    genres = __get_all_genre_list__()
    for genre in genres:
        tf_value = __get_tf_for_genre__(genre)
        for entry in tf_value:
            entry_dict = {'genre': genre}
            for key, value in entry.iteritems():
                entry_dict[key] = value
            tf_list.append(entry_dict)
        tf_idf_value = __get_tfidf_for_genre__(genre)
        for entry in tf_idf_value:
            entry_dict = {'genre': genre}
            for key, value in entry.iteritems():
                entry_dict[key] = value
            tf_idf_list.append(entry_dict)

    tf_data_frame = Data.pd.DataFrame(tf_list)
    tf_idf_data_frame = Data.pd.DataFrame(tf_idf_list)
    tf_data_frame['tfidfweight'] = tf_data_frame.apply(
        lambda new_row: tf_idf_data_frame[tf_idf_data_frame['tag'] == new_row['tag']].iloc[0].tfidfweight, axis=1)
    Data.save_df(tf_data_frame, 'Genre-Model.csv')
