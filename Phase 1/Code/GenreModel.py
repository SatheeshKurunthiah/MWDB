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
        for index2, row2 in records.iterrows():
            if row2['tag'] not in tags_list:
                tags_list.append(row2['tag'])

    return {'tags': tags_list, 'weight': tags_weight}


# Returns all genres in DB
def __get_all_genre_list__():
    genre_list = []
    for index, row in Data.ml_movies.iterrows():
        for genre in row['genres'].split('|'):
            if genre not in genre_list:
                genre_list.append(genre)

    return genre_list


# Returns list of unique tags (time stamp weight added for multiple entries) in given movie set
# count of tags (based of tag timestamp) in movies of given genre / total count of movies in given genre
def __get_all_tag_weight(movies, count):
    tags = __get_all_tags_in_genre__(movies)
    tag_list = {}
    for tag in tags['tags']:
        rec = __genreTagTable[__genreTagTable['tag'] == tag]
        for index, row in rec.iterrows():
            time_weight = round(row.weight / tags['weight'], 6) * 1000
            if tag in tag_list:
                tag_list[tag] += round(time_weight / count, 6)
            else:
                tag_list[tag] = round(time_weight / count, 6)

    return tag_list


# Computes TF Values for each tag in movie and aggregates results of all movies under given genre and prints the result
def __get_tf_for_genre__(genre):
    movies = Data.ml_movies[Data.ml_movies['genres'].map(lambda genres: genre in genres)]
    all_tag_list = []
    tf_value = __get_all_tag_weight(movies, len(movies.groupby('movieid')))

    for key, value in sorted(tf_value.items(), key=lambda x: x[1], reverse=True):
        all_tag_list.append({'tag': str(key), 'tfweight': str(round(value, 8))})
        print('  <' + str(key) + ', ' + str(round(value, 8)) + '>')

    return all_tag_list


# Computes TF IDF values
# count of movies in DB / count of tags (based of tag timestamp) in movies in DB
def __get_tfidf_for_genre__(genre):
    movies = Data.ml_movies[Data.ml_movies['genres'].map(lambda genres: genre in genres)]
    movies_count = len(Data.ml_movies.groupby('movieid'))
    all_tag_count = len(__get_all_tag_weight(Data.ml_movies, movies_count))
    tf_value = __get_all_tag_weight(movies, len(movies))
    tag_list = {}
    all_tag_list = []
    log_value = round(math.log((movies_count / all_tag_count), 2), 8)
    for tag, weight in tf_value.iteritems():
        tfidf_value = weight * log_value
        if tag in tag_list.keys():
            tag_list[tag] += tfidf_value
        else:
            tag_list[tag] = tfidf_value

    for key, value in sorted(tag_list.items(), key=lambda x: x[1], reverse=True):
        all_tag_list.append({'tag': key, 'tfidfweight': round(value, 8)})
        print('  <' + str(key) + ', ' + str(round(value, 8)) + '>')

    return all_tag_list


# Calls appropriate method based on model
def get_movies_by_genre(genre, model):
    if model == 'tf':
        __get_tf_for_genre__(genre)
    if model == 'tf-idf':
        __get_tfidf_for_genre__(genre)


# Saves genre model in output folder
def process_all_genre():
    genre_list = []
    genres = __get_all_genre_list__()

    for genre in genres:
        tf_value = __get_tf_for_genre__(genre)
        tf_idf_value = __get_tfidf_for_genre__(genre)
        genre_list.append({'genre': genre, 'tfweight': tf_value, 'tfidfweight': tf_idf_value})

    frame = Data.pd.DataFrame(genre_list)
    Data.save_df(frame, 'Genre-Model.csv')
