from __future__ import division
import LoadData as Data
import math

# Task 4: Get all movies under given two genres and get differentiating tags with their weights
# Combines parameters of ml-tags, ml-movies and genome-tags
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
        for index2, row2 in records.iterrows():
            tags_weight += row2.weight
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
# count of tags (based of tag timestamp) in movies of given genre / total count of tags in given genre
def __get_all_tag_weight(movies, genre, genre2):
    tags = __get_all_tags_in_genre__(movies)
    tag_list = {}
    for tag in tags['tags']:
        rec = __genreTagTable[(__genreTagTable['tag'] == tag) & (__genreTagTable['genres'].str.contains(genre))]
        if genre2 != '':
            rec = __genreTagTable[(__genreTagTable['tag'] == tag) & ((__genreTagTable['genres'].str.contains(genre)) | (__genreTagTable['genres'].str.contains(genre2)))]
        for index, row in rec.iterrows():
            time_weight = round(row.weight / tags['weight'], 6) * 1000
            tag_count = len(movies.groupby('movieid'))
            if tag_count > 0:
                if tag in tag_list:
                    tag_list[tag] += round(time_weight / tag_count, 6)
                else:
                    tag_list[tag] = round(time_weight / tag_count, 6)

    return tag_list


# Computes TF IDF DIFF value for given 2 genres
# total genre count / total genres count having given tag in Genre1 + Genre2
def __get_tfidf_for_genre__(movies1, movies2, genre1):
    tf_value = __get_all_tag_weight(movies1, genre1, '')
    movies_union = Data.pd.concat([movies1, movies2])
    genre_count = len(Data.pd.DataFrame(movies_union['genres'].str.split('|').tolist()).stack().unique())
    all_tag_list = []
    tag_list = {}
    for tag, weight in tf_value.iteritems():
        total_genres_with_tag = len(Data.pd.DataFrame(
            movies_union[(movies_union['tag'] == tag)]['genres'].str.split('|').tolist()).stack().unique())
        if total_genres_with_tag > 0:
            log_value = round(math.log((genre_count / total_genres_with_tag), 2), 8)
            tfidf_value = weight * log_value
            if tag in tag_list.keys():
                tag_list[tag] += tfidf_value
            else:
                tag_list[tag] = tfidf_value

    for key, value in sorted(tag_list.items(), key=lambda x: x[1], reverse=True):
        all_tag_list.append({'tag': key, 'tfidfweight': round(value, 8)})
        print('  <' + str(key) + ', ' + str(round(value, 8)) + '>')

    return all_tag_list


# Calculates P-DIFF 1
# R - no of movies in Genre1
# M - no of movies in Genre1 + Genre2
# r_dict - A Dictionary (Key: Tag, Value: no of movies in Genre1 containing this Tag)
# m_dict - A Dictionary (Key: Tag, Value: no of movies in Genre2 containing this Tag)
def __get_pdiff1_by_genre__(movie_set_1, movie_set_2):
    tags_in_genre1 = movie_set_1.groupby('moviename')['tag'].apply(list)
    tags_in_genre2 = movie_set_2.groupby('moviename')['tag'].apply(list)
    all_tag_list = []

    r_dict = {}
    for tags_g1 in tags_in_genre1:
        for a in tags_g1:
            if a not in r_dict:
                r_dict[a] = 0
    for key in r_dict:
        r_count = 0
        for tags_g1 in tags_in_genre1:
            if key in tags_g1:
                r_count += 1
        r_dict[key] = r_count

    m_dict = {}
    for tags_g1 in tags_in_genre1:
        for a in tags_g1:
            if a not in m_dict:
                m_dict[a] = 0
    for key in m_dict:
        m_count = 0
        for tags_g2 in tags_in_genre2:
            if key in tags_g2:
                m_count += 1
        m_dict[key] = m_count

    R = len(movie_set_1)
    M = len(movie_set_2) + len(movie_set_1)

    tag_weight = {}
    for tag in r_dict:
        if tag in r_dict and tag in m_dict:
            r = r_dict[tag]
            m = m_dict[tag]
            p1n = (r / (R - r))
            p1d = ((m - r) / (M - m - R + r))
            if p1d != 0:
                p2 = abs((r / R) - ((m - r) / (M - R)))
                pdiff = math.log(abs(p1n / p1d), 2) * p2
                tag_weight[tag] = pdiff

    for key, value in sorted(tag_weight.items(), key=lambda x: x[1], reverse=True):
        all_tag_list.append({'tag': key, 'pdiff1weight': round(value, 8)})
        print('  <' + str(key) + ', ' + str(value) + '>')

    return all_tag_list


# Calculates P-DIFF 2
# R - no of movies in Genre1
# M - no of movies in Genre1 + Genre2
# r_dict - A Dictionary (Key: Tag, Value: no of movies in Genre2 not containing this Tag)
# m_dict - A Dictionary (Key: Tag, Value: no of movies in Genre1 not containing this Tag)
def __get_pdiff2_by_genre__(movie_set_1, movie_set_2):
    tags_in_genre1 = movie_set_1.groupby('moviename')['tag'].apply(list)
    tags_in_genre2 = movie_set_2.groupby('moviename')['tag'].apply(list)
    all_tag_list = []

    r_dict = {}
    for tags_g1 in tags_in_genre1:
        for a in tags_g1:
            if a not in r_dict:
                r_dict[a] = 0
    for key in r_dict:
        r_count = 0
        for tags_g2 in tags_in_genre2:
            if key not in tags_g2:
                r_count += 1
        r_dict[key] = r_count

    m_dict = {}
    for tags_g1 in tags_in_genre1:
        for a in tags_g1:
            if a not in m_dict:
                m_dict[a] = 0
    for key in m_dict:
        m_count = 0
        for tags_g1 in tags_in_genre1:
            if key not in tags_g1:
                m_count += 1
        m_dict[key] = m_count

    R = len(movie_set_1)
    M = len(movie_set_2) + len(movie_set_1)

    tag_weight = {}
    for tag in r_dict:
        if tag in r_dict and tag in m_dict:
            r = r_dict[tag]
            m = m_dict[tag]
            p1n = (r / (R - r))
            p1d = ((m - r) / (M - m - R + r))
            p2 = abs((r / R) - ((m - r) / (M - R)))
            pdiff = math.log(abs(p1n / p1d), 2) * p2
            tag_weight[tag] = pdiff

    for key, value in sorted(tag_weight.items(), key=lambda x: x[1], reverse=True):
        all_tag_list.append({'tag': key, 'pdiff2weight': round(value, 8)})
        print('  <' + str(key) + ', ' + str(value) + '>')

    return all_tag_list


# Calls appropriate method based on model
def get_movies_by_genre(genre1, genre2, model):
    movies_set_1 = __genreTagTable[__genreTagTable['genres'].map(lambda genres: genre1 in genres)]
    movies_set_2 = __genreTagTable[__genreTagTable['genres'].map(lambda genres: genre2 in genres)]

    if model == 'tf-idf-diff':
        return __get_tfidf_for_genre__(movies_set_1, movies_set_2, genre1)
    if model == 'p-diff1':
        return __get_pdiff1_by_genre__(movies_set_1, movies_set_2)
    if model == 'p-diff2':
        return __get_pdiff2_by_genre__(movies_set_1, movies_set_2)


# Saves genre model in output folder
def process_all_genre():
    tf_idf_list = []
    p_diff1_list = []
    p_diff2_list = []
    movies_set_1 = __genreTagTable[__genreTagTable['genres'].map(lambda genres: 'Mystery' in genres)]
    movies_set_2 = __genreTagTable[__genreTagTable['genres'].map(lambda genres: 'Comedy' in genres)]
    tf_idf_value = __get_tfidf_for_genre__(movies_set_1, movies_set_2, 'Mystery')
    for entry in tf_idf_value:
        entry_dict = {'genre': 'Mystery'}
        for key, value in entry.iteritems():
            entry_dict[key] = value
        tf_idf_list.append(entry_dict)
    p_diff1_value = __get_pdiff1_by_genre__(movies_set_1, movies_set_2)
    for entry in p_diff1_value:
        entry_dict = {'genre': 'Mystery'}
        for key, value in entry.iteritems():
            entry_dict[key] = value
        p_diff1_list.append(entry_dict)
    p_diff2_value = __get_pdiff2_by_genre__(movies_set_1, movies_set_2)
    for entry in p_diff2_value:
        entry_dict = {'genre': 'Mystery'}
        for key, value in entry.iteritems():
            entry_dict[key] = value
        p_diff2_list.append(entry_dict)

    tf_idf_data_frame = Data.pd.DataFrame(tf_idf_list)
    p_diff1_data_frame = Data.pd.DataFrame(p_diff1_list)
    p_diff2_data_frame = Data.pd.DataFrame(p_diff2_list)
    c1 = Data.pd.merge(tf_idf_data_frame, p_diff1_data_frame, on='tag')
    final = Data.pd.merge(c1,p_diff2_data_frame, on='tag')
    final.columns = ['genre', 'tag', 'tfidfweight', 'y', 'pdiff1weight', 'z', 'pdiff2weight']
    Data.save_df(final[['genre', 'tag', 'tfidfweight', 'pdiff1weight', 'pdiff2weight']], 'Genre-Diff-Model.csv')
