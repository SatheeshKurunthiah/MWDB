from __future__ import division
import LoadData as Data
import math

# Task 1: Get all movies done by actor
# Parameters of combined tables of Movie-actor, ml-tags and genome-tags
__actorTableParameters = ['actorid', 'moviename', 'movieid', 'actor_movie_rank', 'actorweight']
__tagTableParameters = ['movieid', 'tagid', 'tag', 'timestamp', 'weight']
__actorTagTableParameters = __actorTableParameters + ['tagid', 'tag', 'timestamp', 'weight']

__actorTable = Data.pd.merge(Data.movie_actors, Data.ml_movies, on='movieid')[__actorTableParameters]
__tagTable = Data.pd.merge(Data.ml_tags, Data.genome_tag, on='tagid')[__tagTableParameters]
__actorTagTable = Data.pd.merge(__actorTable, __tagTable, on='movieid')[__actorTagTableParameters]


# Returns all movies done by actor
def __get_movies_by_actor__(actor_id):
    return __actorTable[__actorTable['actorid'] == actor_id]


# count of tags (based of tag timestamp and actor rank) in given movie / total count of tags in given movie
def __compute_tf_tag_weight__(actor_id, movie_id):
    records = __actorTagTable[(__actorTagTable['actorid'] == actor_id) &
                              (__actorTagTable['movieid'] == movie_id)].sort_values(['timestamp'], ascending=False)
    tag_list = {}
    count = len(records)
    for index, row in records.iterrows():
        weight = row.weight * row['actorweight']
        if row['tag'] not in tag_list.keys():
            tag_list[row['tag']] = weight
        else:
            tag_list[row['tag']] += weight

    total_tag_weight = sum(tag_list.values())
    for tag, weight in tag_list.iteritems():
        tf_weight = (weight / total_tag_weight)
        tag_list[tag] = tf_weight / count

    return tag_list


# Computes TF Values for each tag in movie and aggregates results of all movies done by actor and prints the result
def __get_tf_info__(actor_id):
    movies = __get_movies_by_actor__(actor_id)
    all_tag_list = []
    tag_list = {}
    m_count = 0
    for index, row in movies.iterrows():
        m_count += 1
        tf_list = __compute_tf_tag_weight__(actor_id, row['movieid'])
        for key, value in tf_list.items():
            if key in tag_list:
                tag_list[key] += value
            else:
                tag_list[key] = value

    for key, value in sorted(tag_list.items(), key=lambda x: x[1], reverse=True):
        all_tag_list.append({'actorid': actor_id, 'tag': str(key), 'tfweight': str(value)})
        print('  <' + str(key) + ', ' + str(value) + '>')

    return all_tag_list


# Computes TF IDF values
# count of actors in DB / count of actors associated with given tag in DB
def __get_tfidf_info__(actor_id):
    total_actors_count = len(Data.actor_info['actorid'].unique())
    movies = __get_movies_by_actor__(actor_id)
    tag_list = {}
    all_tag_list = []

    for index, row in movies.iterrows():
        tf_value = __compute_tf_tag_weight__(actor_id, row['movieid'])
        for tag, weight in tf_value.iteritems():
            tag_id = __tagTable[__tagTable['tag'] == tag].iloc[0].tagid
            total_actors_with_given_tag = len(__actorTagTable[__actorTagTable['tagid'] == tag_id].groupby('actorid'))
            if total_actors_with_given_tag > 0:
                idf_value = math.log(total_actors_count/total_actors_with_given_tag, 2) * weight
                if tag in tag_list:
                    tag_list[tag] += idf_value
                else:
                    tag_list[tag] = idf_value

    for key, value in sorted(tag_list.items(), key=lambda x: x[1], reverse=True):
        all_tag_list.append({'actorid': actor_id, 'tag': key, 'tfidfweight': value})
        print('  ' + str(key) + ', ' + str(value) + '')

    return all_tag_list


# Calls appropriate method based on model
def get_actor_info(actor_id, model):
    if model == 'tf':
        __get_tf_info__(actor_id)
    if model == 'tf-idf':
        __get_tfidf_info__(actor_id)


# Saves actor model in output folder
def processactormodel():
    tf_list = []
    tf_idf_list = []
    for index, row in Data.actor_info.iterrows():
        tf_data = __get_tf_info__(row['actorid'])
        for entry in tf_data:
            entry_dict = {}
            for key, value in entry.iteritems():
                entry_dict[key] = value
            tf_list.append(entry_dict)

        tf_idf_data = __get_tfidf_info__(row['actorid'])
        for entry in tf_idf_data:
            entry_dict = {}
            for key, value in entry.iteritems():
                entry_dict[key] = value
            tf_idf_list.append(entry_dict)

    tf_data_frame = Data.pd.DataFrame(tf_list)
    tf_idf_data_frame = Data.pd.DataFrame(tf_idf_list)
    tf_data_frame['tfidfweight'] = tf_data_frame.apply(
        lambda new_row: tf_idf_data_frame[tf_idf_data_frame['tag'] == new_row['tag']].iloc[0].tfidfweight, axis=1)

    Data.save_df(tf_data_frame, 'Actor-Model.csv')
