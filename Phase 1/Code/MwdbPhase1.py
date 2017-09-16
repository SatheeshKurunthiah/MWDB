import sys
import ActorModel as Actor
import GenreModel as Genre
import UserModel as User
import GenreUnionModel as GenreUnion

# This file checks for validity of command line arguments and calls appropriate method
__class = sys.argv[1]


def __actor_model__():
    if len(sys.argv) < 4:
        print('Not enough argument..')
        return
    __actor_id = int(sys.argv[2])
    __model = sys.argv[3]
    if __model is None or __actor_id is None:
        print('Not valid arguments')
        return
    Actor.get_actor_info(__actor_id, __model)


def __genre_model__():
    if len(sys.argv) < 4:
        print('Not enough argument..')
        return
    __genre_name = sys.argv[2]
    __model = sys.argv[3]
    if __model is None or __genre_name is None:
        print('Not valid arguments')
        return
    Genre.get_movies_by_genre(__genre_name, __model)


def __user_model__():
    if len(sys.argv) < 4:
        print('Not enough argument..')
        return
    __user_id = int(sys.argv[2])
    __model = sys.argv[3]
    if __model is None or __user_id is None:
        print('Not valid arguments')
        return
    User.get_movies_by_user_id(__user_id, __model)


def __diff_genre_model__():
    if len(sys.argv) < 5:
        print('Not enough argument..')
        return
    __genre_names = [str(sys.argv[2]), str(sys.argv[3])]
    __model = sys.argv[4]
    if __model is None or __genre_names is None or len(__genre_names) != 2:
        print('Not valid arguments')
        return
    GenreUnion.get_movies_by_genre(__genre_names[0], __genre_names[1], __model)

__options = {
    'print_actor_vector': __actor_model__,
    'print_genre_vector': __genre_model__,
    'print_user_vector': __user_model__,
    'differentiate_genre': __diff_genre_model__
}

__options[__class]()
