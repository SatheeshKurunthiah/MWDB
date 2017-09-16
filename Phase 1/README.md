# Pre-Requisites

* Python
* Pandas (You can install using pip <pip install pandas>)

# Run Application

Open cmd prompt and navigate to `Code\` folder

    * Task 1: Get all tags that define a given Actor

        Type `python MwdbPhase1.py print_actor_vector <actor-id> <model>`
        Eg: `python MwdbPhase1.py print_actor_vector 67729 tf` or `python MwdbPhase1.py print_actor_vector 67729 tf-idf`

    * Task 2: Get all tags that define a given Genre

        Type `python MwdbPhase1.py print_genre_vector <genre-name> <model>`
        Eg: `python MwdbPhase1.py print_genre_vector Action tf` or `python MwdbPhase1.py print_genre_vector Comedy tf-idf`
        Note `genre-name` is case sensitive. Use exact case as in DB

    * Task 3: Get all tags that define a given User

        Type `python MwdbPhase1.py print_genre_vector <user-id> <model>`
        Eg: `python MwdbPhase1.py print_user_vector 144 tf` or `python MwdbPhase1.py print_user_vector 144 tf-idf`

    * Task 4: Get all differentiating tags in two given Genres

        Type `python MwdbPhase1.py differentiate_genre <genre1> <genre2> <model>`
        Eg: `python MwdbPhase1.py differentiate_genre Action Comedy tf-idf-diff` or 
        `python MwdbPhase1.py differentiate_genre Action Comedy p-diff1` or
        `python MwdbPhase1.py differentiate_genre Action Comedy p-diff2`
