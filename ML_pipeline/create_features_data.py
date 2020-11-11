import pandas as pd
import numpy as np
import itertools

START_INDEX = 0
END_INDEX = -1


def make_tag_averages(user_df, tag_df, tag):

    tag_df = user_df.merge(tag_df[tag_df["tagId"] == tag])
    # get sum of ratings with tag tag (weighted by tag value)
    denom = tag_df["relevance"].sum()

    # get count of ratings with tag tag (weighted by tag value)
    numer = (tag_df["relevance"] * tag_df["rating"]).sum()

    # Average for rating of movie with that genre, minus that specific movie. Weighted by relevance of tag to movies
    av_list = (numer - tag_df["rating"]*tag_df["relevance"])/(denom - tag_df["relevance"])

    # av_list = av_list.replace([np.inf, -np.inf], np.nan)
    av_list.fillna(0, inplace=True)

    return av_list

def make_genre_averages(user_df, genre):

    genre_df = user_df[user_df['genres'].apply(lambda x: genre in x)]
    # Get sum of ratings with genre genre
    denom = genre_df.shape[0]
    
    # Get count of ratings with genre genre
    numer = genre_df["rating"].sum()

    # Average for ratings of movie with that genre, minus that specific movie
    av_list = (numer - genre_df["rating"])/(denom - 1)

    # av_list = av_list.replace([np.inf, -np.inf], np.nan)
    av_list.fillna(0, inplace=True)

    return av_list



def get_movie_df(file="ml-25m/movies.csv"):

    df = pd.read_csv(file)
    df["genres"] = df["genres"].apply(lambda row: row.split("|"))
    return df


def flatten_and_unique_list(list_of_lists):
    return list(set(itertools.chain(*list_of_lists)))


TAG_DF = pd.read_csv("ml-25m/genome-scores.csv")
names = pd.read_csv('ml-25m/genome-tags.csv')

TAG_NAME_DICT = dict(zip(names["tagId"], names["tag"]))

TAG_DF = TAG_DF.merge(names)
UNIQUE_TAGS = TAG_DF["tagId"].unique()
def get_tags_df(movies_list):
    return TAG_DF[TAG_DF['movieId'].isin(movies_list)]

UNIQUE_GENRES = list(set(itertools.chain(*get_movie_df()["genres"])))
def create_user_ratings(df, userId):

    # filter down to a user
    user_df = df[df["userId"] == userId]

    # Merge with movie genres
    user_df = user_df.merge(get_movie_df())

    # Get movie tags scores
    tag_df = get_tags_df(user_df["movieId"].unique())


    # Calculate tag rating averages for each tag
    tag_av_dict = {}
    for tag in UNIQUE_TAGS:
        col_name = f"{TAG_NAME_DICT[tag]}_tag_av_rating"
        tag_av_dict[col_name] = make_tag_averages(user_df, tag_df, tag)

    # Calculate genre rating averages for each genre
    genre_av_dict = {}
    genre_list = flatten_and_unique_list(user_df["genres"])
    
    for genre in UNIQUE_GENRES:
        col_name = f"{genre}_genre_av_rating"
        genre_av_dict[col_name] = make_genre_averages(user_df, genre)


    # Dummy columns for indicating if the movie is that genre
    genre_dummy_df = pd.get_dummies(user_df["genres"].apply(pd.Series).stack(), prefix='', prefix_sep='').sum(level=0)
    genre_dummy_df = genre_dummy_df.T.reindex(UNIQUE_GENRES).T.fillna(0)


    # Dummy columns for indicating if the movie has that tag (weighted by relevance)
    tag_dummy_df = tag_df.pivot(index='movieId', columns='tag', values='relevance')
    # tag_dummy_df = tag_dummy_df.set_index('movieId')
    tag_dummy_df = tag_dummy_df.reindex(index=user_df['movieId'])
    tag_dummy_df = tag_dummy_df.reset_index()


    # Column for user id
    user_ids = user_df["userId"]

    # Column for ratings
    ratings = user_df["rating"]

    # Column for movieId
    movie_ids = user_df["movieId"]
    
    # Put everythin together
    final_dict = tag_av_dict
    final_dict.update(genre_av_dict)
    final_dict["userId"] = user_ids
    final_dict["rating"] = ratings
    final_dict["movieId"] = movie_ids

    final_df = pd.DataFrame(final_dict)

    print(final_df.shape)
    print(genre_dummy_df.shape)
    print(tag_dummy_df.shape)

    final_df = final_df.merge(genre_dummy_df, left_index=True, right_index=True)

    final_df = final_df.merge(tag_dummy_df, left_index=True, right_index=True)

    print(final_df.shape)
    # Make csv with the feature vectors and the ratings
    final_df.to_csv(f"features/{userId}_feature_vecs.csv", index=False)




def main():

    df = pd.read_csv("ml-25m/ratings.csv")
    print(df.shape)
    num_users = len(df["userId"].unique())
    num_movies = len(df["movieId"].unique())

    print(num_users)
    print(num_movies)

    print( df.shape[0] / (num_users * num_movies))

    unique_users = df["userId"].unique()

    if END_INDEX == -1:
        END_INDEX = num_users

    for user in unique_users[START_INDEX:END_INDEX]:
        print(user)
        create_user_ratings(df, user)

    exit()




if __name__ == "__main__":
    main()






# for user in df["userId"].unique():
#   my_shape = df[df["userId"] == user].shape
#   my_dict = {"userId": user, "freqs": my_shape[0]}


# pd.DataFrame(user_freqs).to_csv("user_freqs.csv", index=False)



# movie_freqs = []
# print(len(df["movieId"].unique()))
# for movie in df["movieId"].unique():
#   my_shape = df[df["movieId"] == movie].shape
#   my_dict = {"movieId": movie, "freqs": my_shape[0]}





# pd.DataFrame(movie_freqs).to_csv("movie_freqs.csv", index=False)


