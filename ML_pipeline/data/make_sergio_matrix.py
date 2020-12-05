import pandas as pd
import numpy as np
import itertools


def get_movie_df(file="movies.csv"):

    df = pd.read_csv(file)
    df["genres"] = df["genres"].apply(lambda row: row.split("|"))
    return df


def flatten_and_unique_list(list_of_lists):
    return list(set(itertools.chain(*list_of_lists)))


TAG_DF = pd.read_csv("genome-scores.csv")
names = pd.read_csv('genome-tags.csv')

TAG_NAME_DICT = dict(zip(names["tagId"], names["tag"]))

TAG_DF = TAG_DF.merge(names)
UNIQUE_TAGS = TAG_DF["tagId"].unique()
def get_tags_df(movies_list):
    return TAG_DF[TAG_DF['movieId'].isin(movies_list)]

UNIQUE_GENRES = list(set(itertools.chain(*get_movie_df()["genres"])))
def make_features_matrix(df, n=100):

    # Get user features from df
    # prep list of features that we will extract from user_df
    feature_list = []
    for tag in UNIQUE_TAGS:
        col_name = f"{TAG_NAME_DICT[tag]}_tag_av_rating"
        feature_list.append(col_name)
    
    for genre in UNIQUE_GENRES:
        col_name = f"{genre}_genre_av_rating"
        feature_list.append(col_name)

    feature_list.extend(["userId", "rating"])


    user_feats = df.iloc[[0]]
    user_feats = user_feats[feature_list]
    user_feats["rating"] = 0
    # user_feats["movieId"] = user_feats["movieId_x"]
    # user_feats = user_feats.drop("movieId_x")
    # user_feats["userId"] = df_.iloc["userId"]

    # Dummy columns for indicating if the movie is that genre
    genres_df = get_movie_df()
    genres_df = genres_df.head(n)
    genre_dummy_df = pd.get_dummies(genres_df["genres"].apply(pd.Series).stack(), prefix='', prefix_sep='').sum(level=0)
    genre_dummy_df = genre_dummy_df.T.reindex(UNIQUE_GENRES).T.fillna(0)


    # Dummy columns for indicating if the movie has that tag (weighted by relevance)
    tag_df = get_tags_df(genres_df["movieId"].unique())
    tag_dummy_df = tag_df.pivot(index='movieId', columns='tag', values='relevance')
    # tag_dummy_df = tag_dummy_df.set_index('movieId')
    tag_dummy_df = tag_dummy_df.reindex(index=genres_df['movieId'])
    tag_dummy_df = tag_dummy_df.reset_index()




    final_df = genre_dummy_df.merge(tag_dummy_df, left_index=True, right_index=True)
    final_df = final_df.merge(genres_df[["movieId"]], left_index=True, right_index=True)
    final_df = final_df.assign(**user_feats.iloc[0])

    final_df.fillna(0, inplace=True)

    print(final_df.shape)
    # Make csv with the feature vectors and the ratings
    final_df = final_df[list(df)]
    final_df.to_csv(f"sergio_feature_matrix.csv", index=False)


def main():

    df = pd.read_csv("3551_feature_vecs.csv")

    make_features_matrix(df)




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


