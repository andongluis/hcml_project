
from sklearn.linear_model import LinearRegression

import pandas as pd


import numpy as np

INPUT_FILE = "features/3640_feature_vecs.csv"
#INPUT_FILE = "features/3742_feature_vecs.csv"

df = pd.read_csv(INPUT_FILE)

# Ratings
Y = df["rating"]

# Drop ratings, movie, user columns from traning. I made a booboo and left two columns with movieId when I was merging stuff
X = df.drop(["rating","movieId_x","movieId_y","userId"], axis=1)


# Fill NaN's will 0's, justin case
X.fillna(0, inplace=True)


# Mean normalization
# X=(X-X.mean())/X.std()

# Min max normalization (somethings std will give errors)
X=(X-X.min())/(X.max()-X.min())

X.fillna(0, inplace=True)

regressor = LinearRegression()
regressor.fit(X,Y)

weights = regressor.coef_

print(weights)
print(len(weights))


print(np.average(weights))
print(np.std(weights))

weights.sort()
print(weights[0:5])
print(weights[-5:])

regressor.coef_[0] = 123

print(weights)