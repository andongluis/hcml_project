import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def explainer(feature1, feature2):
    st = 'Your recommendation was '+determiner(feature1[1],feature2[1])+'most significantly impacted by '+ feature[0]
    pass

def determiner(sd1, sd2):
    close_choice = {
        'roughly': ['roughly', 'approximately', 'nearly', ''],
        'evenly': ['evenly', 'equally']
    }
    uneven_choice = {
        'most': ['most significantly', 'mostly'],
        'impacted': ['impacted', 'driven', 'affected']
        'second': ['']
    }
    if abs(sd1-sd2)<2:
        return random.choice(choice['roughly'])+' '+random.choice(choice['evenly'])
    else:


def provide_explanation(features_and_weights, weights):
    # Keep only non-zero weights
    weights = [w for w in weights if w != 0]

    # Calculate standard deviation and average for weights and pair 
    sd = np.std(weights)
    avg = np.average(weights)
    feature_from_sd = [(f[0], abs((f[1]-avg)/sd), 1 if f[1]>=0 else -1) for f in features_and_weights]

    explanation = explainer(feature_from_sd[0], feature_from_sd[1])

    return explanation
    
def highest_weight_features(weights, input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        features = lines[0].split(',')
        sample_movie = []
        for x in lines[10].split(','):
            try:
                sample_movie.append(float(x))
            except ValueError:
                sample_movie.append(0)
        print(len(sample_movie))

        # Remove features without weights
        to_remove = ["rating","movieId_x","movieId_y","userId"]
        for r in to_remove:
            idx = features.index(r)
            features.pop(idx)
            sample_movie.pop(idx)

        sample_movie = np.array(sample_movie)


        # Recalculated weights
        recalculated_weights = np.multiply(weights, np.array(sample_movie))
        
        # Get absolute values for all weights for sorting reasons
        weights_abs = [(abs(w),(True if w>=0 else False)) for w in recalculated_weights]

        # Match each weight with its feature
        features_and_weights = zip(features, weights_abs)

        # Sort and get the two features with highest magnitude
        features_and_weights_sorted = sorted(features_and_weights, key = lambda x: x[1][0], reverse = True)

        # Get the original sign of the weights
        top_two = ([(feature_and_weight[0], (feature_and_weight[1][0] if feature_and_weight[1][1] 
                    else -feature_and_weight[1][0])) for feature_and_weight in features_and_weights_sorted[:2]])

        return top_two

        

def main():
    INPUT_FILE = "features/3640_feature_vecs.csv"

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

    features = highest_weight_features(regressor.coef_, INPUT_FILE)
    provide_explanation(features, regressor.coef_)


if __name__=='__main__':
    main()