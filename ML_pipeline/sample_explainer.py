import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def explainer(score, feature1, feature2):
    feature1_sd = feature1[1]
    feature1_sign = feature1[2]
    feature1 = feature1[0]
    feature2_sd = feature2[1]
    feature2_sign = feature2[2]
    feature2 = feature2[0]
    score = 'This movie received a score of '+str(score)+' on a 1-5 scale.'
    what = 'Your recommendation was '+determiner(feature1,feature2, close(feature1_sd, feature2_sd))+'.'
    how = signs(feature1, feature2, feature1_sign, feature2_sign)
    pass

def close(sd1, sd2):
    return abs(sd1-sd2)<2

def signs(feature1, feature2, feature_sign1, feature_sign2):
    contrast = False
    f1_dir = ('increased' if feature_sign1 == 1 else 'decreased')
    f2_dir = ('increased' if feature_sign2 == 1 else 'decreased')
    f_dirs = [f1_dir, f2_dir]
    features = [(feature1 + ' ' + f1_dir), (feature2 + ' ' + f2_dir)]
    num = random.randint(0,1)

    word_choice = {
        'choice1': ['taking the other features into account', 'adjusting for the other features'],
        'choice2': ['both', '']
    }
    contrast_choice = {
        'contrast': ['While', 'Whereas', '']
    }
    cont_choice = random.choice(contrast_choice['contrast'])
    contrast = cont_choice != ''

    if feature_sign1 == feature_sign2:
        return ('After '+random.choice(word_choice['choice1'])+', '+random.choice(word_choice['choice2'])
            +feature1+' and '+feature2+' '+ f1_dir +' the score.')
    else:
        return (cont_choice+' '+features[num]+' '+f_dirs[num]+(' the score' if random.randint(0,1)==1 else '')+', '+('while ' if contrast else '')+
            features[(num+1)%2] + ' '+f_dirs[(num+1)%2])+' the score.'

def determiner(feature1, feature2, close):
    close_choice = {
        'roughly': ['roughly', 'approximately', 'nearly', ''],
        'evenly': ['evenly', 'equally']
    }
    most = {
        'most': ['most','most significantly', 'mostly', 'primarily']
    }
    impact = {
        'impact': ['impacted', 'driven', 'affected','influenced']
    }
    second = {
        'addition': []
        'second': ['second most', 'secondarily it']
    }
    if close:
        return (random.choice(most['most'])+' and '+random.choice(choice['roughly'])+' '+random.choice(choice['evenly'])+' 'random.choice(impact['impact'])
            +' by '+feature1+' and '+feature2)
        )
    else:
        return random.choice(uneven_choice['first'])+' '+random.choice(impact['impact'])


def provide_explanation(features_and_weights, weights):
    # Keep only non-zero weights
    weights = [w for w in weights if w != 0]

    # Calculate standard deviation and average for weights and pair 
    sd = np.std(weights)
    avg = np.average(weights)
    feature_from_sd = [(f[0], abs((f[1]-avg)/sd), 1 if f[1]>=0 else -1) for f in features_and_weights]

    explanation = explainer(feature_from_sd, feature_from_sd)

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