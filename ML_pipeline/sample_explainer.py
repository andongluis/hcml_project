import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# function to generate nlg explanation for the score using the impact from the top 2 impactful features
def explainer(score, feature1, feature2):
    feature1_z, feature1_sign, feature1 = feature1[1], feature1[2], feature1[0]
    feature2_z, feature2_sign, feature2 = feature2[1], feature2[2], feature2[0]
    score = 'This movie received a score of '+str(score)+' on a 1-5 scale.'
    what = 'Your recommendation was '+determiner(feature1,feature2, feature1_z, feature2_z)+'.'
    how = signs(feature1, feature2, feature1_sign, feature2_sign)
    return score + ' ' + what + ' ' + how

# check if the top 2 features are roughly equal
def close(z1, z2):
    return abs(z1-z2)<2

# check if the top feature impacted the score much more than the 2nd feature
def dominated(z1, z2):
    return abs(z1-z2)>10

# generate nlg explanation about whether the top 2 features increased or decreased the score
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
        return (cont_choice+' '+features[num]+' '+f_dirs[num]+(' the score' if random.randint(0,1)==1 else '')+', '+('' if contrast else 'while ')+
            features[(num+1)%2] + ' '+f_dirs[(num+1)%2])+' the score.'

# generate nlg explanation for the relative impact the top 2 features had on the score
def determiner(feature1, feature2, feature1_z, feature2_z):
    close_choice = {
        'roughly': ['roughly', 'approximately', 'nearly', ''],
        'evenly': ['evenly', 'equally']
    }
    most = ['most','most significantly', 'more']
    impact = {
        'impact': ['impacted', 'driven', 'affected','influenced'],
        'impacted': ['impacted', 'affected', 'influenced', 'drove'],
        'effect': ['effect', 'impact']
    }
    second = {
        'addition': []
        'second': ['second most', 'secondarily it']
    }
    magnitude = {
        'biggest': ['biggest','greatest','highest','most'],
        'bigger': ['a bigger','a greater','a higher', 'most of the']
    }
    dominated_choice = {
        'dominated': ['dominated', 'much more impacted', 'significantly more affected']
    }
    close = close(feature1_z, feature2_z)
    dominated = dominated(feature1_z, feature2_z)
    if close and feature1_z >= 2:
        return (random.choice(most)+' and '+random.choice(choice['roughly'])+' '+random.choice(choice['evenly'])+' 'random.choice(impact['impact'])
            +' by '+feature1+' and '+feature2)
        )
    elif close:
        return ('not significantly '+random.choice(impact['impact']+' by any feature, but '+feature1+' and '+feature2+' had the '+random.choice(magnitude['biggest'])+' ' 
            + random.choice(impact['effect'])+' on the score.')
    elif dominated:
        sig = feature2_z >= 2
        comma = random.randint(0,1)>0
        but_str = ['. But ', ', but ', '. ']
        return (random.choice(dominated_choice['dominated']) + ' by ' +feature1 + (random.choice(but_str) + 
            feature2 + ' also had significant '+random.choice(impact['effect']+'.') if sig else '. ' + feature2 + ' also ' + random.randint(impact['impacted']) + ' the score.')
    else:
        phrase1 = feature1 + ' ' + random.choice(impact['impacted']) + ' the score '+
        phrasing = [phrase1, phrase2]
        return (feature1 + ' ' + random.choice(impact['impacted']) + ' the score '+)
        




# calculate z-scores and create feature objects (holding feature name, z-score and weight sign)
# then calls explainer function
def provide_explanation(features_and_weights, weights):
    # Keep only non-zero weights
    weights = [w for w in weights if w != 0]

    # Calculate standard deviation and average for weights and pair 
    sd = np.std(weights)
    avg = np.average(weights)
    feature_from_z = [(f[0], abs((f[1]-avg)/sd), 1 if f[1]>=0 else -1) for f in features_and_weights]

    return explainer(feature_from_z, feature_from_z)
    
# get the 2 movies with the greatest impact on final score, together with their weight for a sample movie and the sign of their weights
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
        weights_abs = [(abs(w),w>=0) for w in recalculated_weights]

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