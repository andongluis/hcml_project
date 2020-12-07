import csv
import pandas as pd
import numpy as np
import random
import custom_model
from sklearn.linear_model import LinearRegression

# function to generate nlg explanation for the score using the impact from the top 2 impactful features

class Explainer():

    def __init__(self):
        self.recommendations = None
        self.model = None
        self.feature_list = None


    def get(self):
        INPUT_FILE = "features/3640_feature_vecs.csv"

        if not self.model:
            self.model = self.get_regression_model(INPUT_FILE)

        NUM_RECOMMENDATIONS = 5

        self.recommendations = self.model.n_recommendations(NUM_RECOMMENDATIONS)

        for rec in self.recommendations:
            features, sample_movie = self.highest_weight_features(self.model.regressor.coef_, INPUT_FILE, rec['row'])
            rec['explanation'] = self.provide_explanation(features, self.model.regressor.coef_, self.model.regressor, sample_movie, rec['title'], rec['rating'])
            rec['top_feature'] = features[0][0]
            rec['next_feature'] = features[1][0]
            rec.pop('row')

        return {'recommendations': self.recommendations}

    def mask_at_index(self, feature):
        idx = self.feature_list.index(feature)
        self.model.mask_at_index(idx)
        print(self.model.regressor.coef_[idx])

    def unmask_at_index(self, feature):
        idx = self.feature_list.index(feature)
        self.model.unmask_at_index(idx)
        print(self.model.regressor.coef_[idx])

    def reset_parameters(self):
        self.model.reset_parameters()

    def get_regression_model(self, INPUT_FILE):
        model = custom_model.Custom_Model()
        model.train_with_file(INPUT_FILE)
        return model


    def explainer(self, movie_name, score, feature1, feature2):
        """Main NLG function, explains movie score

        Args:
            movie_name (string): title of the movie the prediction was made on
            score (float): regressor predicted score for movie
            feature1 (string): feature from the movie that most influenced the score
            feature2 (string): feature from the movie that second most influenced the score

        Returns:
            string: explanation for the score the movie got using the provided features
        """
        feature1_z, feature1_sign, feature1 = feature1[1], feature1[2], feature1[0]
        feature2_z, feature2_sign, feature2 = feature2[1], feature2[2], feature2[0]
        features = [feature1, feature2]
        feature1, feature2 = [self.ppfeature(feature) for feature in features]
        score = movie_name + ' received a score of '+str(round(score, 1))+' on a 1-5 scale.'
        what = 'This predicted rating was '+self.determiner(feature1, feature2, feature1_z, feature2_z)
        how = self.signs(feature1, feature2, feature1_sign, feature2_sign)
        return score + ' ' + what + ' ' + how

    # pretty print of genre or tag
    def ppfeature(self, feature):
        """Helper function to generate nice looking string from features

        Args:
            feature (string): feature from a movie

        Returns:
            string: Nicer looking output given feature
        """
        return ('your previous ratings for movies with tags containing the term \''+feature.split('_')[0] +'\'' if 'tag_av' in feature
            else 'your previous ratings movies in the genre \''+feature.split('_') +'\'' if 'av_rating' in feature
            else 'tags containing the term \''+feature +'\'' if not feature[0].isupper()
            else 'movies in the genre \''+feature +'\'')


    # check if the top 2 features are roughly equal
    def close(self, z1, z2):
        """Function to determine if difference between z-scores is small

        Args:
            z1 (float): (pseudo) z-score for feature 1
            z2 (float): (pseudo) z-score for feature 2

        Returns:
            boolean: z-scores are roughly equal
        """
        return abs(z1-z2)<2

    # check if the top feature impacted the score much more than the 2nd feature
    def dominated(self, z1, z2):
        """Function to determine if difference between z-scores is very large

        Args:
            z1 (float): (pseudo) z-score for feature 1
            z2 (float): (pseudo) z-score for feature 2

        Returns:
            boolean: one z-score is much larger than the other
        """
        return abs(z1-z2)>10

    # generate nlg explanation about whether the top 2 features increased or decreased the score
    def signs(self, feature1, feature2, feature_sign1, feature_sign2):
        """Function to explain whether feature1 and feature2 positively or negatively affected the score

        Args:
            feature1 (string): feature from the movie that most influenced the score
            feature2 (string): feature from the movie that second most influenced the score
            feature_sign1 (integer): 1 for positive feature weight or -1 for negative feature weight
            feature_sign2 (integer): 1 for positive feature weight or -1 for negative feature weight

        Returns:
            string: NLG explanation for how (positively or negatively) feature 1 and feature 2 affected the
                regression provided score
        """
        contrast = False
        f1_dir = ('increased' if feature_sign1 == 1 else 'decreased')
        f2_dir = ('increased' if feature_sign2 == 1 else 'decreased')
        features = [(feature1 + ' ' + f1_dir), (feature2 + ' ' + f2_dir)]
        num = random.randint(0,1)

        word_choice = {
            'choice1': ['taking the other features into account', 'adjusting for the other features'],
            'choice2': ['both ', '']
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
            return (cont_choice+' '+features[num]+(' the score' if random.randint(0,1)==1 else '')+', '+('' if contrast else 'while ')+
                features[(num+1)%2] +' the score.')

    # generate nlg explanation for the relative impact the top 2 features had on the score
    def determiner(self, feature1, feature2, feature1_z, feature2_z):
        """Explains the relative impact of the top 2 features compared to the others

        Args:
            feature1 (string): the feature with the greatest impact on the score
            feature2 (string): the feature with the second greatest impact on the score
            feature1_z (float): z-score for feature 1 (relative to all other features)
            feature2_z (float): z-score for feature 2 (relative to all other features)

        Returns:
            string: Explanation about the impact of the top 2 features in terms of magnitude
        """
        close_choice = {
            'roughly': ['roughly', 'approximately', 'nearly'],
            'evenly': ['evenly', 'equally']
        }
        most = ['most','most significantly', 'more']
        impact = {
            'driven': ['impacted', 'driven', 'affected','influenced'],
            'impacted': ['impacted', 'affected', 'influenced', 'drove'],
            'effect': ['effect', 'impact']
        }
        magnitude = {
            'biggest': ['biggest','greatest','highest','most'],
            'bigger': ['a bigger','a greater','a higher', 'most of the']
        }
        dominated_choice = {
            'dominated': ['dominated', 'much more impacted', 'significantly more affected']
        }
        is_close = self.close(feature1_z, feature2_z)
        is_dominated = self.dominated(feature1_z, feature2_z)
        but_str = ['. But ', ', but ', '. ']
        sig = feature2_z >= 2
        if is_close and feature1_z >= 2:
            return (random.choice(most)+' and '+random.choice(close_choice['roughly'])+' '+random.choice(close_choice['evenly'])+' '+random.choice(impact['driven'])
                +' by '+feature1+' and '+feature2)+'.'
        elif is_close:
            return ('not significantly '+random.choice(impact['driven'])+' by any feature, but '+feature1+' and '+feature2+' had the '+random.choice(magnitude['biggest'])+' ' 
                + random.choice(impact['effect'])+' on the score.')
        elif is_dominated:
            return (random.choice(dominated_choice['dominated']) + ' by ' +feature1 + random.choice(but_str) + 
                feature2 + ' also had significant '+ (random.choice(impact['effect'])+'. ') if sig else '. ' + feature2 + ' also ' + random.choice(impact['impacted']) + ' the score.')
        else:
            phrase = (random.choice(impact['driven']) + ' ' + random.choice(most) + ' by ' + 
                feature1 + random.choice(but_str) + feature2)
            phrase1 = ' '+random.choice(impact['impacted']) + ' the score ' + ('significantly ' if sig else '') + 'as well.'
            phrase2 = ' also had ' + ('significant ' if sig else '') + random.choice(impact['effect']) + '.' 
            phrasing = [phrase1, phrase2]
            return phrase + random.choice(phrasing)

    # calculate z-scores and create feature objects (holding feature name, z-score and weight sign)
    # then calls explainer function
    def provide_explanation(self, features_and_weights, weights, regressor, sample_movie, movie_name, score = None):
        """Superfunction that calculates the score, z-scores for every feature, and which features
            most heavily influenced the score for sample_movie and calls the explainer function 
            for an NLG explanation

        Args:
            features_and_weights (list): list of feature,weight tuples
            weights (list): list of weights (coefficients from the regression model)
            regressor (sklearn LinearRegression object): the trained regression model
            sample_movie (nd_array): feature vector for movie sample_movie
            movie_name (string): title of the movie explanation is made for
            score (float): predicted rating for movie

        Returns:
            string: Explanation returned by explainer function
        """
        # Keep only non-zero weights
        weights = [w for w in weights if w != 0]

        # Calculate standard deviation and average for weights and pair 
        sd = np.std(weights)
        avg = np.average(weights)
        feature_from_z = [(f[0], abs((f[1]-avg)/sd), 1 if f[1]>=0 else -1) for f in features_and_weights]
        if not score:
            score = regressor.predict(sample_movie.reshape(1,-1))
        print(score)

        return self.explainer(movie_name, score, *feature_from_z)
        
    # get the 2 features with the greatest impact on final score, together with their weight for a sample movie and the sign of their weights
    def highest_weight_features(self, weights, input_file, sample_movie = []):
        """Function to calculate the 2 features with the greatest impact on a score for movie sample_movie
            if no sample movie is given we collect one from the table

        Args:
            weights (list): feature vector of weights from the model
            input_file (string): input file location

        Returns:
            tuple: tuple consisting of a tuple with the top 2 features weights, and a weight vector for movie
                sample_movie
        """
        with open(input_file, 'r') as f:
            df = pd.read_csv(input_file)
            df = df.drop(["rating","movieId_x","movieId_y","userId"], axis=1)
            df = (df-df.min())/(df.max()-df.min())
            df = df.fillna(0)
            if len(sample_movie) == 0:
                sample_movie = np.array(df.iloc[7])
            features = df.columns.tolist()
            self.feature_list = features

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

            return top_two, sample_movie

# api.add_resource(Explainer, '/explainer')



# def main():
#     explainer = Explainer()
#     do = True
#     masked_features = []
#     while do:
#         explainer.get()
#         for r in explainer.recommendations:
#             print(r['explanation'])
#             print(r['top_feature'])
#             print(r['next_feature'])
#         ans = input('Do you want to mask a feature? y/n')
#         if ans == 'y':
#             feature = input('Which feature do you want to mask?')
#             explainer.mask_at_index(feature)
#             masked_features.append(feature)
#         if len(masked_features) > 0:
#             ans = input('Do you want to unmask a feature? y/n')
#             if ans == 'y':
#                 print('Currently masked features:')
#                 print(masked_features)
#                 feature = input('Which feature do you want to unmask?')
#                 explainer.unmask_at_index(feature)
#                 masked_features.remove(feature)
#         if len(masked_features) > 0:
#             ans = input('Do you want to reset the features? y/n')
#             if ans == 'y':
#                 explainer.reset_parameters()
            
#         ans = input('Do you want to continue? y/n')
#         if ans == 'n':
#             do = False
    
# #     print(recommendations)



# if __name__=='__main__':
#     main()