import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class Custom_Model(object):

	def __init__(self):
		super(Custom_Model, self).__init__()
		self.regressor = LinearRegression()
		self.original_parameters = None
		self.masked_parameters = None # 1 if masked, 0 otherwise
		self.n_features = 0

	def train(self, X, Y):
		self.regressor.fit(X, Y)
		self.original_parameters = self.regressor.coef_
		self.n_features = len(self.original_parameters)
		self.masked_parameters = [0] * self.n_features

	def train_with_file(self, file):

		df = pd.read_csv(file)
		Y = df["rating"]
		X = df.drop(["rating","movieId_x","movieId_y","userId"], axis=1)

		X.fillna(0, inplace=True)
		X = (X-X.min()) / (X.max()-X.min())
		X.fillna(0, inplace=True)

		self.train(X, Y)

	def mask_at_index(self, index):
		if index >= self.n_features: return
		self.regressor.coef_[index] = 0
		self.masked_parameters[index] = 1

	def unmask_at_index(self, index):
		if index >= self.n_features: return
		self.regressor.coef_[index] = self.original_parameters[index]
		self.masked_parameters[index] = 0

	def reset_parameters(self):
		self.regressor.coef_ = self.original_parameters
		self.masked_parameters = [0] * self.n_features

	def parameter_relevance(self):
		ascending = np.argsort(self.regressor.coef_)
		return np.flip(ascending)

	def n_most_relevant(self, n):
		return self.parameter_relevance()[:n]

	def predict(self, unseen_user):
		"""
		TODO: prediction function. Should return a list of movie ids or titles, either one is fine.
		"""
		pass

"""
model = Custom_Model()
model.train_with_file("features/3640_feature_vecs.csv")

# Print original values
print(model.original_parameters)
print(model.masked_parameters[:10])

# Mask first parameter and print
model.mask_at_index(0)
print(model.original_parameters)
print(model.masked_parameters[:10])

# Unmask first parameter and print
model.unmask_at_index(0)
print(model.original_parameters)
print(model.masked_parameters[:10])

# Check vector sizes
print(len(model.original_parameters))
print(len(model.masked_parameters))

# Check first index of parameter relevance returns max value
print(model.regressor.coef_.max())
max_index = model.parameter_relevance()[0]
print(model.regressor.coef_[max_index])

# Print 10 most relevant feature indexes
print(model.n_most_relevant(10))
"""