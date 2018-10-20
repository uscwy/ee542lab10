# copyright: yueshi@usc.edu
import pandas as pd 
import hashlib
import os 
from utils import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


from sklearn.feature_selection import SelectFromModel
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from utils import logger
#def lassoSelection(X,y,)

def lassoSelection(X_train, y_train, n):
	'''
	Lasso feature selection.  Select n features. 
	'''
	#lasso feature selection
	#print (X_train)
	return [4, 5, 8, 10, 13, 26, 49, 72, 78, 88, 90, 92, 93, 119, 141, 175, 180, 191, 194, 195, 201, 203, 204, 229, 232, 233, 239, 240, 245, 248, 249, 255, 264, 266, 270, 272, 273, 286, 296, 299, 302, 304, 305, 306, 309, 325, 327, 329, 332, 339, 344, 352, 381, 387, 406, 426, 429, 448, 458, 464, 470, 477, 482, 483, 492, 493, 495, 496, 497, 498, 500, 503, 505, 513, 514, 515, 532, 539, 544, 566, 588, 593, 595, 615, 623, 633, 638, 645, 646, 676, 677, 680, 692, 710, 764, 777, 784, 810, 813, 814, 834, 836, 847, 860, 880, 884, 888, 894, 900, 907, 911, 950, 957, 958, 969, 991, 996, 1004, 1038, 1041, 1048, 1063, 1072, 1078, 1079, 1091, 1102, 1111, 1135, 1141, 1148, 1152, 1232, 1251, 1267, 1274, 1289, 1305, 1316, 1337, 1342, 1362, 1363, 1364, 1369, 1376, 1378, 1387, 1395, 1402, 1406, 1410, 1412, 1447, 1461, 1475, 1487, 1504, 1507, 1509, 1516, 1524, 1544, 1546, 1560, 1584, 1588, 1638, 1644, 1665, 1677, 1695, 1717, 1720, 1722, 1733, 1747, 1750, 1771, 1786, 1791, 1834, 1843, 1848, 1859, 1860, 1872, 1874, 1875, 1879]
	clf = LassoCV()
	sfm = SelectFromModel(clf, threshold=0)
	sfm.fit(X_train, y_train)
	X_transform = sfm.transform(X_train)
	n_features = X_transform.shape[1]
	#return [5, 8, 10, 26, 78, 88, 92, 119, 141, 180, 191, 194, 195, 201, 204, 229, 232, 233, 240, 248, 249, 255, 270, 272, 273, 286, 296, 299, 302, 305, 306, 309, 325, 327, 329, 332, 339, 344, 352, 387, 429, 458, 477, 482, 483, 492, 495, 496, 497, 498, 500, 505, 515, 539, 588, 595, 615, 638, 645, 646, 680, 692, 764, 834, 847, 860, 880, 884, 888, 894, 911, 957, 958, 969, 991, 1041, 1063, 1072, 1078, 1079, 1091, 1102, 1111, 1135, 1148, 1152, 1251, 1274, 1289, 1305, 1316, 1337, 1342, 1362, 1363, 1364, 1369, 1395, 1410, 1447, 1461, 1504, 1509, 1544, 1560, 1644, 1665, 1677, 1717, 1720, 1722, 1750, 1771, 1791, 1834, 1848, 1860, 1872, 1874, 1875, 1879]
        #return [5, 8, 10, 26, 78, 88, 119, 141, 180, 191, 194, 195, 204, 232, 240, 248, 270, 273, 286, 296, 302, 305, 306, 325, 327, 329, 339, 344, 352, 477, 482, 492, 495, 496, 497, 498, 500, 515, 539, 588, 595, 615, 638, 646, 680, 847, 884, 911, 969, 991, 1072, 1078, 1102, 1148, 1251, 1289, 1316, 1337, 1342, 1362, 1364, 1369, 1395, 1461, 1504, 1509, 1644, 1665, 1677, 1717, 1722, 1750, 1771, 1791, 1834, 1848, 1872, 1875, 1879]
	#print(n_features)
	while n_features > n:
		sfm.threshold += 0.01
		X_transform = sfm.transform(X_train)
		n_features = X_transform.shape[1]
	features = [index for index,value in enumerate(sfm.get_support()) if value == True  ]
	logger.info("selected features are {}".format(features))
	return features


def specificity_score(y_true, y_predict):
	'''
	true_negative rate
	'''
	true_negative = len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[0]==pair[1] and pair[0]==0 ])
	real_negative = np.count_nonzero(y_true == 0)
	#real_negative = len(y_true) - sum(y_true)
	return true_negative / real_negative 

def model_fit_predict(X_train,X_test,y_train,y_test):

	np.random.seed(201)
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.svm import SVC
	from sklearn.metrics import precision_score
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import f1_score
	from sklearn.metrics import recall_score
	models = {
	#	'LogisticRegression': LogisticRegression(),
	#	'ExtraTreesClassifier': ExtraTreesClassifier(),
		'RandomForestClassifier': RandomForestClassifier(),
    	#'AdaBoostClassifier': AdaBoostClassifier(),
    	#'GradientBoostingClassifier': GradientBoostingClassifier(),
    	#'SVC': SVC()
	}
	tuned_parameters = {
	#	'LogisticRegression':{'C': [1, 10]},
	#	'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
		'RandomForestClassifier': { 'n_estimators': [100, 500] },
    	#'AdaBoostClassifier': { 'n_estimators': [16, 32] },
    	#'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    	#'SVC': {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
	}
	scores= {}
	for key in models:
		print (key)
		clf = GridSearchCV(models[key], tuned_parameters[key], scoring=None,  refit=True, cv=10)
		#print (clf.best_params_)
		clf.fit(X_train,y_train)
		y_test_predict = clf.predict(X_test)
		precision = precision_score(y_test, y_test_predict, average='micro')
		accuracy = accuracy_score(y_test, y_test_predict)
		f1 = f1_score(y_test, y_test_predict,average='micro')
		recall = recall_score(y_test, y_test_predict,average='micro')
		specificity = specificity_score(y_test, y_test_predict)
		scores[key] = [precision,accuracy,f1,recall,specificity]
	#print(scores)
	return scores



def draw(scores):
	'''
	draw scores.
	'''
	import matplotlib.pyplot as plt
	logger.info("scores are {}".format(scores))
	ax = plt.subplot(111)
	precisions = []
	accuracies =[]
	f1_scores = []
	recalls = []
	categories = []
	specificities = []
	N = len(scores)
	ind = np.arange(N)  # set the x locations for the groups
	width = 0.1        # the width of the bars
	for key in scores:
		categories.append(key)
		precisions.append(scores[key][0])
		accuracies.append(scores[key][1])
		f1_scores.append(scores[key][2])
		recalls.append(scores[key][3])
		specificities.append(scores[key][4])

	precision_bar = ax.bar(ind, precisions,width=0.1,color='b',align='center')
	accuracy_bar = ax.bar(ind+1*width, accuracies,width=0.1,color='g',align='center')
	f1_bar = ax.bar(ind+2*width, f1_scores,width=0.1,color='r',align='center')
	recall_bar = ax.bar(ind+3*width, recalls,width=0.1,color='y',align='center')
	specificity_bar = ax.bar(ind+4*width,specificities,width=0.1,color='purple',align='center')

	print(categories)
	ax.set_xticks(np.arange(N))
	ax.set_xticklabels(categories)
	ax.legend((precision_bar[0], accuracy_bar[0],f1_bar[0],recall_bar[0],specificity_bar[0]), ('precision', 'accuracy','f1','sensitivity','specificity'))
	ax.grid()
	plt.show()

if __name__ == '__main__':


	data_dir ="../data/"

	data_file = data_dir + "miRNA_matrix.csv"

	df = pd.read_csv(data_file)
	# print(df)
	y_data = df.pop('label').values

	df.pop('file_id')

	columns =df.columns
	#print (columns)
	X_data = df.values
	
	# split the data to train and test set
	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
	

	#standardize the data.
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	# check the distribution of tumor and normal sampels in traing and test data set.
	train_tumor_count = np.count_nonzero(y_train > 0)
	test_tumor_count = np.count_nonzero(y_test > 0)
	logger.info("Percentage of tumor cases in training set is {}".format(train_tumor_count/len(y_train)))
	logger.info("Percentage of tumor cases in test set is {}".format(test_tumor_count/len(y_test)))
	n = 200
	feaures_columns = lassoSelection(X_train, y_train, n)



	scores = model_fit_predict(X_train[:,feaures_columns],X_test[:,feaures_columns],y_train,y_test)

	draw(scores)
	#lasso cross validation
	# lassoreg = Lasso(random_state=0)
	# alphas = np.logspace(-4, -0.5, 30)
	# tuned_parameters = [{'alpha': alphas}]
	# n_fold = 10
	# clf = GridSearchCV(lassoreg,tuned_parameters,cv=10, refit = False)
	# clf.fit(X_train,y_train)




 




