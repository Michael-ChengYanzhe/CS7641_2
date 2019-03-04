import numpy as np
import pandas as pd
import time
import graphviz
import pydotplus

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, classification_report
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


def main():
	df = pd.read_csv("RandomHillClimb_LOG.csv")
	df = df.iloc[:-1]

	for i in range(df.index.max()):
		if i %50 != 0:
			df.drop([i],inplace=True)











	plt.figure()
	plt.title('Random Hill Climbing Learning Accuracy')
	plt.plot(df["acc_tst"], label="Testing Accuracy")
	plt.plot(df["acc_trg"], label="Training Accuracy")
	plt.xlabel("Iteration(*10)")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0.5,1)
	plt.grid()
	plt.savefig('../fig/RHC_acc.png')

	plt.figure()
	plt.title('Random Hill Climbing Learning Mean Squared Error')
	plt.plot(df["MSE_tst"], label="Testing Error")
	plt.plot(df["MSE_trg"], label="Training Error")
	plt.xlabel("Iteration(*10)")
	plt.ylabel("Error")
	plt.legend(loc="best")
	plt.ylim(0,0.2)
	plt.grid()
	plt.savefig('../fig/RHC_err.png')




def plot_learning_curve(clf, title, file_name, xTrain, yTrain,
                        n_jobs=None, train_sizes=np.linspace(0.01, 1.0, 50)):

	plt.figure()
	plt.title(title + 'Learning Curve')

	train_sizes, train_scores, test_scores = learning_curve(
		clf, xTrain, yTrain, cv=10, n_jobs=n_jobs, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="deepskyblue")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="orange")
	plt.plot(train_sizes, train_scores_mean, '^-', color="deepskyblue", label="Training score")
	plt.plot(train_sizes, test_scores_mean, '^-', color="orange", label="Cross-validation score")
	plt.xlabel("Training Size")
	plt.ylabel("Accuracy Score")
	plt.legend(loc="best")
	plt.savefig('ex1_' + file_name + 'curve.png')
	plt.tight_layout()


	return plt





if __name__ == "__main__":
	main()
	
