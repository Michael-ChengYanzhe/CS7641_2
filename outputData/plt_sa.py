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

	df1 = pd.read_csv("SA_1000000.0_0.99_LOG.csv")
	df1 = df1.iloc[:-1]
	for i in range(df1.index.max()):
		if i %70 != 0:
			df1.drop([i],inplace=True)
	df2 = pd.read_csv("SA_100000000.0_0.99_LOG.csv")
	df2 = df2.iloc[:-1]
	for i in range(df2.index.max()):
		if i %70 != 0:
			df2.drop([i],inplace=True)
	df3 = pd.read_csv("SA_10000000000.0_0.99_LOG.csv")
	df3 = df3.iloc[:-1]
	for i in range(df3.index.max()):
		if i %70 != 0:
			df3.drop([i],inplace=True)
	df4 = pd.read_csv("SA_1e+12_0.99_LOG.csv")
	df4 = df4.iloc[:-1]
	for i in range(df4.index.max()):
		if i %70 != 0:
			df4.drop([i],inplace=True)
	plt.figure()
	plt.title('Simulate Annealing Training Accuracy(Cooling Exp = 0.99)')
	plt.plot(df1["acc_trg"], label="T = 1e6")
	plt.plot(df2["acc_trg"], label="T = 1e8")
	plt.plot(df3["acc_trg"], label="T = 1e10")
	plt.plot(df4["acc_trg"], label="T = 1e12")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0.5,1)
	plt.grid()
	plt.savefig('../fig/SA_0.99_trg_acc.png')


	plt.figure()
	plt.title('Simulate Annealing Testing Accuracy(Cooling Exp = 0.99)')
	plt.plot(df1["acc_tst"], label="T = 1e6")
	plt.plot(df2["acc_tst"], label="T = 1e8")
	plt.plot(df3["acc_tst"], label="T = 1e10")
	plt.plot(df4["acc_tst"], label="T = 1e12")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0.5,1)
	plt.grid()
	plt.savefig('../fig/SA_0.99_tst_acc.png')

	plt.figure()
	plt.title('Simulate Annealing Training Error(Cooling Exp = 0.99)')
	plt.plot(df1["MSE_trg"], label="T = 1e6")
	plt.plot(df2["MSE_trg"], label="T = 1e8")
	plt.plot(df3["MSE_trg"], label="T = 1e10")
	plt.plot(df4["MSE_trg"], label="T = 1e12")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0,0.3)
	plt.grid()
	plt.savefig('../fig/SA_0.99_trg_err.png')

	plt.figure()
	plt.title('Simulate Annealing Training Error(Cooling Exp = 0.99)')
	plt.plot(df1["MSE_tst"], label="T = 1e6")
	plt.plot(df2["MSE_tst"], label="T = 1e8")
	plt.plot(df3["MSE_tst"], label="T = 1e10")
	plt.plot(df4["MSE_tst"], label="T = 1e12")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0,0.3)
	plt.grid()
	plt.savefig('../fig/SA_0.99_tst_err.png')




	df97 = pd.read_csv("SA_10000000000.0_0.97_LOG.csv")
	df97 = df97.iloc[:-1]
	for i in range(df97.index.max()):
		if i %70 != 0:
			df97.drop([i],inplace=True)

	df95 = pd.read_csv("SA_10000000000.0_0.95_LOG.csv")
	df95 = df95.iloc[:-1]
	for i in range(df95.index.max()):
		if i %70 != 0:
			df95.drop([i],inplace=True)

	df9 = pd.read_csv("SA_10000000000.0_0.9_LOG.csv")
	df9 = df9.iloc[:-1]
	for i in range(df9.index.max()):
		if i %70 != 0:
			df9.drop([i],inplace=True)

	df8 = pd.read_csv("SA_10000000000.0_0.8_LOG.csv")
	df8 = df8.iloc[:-1]
	for i in range(df8.index.max()):
		if i %70 != 0:
			df8.drop([i],inplace=True)
	df6 = pd.read_csv("SA_10000000000.0_0.6_LOG.csv")
	df6 = df6.iloc[:-1]
	for i in range(df6.index.max()):
		if i %70 != 0:
			df6.drop([i],inplace=True)
	df40 = pd.read_csv("SA_10000000000.0_0.4_LOG.csv")
	df40 = df40.iloc[:-1]
	for i in range(df40.index.max()):
		if i %70 != 0:
			df40.drop([i],inplace=True)

	plt.figure()
	plt.title('Simulate Annealing Training Accuracy(T = 1e10)')
	plt.plot(df3["acc_trg"], label="Cooling Exp = 0.99")
	plt.plot(df97["acc_trg"], label="Cooling Exp = 0.97")
	plt.plot(df95["acc_trg"], label="Cooling Exp = 0.95")
	plt.plot(df9["acc_trg"], label="Cooling Exp = 0.9")
	plt.plot(df8["acc_trg"], label="Cooling Exp = 0.8")
	plt.plot(df6["acc_trg"], label="Cooling Exp = 0.6")
	plt.plot(df40["acc_trg"], label="Cooling Exp = 0.4")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0.5,1)
	plt.grid()
	plt.savefig('../fig/SA_1e10_trg_acc.png')


	plt.figure()
	plt.title('Simulate Annealing Testing Accuracy(T = 1e10)')
	plt.plot(df3["acc_tst"], label="Cooling Exp = 0.99")
	plt.plot(df97["acc_tst"], label="Cooling Exp = 0.97")
	plt.plot(df95["acc_tst"], label="Cooling Exp = 0.95")
	plt.plot(df9["acc_tst"], label="Cooling Exp = 0.9")
	plt.plot(df8["acc_tst"], label="Cooling Exp = 0.8")
	plt.plot(df6["acc_tst"], label="Cooling Exp = 0.6")
	plt.plot(df40["acc_tst"], label="Cooling Exp = 0.4")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0.5,1)
	plt.grid()
	plt.savefig('../fig/SA_1e10_tst_acc.png')

	plt.figure()
	plt.title('Simulate Annealing Training Error(T = 1e10)')
	plt.plot(df3["MSE_trg"], label="Cooling Exp = 0.99")
	plt.plot(df97["MSE_trg"], label="Cooling Exp = 0.97")
	plt.plot(df95["MSE_trg"], label="Cooling Exp = 0.95")
	plt.plot(df9["MSE_trg"], label="Cooling Exp = 0.9")
	plt.plot(df8["MSE_trg"], label="Cooling Exp = 0.8")
	plt.plot(df6["MSE_trg"], label="Cooling Exp = 0.6")
	plt.plot(df40["MSE_trg"], label="Cooling Exp = 0.4")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0,0.3)
	plt.grid()
	plt.savefig('../fig/SA_1e10_trg_err.png')

	plt.figure()
	plt.title('Simulate Annealing Training Error(T = 1e10)')
	plt.plot(df3["MSE_tst"], label="Cooling Exp = 0.99")
	plt.plot(df97["MSE_tst"], label="Cooling Exp = 0.97")
	plt.plot(df95["MSE_tst"], label="Cooling Exp = 0.95")
	plt.plot(df9["MSE_tst"], label="Cooling Exp = 0.9")
	plt.plot(df8["MSE_tst"], label="Cooling Exp = 0.8")
	plt.plot(df6["MSE_tst"], label="Cooling Exp = 0.6")
	plt.plot(df40["MSE_tst"], label="Cooling Exp = 0.4")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0,0.3)
	plt.grid()
	plt.savefig('../fig/SA_1e10_tst_err.png')






	# df = pd.read_csv("RandomHillClimb_LOG.csv")
	# df = df.iloc[:-1]
	# plt.figure()
	# plt.title('Random Hill Climbing Learning Accuracy')
	# plt.plot(df["acc_tst"], label="Testing Accuracy")
	# plt.plot(df["acc_trg"], label="Training Accuracy")
	# plt.xlabel("Iteration")
	# plt.ylabel("Accuracy")
	# plt.legend(loc="best")
	# plt.ylim(0.5,1)
	# plt.grid()
	# plt.savefig('../fig/RHC_acc.png')

	# plt.figure()
	# plt.title('Random Hill Climbing Learning Mean Squared Error')
	# plt.plot(df["MSE_tst"], label="Testing Error")
	# plt.plot(df["MSE_trg"], label="Training Error")
	# plt.xlabel("Iteration")
	# plt.ylabel("Error")
	# plt.legend(loc="best")
	# plt.ylim(0,0.2)
	# plt.grid()
	# plt.savefig('../fig/RHC_err.png')

def read_and_plot(file, save):
	df = pd.read_csv(file)
	df = df.iloc[:-1]
	plt.figure()
	plt.title('Random Hill Climbing Learning Accuracy')
	plt.plot(df["acc_tst"], label="Testing Accuracy")
	plt.plot(df["acc_trg"], label="Training Accuracy")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0.5,1)
	plt.grid()
	plt.savefig('../fig/' + save + '_acc.png')

	plt.figure()
	plt.title('Random Hill Climbing Learning Mean Squared Error')
	plt.plot(df["MSE_tst"], label="Testing Error")
	plt.plot(df["MSE_trg"], label="Training Error")
	plt.xlabel("Iteration")
	plt.ylabel("Error")
	plt.legend(loc="best")
	plt.ylim(0,0.2)
	plt.grid()
	plt.savefig('../fig/' + save + '_err.png')




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
	
