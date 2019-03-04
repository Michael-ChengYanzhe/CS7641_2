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

	df1 = pd.read_csv("GA_10_5_5_LOG.csv")
	df1 = df1.iloc[:-1]
	for i in range(df1.index.max()):
		if i %100 != 0:
			df1.drop([i],inplace=True)
	df2 = pd.read_csv("GA_50_5_5_LOG.csv")
	df2 = df2.iloc[:-1]
	for i in range(df1.index.max()):
		if i %100 != 0:
			df2.drop([i],inplace=True)
	df3 = pd.read_csv("GA_100_5_5_LOG.csv")
	df3 = df3.iloc[:-1]
	for i in range(df1.index.max()):
		if i %100 != 0:
			df3.drop([i],inplace=True)
	plt.figure()
	plt.title('Genetic Algorithm Training Accuracy(Mate,Mutate = 5,5)')
	plt.plot(df1["acc_trg"], label="P = 10")
	plt.plot(df2["acc_trg"], label="P = 50")
	plt.plot(df3["acc_trg"], label="P = 100")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0.3,1)
	plt.grid()
	plt.savefig('../fig/GA_55_trg_acc.png')


	plt.figure()
	plt.title('Simulate Annealing Testing Accuracy(Mate,Mutate = 5,5)')
	plt.plot(df1["acc_tst"], label="P = 10")
	plt.plot(df2["acc_tst"], label="P = 50")
	plt.plot(df3["acc_tst"], label="P = 100")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0.3,1)
	plt.grid()
	plt.savefig('../fig/GA_55_tst_acc.png')

	plt.figure()
	plt.title('Simulate Annealing Training Error(Mate,Mutate = 5,5)')
	plt.plot(df1["MSE_trg"], label="P = 10")
	plt.plot(df2["MSE_trg"], label="P = 50")
	plt.plot(df3["MSE_trg"], label="P = 100")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0,0.3)
	plt.grid()
	plt.savefig('../fig/GA_55_trg_err.png')

	plt.figure()
	plt.title('Simulate Annealing Testing Error(Mate,Mutate = 5,5)')
	plt.plot(df1["MSE_tst"], label="P = 10")
	plt.plot(df2["MSE_tst"], label="P = 50")
	plt.plot(df3["MSE_tst"], label="P = 100")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0,0.3)
	plt.grid()
	plt.savefig('../fig/GA_55_tst_err.png')

	plt.figure()
	plt.title('Simulate Annealing Training Time(Mate,Mutate = 5,5)')
	plt.plot(df1["elapsed"], label="P = 10")
	plt.plot(df2["elapsed"], label="P = 50")
	plt.plot(df3["elapsed"], label="P = 100")
	plt.xlabel("Iteration")
	plt.ylabel("Time Consuming")
	plt.legend(loc="best")
	# plt.ylim(0,0.3)
	plt.grid()
	plt.savefig('../fig/GA_55_time.png')


	df4 = pd.read_csv("GA_100_10_5_LOG.csv")
	df4 = df4.iloc[:-1]
	for i in range(df4.index.max()):
		if i %100 != 0:
			df4.drop([i],inplace=True)
	df5 = pd.read_csv("GA_100_10_10_LOG.csv")
	df5 = df5.iloc[:-1]
	for i in range(df5.index.max()):
		if i %100 != 0:
			df5.drop([i],inplace=True)
	df6 = pd.read_csv("GA_100_20_5_LOG.csv")
	df6 = df6.iloc[:-1]
	for i in range(df6.index.max()):
		if i %100 != 0:
			df6.drop([i],inplace=True)
	df7 = pd.read_csv("GA_100_20_20_LOG.csv")
	df7 = df7.iloc[:-1]
	for i in range(df7.index.max()):
		if i %100 != 0:
			df7.drop([i],inplace=True)


	plt.figure()
	plt.title('Genetic Algorithm Training Accuracy(P = 100)')
	plt.plot(df3["acc_trg"], label="Mate,Mutate = 5,5")
	plt.plot(df4["acc_trg"], label="Mate,Mutate = 10,5")
	plt.plot(df5["acc_trg"], label="Mate,Mutate = 10,10")
	plt.plot(df6["acc_trg"], label="Mate,Mutate = 20,5")
	plt.plot(df7["acc_trg"], label="Mate,Mutate = 20,20")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0.3,1)
	plt.grid()
	plt.savefig('../fig/GA_100_trg_acc.png')


	plt.figure()
	plt.title('Simulate Annealing Testing Accuracy(P = 100)')
	plt.plot(df3["acc_tst"], label="Mate,Mutate = 5,5")
	plt.plot(df4["acc_tst"], label="Mate,Mutate = 10,5")
	plt.plot(df5["acc_tst"], label="Mate,Mutate = 10,10")
	plt.plot(df6["acc_tst"], label="Mate,Mutate = 20,5")
	plt.plot(df7["acc_tst"], label="Mate,Mutate = 20,20")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0.3,1)
	plt.grid()
	plt.savefig('../fig/GA_100_tst_acc.png')

	plt.figure()
	plt.title('Simulate Annealing Training Error(P = 100)')
	plt.plot(df3["MSE_trg"], label="Mate,Mutate = 5,5")
	plt.plot(df4["MSE_trg"], label="Mate,Mutate = 10,5")
	plt.plot(df5["MSE_trg"], label="Mate,Mutate = 10,10")
	plt.plot(df6["MSE_trg"], label="Mate,Mutate = 20,5")
	plt.plot(df7["MSE_trg"], label="Mate,Mutate = 20,20")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0,0.3)
	plt.grid()
	plt.savefig('../fig/GA_100_trg_err.png')

	plt.figure()
	plt.title('Simulate Annealing Testing Error(P = 100)')
	plt.plot(df3["MSE_tst"], label="Mate,Mutate = 5,5")
	plt.plot(df4["MSE_tst"], label="Mate,Mutate = 10,5")
	plt.plot(df5["MSE_tst"], label="Mate,Mutate = 10,10")
	plt.plot(df6["MSE_tst"], label="Mate,Mutate = 20,5")
	plt.plot(df7["MSE_tst"], label="Mate,Mutate = 20,20")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0,0.3)
	plt.grid()
	plt.savefig('../fig/GA_100_tst_err.png')

	plt.figure()
	plt.title('Simulate Annealing Training Time with Different Parameter')
	# plt.plot(df3["elapsed"], label="P,Mate,Mutate = 100,5,5")
	plt.plot(df4["elapsed"], label="P,Mate,Mutate = 100,10,5")
	plt.plot(df5["elapsed"], label="P,Mate,Mutate = 100,10,10")
	plt.plot(df6["elapsed"], label="P,Mate,Mutate = 100,20,5")
	plt.plot(df7["elapsed"], label="P,Mate,Mutate = 100,20,20")
	plt.plot(df2["elapsed"], label="P,Mate,Mutate = 50,5,5")
	plt.plot(df1["elapsed"], label="P,Mate,Mutate = 10,5,5")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	# plt.ylim(0,0.3)
	plt.grid()
	plt.savefig('../fig/GA_100_time.png')

	






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
	
