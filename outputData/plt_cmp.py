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

	dfga = pd.read_csv("GA_100_20_20_LOG.csv")
	dfga = dfga.iloc[:-1]
	for i in range(dfga.index.max()):
		if i %100 != 0:
			dfga.drop([i],inplace=True)
	dfrhc = pd.read_csv("RandomHillClimb_LOG.csv")
	dfrhc = dfrhc.iloc[:-1]
	for i in range(dfrhc.index.max()):
		if i %50 != 0:
			dfrhc.drop([i],inplace=True)
	dfsa = pd.read_csv("SA_100000000.0_0.99_LOG.csv")
	dfsa = dfsa.iloc[:-1]
	for i in range(dfsa.index.max()):
		if i %70 != 0:
			dfsa.drop([i],inplace=True)

	plt.figure()
	plt.title('Testing Accuracy Comparing')
	plt.plot(dfga["acc_tst"], label="GA,P=100, Mutate=20,Mate=20")
	plt.plot(dfrhc["acc_tst"], label="RHC")
	plt.plot(dfsa["acc_tst"], label="SA,T=1e8,CE=0.97")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.ylim(0.3,1)
	plt.grid()
	plt.savefig('../fig/CMP_tst_acc.png')


	plt.figure()
	plt.title('Simulate Annealing Training Time(Mate,Mutate = 5,5)')
	plt.plot(dfga["elapsed"], label="GA,P=100, Mutate=20,Mate=20")
	plt.plot(dfrhc["elapsed"], label="RHC")
	plt.plot(dfsa["elapsed"], label="SA,T=1e8,CE=0.97")
	plt.xlabel("Iteration")
	plt.ylabel("Time Consuming")
	plt.legend(loc="best")
	# plt.ylim(0,0.3)
	plt.grid()
	plt.savefig('../fig/CMP_time.png')


	



if __name__ == "__main__":
	main()
	
