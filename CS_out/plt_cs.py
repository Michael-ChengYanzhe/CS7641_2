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

	dfga = pd.read_csv("CS_GA100_30_30_1_LOG.csv")
	dfga = dfga.iloc[:-1]
	for i in range(dfga.index.max()):
		if i %20 != 0:
			dfga.drop([i],inplace=True)
	dfrhc = pd.read_csv("CS_RHC_1_LOG.csv")
	dfrhc = dfrhc.iloc[:-1]
	for i in range(dfrhc.index.max()):
		if i %50 != 0:
			dfrhc.drop([i],inplace=True)
	dfsa = pd.read_csv("CS_SA0.55_1_LOG.csv")
	dfsa = dfsa.iloc[:-1]
	for i in range(dfsa.index.max()):
		if i %50 != 0:
			dfsa.drop([i],inplace=True)
	dfmm = pd.read_csv("CS_MIMIC100_50_0.9_1_LOG.csv")
	dfmm = dfmm.iloc[1:-1]
	for i in range(2,dfmm.index.max()):
		if i %50 != 0:
			dfmm.drop([i],inplace=True)

	plt.figure()
	plt.title('Fitness" Comparing(Continuous Peaks)')
	plt.plot(dfga["fitness"], label="GA")
	plt.plot(dfrhc["fitness"], label="RHC")
	plt.plot(dfsa["fitness"], label="SA")
	plt.plot(dfmm["fitness"], label="MIMIC")
	plt.xlabel("Iteration(*10)")
	plt.ylabel("Fitness")
	plt.legend(loc="best")
	# plt.ylim(0,1)
	plt.grid()
	plt.savefig('fig/CP_acc.png')


	plt.figure()
	plt.title('Training Time Comparing(Continuous Peaks)')
	plt.plot(dfga["time"], label="GA")
	plt.plot(dfrhc["time"], label="RHC")
	plt.plot(dfsa["time"], label="SA")
	plt.plot(dfmm["time"], label="MIMIC")
	plt.xlabel("Iteration(*10)")
	plt.ylabel("Time Consuming")
	plt.legend(loc="best")
	# plt.ylim(0,0.3)
	plt.grid()
	plt.savefig('fig/CP_time.png')


	



if __name__ == "__main__":
	main()
	
