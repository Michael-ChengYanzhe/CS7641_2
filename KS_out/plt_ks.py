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

	dfga = pd.read_csv("KS_GA100_50_10_1_LOG.csv")
	dfga = dfga.iloc[:-1]
	for i in range(dfga.index.max()):
		if i %10 != 0:
			dfga.drop([i],inplace=True)
	dfrhc = pd.read_csv("KS_RHC_1_LOG.csv")
	dfrhc = dfrhc.iloc[:-1]
	for i in range(dfrhc.index.max()):
		if i %10 != 0:
			dfrhc.drop([i],inplace=True)
	dfsa = pd.read_csv("KS_SA0.75_1_LOG.csv")
	dfsa = dfsa.iloc[:-1]
	for i in range(dfsa.index.max()):
		if i %10 != 0:
			dfsa.drop([i],inplace=True)
	dfmm = pd.read_csv("KS_MIMIC100_50_0.3_1_LOG.csv")
	dfmm = dfmm.iloc[:-1]
	for i in range(dfmm.index.max()):
		if i %10 != 0:
			dfmm.drop([i],inplace=True)

	plt.figure()
	plt.title('Fitness" Comparing(Knapsack)')
	plt.plot(dfga["fitness"], label="GA")
	plt.plot(dfrhc["fitness"], label="RHC")
	plt.plot(dfsa["fitness"], label="SA")
	plt.plot(dfmm["fitness"], label="MIMIC")
	plt.xlabel("Iteration(*10)")
	plt.ylabel("Fitness")
	plt.legend(loc="best")
	# plt.ylim(0,1)
	plt.grid()
	plt.savefig('fig/KS_acc.png')


	plt.figure()
	plt.title('Training Time Comparing(Knapsack)')
	plt.plot(dfga["time"], label="GA")
	plt.plot(dfrhc["time"], label="RHC")
	plt.plot(dfsa["time"], label="SA")
	plt.plot(dfmm["time"], label="MIMIC")
	plt.xlabel("Iteration(*10)")
	plt.ylabel("Time Consuming")
	plt.legend(loc="best")
	# plt.ylim(0,0.3)
	plt.grid()
	plt.savefig('fig/KS_time.png')



	df3 = pd.read_csv("KS_MIMIC100_50_0.3_1_LOG.csv")
	df3 = df3.iloc[:-1]
	for i in range(df3.index.max()):
		if i %70 != 0:
			df3.drop([i],inplace=True)

	df1 = pd.read_csv("KS_MIMIC100_50_0.1_1_LOG.csv")
	df1 = df1.iloc[:-1]
	for i in range(df1.index.max()):
		if i %70 != 0:
			df1.drop([i],inplace=True)
	df5 = pd.read_csv("KS_MIMIC100_50_0.5_1_LOG.csv")
	df5 = df5.iloc[:-1]
	for i in range(df5.index.max()):
		if i %70 != 0:
			df5.drop([i],inplace=True)
	df7 = pd.read_csv("KS_MIMIC100_50_0.7_1_LOG.csv")
	df7 = df7.iloc[:-1]
	for i in range(df7.index.max()):
		if i %70 != 0:
			df7.drop([i],inplace=True)
	df9 = pd.read_csv("KS_MIMIC100_50_0.9_1_LOG.csv")
	df9 = df9.iloc[:-1]
	for i in range(df9.index.max()):
		if i %70 != 0:
			df9.drop([i],inplace=True)
	plt.figure()
	plt.title('MIMIC Fitness Comparison(Knapsack)')
	plt.plot(df1["fitness"], label="m = 1")
	plt.plot(df3["fitness"], label="m = 3")
	plt.plot(df5["fitness"], label="m = 5")
	plt.plot(df7["fitness"], label="m = 7")
	plt.plot(df9["fitness"], label="m = 9")
	plt.xlabel("Iteration")
	plt.ylabel("Fitness")
	plt.legend(loc="best")
	plt.grid()
	plt.savefig('fig/KS_M_Fit.png')

	plt.figure()
	plt.title('MIMIC Training Time Comparison(Knapsack)')
	plt.plot(df1["time"], label="m = 1")
	plt.plot(df3["time"], label="m = 3")
	plt.plot(df5["time"], label="m = 5")
	plt.plot(df7["time"], label="m = 7")
	plt.plot(df9["time"], label="m = 9")
	plt.xlabel("Iteration")
	plt.ylabel("Time")
	plt.legend(loc="best")
	plt.grid()
	plt.savefig('fig/KS_M_Time.png')









	



if __name__ == "__main__":
	main()
	
