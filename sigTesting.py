import numpy as np
from numpy import genfromtxt
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
from scipy import stats
import csv


class SignificanceTesting(object):
	def __init__(self, filePath):
		self.filePath = filePath
		self.loadData()

	def loadData(self):
		self.models_scores = ['Baseline_R2', 'Baseline+Fusion_R2', 'Baseline+Ordering_R2',\
							  'Baseline+Ordering+Fusion_R2', 'Baseline_RSU4',	'Baseline+Fusion_RSU4' \
							  'Baseline+Ordering_RSU4', 'Baseline+Ordering+Fusion_RSU4']
		self.data = genfromtxt(self.filePath, delimiter=',')[1:].T
                self.data = np.asarray(self.data, dtype=np.float32)
                self.distributional_stats(self.models_scores,self.data)


        def distributional_stats(self,headers,data):
                dist_score = dict()
                for i,header_title in enumerate(headers):
                    dist_score[header_title]=[]
                    mean_score=np.mean(data[i])
                    median_score=np.median(data[i])
                    mode_score=stats.mode(data[i])
                    min_score=np.amin(data[i])
                    max_score=np.amax(data[i])
                    dist_score[header_title].append(mean_score)
                    dist_score[header_title].append(median_score)
                    dist_score[header_title].append(mode_score)
                    dist_score[header_title].append(min_score)
                    dist_score[header_title].append(max_score)
                print dist_score


	def ksTest(self, listA, listB):
		value, pvalue = ks_2samp(listA,listB)
		return pvalue

	def tTest(self, listA, listB):
		value, pvalue = ttest_ind(listA,listB)
		return pvalue

	def wilcoxonTest(self, listA, listB):
		T, pvalue = wilcoxon(listA,listB)
		return pvalue

	def writeOutput(self):
		resultsFile = open('SigTestResults.csv', 'w')
		w = 6
		h = 14
		resultsData = [[0 for x in range(w)] for y in range(h)] 
		resultsData[0] = ['metric','model', 'mean diff', 'P(T test)', 'P(wilcoxon test)' ,'P(ks test)']
		for row in xrange(1,7):
			resultsData[row][0]= 'ROUGE-2'
		for row in xrange(7,14):
			resultsData[row][0]= 'ROUGE-SU4'
		resultsData[1][1] = 'Baseline & Fusion'
		resultsData[2][1] = 'Baseline & Ordering'
		resultsData[3][1] = 'Fusion & Ordering'
		resultsData[4][1] = 'Baseline & Ordering+Fusion'
		resultsData[5][1] = 'Ordering & Ordering+Fusion'
		resultsData[6][1] = 'Fusion & Ordering+Fusion'
		resultsData[7][1] = 'Baseline & Fusion'
		resultsData[8][1] = 'Baseline & Ordering'
		resultsData[9][1] = 'Fusion & Ordering'
		resultsData[10][1] = 'Baseline & Ordering+Fusion'
		resultsData[11][1] = 'Ordering & Ordering+Fusion'
		resultsData[12][1] = 'Fusion & Ordering+Fusion'

		with resultsFile:
			writer = csv.writer(resultsFile)
			writer.writerows(resultsData)
		resultsFile.close()


	def boxingPlot(self):
		plt.figure()
		plt.boxplot(self.data.T)
		plt.show()
		


if __name__ == '__main__':
	filePath = "ROUGE_SCORES.csv"
	sigInstance = SignificanceTesting(filePath)
	#ks = sigInstance.ksTest()
	#t = sigInstance.tTest()
	#w = sigInstance.wilcoxonTest()
	pl = sigInstance.writeOutput()


