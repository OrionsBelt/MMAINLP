# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 07:46:35 2019

@author: kishite
"""

import re
import pandas as pd
import numpy as np

from os.path import join

from nltk.tokenize import word_tokenize

from resources import Res
from pipeData import Pipe
from featureEng import Vec


class Data():
    
    """
    	Initalizes resource class with feature engineering
        as:
		None
        urns:
		None
	"""
    def __init__(self):

        self.res = Res()
        self.vector = Vec()
        self.pipe = Pipe()
        
    """
    Read csv
    """
    def readCorp(self, fname):
        df = pd.read_csv(fname, encoding = "iso-8859-1")
        return (df)
    
    """
			Creates training data set with labels appended to beginging of each label

			Paras:
				df: datafframe
			Returns:
				None
	""" 
    def createTrainingCorpus(self, dfVec, fname):
		
        df = self.readCorp(fname)
        df_feature = pd.concat([df,dfVec], axis=1)
        df_feature.fillna(0, inplace=True)
        df_feature.to_csv (r'C:\Users\kishite\Documents\Education\Queens\MMAI\MMAI891\Project\Ppython\Final\DataFeat\BGIS_Vendor_1hot_feature_LDA.csv', index = None, header=True)
        print("DF: ", df_feature)
        return(df_feature)
        
    """
			Runs DataSets class and creates training set

			Paras: 
				None
			Returns:
				None
	"""	
    def createSet(self,fnamePre, fname):
		
        #df = self.vector.matrix(fnamePre)
        df = self.readCorp(fnamePre)
        print("Pre:", df)
        df_feat=self.createTrainingCorpus(df, fname)
        return(df_feat)
        
if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    data = Data()
    df_F=data.createSet(r'Final\Data\BGIS_Vendor_scaled1hot_wo_description.csv', r'Final\LDA\LDA.csv')
    np.isnan(df_F.values.any())
        
        
        
        
    
    
