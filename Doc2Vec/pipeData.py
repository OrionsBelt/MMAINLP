# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 22:37:13 2019

@author: kishite
"""

import re
import pandas as pd

from os.path import join
from os import listdir, makedirs
from pathos.multiprocessing import ProcessingPool as Pool

from resources import Res
from pprint import pprint

class Pipe():

    def __init__(self):
        """
            Initalizes DataProcessing class with utilities and parallel processing
            Paras:
                None
            Returns:
                None
        """
        self.res = Res()
        self.pool = Pool()

    def getDescription(self, description):
    
        """
            Retuns list of cleaned description from dataframe list column
            Paras:
                summaries: list of work order desciptions
            Returns:
                summaries: list of cleaned descriptions
        """
        description = [re.sub(r"[^a-zA-Z]", " ", description[i].lower()) for i in range(len(description))]
        #print("Desc: ", description)
        #return list(self.pool.map(self.res.clean_text, description))
        return list(self.pool.map(self.res.clean_text, description))

    def createDataframe(self, description):
        """
            Creates dataframe class of cleaned descriptions.
            Paras:
                reviews: cleaned concated reviews
            Returns:
                ratings: ratings of the reviews
        """
        return pd.DataFrame({"descriptions": description})

    def ProcessData(self, column_names = ["Description_Document"]):
        """
            Runs DataProcessing class
            Paras:
                None
            Returns:_
                None
        """
         
        dataframe = self.res.loadData(r"./Final/Data", column_names)
        #print("D: ", dataframe["Description_Document"])
        description = self.getDescription(list(dataframe["Description_Document"]))
        #Sprint("des: ", description)
#        review = self.getReview(list(dataframe["reviewText"]))
#        reviews = self.utls.concate_columns(summaries, review)
#        rating = list(dataframe["overall"])
        return self.createDataframe(description)

if __name__ == "__main__":
    pd.set_option('display.max_colwidth', -1)
    Preprocessing = Pipe()
    dataframe = Preprocessing.ProcessData()
    print("df: ", dataframe)
    export_csv = dataframe.to_csv (r'C:\Users\kishite\Documents\Education\Queens\MMAI\MMAI891\Project\Ppython\Final\DataPre\bgis_vendorPre_words.csv', index = None, header=True)
  
  
  
