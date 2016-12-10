#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import json
import numpy as np
import os
import redditor_item
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

'''
We have a sample set of approximately 200 Reddit users who are rated, on a scale of 1-10, 
for how strongly does their comment history project each of 8 different moods.
The json files described in line 20 refer to this set, one for each user.
The text files described on line 29 separate these ratings by mood.
'''
moods = ["nerve", "pulse", "humor", "romance", "clever", "deep", "cliche", "odd"]
json_files = [x for x in os.listdir(os.getcwd()) if x.find('.json') >= 0]

'''
For each mood, train linear regression model on corpus of user comments using
bag of words feature representation with TFIDF weighting
X = words, y = moods
'''
def linreg_model(mood):
    X = list(open('mood_corpus.txt'))
    y = np.loadtxt('mood_ratings_{0}.txt'.format(mood))
    tf_vect = TfidfVectorizer()
    regr = linear_model.LinearRegression()
    X_train_tf = tf_vect.fit_transform(X) 
    regr.fit(X_train_tf,y)
    joblib.dump(regr, '{0}_model.pkl'.format(mood))

'''
This requires a model for a predetermined mood to be trained and serialized using linreg_model.
@param docs: comment history for sample user
@param vectorizer: the TFIDF-weighted word matrix collected from training set and serialized
@param regr: fitted model, X = feature representation of dictionary, Y = users' mood ratings
in training set
@return: how strongly does the comment history project the predetermined mood?
'''
def predict(docs, vectorizer, regr):
    if isinstance(docs, str):
        docs = [docs] #must be in list form, treat string as 1 document
    docmat = vectorizer.transform(docs)
    return regr.predict(docmat)

'''
The code below was executed only once. It parsed an XML file containing
8 mood ratings for a subset of movies and produced a corpus where
each document consisted of the tags corresponding to a movie and was
labeled according to its predefined mood rating (once for each different
mood)
'''
##def get_mood_ratings(category): //produce corpora
##    mood_array = []
##    for plot in plots.findall("./table/outputs/output"):
##        imdbid = plot.get('id')
##        tags = plot.find('tags')
##        row_of_tags = ""
##        for tag in tags.findall('tag'):
##            row_of_tags += tag.text.replace("-", " ") + " "
##        example = plots.find('./table/examples/example[@output="{0}"]'.format(imdbid))
##        vital_info = example.text.split(';')
##        rating = float(vital_info[category])
##        try:
##                desc = plot.find('result').find('text').text
##                #descs.write(desc.replace("\n", " ") + "\n")
##                mood_array.append(rating)
##        except:
##                continue
##        print(imdbid)
##        #descs.write(row_of_tags + "\n")
##        mood_array.append(rating)
##        try:
##            j_output = movies[str(int(imdbid))]
##            synopsis = j_output['synopsis']
##        except:
##            continue
##        if synopsis:
##            #descs.write(synopsis.replace("\n", " ") + "\n")
##            mood_array.append(rating)
##    np.savetxt('{0}_movietwist_ratings.txt'.format(moods[category]), mood_array)
##    return len(mood_array)



