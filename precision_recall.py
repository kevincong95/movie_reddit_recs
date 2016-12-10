from __future__ import division
from libshorttext.analyzer import *
from libshorttext.classifier import *
from libshorttext.converter import *
import imdb
import pickle
import numpy as np

ia = imdb.IMDb()
revs = list(open('test_revs.txt'))
urls = list(open('test_urls.txt'))
#A list of movie ID's corresponding to movies with no genres listed under IMDb
bad_ids = ['0319477', '0091649', '0116056', '0192013', '0274509', '0118030', 
           '0111362', '0227912', '0311066', '0160395', '0261249', '0241498', 
           '0454746', '0368972', '0084567', '0326870', '0239058', '0337534', 
           '0159847', '0314984', '0120451', '0224156', '0319477', '0196279', 
           '0211716', '0192317', '0091649', '0116056', '0164211', '0192013']
analyzer = Analyzer('genres_train_logreg.model')
genre_dict = pickle.load(open("genre_dict.dict", "rb"))

#0 <= index <= 24999; from the corresponding line in urls, retrieve the 7-digit number corresponding to the movie ID
def get_id(index):
    url = urls[index]
    return fil_id(url)

def fil_id(url):
    return filter(str.isdigit, url)

#0 <= index <= 24999. 
#@param multiple: If false, returns (1,1) if the model Analyzer classified this particular review correctly, (0,1) if not,
#and (0,0) if the review corresponds to a movie with no listed genres.
#If true, checks the top n labels, where N is the number of genres listed under IMDb. Returns (M,N) where M is the number of
#top genres from Analyzer matching a genre in the IMDb listing.
def precision_recall(index, multiple = False):
    review = revs[index]
    imdbid = get_id(index)
    if imdbid in bad_ids:
        return 0,0
    if imdbid not in genre_dict:
        genres = ia.get_movie(imdbid).get('genre')
        if not genres:
            return 0,0
        else:
            genre_dict[imdbid] = map(str, genres)
    else:
        genres = genre_dict[imdbid]
    analyzer.analyze_single(review, amount = len(genres), output = 'tmp.txt')
    results = open('tmp.txt').readline().split()
    if multiple:
        correct = list(set(results).intersection(genres))
        return len(correct), len(genres)
    correct = 1 if results[0] in genres else 0
    return correct, 1

#For the first index reviews, runs precision-recall on each one of them and aggregate the results.
#@param multiple: See above method.
#@return: A float representing either the rate at which the top genre label identified matched IMDb, or the top N genre labels
#were identified. N is the number of genres listed under IMDb. 
def overall_accuracy(index = len(revs), multiple = False):
    correct = 0
    available = 0
    try:
        for i in range(index):
            hit, genres = precision_recall(i, multiple)
            correct += hit
            available += genres
    except:
        pickle.dump(genre_dict, open("genre_dict.dict", "wb"))
    finally:
        print(len(genre_dict))
        return correct, available, correct / available

#Creates an empty square matrix the size of the number of available genre labels.
def gen_matrix():
    n = len(analyzer.labels)
    mat = np.zeros((n,n))

result_matrix = gen_matrix()

#For the genres in targets, create a confusion matrix of classification accuracy.
#@param targets: A list of genre labels, length >= 2
#@return: A matrix M, where M_ij is the number of instances actually labeled targets[i] classified with label targets[j].
def confusion_matrix(targets):
    try:
        indices = [analyzer.labels.index(target) for target in targets]
    except ValueError:
        print("Unrecognized label.")
    n = len(targets)
    mat = np.zeros(n,n)
    for i in range(n):
        row = indices[i]
        for j in range(n):
            col = indices[j]
            mat[i][j] = result_matrix[row][col]
            return mat
