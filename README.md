# movie_reddit_recs

Two separate models have been trained for genre identification, one of which does not consider "Drama" as a genre label due to its overarching prominence in IMDb.

redditor_item.py: A class representing a Reddit user. Contains methods to access user's comment and submission history.

redd_dicts.py: Contains reference information about Reddit content, including the hottest subreddits and common Reddit acronyms.

genres_train_logreg.model: A logistical regression model generated using the library libshorttext (Hsiang-Fu Yu, Chia-Hua Ho, Yu-Chin Juan, 
Chih-Jen Lin, of National Taiwan University and University of Texas-Austin) by training on a set of 75,000 movie reviews.

test_revs.txt: A list of movie reviews on which logreg.model can be tested.

test_urls.txt: The URLS corresponding to the reviews in test_revs.txt.

genre_predict: Predictions made for the test set.

genre_predict_df: Predictions made for the test set when "Drama" is removed from the set of available labels. 

precision_recall.py: After using libshorttext to make predictions on the test data, compute the precision and recall rate of each type of label (27 in all)

confusion_matrix.xlsb: A confusion matrix M consisting of all available genres. M[i,j] is the number of instances with label i that were predicted to have label j. 

dramafritt_matrix.xlsb: Same as confusion_matrix.xlsb, but does not consider "Drama" as a label.

mood_reg.py: Trains linear regression models on data mapping movie synopses and tags to ratings ranging from 1 to 10 for 8 different moods. 
