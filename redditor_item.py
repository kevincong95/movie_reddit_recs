import praw
import numpy as np
import nltk
import redddicts
import matplotlib.pyplot as plt
import datetime as dt
from textblob import TextBlob as tb
from textblob.sentiments import NaiveBayesAnalyzer
from collections import Counter
from datetime import datetime
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import *

r = praw.Reddit('kingcong95') #this is my Reddit username
def_bins = [0,10,20,30,40,50,60,70,80,90,100] #most Reddit comments are under 100 words long
stoplist = stopwords.words('english')

#The *args param is a list of subreddit topics on which we may filter. By default, all comments/submissions are considered.
class RedditorItem():

    # @param username is just a string pertaining to the user account we wish to access
    def __init__(self, username):
        self.collector = r
        self.username = username
        self.account = r.get_redditor(username)
        self.submitted = list(self.account.get_submitted('new', 'all', limit = None))
        self.comments = list(self.account.get_comments('new', 'all', limit = None))

    def __str__(self):
        return self.username

    def equals(self, other):
        return self.username == other.username

    #return a list of praw.objects.Comment objects pertaining to this Redditor
    #given a praw.objects.Comment object c, to get c's entirety, call c.body
    def getComments(self, *args):
        if not args:
            return self.comments
        topicList = list(args)
        return list(filter(lambda x : x.subreddit._case_name in topicList, self.comments))

    #return a list of comments in string form; remove punctuation and hyperlinks, translate Reddit-specific acronyms 
    def fullComments(self, *args):
        comments = self.getComments(*args)
        bodies = [x.body for x in comments]
        return [clean(y) for y in bodies]

    #a praw.objects.Submission object works the same way as a Comment object; to get full title of an object s, call s.body
    def getSubmitted(self, *args):
        if not args:
            return self.submitted
        topicList = list(args)
        return list(filter(lambda x : x.subreddit._case_name in topicList, self.submitted))

    #again, free of hyperlinks and Reddit jargon
    def fullSubmittedTitles(self, *args):
        submitted = self.getSubmitted(*args)
        titles = [x.title for x in submitted]
        return [clean(y) for y in titles]

    #outputs a file consisting of this Redditor's comments
    def commentsCorpus(self, *args):
        comments = iter(self.fullComments(*args))
        f = open(str(self) + '_comments.txt', 'w')
        a = next(comments)
        while a:
            try:
                f.write(a + "\n")
            except UnicodeEncodeError:
                pass
            try:
                a = next(comments, None)
            except StopIteration:
                print('Comments saved to disk.')

    #plot a histogram separating this Redditor's comments by length in words
    def plotCommentLen(self, *args):
        comments = self.fullComments(*args)
        lengths = [len(c.split()) for c in comments]
        bins = def_bins
        maxLength = max(lengths)
        largest = 100
        while largest < maxLength:
            largest += 100
            bins.append(largest)
        plt.hist(lengths, bins, rwidth=0.8)
        plt.xlabel('Length of comment')
        plt.ylabel('Count')
        title = str(self) + "'s comment lengths"
        if args:
            topics = list(args)
            title += " for topics: "
            space = " "
            title += space.join(topics)
        plt.title(title)
        plt.show()

    #plots a line graph, with time as independent variable, of occurrences of this Redditor's comments
    def plotCommentTimes(self, *args):
        times = self.commentTimes(*args)
        times.reverse()
        commentsSoFar = list(range(1, len(times) + 1))
        dts = [dt.datetime.fromtimestamp(DT) for DT in times]
        plt.plot(dts, commentsSoFar, 'o-')
        plt.xlabel('Date')
        plt.ylabel('Number of comments')
        title = (str(self) + "'s comment history")
        if args:
            topics = list(args)
            title += " for topics: "
            space = " "
            title += space.join(topics)
        plt.title(title)
        plt.show()

    #outputs a file consisting of this Redditor's submitted post titles
    def subTitleCorpus(self, *args):
        titles = iter(self.fullSubmittedTitles(*args))
        f = open(str(self) + '_sub_titles.txt', 'w')
        a = next(titles)
        while a:
            try:
                f.write(a + "\n")
            except UnicodeEncodeError:
                pass
            a = next(titles, None)
        print('Submission titles saved to disk.')

    #returns a list consisting of this Redditor's submitted post bodies
    def fullSubmittedBodies(self, *args):
        submitted = self.getSubmitted(*args)
        bodies = []
        for x in submitted:
            if not x.selftext:
                bodies.append('  ')
            else:
                bodies.append(x.selftext)
        return [clean(y) for y in bodies]

    #outputs a file consisting of this Redditor's submitted post bodies
    def subBodyCorpus(self, *args):
        bodies = iter(self.fullSubmittedBodies(*args))
        f = open(str(self) + '_sub_bodies.txt', 'w')
        a = next(bodies)
        while a:
            try:
                f.write(a + "\n")
            except UnicodeEncodeError:
                pass
            a = next(bodies, None)
        print('Submission bodies saved to disk.')

    #Returns a list of topics to which this Redditor has made a post, sorted by count.
    def submittedTopics(self):
        topicList = [x.subreddit._case_name for x in self.getSubmitted()]
        return Counter(topicList)

    #Returns a list of topics to which this Redditor has written a comment, sorted by count.
    def commentedTopics(self):
        topicList = [x.subreddit._case_name for x in self.getComments()]
        return Counter(topicList)

    #Returns a list of UTC timestamps, most to least recent, at which this Redditor has made a post.
    def submitTimes(self, *args):
        return [x.created_utc for x in self.getSubmitted(*args)]

    #Returns a list of UTC timestamps, most to least recent, at which this Redditor has written a comment.
    def commentTimes(self, *args):
        return [x.created_utc for x in self.getComments(*args)]

    #For the given item (must be a Comment or Submission on Reddit), returns a recency rating.
    #@param item: Comment or Submission to inspect.
    #@return: A floating point value between 0 and 1; the closer to 1, the more recently this activity was recorded.
    def recencyWeight(self, item):
        try:
            myTime = item.created_utc
        except:
            print("Usage: The inputted item is not a Reddit comment or submission. Please try again.")
            return
        if isinstance(item, praw.objects.Comment):
            #1 day before the first observable activity so that the first activity does not have a weight of 0
            firstTime = self.commentTimes()[-1] - 86400 
        elif isinstance(item, praw.objects.Submission):
            firstTime = self.submitTimes()[-1] - 86400
        now = datetime.utcnow().timestamp()
        accountAge = now - firstTime
        return (myTime - firstTime) / accountAge

    #Returns a list of sentiments for each comment this Redditor has written.
    def commentSentiments(self, *args):
        comments = self.fullComments(*args)
        blobList = [tb(text, analyzer = NaiveBayesAnalyzer()) for text in comments]
        return [blob.sentiment for blob in blobList]

    #Returns a list of sentiments for the text portion of each post this Redditor has made.
    def submittedSentiments(self, *args):
        posts = self.fullSubmittedTitles(*args)
        blobList = [tb(text, analyzer = NaiveBayesAnalyzer()) for text in posts]
        return [blob.sentiment for blob in blobList]        

#Translate typical Reddit jargon and remove hyperlinks before processing comment.
#@param comment: Original comment from Reddit thread
#return: Comment with Reddit jargon (see redd_dicts.py) and hyperlinks removed
def clean(comment):
    if comment.isspace():
        return comment
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(comment)
    refined = []
    for word in words:
        x = word.lower()
        if x in redddicts.jargon:
            refined += redddicts.jargon[x].split()
        elif ("http" in x and "://" in x) or (x == "br"):
            continue
        else:
            refined.append(x)
    return " ".join(refined)

#Remove HTML tags before processing comment.
#@param comment: Original comment from Reddit thread
#return: Comment with HTML tags removed
def cleanhtml(raw_html):
    cleanr =re.compile('<.*?>')
    cleantext = re.sub(cleanr,' ', raw_html)
    return cleantext
