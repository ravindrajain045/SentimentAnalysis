import nltk
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.svm import  LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self,*classifers):
        self._classfiers = classifers
    def classify(self,features):
        votes = []
        for c in self._classfiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self,features):
        votes = []
        for c in self._classfiers:
            v = c.classify(features)
            votes.append(v)
        conf = votes.count(mode(votes))/len(votes)
        return conf


short_pos = open("C:\\Users\\Ravindra Jain\\Downloads\\short_reviews\\positive.txt","r").read()
short_neg = open("C:\\Users\Ravindra Jain\\Downloads\\short_reviews\\negative.txt","r").read()

#documents = []
#all_words = []

#allowed_word_types = ["J","R","V"]
#for r in short_pos.split('\n'):
#    documents.append((r , "pos"))
#    words = word_tokenize(r)
#    pos = nltk.pos_tag(words)
#    for w in pos:
#        if w[1][0] in allowed_word_types:
#            all_words.append(w[0].lower())

#for r in short_neg.split('\n'):
#    documents.append((r , "neg"))
#    words = word_tokenize(r)
#    pos = nltk.pos_tag(words)
#    for w in pos:
#        if w[1][0] in allowed_word_types:
#            all_words.append(w[0].lower())

#save_classifier1 = open("documents.pickle","wb")
#pickle.dump(documents , save_classifier1)
#save_classifier1.close()

classifier1 = open("documents.pickle","rb")
documents = pickle.load(classifier1)
classifier1.close()


#all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(50))
#print(all_words["sexy"])

#word_features = list(all_words.keys())[:6000]

#save_classifier3 = open("word_features.pickle","wb")
#pickle.dump(word_features , save_classifier3)
#save_classifier3.close()

classifier3_f = open("word_features.pickle","rb")
word_features = pickle.load(classifier3_f)
classifier3_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = w in words
    return features

#featuresets = [(find_features(rev) , category) for (rev , category) in documents]
#random.shuffle(featuresets)
#save_classifier2 = open("featuresets.pickle","wb")
#pickle.dump(featuresets , save_classifier2)
#save_classifier2.close()

classifier2_f = open("featuresets.pickle","rb")
featuresets = pickle.load(classifier2_f)
classifier2_f.close()
random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_test = featuresets[10000:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier_f = open("naivebayes1.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
#print("Naive Bayes Classifier Accuracy : ", (nltk.classify.accuracy(classifier , testing_test))*100)
#classifier.show_most_informative_features(16)
#save_classifier = open("naivebayes1.pickle","wb")
#pickle.dump(classifier , save_classifier)
#save_classifier.close()

#MNB_classifier = SklearnClassifier(MultinomialNB())
#MNB_classifier.train(training_set)
#print("Multinomial classifier accuracy :", nltk.classify.accuracy(MNB_classifier,testing_test))
#save_classifier4 = open("Multinomial.pickle","wb")
#pickle.dump(MNB_classifier , save_classifier4)
#save_classifier4.close()
classifier4 = open("Multinomial.pickle","rb")
MNB_classifier = pickle.load(classifier4)
classifier4.close()


#BN_classifier = SklearnClassifier(BernoulliNB())
#BN_classifier.train(training_set)
#print("Bernoulli classifier accuracy :",nltk.classify.accuracy(BN_classifier,testing_test))
#save_classifier5 = open("Bernoulli.pickle","wb")
#pickle.dump(BN_classifier , save_classifier5)
#save_classifier5.close()
classifier5 = open("Bernoulli.pickle","rb")
BN_classifier = pickle.load(classifier5)
classifier5.close()

#LR_classifier = SklearnClassifier(LogisticRegression())
#LR_classifier.train(training_set)
#print("LogisticRegression classifier accuracy :",nltk.classify.accuracy(LR_classifier,testing_test))
#save_classifier6 = open("LinearSVC.pickle","wb")
#pickle.dump(LR_classifier , save_classifier6)
#save_classifier6.close()
classifier6 = open("LinearSVC.pickle","rb")
LR_classifier = pickle.load(classifier6)
classifier6.close()

#SG_classifier = SklearnClassifier(SGDClassifier())
#SG_classifier.train(training_set)
#print("SGDClassifier classifier accuracy :",nltk.classify.accuracy(SG_classifier,testing_test))
#save_classifier7 = open("SG.pickle","wb")
#pickle.dump(SG_classifier , save_classifier7)
#save_classifier7.close()
classifier7 = open("SG.pickle","rb")
SG_classifier = pickle.load(classifier7)
classifier7.close()


#SVC_classifier = SklearnClassifier(SVC())
#SVC_classifier.train(training_set)
#print("SVC classifier accuracy :",nltk.classify.accuracy(SVC_classifier,testing_test))

#LSVC_classifier = SklearnClassifier(LinearSVC())
#LSVC_classifier.train(training_set)
#print("LinearSVC classifier accuracy :",nltk.classify.accuracy(LSVC_classifier,testing_test))
#save_classifier8 = open("LSVC.pickle","wb")
#pickle.dump(LSVC_classifier , save_classifier8)
#save_classifier8.close()
classifier8 = open("LSVC.pickle","rb")
LSVC_classifier = pickle.load(classifier8)
classifier8.close()

#NuSVC_classifier = SklearnClassifier(NuSVC())
#NuSVC_classifier.train(training_set)
#print("NuSVC classifier accuracy :",nltk.classify.accuracy(NuSVC_classifier,testing_test))
#save_classifier9 = open("NuSVC.pickle","wb")
#pickle.dump(NuSVC_classifier , save_classifier9)
#save_classifier9.close()
classifier9 = open("NuSVC.pickle","rb")
NuSVC_classifier = pickle.load(classifier9)
classifier9.close()

voted_classifier = VoteClassifier(classifier,MNB_classifier,BN_classifier,LR_classifier,SG_classifier,LSVC_classifier,NuSVC_classifier)
#print("Voted classifier accuracy :",nltk.classify.accuracy(voted_classifier,testing_test))

#print("category : ",voted_classifier.classify(testing_test[0][0]),"confidence : ",voted_classifier.confidence(testing_test[0][0]))

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
