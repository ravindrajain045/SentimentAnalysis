from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

st = StanfordNERTagger('C:\\Users\\Ravindra Jain\\Downloads\\stanford-ner-2015-12-09 (1)\\stanford-ner-2015-12-09\\classifiers\\english.all.3class.distsim.crf.ser.gz',
					   'C:\\Users\\Ravindra Jain\\Downloads\\stanford-ner-2015-12-09 (1)\\stanford-ner-2015-12-09\\stanford-ner-3.6.0.jar',
					   encoding='utf-8')

text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'

tokenized_text = word_tokenize(text)
classified_text = st.tag(tokenized_text)

print(classified_text)
