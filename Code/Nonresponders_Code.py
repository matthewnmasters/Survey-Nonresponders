# -*- coding: utf-8 -*-
"""
Topic Modeling of qualitative responses from individuals that did not return CPS-3 Surveys.
Author: Matt Masters
Date: 2021.FEB.24
"""

#########################################################################################
#Set working directory                                                                  #
#########################################################################################
import os 
os.chdir('C:/Users/matth/Documents/GitHub/Survey-Nonresponders/Data')

#########################################################################################
#Import the .CSV files                                                                  #
#########################################################################################

import pandas as pd
reasons = pd.read_csv('Reasons.csv', encoding='utf-8-sig')
suggestions = pd.read_csv('Suggestions.csv', encoding='utf-8-sig')

#########################################################################################
#Make lowercase                                                                         #
#########################################################################################

reasons["Response"] = reasons["Response"].str.lower()
suggestions["Response"] = suggestions["Response"].str.lower()

#########################################################################################
#Make them lists instead of a dataframe                                                 #
#########################################################################################

reasonsList = list(reasons["Response"])
suggestionsList = list(suggestions["Response"])

#########################################################################################
#Part-of-Speech (POS) tag the words, then lemmatize (make root word) them               #
#########################################################################################

"""We may not actually want to lemmatize, since we will be making bigrams. Check """

from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer

lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

#Function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):
    #Tokenize the sentence and find the POS tag for each token
    nltk_tagged = pos_tag(tokenizer.tokenize(sentence))  
    #Tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #If there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #Else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


reasonsLemmatized = [lemmatize_sentence(doc) for doc in reasonsList]
suggestionsLemmatized = [lemmatize_sentence(doc) for doc in suggestionsList]


#Need to tokenize again

for idx in range(len(reasonsLemmatized)):
    reasonsLemmatized[idx] = tokenizer.tokenize(reasonsLemmatized[idx])
    
for idx in range(len(suggestionsLemmatized)):
    suggestionsLemmatized[idx] = tokenizer.tokenize(suggestionsLemmatized[idx])
    
"""Remove numbers? Remove words less than X number of characters? Remove 'um' and 'uh'? """

#########################################################################################
#Remove stop words, after downloading the stopwords data                                #
#nltk.download('stopwords')                                                             #
#########################################################################################

"""We may not want to remove stop words..."""

from nltk.corpus import stopwords

reasonsNoStop = [[token for token in reply if token not in stopwords.words('english')] for reply in reasonsLemmatized]
suggestionsNoStop = [[token for token in reply if token not in stopwords.words('english')] for reply in suggestionsLemmatized]

#########################################################################################
#Compute Bigrams                                                                        #
#########################################################################################

"""May want trigrams, and with stopwords added back in"""

from gensim.models import Phrases

reasonsBigram = list(Phrases(reasonsNoStop, min_count=5)[reasonsNoStop])
suggestionsBigram = list(Phrases(suggestionsNoStop, min_count=5)[suggestionsNoStop])

#########################################################################################
#Create a dictionary and corpus                                                         #
#########################################################################################

"""Can filter how often a word must appear or how many replies it must be in"""

from gensim.corpora import Dictionary

reasonsDictionary = Dictionary(reasonsBigram)
suggestionsDictionary = Dictionary(suggestionsBigram)

reasonsCorpus = [reasonsDictionary.doc2bow(doc) for doc in reasonsBigram]
suggestionsCorpus = [suggestionsDictionary.doc2bow(doc) for doc in suggestionsBigram]

# Let's see how many tokens and documents we have to train on
print('Number of unique tokens: %d' % len(reasonsDictionary))
print('Number of documents: %d' % len(reasonsCorpus))

print('Number of unique tokens: %d' % len(suggestionsDictionary))
print('Number of documents: %d' % len(suggestionsCorpus))

#########################################################################################
#Do a montecarlo simulation to see about how many topics we think we have in reasons    #
#########################################################################################


from gensim.models import LdaModel
from gensim.models import CoherenceModel


# Set training parameters.
chunksize = 811
passes = 100
iterations = 10000
eval_every = None  # evaluate perplexity

# Make a index to word dictionary.
temp = reasonsDictionary[0]  # This is only to "load" the dictionary.
id2word = reasonsDictionary.id2token

#montecarlo showed 3 topics as being a decent spot for c_v coherence
for i in range(2,51,1):
    model = LdaModel(
        corpus=reasonsCorpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=i,
        passes=passes,
        eval_every=eval_every
        )

# Calculate coherence
    coherence_model_lda = CoherenceModel(model=model, texts=reasonsBigram, dictionary=reasonsDictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Num Topics: ', i, 'coherence is: ', coherence_lda)
    
    
#########################################################################################
#Now do suggestions                                                                     #
#########################################################################################

# Set training parameters.
chunksize = 1541
passes = 100
iterations = 10000
eval_every = None  # evaluate perplexity


# Make a index to word dictionary.
temp = suggestionsDictionary[0]  # This is only to "load" the dictionary.
id2word = suggestionsDictionary.id2token

#montecarlo showed 8 topics as being a decent spot for c_v coherence
for i in range(2,51,1):
    model = LdaModel(
        corpus=suggestionsCorpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=i,
        passes=passes,
        eval_every=eval_every
        )

# Calculate coherence
    coherence_model_lda = CoherenceModel(model=model, texts=suggestionsBigram, dictionary=suggestionsDictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Num Topics: ', i, 'coherence is: ', coherence_lda)
    
################################################################################################
#Now run the reasons model with one of the identified "sweet spots" and then get topic words   #
################################################################################################

chunksize = 811
passes = 1000
iterations = 100000
eval_every = None  # evaluate perplexity
num_topics = 3

# Make a index to word dictionary.
temp = reasonsDictionary[0]  # This is only to "load" the dictionary.
id2word = reasonsDictionary.id2token

#montecarlo showed 3 topics as being a decent spot for c_v coherence

reasonsModel = LdaModel(
    corpus=reasonsCorpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
    )

# Calculate coherence
coherence_model_lda = CoherenceModel(model=reasonsModel, texts=reasonsBigram, dictionary=reasonsDictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Num Topics: ', num_topics, 'coherence is: ', coherence_lda)

reasonsModel.save('C:/Users/matth/Documents/GitHub/Survey-Nonresponders/Data/reasonsModel_save')
reasonsModel = LdaModel.load('C:/Users/matth/Documents/GitHub/Survey-Nonresponders/Data/reasonsModel_save')

for idx, topic in reasonsModel.print_topics(-1,50):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")
    
################################################################################################
#Now run the suggestions with one of the identified "sweet spots" and then get topic words     #
################################################################################################

chunksize = 1541
passes = 1000
iterations = 100000
eval_every = None  # evaluate perplexity
num_topics = 8

# Make a index to word dictionary.
temp = suggestionsDictionary[0]  # This is only to "load" the dictionary.
id2word = suggestionsDictionary.id2token

#montecarlo showed 8 topics as being a decent spot for c_v coherence

suggestionsModel = LdaModel(
    corpus=suggestionsCorpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
    )

# Calculate coherence
coherence_model_lda = CoherenceModel(model=suggestionsModel, texts=suggestionsBigram, dictionary=suggestionsDictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Num Topics: ', num_topics, 'coherence is: ', coherence_lda)

suggestionsModel.save('C:/Users/matth/Documents/GitHub/Survey-Nonresponders/Data/suggestionsModel_save')
suggestionsModel = LdaModel.load('C:/Users/matth/Documents/GitHub/Survey-Nonresponders/Data/suggestionsModel_save')

for idx, topic in suggestionsModel.print_topics(-1,50):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")
    
#########################################################################################
#Create transformed corpuses (corpi?)                                                   #
#########################################################################################

txReasons = pd.DataFrame(reasonsModel.get_document_topics(reasonsCorpus))
txSuggestions = pd.DataFrame(suggestionsModel.get_document_topics(suggestionsCorpus))

#Now merge with the original data
txReasons["Response"] = reasonsList
txSuggestions["Response"] = suggestionsList

#Split up the tuples into separate columns
txReasons["TopicA"], txReasons["PercA"] = txReasons[0].str
txReasons["TopicB"], txReasons["PercB"] = txReasons[1].str
txReasons["TopicC"], txReasons["PercC"] = txReasons[2].str

txSuggestions["TopicA"], txSuggestions["PercA"] = txSuggestions[0].str
txSuggestions["TopicB"], txSuggestions["PercB"] = txSuggestions[1].str
txSuggestions["TopicC"], txSuggestions["PercC"] = txSuggestions[2].str
txSuggestions["TopicD"], txSuggestions["PercD"] = txSuggestions[3].str
txSuggestions["TopicE"], txSuggestions["PercE"] = txSuggestions[4].str
txSuggestions["TopicF"], txSuggestions["PercF"] = txSuggestions[5].str
txSuggestions["TopicG"], txSuggestions["PercG"] = txSuggestions[6].str
txSuggestions["TopicH"], txSuggestions["PercH"] = txSuggestions[7].str

#remove the original tuples
txReasons = txReasons[txReasons.columns[3:10]]
txSuggestions = txSuggestions[txSuggestions.columns[8:25]]

#########################################################################################
#Write to disk. Manipulation and categorizing can be done easier in R                   #
#########################################################################################

txReasons.to_csv("modeledReasons.csv", index=False, encoding='utf-8-sig')
txSuggestions.to_csv("modeledSuggestions.csv", index=False, encoding='utf-8-sig')
