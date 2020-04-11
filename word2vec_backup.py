# Python program to generate word vectors using Word2Vec 
  
# importing all necessary modules 
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
import nltk
from sklearn.decomposition import PCA
warnings.filterwarnings(action = 'ignore') 

import gensim 
from gensim.models import Word2Vec, KeyedVectors
#from matplotlib import pyplot
import matplotlib.pyplot as plt

# from tutorial https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

sample = open("./ash_wednesday.txt", "r") 
s = sample.read() 
  
# Replaces escape character with space 
f = s.replace("\n", " ") 
  
data = [] 
  
# iterate through each sentence in the file 
for i in sent_tokenize(f): 
    temp = [] 
      
    # tokenize the sentence into words 
    for j in word_tokenize(i): 
        temp.append(j.lower()) 
  
    data.append(temp) 

sentences = data
# train model
eliot_model = Word2Vec(sentences, min_count=1)

# summarize vocabulary
words = list(eliot_model.wv.vocab)
print(words)
print(f'---words {words}')

# save model
#model.save('model.bin')

# load model
filename = './GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
# summarize the loaded model
print('----MODEL')
print(model)
#new_model = Word2Vec.load('model.bin')
#print(new_model)

X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])

for i, word in enumerate(words):
  print(f'i {i}, word: {word}')
  plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()
'''
# Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              size = 100, window = 5) 
  
# Print results 
print("Cosine similarity between 'alice' " + 
               "and 'wonderland' - CBOW : ", 
    model1.similarity('hope', 'power')) 
      
print("Cosine similarity between 'alice' " +
                 "and 'machines' - CBOW : ", 
      model1.similarity('hope', 'power')) 
  
# Create Skip Gram model 
model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, 
                                             window = 5, sg = 1) 
  
# Print results 
print("Cosine similarity between 'alice' " +
          "and 'wonderland' - Skip Gram : ", 
    model2.similarity('hope', 'power')) 
      
print("Cosine similarity between 'alice' " +
            "and 'machines' - Skip Gram : ", 
      model2.similarity('hope', 'power')) 
'''
z