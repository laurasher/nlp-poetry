# Python program to generate word vectors using Word2Vec 
  
# importing all necessary modules 
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
import nltk
import pandas as pd
from sklearn.decomposition import PCA
warnings.filterwarnings(action = 'ignore') 

import gensim 
from gensim.models import Word2Vec, KeyedVectors
#from matplotlib import pyplot
import matplotlib.pyplot as plt

# from tutorial https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
poem = 'ash_wednesday'
sample = open(f"./{poem}.txt", "r") 
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
model = Word2Vec(sentences, min_count=1)

# summarize the loaded model
print(model)

# summarize vocabulary
words = list(model.wv.vocab)

#model.save('model.bin')

X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])

data_to_df = []

for i, word in enumerate(words):
  plt.annotate(word, xy=(result[i, 0], result[i, 1]))
  print(f"RESULT {word} {result[i, 0], result[i, 1]}")
  x_val = result[i, 0]
  y_val = result[i, 1]
  data_to_df.append([word, x_val, y_val])

#plt.show()
df = pd.DataFrame(data_to_df, columns = ['Word', 'x_val', 'y_val'])
df.to_csv(f'./{poem}.csv')

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