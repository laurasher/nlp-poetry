import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, WheelZoomTool
import math
from bokeh.io import export_svgs
from bokeh.io import export_png
from bokeh.transform import dodge
from bokeh.resources import CDN
from bokeh.embed import file_html, json_item

# Python program to generate word vectors using Word2Vec 
  
# importing all necessary modules 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import warnings 
import nltk
import pandas as pd
from sklearn.decomposition import PCA
warnings.filterwarnings(action = 'ignore') 

import gensim 
from gensim.models import Word2Vec, KeyedVectors
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import json


def style_plots(fig):
    fig.background_fill_color = None
    fig.border_fill_color = None
    fig.toolbar.logo = None
    fig.toolbar_location = None
    fig.legend.background_fill_alpha = 0.3
    fig.outline_line_color = None
    fig.xgrid.grid_line_color = "white"
    fig.ygrid.grid_line_color = "white"
    fig.xgrid.grid_line_alpha = 0
    fig.ygrid.grid_line_alpha = 0
    fig.yaxis.visible = False
    fig.xaxis.visible = False
    fig.title.text_font_size = "9pt"
    fig.title.text_font_style = "normal"
    fig.title.visible = None
    fig.yaxis.major_label_text_color = 'white'
    fig.xaxis.major_label_text_color = 'white'
    fig.xaxis.major_label_text_font_size = '14pt'
    fig.yaxis.major_label_text_font_size = '14pt'
    return fig


# from tutorial https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
poems = ['ash_wednesday','dry_salvages','the_waste_land','east_coker','little_gidding','burnt_norton','choruses_from_the_rock','the_hollow_men']
data = []
temp = []
nltk.download('stopwords')
nltk.download("punkt")
stop_words = set(stopwords.words('english')) 

for poem in poems:
	sample = open(f"./{poem}.txt", "r") 
	s = sample.read() 

	# Replaces escape character with space 
	f = s.replace("\n", " ") 
	
	# iterate through each sentence in the file 
	for i in sent_tokenize(f): 
	 	# tokenize the sentence into words 
	 	for j in word_tokenize(i):
	 		if j.isalnum() and not j.lower() in stop_words:
	 			temp.append(j.lower()) 
	 	data.append(temp) 

	sentences = data


	# train model
	model = Word2Vec(sentences, min_count=1)

	# summarize the loaded model
	#print(model)

	# summarize vocabulary
	words = list(model.wv.vocab)
	#print(words)
	#model.save('model.bin')

	X = model[model.wv.vocab]
	pca = PCA(n_components=2)
	result = pca.fit_transform(X)


	###### Bokeh plot
	source = ColumnDataSource(data=dict(x=result[:, 0], y=result[:, 1], word=words))
	hover = HoverTool(tooltips=[
	    ('', '@word'),
	])

	p = figure(plot_width=300, plot_height=300, tools=['wheel_zoom', 'pan', hover])
	p.circle('x', 'y', source=source, size=2, color="black", line_alpha = 0, alpha=0.5)

	'''
	p = figure(plot_height=200, plot_width=200, title=f"{poem}",
           toolbar_location=None, tools="")
	p.circle(result[:, 0], result[:, 1], size=2, color="black", line_alpha = 0, alpha=0.5)
	'''

	p = style_plots(p)
	#p.add_tools(WheelZoomTool())
	p.toolbar.active_scroll = p.select_one(WheelZoomTool)
	data_to_df = []

	for i, word in enumerate(words):
	  #plt.annotate(word, xy=(result[i, 0], result[i, 1]))
	  #print(f"RESULT {word} {result[i, 0], result[i, 1]}")
	  x_val = result[i, 0]
	  y_val = result[i, 1]
	  data_to_df.append([word, x_val, y_val])

	#plt.show()
	df = pd.DataFrame(data_to_df, columns = ['Word', 'x_val', 'y_val'])
	df.to_csv(f'./individual_training/{poem}.csv')

	#p.output_backend = "svg"
	#export_svgs(p, filename=f"./individual_training/{poem}.svg")

	#html = file_html(p, CDN, f"individual_training_{poem}")
	plot_json = json.dumps(json_item(p, f"{poem}"))
	f = open(f"./web/static/{poem}.json", "w")
	print(f"Writing ./web/static/{poem}.json...")
	#f = open(f"./web/static/{poem}.html", "w")
	#print(f"Writing ./web/static/{poem}.html...")
	f.write(plot_json)
	f.close()




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