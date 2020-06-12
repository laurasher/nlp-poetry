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
#poems = ['ash_wednesday','dry_salvages','the_waste_land','east_coker','little_gidding','burnt_norton','choruses_from_the_rock','the_hollow_men', 
#'the_country_of_marriage', 'the_peace_of_wild_things', 'what_we_need_is_here', 'the_man_born_to_farming', 'sabbaths_2001', 'silence', 'the_wish_to_be_generous', 'water'
#]
poems = [
'ash_wednesday','dry_salvages','the_waste_land','east_coker','little_gidding','burnt_norton','choruses_from_the_rock','the_hollow_men'
]
data = []
temp = []
nltk.download('stopwords')
nltk.download("punkt")
stop_words = set(stopwords.words('english')) 

mode = 'svg'
#mode = 'bokeh'

for poem in poems:
	data = []
	temp = []
	sentences = []
	s = []
	sample = open(f"./data/{poem}.txt", "r") 
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
	#model.save('model.bin')

	X = model[model.wv.vocab]
	pca = PCA(n_components=2)
	result = pca.fit_transform(X)


	###### Bokeh plot
	source = ColumnDataSource(data=dict(x=result[:, 0], y=result[:, 1], word=words))
	hover = HoverTool(tooltips=[
	    ('', '@word')
	])

	p = figure(plot_width=300, plot_height=300, tools=['wheel_zoom', 'pan', hover],
		x_range=(-np.min(result[:, 0])*2.5, np.min(result[:, 0])*2), y_range=(-np.min(result[:, 1])*2, np.min(result[:, 1])*2.5))

	if mode is 'svg':
		p.circle('x', 'y', source=source, size=1, color='black', alpha=1,
		line_color='black', line_width = 0.3, line_alpha = 0)

	if mode is 'bokeh':
		p.circle('x', 'y', source=source, size=2, color="black", line_alpha=0, alpha=0.5)
		p.circle('x', 'y', source=source, size=5, color='black', alpha=0,
			line_color='black', line_width = 0.2, line_alpha = 0)


	p = style_plots(p)
	p.toolbar.active_scroll = p.select_one(WheelZoomTool)
	data_to_df = []

	for i, word in enumerate(words):
	  x_val = result[i, 0]
	  y_val = result[i, 1]
	  data_to_df.append([word, x_val, y_val])

	df = pd.DataFrame(data_to_df, columns = ['Word', 'x_val', 'y_val'])
	df.to_csv(f'./individual_training/{poem}.csv')


	if mode is 'bokeh':
		plot_json = json.dumps(json_item(p, f"{poem}"))
		f = open(f"./web/static/{poem}.json", "w")
		print(f"Writing ./web/static/{poem}.json...")
		f.write(plot_json)
		f.close()

	if mode is 'svg':
		p.output_backend = "svg"
		export_svgs(p, filename=f"./individual_training/{poem}.svg")
		print(f"Writing ./individual_training/{poem}.svg...")

