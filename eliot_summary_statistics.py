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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

warnings.filterwarnings(action = 'ignore') 

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

lemmatizer = WordNetLemmatizer() 

# from tutorial https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
poems = ['ash_wednesday','dry_salvages','the_waste_land','east_coker','little_gidding','burnt_norton','choruses_from_the_rock','the_hollow_men']
pub_year_df = pd.read_csv(f'./data/eliot_poetry_year.csv')

#nltk.download('stopwords')
#nltk.download("punkt")
#nltk.download('wordnet')

stop_words = set(stopwords.words('english')) 
ignore_words = ['u', 'man', 'dead', 'iii', 'ii', 'v']

mode = 'svg'
#mode = 'bokeh'

df_dict = {'word': [], 'poem_title': [], 'year': []}

data = []

for poem in poems:
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
	 		if j.isalnum() and not j.lower() in stop_words and not j.lower in ignore_words:
	 			j = j.lower()
	 			stems = wn._morphy(j, wn.NOUN)
	 			if len(stems) >= 1:
	 				j = stems[0]
	 			#if len(stems) == 1:
	 			#	print(stems)
	 			if j not in ignore_words:
		 			temp.append(j)
				 	df_dict['word'].append(j)
				 	df_dict['year'].append(pub_year_df.loc[pub_year_df['poem_title']==poem, 'pub_year'].values[0])
				 	df_dict['poem_title'].append(poem)

df = pd.DataFrame.from_dict(df_dict)
uniques = df['word'].unique().tolist()
print(f"df length: {len(df)}")
print(f"num unique words: {len(uniques)}")

n = 20
print(f"top 20 words: {df['word'].value_counts()[:n].index.tolist()}")

###### Bokeh plot
'''
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
'''
