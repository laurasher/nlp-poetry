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
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

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
    fig.title.text_font_style = "bold"
    fig.title.text_color = "#394d7e"
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
#poems = ['ash_wednesday','dry_salvages','the_waste_land','east_coker','little_gidding','burnt_norton',
#'choruses_from_the_rock','the_hollow_men','gerontion','animula','portrait_of_a_lady','whispers_of_immortality']
poems = ['dry_salvages','east_coker','little_gidding','burnt_norton']
pub_year_df = pd.read_csv(f'./data/poem_pub_year.csv')

data = []
temp = []
#nltk.download('stopwords')
#nltk.download("punkt")

stop_words = set(stopwords.words('english'))
ignore_words = ['u', 'man', 'dead', 'iii', 'ii', 'iv', 'v', 'ha', 'wa']

#mode = 'svg'
mode = 'bokeh'

total_words = []

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
	 		j = j.lower()
	 		stems = wn._morphy(j, wn.NOUN)
	 		if len(stems) >= 1:
	 			j = stems[0]
	 		if j.isalnum() and not j.lower() in stop_words and j not in ignore_words:
	 			temp.append(j)
	 			total_words.append(j)
	 		#if j.isalnum() and not j.lower() in stop_words:
	 		#	temp.append(j.lower()) 
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

	word_search_list = ['sea']

	# Non selected word colors
	#words_df = pd.DataFrame({'words': words, 'color': '#394d7e', 'size': 2})
	words_df = pd.DataFrame({'words': words, 'color': '#DBA68F', 'size': 2})
	
	# Color selected words
	#words_df.loc[words_df['words'].str.contains('|'.join(word_search_list)), 'color'] = '#cf6233'
	#words_df.loc[words_df['words'].str.contains('|'.join(word_search_list)), 'color'] = '#5D7D2C'
	#words_df.loc[words_df['words'].str.match('|'.join(word_search_list)), 'color'] = '#B56548'
	#words_df.loc[words_df['words'].str.match('|'.join(word_search_list)), 'size'] = 20
	words_df.loc[words_df['words'] == 'fire', 'color'] = '#B56548' 
	words_df.loc[words_df['words'] == 'fire', 'size'] = 20

	###### Bokeh plot
	source = ColumnDataSource(data=dict(x=result[:, 0], y=result[:, 1], word=words, 
		fill=list(words_df['color']), size=list(words_df['size'])))
	hover = HoverTool(tooltips=[
	    ('', '@word')
	])

	p = figure(plot_width=300, plot_height=300, #tools=['wheel_zoom', 'pan', hover],
		tools=[hover],
		title=f"{poem.replace('_', ' ')}, {pub_year_df.loc[pub_year_df['poem_title']==poem, 'pub_year'].values[0]}",
		x_range=(-np.min(result[:, 0])*2.5, np.min(result[:, 0])*2), y_range=(-np.min(result[:, 1])*2, np.min(result[:, 1])*2.5))

	if mode is 'svg':
		p.circle('x', 'y', source=source, size=1, color='black', alpha=1,
		line_color='black', line_width = 0.3, line_alpha = 0)

	if mode is 'bokeh':
		p.circle('x', 'y', source=source, #size=2, color="black",
			color= 'fill', size = 'size', line_alpha=0, alpha=0.6)
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

######## ######## Create word columns ######## ########
word_column_categories = ['geologic','time','communication','botanical or animal','relationship','light','lamentation','religion or god','physicality','body','movement','arts','direction','emotion','clothing']
column_ordering = pd.DataFrame.from_dict({'category': ['geologic','time','communication','botanical or animal','relationship','light','lamentation','religion or god','physicality','body','movement','arts','direction','emotion','clothing'],
	#'column_num': [5,1,8,2,9,3,6,4,7]
	'column_num': [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]
})

total_words_unique = set(total_words)
df = pd.DataFrame(total_words_unique,columns=['word'])
df.to_csv('./all_words.csv')
df_cats = pd.read_csv('word_column_categories.csv').dropna().reset_index().drop(['index'],axis=1)
print(df_cats)
print(column_ordering)
df_cats = df_cats.merge(column_ordering, left_on='category', right_on='category', how='outer')
print(df_cats)
print(f"Total unique words in this corpus {len(total_words_unique)}")
json_out = []
#num_cols = 63
#num_rows = math.ceil(len(total_words_unique)/num_cols)
num_rows = 20
print("NUM ROWS")
print(num_rows)
col_count = 0
row_count = 0
col_num = 0
col_word_count = {
	'geologic': 0,
	'time': 0,
	'communication': 0,
	'botanical or animal': 0,
	'relationship': 0,
	'light': 0,
	'lamentation': 0,
	'religion or god': 0,
	'physicality': 0,
	'body': 0,
	'movement': 0,
	'arts': 0,
	'direction': 0,
	'emotion': 0,
	'clothing': 0
}

for word in total_words_unique:
	if df_cats.loc[df_cats['word']==word,'column_num'].any():
		# Get the word's category
		col_cat = df_cats.loc[df_cats['word']==word,'category'].values[0]

		# If the row count for that column os greater than rows, reset word count and increment column num
		if col_word_count[col_cat] >= num_rows:
			print(f'Resetting row count for {col_cat}')
			print(col_word_count)
			df_cats.loc[df_cats['category']==col_cat,'column_num'] = df_cats.loc[df_cats['category']==col_cat,'column_num']+1
			col_word_count[col_cat] = 0
			#row_count = col_word_count[col_cat]
			#col_num = int(df_cats.loc[df_cats['word']==word,'column_num'].values[0])
			#print(f'\nAfter reset {col_word_count}\n')
			#print(f"-------- {word}")
			#print(f"{df_cats[df_cats['word']==word]}")
			#print(f"category: {col_cat}")
			#print(f"col: {col_num}")
			#print(f"row: {row_count}")
		
		col_num = int(df_cats.loc[df_cats['word']==word,'column_num'].values[0])
		col_word_count[col_cat] = col_word_count[col_cat] + 1
		row_count = col_word_count[col_cat]

		if ~col_num % 2:
			print(f"-------- {word}")
			print(f"{df_cats[df_cats['word']==word]}")
			print(f"category: {col_cat}")
			print(f"col: {col_num}")
			print(f"row: {row_count}")

		json_out.append({
			'word': word,
			'col': col_num,
			'row': row_count,
			'count': total_words.count(word),
			'category': col_cat,
			'poems_that_use': [],
			'per_poem_count': [],
		})
		#row_count = col_word_count[col_cat]
		#col_num = int(df_cats.loc[df_cats['word']==word,'column_num'].values[0])
		#col_word_count[col_cat] = col_word_count[col_cat] + 1
print(df_cats)
print(col_word_count)
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
	 		j = j.lower()
	 		stems = wn._morphy(j, wn.NOUN)
	 		if len(stems) >= 1:
	 			j = stems[0]
	 		if j.isalnum() and not j.lower() in stop_words and j not in ignore_words:
	 			temp.append(j)
	 			total_words.append(j)
	 		#if j.isalnum() and not j.lower() in stop_words:
	 		#	temp.append(j.lower()) 
	 	data.append(temp)

	sentences = data
	uniques = set(data[0])
	for word in uniques:
		for item in json_out:
			if item['word'] == word:
				item['poems_that_use'].append(poem)
				item['per_poem_count'].append({'poem': poem, 'per_poem_count': data[0].count(word)})
with open(f'./web/static/all_words.json', 'w') as outfile:
        json.dump(json_out, outfile)
        outfile.close()
