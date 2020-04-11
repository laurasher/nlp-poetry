
from flask import Flask, render_template, request
import json
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
	return render_template("home.html")
'''
@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')
''' 
poems = ['ash_wednesday','dry_salvages','the_waste_land','east_coker','little_gidding','burnt_norton','choruses_from_the_rock','the_hollow_men']

@app.route("/get_ash_wednesday", methods=["GET"])
def get_ash_wednesday():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'ash_wednesday.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

@app.route("/get_dry_salvages", methods=["GET"])
def get_dry_salvages():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'dry_salvages.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

@app.route("/get_the_waste_land", methods=["GET"])
def get_the_waste_land():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'the_waste_land.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

@app.route("/get_east_coker", methods=["GET"])
def get_east_coker():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'east_coker.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

@app.route("/get_little_gidding", methods=["GET"])
def get_little_gidding():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'little_gidding.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

@app.route("/get_burnt_norton", methods=["GET"])
def get_burnt_norton():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'burnt_norton.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

@app.route("/get_choruses_from_the_rock", methods=["GET"])
def get_choruses_from_the_rock():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'choruses_from_the_rock.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

@app.route("/get_the_hollow_men", methods=["GET"])
def get_the_hollow_men():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'the_hollow_men.json')
        with open(filename) as f:
        	data = json.load(f)
        return data
                
if __name__ == '__main__':
    app.run(debug=True)
