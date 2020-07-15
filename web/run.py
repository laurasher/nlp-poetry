
from flask import Flask, render_template, request
import json
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
	return render_template("home.html")

@app.route('/d3_experiment', methods=['GET'])
def d3_experiment():
    return render_template("d3_experiment.html")

@app.route('/d3_exp_ash_wednesday', methods=['GET'])
def d3_exp_ash_wednesday():
    return render_template("d3_exp_ash_wednesday.html")

@app.route('/d3_exp_east_coker', methods=['GET'])
def d3_exp_east_coker():
    return render_template("d3_exp_east_coker.html")

@app.route('/d3_exp_little_gidding', methods=['GET'])
def d3_exp_little_gidding():
    return render_template("d3_exp_little_gidding.html")

@app.route('/d3_exp_burnt_norton', methods=['GET'])
def d3_exp_burnt_norton():
    return render_template("d3_exp_burnt_norton.html")

@app.route('/d3_exp_dry_salvages', methods=['GET'])
def d3_exp_dry_salvages():
    return render_template("d3_exp_dry_salvages.html")

@app.route('/eliot', methods=['GET'])
def eliot():
	return render_template("eliot.html")

@app.route('/berry', methods=['GET'])
def berry():
	return render_template("berry.html")

@app.route('/all_words_burnt_norton', methods=['GET'])
def all_words_burnt_norton():
    return render_template("all_words_burnt_norton.html")

@app.route('/all_words_east_coker', methods=['GET'])
def all_words_east_coker():
    return render_template("all_words_east_coker.html")

@app.route('/all_words_little_gidding', methods=['GET'])
def all_words_little_gidding():
    return render_template("all_words_little_gidding.html")

@app.route('/all_words_dry_salvages', methods=['GET'])
def all_words_dry_salvages():
    return render_template("all_words_dry_salvages.html")
'''
@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')
''' 

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

@app.route("/get_gerontion", methods=["GET"])
def get_gerontion():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'gerontion.json')
        with open(filename) as f:
            data = json.load(f)
        return data

@app.route("/get_animula", methods=["GET"])
def get_animula():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'animula.json')
        with open(filename) as f:
            data = json.load(f)
        return data

@app.route("/get_whispers_of_immortality", methods=["GET"])
def get_whispers_of_immortality():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'whispers_of_immortality.json')
        with open(filename) as f:
            data = json.load(f)
        return data

@app.route("/get_portrait_of_a_lady", methods=["GET"])
def get_portrait_of_a_lady():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'portrait_of_a_lady.json')
        with open(filename) as f:
            data = json.load(f)
        return data

@app.route("/get_the_peace_of_wild_things", methods=["GET"])
def get_the_peace_of_wild_things():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'the_peace_of_wild_things.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

@app.route("/get_the_country_of_marriage", methods=["GET"])
def get_the_country_of_marriage():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'the_country_of_marriage.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

@app.route("/get_what_we_need_is_here", methods=["GET"])
def get_what_we_need_is_here():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'what_we_need_is_here.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

@app.route("/get_the_man_born_to_farming", methods=["GET"])
def get_the_man_born_to_farming():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'the_man_born_to_farming.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

@app.route("/get_sabbaths_2001", methods=["GET"])
def get_sabbaths_2001():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'sabbaths_2001.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

@app.route("/get_silence", methods=["GET"])
def get_silence():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'silence.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

@app.route("/get_the_wish_to_be_generous", methods=["GET"])
def get_the_wish_to_be_generous():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'the_wish_to_be_generous.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

@app.route("/get_water", methods=["GET"])
def get_water():
    if request.method == "GET":
        filename = os.path.join(app.static_folder, 'water.json')
        with open(filename) as f:
        	data = json.load(f)
        return data

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
