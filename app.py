import os
import joblib
from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result/', methods=["POST"])
def result():
    cap_shape = request.form.get('cap-shape')
    cap_surface = request.form.get('cap-surface')
    cap_color = request.form.get('cap-color')
    bruises = request.form.get('bruises')
    odor = request.form.get('odor')
    gill_attachment = request.form.get('gill-attachment')
    gill_spacing = request.form.get('gill-spacing')
    gill_size = request.form.get('gill-size')
    gill_color = request.form.get('gill-color')
    stalk_shape = request.form.get('stalk-shape')
    stalk_root = request.form.get('stalk-root')
    stalk_surface_above_ring = request.form.get('stalk-surface-above-ring')
    stalk_surface_below_ring = request.form.get('stalk-surface-below-ring')
    stalk_color_above_ring = request.form.get('stalk-color-above-ring')
    stalk_color_below_ring = request.form.get('stalk-color-below-ring')
    veil_type = request.form.get('veil-type')
    veil_color = request.form.get('veil-color')
    ring_number = request.form.get('ring-number')
    ring_type = request.form.get('ring-type')
    spore_print_color = request.form.get('spore-print-color')
    population = request.form.get('population')
    habitat = request.form.get('habitat')
    
    # Load model
    filepath = '/model/rf_model.model'
    load_model = joblib.load(filepath)

    return render_template('result.html')

if __name__ == "__main__":
    app.run()