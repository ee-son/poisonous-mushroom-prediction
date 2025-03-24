import os
import pandas as pd
import joblib
from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)
variable_form = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
                 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result/', methods=["POST"])
def result():
    # Get features from form
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

    variable_names = [cap_shape, cap_surface, cap_color, bruises, odor,
                  gill_attachment, gill_spacing, gill_size, gill_color,
                  stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring,
                  stalk_color_above_ring, stalk_color_below_ring, veil_type, veil_color,
                  ring_number, ring_type, spore_print_color, population, habitat]
    
    # Load model
    filepath = 'model/rf_model.model'
    load_model = joblib.load(filepath)

    # Make new dataframe for input
    df_input = pd.DataFrame(columns = variable_form)
    df_input.loc[0] = variable_names

    # print result
    result = load_model.predict(df_input)
    for i in result:
        int_result = int(i)
        if(int_result==0):
            decision="Definitely poisonous"
        elif(int_result==1):
            decision="It's not poisonous"
        else:
            decision="We don't know"

    # return output
    return render_template('result.html', cap_shape=cap_shape, cap_surface=cap_surface, cap_color=cap_color, 
                           bruises=bruises, odor=odor, gill_attachment=gill_attachment, gill_spacing=gill_spacing, 
                           gill_size=gill_size, gill_color=gill_color, stalk_shape=stalk_shape, stalk_root=stalk_root, 
                           stalk_surface_above_ring=stalk_surface_above_ring, stalk_surface_below_ring=stalk_surface_below_ring, 
                           stalk_color_above_ring=stalk_color_above_ring, stalk_color_below_ring=stalk_color_below_ring, 
                           veil_type=veil_type, veil_color=veil_color, ring_number=ring_number, ring_type=ring_type, 
                           spore_print_color=spore_print_color, population=population, habitat=habitat, decision=decision)

if __name__ == "__main__":
    app.run()