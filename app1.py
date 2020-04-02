# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 08:56:08 2020

@author: dell
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
# prediction function 
def ValuePredictor(to_predict_list): 
	to_predict = np.array(to_predict_list).reshape(1, 7) 
	loaded_model = pickle.load(open("model.pkl", "rb")) 
	result = loaded_model.predict(to_predict) 
	return result[0] 

# app
app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods = ['POST']) 
def result(): 
	if request.method == 'POST': 
		to_predict_list = request.form.to_dict() 
		to_predict_list = list(to_predict_list.values()) 
		to_predict_list = list(map(int, to_predict_list)) 
		result = ValuePredictor(to_predict_list)		 
		if int(result)== 1: 
			prediction ='default'
		else: 
			prediction ='nondefault'			
		return render_template('result.html', prediction = prediction) 

if __name__ == '__main__':
    app.run(debug=True)
