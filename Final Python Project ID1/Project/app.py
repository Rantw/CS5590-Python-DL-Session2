#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re
from keras.optimizers import RMSprop
#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from load import * 
#initalize our flask app
app = Flask(__name__)

import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf


def init():
	json_file = open('VGG16Berlin50cl.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
	loaded_model.load_weights("VGG16Berlin50clModel.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	optimizer = RMSprop(0.001)
	loaded_model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.get_default_graph()
	return loaded_model, graph

#global vars for easy reusability
global model, graph
#initialize these variables
model, graph = init()
import cv2
import base64
from keras.applications.vgg16 import preprocess_input
#decoding an image from base64 into raw representation
def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	#base64.b64encode(b'base64,(*)')
	#print(imgstr)
	with open('output.png','wb') as output:
		#output.write(imgstr.decode('base64'))
		output.write(base64.b64decode(imgstr))
	

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	#whenever the predict method is called, we're going
	#to input the user drawn character as an image into the model
	#perform inference, and return the classification
	#get the raw data format of the image
	imgData = request.get_data()
	#encode it into a suitable format
	convertImage(imgData)
	print("debug")
	#read the image into memory
	#x = imread('output.png', mode='L')
	x = cv2.imread('output.png')
	#compute a bit-wise inversion so black becomes white and vice versa
	#x = np.invert(x)
	#img = cv2.imread(path)
	#x = cv2.resize(x, (224, 224), 3)
	#make it the right size
	#x = imresize(x,(224, 224))
	#imshow(x)
	#convert to a 4D tensor to feed into our model
	#import tensorflow as tf
	#x = tf.reshape(x, [1, 224, 224, 3])
	#x = x.reshape(1, 224, 224, 1)

	sample_image2 = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
	sample_image2 = cv2.resize(sample_image2, (224, 224))
	sample_image_processed2 = np.expand_dims(sample_image2, axis=0)
	sample_image_processed2 = preprocess_input(sample_image_processed2)
	x = sample_image_processed2

	dict_labels ={'airplane': 0, 'ant': 1, 'armchair': 2, 'basket': 3, 'bear (animal)': 4, 'bell': 5, 'blimp': 6, 'bookshelf': 7,
	 'bulldozer': 8, 'butterfly': 9, 'camel': 10, 'cannon': 11, 'carrot': 12, 'cat': 13, 'chair': 14, 'church': 15,
	 'comb': 16, 'cow': 17, 'crown': 18, 'dragon': 19, 'feather': 20, 'foot': 21, 'hamburger': 22, 'head-phones': 23,
	 'horse': 24, 'ice-cream-cone': 25, 'key': 26, 'laptop': 27, 'megaphone': 28, 'mouse (animal)': 29, 'panda': 30,
	 'person sitting': 31, 'power outlet': 32, 'rainbow': 33, 'sailboat': 34, 'sea turtle': 35, 'skull': 36,
	 'speed-boat': 37, 'squirrel': 38, 'streetlight': 39, 'suv': 40, 'table': 41, 'teddy-bear': 42, 'tiger': 43,
	 'tooth': 44, 'train': 45, 'truck': 46, 'umbrella': 47, 'walkie talkie': 48, 'wine-bottle': 49}

	print("debug2")
	#in our computation graph
	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		print(out)
		print(out.shape)
		print(np.argmax(out,axis=1))
		print("debug3")
		#convert the response to a string
		label = np.argmax(out,axis=1)


		for labels, values in dict_labels.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
			if values == label:
				response = labels
		#response = np.array_str(np.argmax(out, axis=1))
		return response	
	

if __name__ == "__main__":
	#decide what port to run the app in
	#port = int(os.environ.get('PORT', 8085))
	#run the app locally on the givn port
	#app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	app.run(debug=True, port = 8085)
