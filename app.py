from flask import Flask, render_template, request
# from keras.applications.resnet50 import *
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import numpy as np
import os
import json

app = Flask(__name__, template_folder=os.path.abspath("static/"))
global graph
# global sess
# sess = tf.Session()
# graph = tf.get_default_graph()

# set_session(sess)
model = load_model('./COVID_testing.h5')
# model = MobileNet(weights='imagenet')

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
UPLOAD_FOLDER = 'static/'
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(path):
    img = image.load_img(path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # p_ill,p_good = np.around(new_model.predict(x), decimals=2)[0]
    p_ill,p_good = model.predict(x)[0]

    print("COVID ", p_ill)
    print("NORMAL", p_good)

    acc=[p_ill*100, p_good*100]
    print(acc)
    classes = ['COVID-19', 'NORMAL']

    return acc, classes


@app.route("/")
def index():
	# return the rendered template
	return render_template('gui.html')


@app.route("/detect", methods=["GET", "POST"])
def detect():

    if request.method == "GET":
        return render_template("object.html")

    if request.method == "POST":
        image = request.files["x-ray"]
        image_name = image.filename
        path = os.path.join(UPLOAD_FOLDER, image_name)
        image.save(path)
        # path = request.form["path"]':


        # accuracies, classes = predict(path)
        accuracies, classes = predict(path)
        os.remove(path)
	    
        return render_template("object.html", preds=accuracies,
                               classes=json.dumps(classes), img=path)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8080)
