
from flask import Flask,request,render_template
from keras.models import load_model
import numpy as np
global model, graph, c
#import tensorflow as tf
#graph =  tf.get_default_graph()
model = load_model('airline4-copy1.h5')
app = Flask(__name__)
@app.route('/ind1')
def home():
    return render_template('index.html')

@app.route('/ind2')
def index2():
    return render_template('index2.html')

@app.route('/login', methods =['POST']) #when you click submit on html page it is redirection to this url
def login():
    year = request.form['year']
    month = request.form['month']
    passengers = request.form['passengers']
   
    total = [year,month,passengers]
    #with graph.as_default():
    #y_pred = model.predict(x_test)
    from sklearn.preprocessing import RobustScaler
    #rs = RobustScaler()
    rs_pas = RobustScaler()
    y_predict = model.predict(np.array([[total]]))

    scaled_training=rs_pas.fit_transform(y_predict)
    y_pred=rs_pas.inverse_transform(scaled_training.reshape(1,-1))[0][0]*10
    return render_template('index.html' ,showcase = str(round(y_pred)))
    #after typing the name show this name on index.html file where we have created a varibale abc
if __name__ == '__main__':
    app.run(debug=True)





