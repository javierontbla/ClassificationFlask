# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.

from flask import Flask, jsonify, request 
from flask_cors import CORS

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier

# Flask constructor takes the name of 
# current module (__name__) as argument.

# training
mnist = fetch_openml('mnist_784', version=1)
X = mnist["data"]
y = mnist["target"].astype(np.uint8) # convertir str a int
X_new = X.replace([range(1, 255)], 255) # avoid floats
X_train = X_new[:60000]
y_train = y[:60000]
y_train_7 = (y_train == 7)
classifier = SGDClassifier()
classifier.fit(X_train, y_train_7)

app = Flask(__name__)
CORS(app)

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/', methods=['GET', 'POST'])
# ‘/’ URL is bound with hello_world() function.
def hello_world():
      # POST request
    if request.method == 'POST':
        print('Calculating...')
        number = request.get_json()  # parse as JSON
        answer = classifier.predict(np.array([number]))
        return str(answer[0])

    return 'Hello World'
  
# main driver function
if __name__ == '__main__':
  
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()