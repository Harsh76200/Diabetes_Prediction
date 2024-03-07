import os
import pickle
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request, redirect

model: LogisticRegression
file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pkl')

#first procedure then thread
with open(file, 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if not request.method == 'POST':
        redirect('/')
    output = model.predict([(
        request.form['pregnancies'],
        request.form['glucose'],
        request.form['bp'],
        request.form['skinthickness'],
        request.form['insulin'],
        request.form['bmi'],
        request.form['diabetespedigreefunction'],
        request.form['age'],
        )])
    return render_template('output.html', output='Yes' if output else 'No')

if __name__ == "__main__":
    app.run(debug=True)