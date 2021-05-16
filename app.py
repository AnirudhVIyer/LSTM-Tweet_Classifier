# Importing essential libraries
from flask import Flask, render_template, request
from functions import preprocess





app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        inputtweet = request.form['tweet']

  


        
        output = preprocess(inputtweet)
       

         
        return render_template('result.html', prediction=int(output))




@app.route('/return',methods=['GET'])
def back():
    return render_template('index.html')


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=9090)