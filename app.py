from flask import Flask, render_template, request
from main import prediction
import pickle
import numpy as np
app=Flask(__name__)

app = Flask(__name__, template_folder="template")

model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def hello_world():
    return render_template('login.html')

@app.route('/bc', methods=['POST','GET'])
def bc():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    param0=request.form.get('param0')
    param1=request.form.get('param1')
    param2=request.form.get('param2')
    param3=request.form.get('param3')
    param4=request.form.get('param4')
    param5=request.form.get('param5')
    param6=request.form.get('param6')
    param7=request.form.get('param7')
    param8=request.form.get('param8')
    param9=request.form.get('param9')
    param10=request.form.get('param10')
    result=prediction(param0,param1,param2,param3,param4,param5,param6,param7,param8,param9,param10)
    return render_template('home.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
