import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


app = Flask(__name__)



@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload',methods=['GET','POST'])
def upload():
    # Get the uploaded file from the HTML form
    csv_file = request.files['csv_file']
    
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Convert the DataFrame into an HTML table
    table_html = df.to_html(index=False)
    
    # Pass the HTML table to the template
    return render_template('csv_display.html', table_html=table_html)


@app.route('/train_model', methods=['POST'])
def train_model():
    csv_data = request.files['csv_file']
    df = pd.read_csv(csv_data)
    # df.head()
    x = df['Temperature']
    y = df['Revenue']

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

    model=LinearRegression()

    np.array([x_train]).ndim

    model.fit(np.array([x_train]).reshape(-1,1), y_train)

    y_pred = model.predict(np.array(x_test).reshape(-1,1))
    # y_pred
    score = r2_score(y_test,y_pred)
    return render_template('result.html', total_score=f'Score is: {score}')

@app.route('/predict', methods=['GET','POST'])
def predict():
    # Convert input data to float
    temperature = float(request.form.get('temperature'))

    # Make prediction using the model
    prediction = model.predict([[temperature]])

    # Convert prediction to float and round to 2 decimal places
    output = round(float(prediction[0]), 2)

    # Render the result page with the predicted output
    return render_template("result.html", prediction_text=f"Total revenue generated is (hundred crores) Rs. {output}/-")



if __name__ == '__main__':
    app.run(debug=True)