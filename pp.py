from flask import Flask, render_template, request
import pickle 
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#importing the interpretation model 
tokenize=pickle.load(open( r"C:\Users\user\Downloads\refund\refund\models/tok_model.pkl" , "rb" ))
model=pickle.load(open(r"C:\Users\user\Downloads\refund\refund\models/model.pkl", "rb"))
import tensorflow 



app=Flask(__name__)
UPLOAD_FOLDER = r'C:\Users\user\Downloads\refund\refund\uploaded'
  # Replace with your desired upload directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def home():
    return render_template('fileupload.html')
  
@app.route('/test')
def test():
    return(render_template('index.html'))


from tensorflow.keras.preprocessing.sequence import pad_sequences

@app.route("/predict", methods=["POST"])
def predict():
    reason = request.form.get('reason-content')
    tokenized_reason=tokenize.texts_to_sequences ([reason])
    prediction = model.predict(pad_sequences(tokenized_reason, maxlen=50))
    pred=np.argmax(prediction)
    if pred==0:
        prediction='PARTIAL_REFUND'
    else:
        prediction='FULL_REFUND'
    return render_template("index.html", prediction=prediction)


@app.route('/bulk', methods=['GET'])
def index():
    return render_template('fileupload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    # Process the Excel file using pandas
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    print(file_path)
    try:
        df = pd.read_excel(file_path)
        prediction_list=[]
        # Perform operations on the DataFrame
        print(df.head())  # Example: Print the first 5 rows
        df['tokenized'] = df['Reason_FR'].apply(lambda x: tokenize.texts_to_sequences([x]))
        df['padded'] = df['tokenized'].apply(lambda x: pad_sequences(x, maxlen=50))
        for X in range(0,df.shape[0]):
            predictions = model.predict(df.iloc[X,6])
            pred_labels = np.argmax(predictions, axis=1)
            #print(pred_labels[0])
            prediction_list.append(pred_labels[0])
        pd_series = pd.Series(prediction_list)
        df['prediction']=prediction_list
        #df = pd.concat([df, pd_series], axis=1)
        df = df.drop(['tokenized', 'padded'], axis=1)
        df['prediction']=df['prediction'].replace(1,'Full Refund')
        df['prediction']=df['prediction'].replace(0,'Partial Refund')
        df.to_csv('Predicted_data.csv', index=False)
            #df['prediction'] = pd.Series(pred_labels).astype(str).replace({'0': 'PARTIAL_REFUND', '1': 'FULL_REFUND'})
        print(df.head())
    except Exception as e:
        return f"Error processing file: {str(e)}"

    #return df.head()
    table = df.to_html()
    return render_template('Results.html', table=table)

# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=True)