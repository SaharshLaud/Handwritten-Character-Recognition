import cv2
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


# create route for index.html
@app.route('/')
def index():
    return render_template('index.html')
    
# create route for after.html after button click on index.html
@app.route('/after', methods=['GET', 'POST'])
def after():
    # store the uploaded image by user in static folder
    img = request.files['file1']
    img.save('static/file.jpg')


    # preprocess image uploaded by user
     
    img = cv2.imread('static/file.jpg')[:,:,0]
    img = np.invert(np.array([img]))


    # load the CNN model trained previously and make predictions
    model = load_model('model_digit_test.h5') 
    predictions = model.predict(img)  
    final_prediction = np.argmax(predictions)
    return render_template("after.html", data=final_prediction)


if __name__ == "__main__":
    app.run(debug=True)