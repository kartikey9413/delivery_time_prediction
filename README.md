# Food delivery time prediction

### Dataset:
The dataset is a cleaned version of the original dataset submitted by Gaurav Malik on Kaggle. Below are all the features in the dataset:

- **ID:** Order ID number
- **Delivery_person_ID:** ID number of the delivery partner
- **Delivery_person_Age:** Age of the delivery partner
- **Delivery_person_Ratings:** Ratings of the delivery partner based on past deliveries
- **Restaurant_latitude:** The latitude of the restaurant
- **Restaurant_longitude:** The longitude of the restaurant
- **Delivery_location_latitude:** The latitude of the delivery location
- **Delivery_location_longitude:** The longitude of the delivery location
- **Type_of_order:** The type of meal ordered by the customer
- **Type_of_vehicle:** The type of vehicle the delivery partner rides
- **Time_taken(min):** The time taken by the delivery partner to complete the order



# Libraries and Frameworks

The Python libraries used in this project include pandas, numpy, scikit-learn, matplotlib, plotly, pickle, Flask, and other related libraries for data manipulation, model building, visualization, and deployment.

# Data Expolariton and Pre-processing

In exploring the data set I discovered that the distance taken for delivery between the restaurant and the delivery location wasn't given. To rectify this I used the given variable which was the latitude and longitude of the restaurant and delivery location to calculate the distance using **haversine formula** then proceeded to add the distance column to the dataset.

I also explored the data to find relationships between the features. The relationships explored were;

- Relationship between the distance and time taken to deliver the food
- Relationship between the time taken to deliver the food and the age of the delivery partner
- Relationship between the time taken to deliver the food and the ratings of the delivery partner.
- The relationship between the type of food ordered by the customer and the type of vehicle used by the delivery partner affects the delivery time or not

From the exploration, I discovered that the features that contribute most to the food delivery time based on our analysis are:

- Age of the delivery partner
- Ratings of the delivery partner
- Distance between the restaurant and the delivery location

I also performed a little preprocessing that included;

- Revoming columns not needed for prediction
- Saving the new dataset to be used for prediction

# Model Building:
For this project, I using an LSTM neural network model for the task of delivery time prediction.

# Deployment:
The model was deployed on Google Colab IDE using the Flask framework and HTML for the user interface.

```python
from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(5000)"))
```
The below message in the Python shell is seen, which indicates that our App is now hosted and I can run it by clicking on the URL

```python
* https://3ptu6ergm1w-496ff2e9c6d22116-5000-colab.googleusercontent.com/
```

Flask web application

```python
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='/content/templates', static_folder='/content/static')
model = pickle.load(open('/content/model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict",methods = ["POST"])
def predict():
    Delivery_person_Age = int(request.form["Delivery_person_Age"])
    Delivery_person_Ratings = float(request.form["Delivery_person_Ratings"])
    distance = int(request.form["distance"])
    features = np.array([[Delivery_person_Age,Delivery_person_Ratings,distance]])

    prediction = model.predict(features)
    output = np.round(prediction[0], 2)
    return render_template('index.html', prediction_text='Predicted Delivery Time in Minutes = {}'.format(output))


if __name__=="__main__":
   app.run()
```

HERE'S WHAT THE FRONT END LOOKS LIKE:

![chrome_Pe0v1wp06x](https://github.com/MisterAare/delivery_time_prediction/assets/109184556/2c3cab6a-bd63-455a-823f-7bae2fb7c084)
