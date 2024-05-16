# Shaastra-AI/ML-Challenge-TEAM-TECH-01
# COVID-19 LSTM Model

## Project Description
This project aims to predict the number of deaths due to COVID-19 using a Long Short-Term Memory (LSTM) model based on various features such as confirmed cases, population density, and state/union territory. The model is trained on historical COVID-19 data and used to make predictions on test data.

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- numpy
- keras
- matplotlib

## Installation
1. Clone this repository: `git clone https://github.com/your_username/covid-lstm-model.git`
2. Install dependencies: `pip install -r requirements.txt`

## Usage
1. Ensure the training data (`train_data_covid.csv`) and test data (`test_data_covid.csv`) are in CSV format and located in the root directory.
2. Run the script: `python predict_deaths.py`
3. The script will train the LSTM model on the training data, make predictions on the test data, and save the results to `submissions/test.csv`.
4. Optionally, visualize the training and validation loss by running `python plot_loss.py`.

## File Structure
- `train_data_covid.csv`: Training dataset containing historical COVID-19 data.
- `test_data_covid.csv`: Test dataset for making predictions.
- `predict_deaths.py`: Python script to train the LSTM model and make predictions.

## Program
```.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.layers import LSTM, Dense, Dropout

# Assuming the data is loaded in a pandas dataframe called 'df'
df=pd.read_csv("train_data_covid.csv")
df.replace('-', 0)

# Handle missing value (replace with next day for demonstration)
df.fillna(method='ffill', inplace=True)

# Create separate Date and Time columns (corrected format)
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df['Time'] = df['Date'].dt.time

# Extract day of week
df['Day_of_Week'] = df['Date'].dt.weekday

# Calculate daily increase in confirmed cases (assuming 'Confirmed' is cases)
df['Daily_Increase_Confirmed'] = df['Confirmed'].diff()

# Create target variable (assuming 'Deaths' is cumulative)
df['Daily_Deaths'] = df['Deaths'].diff()

# Create a LabelEncoder object
le = LabelEncoder()
# combine the test and train data
test_df=pd.read_csv("test_data_covid.csv")
combine = pd.concat([df,test_df],axis=0)

#fit the encoder
le.fit(combine["State/UnionTerritory"])

# Encode the "State/UnionTerritory" column
df["State/UnionTerritory_Encoded"] = le.transform(df["State/UnionTerritory"])

# Select relevant features (replace with your choices)
features = ['Confirmed', 'Daily_Increase_Confirmed', 'PopulationDensityPerSqKm',"State/UnionTerritory_Encoded"]

# Feature scaling (assuming StandardScaler is suitable)
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
target = 'Deaths'

# Drop rows with NaN values (resulting from diff() operation)
df = df.dropna()

# Split data into input (X) and output (y) sequences
X = df[features].values
y = df[target].values

# Reshape input data for LSTM (samples, time steps, features)
X = X.reshape(X.shape[0], 1, X.shape[1])

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# add test to the train data
X_train = np.concatenate((X_train,X_test),axis=0)
y_train = np.concatenate((y_train,y_test),axis=0)


# Define larger and deeper LSTM model architecture
model = Sequential([
    LSTM(units=1024, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3), # Dropout layer to prevent overfitting
    LSTM(units=512, return_sequences=True),
    Dropout(0.3),
    LSTM(units=512, return_sequences=True),
    Dropout(0.3),
    LSTM(units=512),
    Dropout(0.3),
    Dense(units=256, activation='relu'),
    Dense(units=256, activation='relu'),
    Dense(units=128, activation='relu'), 
    Dense(units=128, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),  
    Dense(units=1)  # Output layer
])

# Compile model
model.compile(optimizer='adam', loss='mae')

# early stopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)

# Fit model
history = model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_split=0.4, callbacks=[early_stopping], verbose=1)

# Evaluate model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

test_data=pd.read_csv("test_data_covid.csv")
test_data.fillna(method='ffill', inplace=True)
test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y-%m-%d')
test_data['Time'] = test_data['Date'].dt.time
test_data['Day_of_Week'] = test_data['Date'].dt.weekday
test_data['Daily_Increase_Confirmed'] = test_data['Confirmed'].diff()
test_data["State/UnionTerritory_Encoded"] = le.transform(test_data["State/UnionTerritory"])
test_data[features] = scaler.transform(test_data[features])
X_new = test_data[features].values
X_new = X_new.reshape(X_new.shape[0], 1, X_new.shape[1])
y_new = model.predict(X_new)
y_new=y_new.astype(int)
y_new

# convert to one dimensional array
y_new = y_new.ravel()

# save to csv with Sno from test_data_covid.csv
output = pd.DataFrame({'Sno': test_data['Sno'], 'Deaths': y_new})

# replace all negative values with 0
output['Deaths'] = output['Deaths'].clip(lower=0)
display(output)
output.to_csv('submissions/test.csv', index=False)

import matplotlib.pyplot as plt
# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

```

## Program Conclusion
The LSTM model demonstrates promising performance in predicting COVID-19 deaths based on historical data. However, the accuracy of predictions may vary depending on various factors such as data quality, feature selection, and model hyperparameters. Further experimentation and fine-tuning may be necessary to improve prediction accuracy.

## Troubleshooting
- If you encounter any issues or errors while running the program, please check the following:
  - Ensure all dependencies are installed correctly.
  - Verify that the input data files (`train_data_covid.csv` and `test_data_covid.csv`) exist in the correct location and are formatted properly.
  - Check for any typos or syntax errors in the code.
