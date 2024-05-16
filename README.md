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

## Program Conclusion
The LSTM model demonstrates promising performance in predicting COVID-19 deaths based on historical data. However, the accuracy of predictions may vary depending on various factors such as data quality, feature selection, and model hyperparameters. Further experimentation and fine-tuning may be necessary to improve prediction accuracy.

## Troubleshooting
- If you encounter any issues or errors while running the program, please check the following:
  - Ensure all dependencies are installed correctly.
  - Verify that the input data files (`train_data_covid.csv` and `test_data_covid.csv`) exist in the correct location and are formatted properly.
  - Check for any typos or syntax errors in the code.

## Acknowledgements
- Special thanks to all contributors and developers of open-source libraries used in this project.
- This project is inspired by the collective efforts to combat the COVID-19 pandemic and aims to contribute to the understanding and prediction of its impact.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
