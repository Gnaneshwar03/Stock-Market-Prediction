# Stock Price Prediction with LSTM

This project aims to predict the future stock prices of a specific company using historical stock data and LSTM (Long Short-Term Memory) neural networks. The code provided in this repository performs the necessary steps for data retrieval, exploration, preparation, model building, and prediction visualization.

## Requirements

To run the code in this repository, the following dependencies need to be installed:

- `yfinance`: A Python library to retrieve historical stock data from Yahoo Finance.
- `numpy`: A fundamental package for scientific computing with Python.
- `pandas`: A library providing high-performance data manipulation and analysis tools.
- `tensorflow`: An open-source machine learning framework for training and deploying deep learning models.
- `plotly`: A graphing library for interactive, publication-quality graphs.

## Usage

1. Data Retrieval and Exploration: The code downloads historical stock data for the desired company using the `yfinance` library. The data is then explored through various analyses such as shape, duplicate index removal, missing value check, and statistical information. The trends in closing values and volume traded are visualized using `plotly`.

2. Data Preparation: The code filters the data to include only the necessary columns and splits it into training and testing sets. The input features are scaled using a `MultiDimensionScaler` class, and the scaler objects are saved for future use. The target values are also scaled using `MinMaxScaler`.

3. Model Building: The code constructs an LSTM-based deep learning model using `tensorflow.keras`. The model architecture consists of bidirectional LSTM layers, dropout layers, and dense layers. The model is compiled with a mean squared error loss function and an optimizer. It is then trained on the training data and the best weights are saved based on the validation loss.

4. Visualization and Evaluation: The code loads the best weights obtained during training and uses the trained model to make predictions on the test set. The predictions and actual values are then scaled back to their original scale. The results are visualized through line graphs using `plotly`. Additionally, the code predicts the stock prices on the entire dataset and visualizes the predictions on the entire data.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
