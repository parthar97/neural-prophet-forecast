# Time Series Forecasting with Neural Prophet

This project uses the NeuralProphet package, which is a neural network-based time series forecasting tool inspired by Facebook Prophet. The application is built using Streamlit.

## Installation

Before running the code, you need to install the necessary Python libraries. You can do this by running the following command:

```bash
pip install pandas numpy neuralprophet matplotlib streamlit
```

## Usage

The application provides two forecasting options: 
1. Forecasting without events
2. Forecasting with events

You can upload your CSV data file using the sidebar file uploader. After uploading the data, you can set different parameters using the sidebar, such as the number of historical data points, the number of epochs, the number of hidden layers, the loss function, and so on.

In the case of forecasting with events, you can specify events and their dates. You can also set the lower and upper windows for the events.

After setting all parameters and choosing the forecasting option, you can run the model to get the forecast. The results will be displayed in the main area of the application, showing the model metrics, forecast values, and trend & seasonality. You can also download the forecast data as a CSV file.

## File Structure

The main Python file is `app.py`, which contains all the Streamlit code for running the application.

## Example

To run the application, use the following command in the terminal:

```bash
streamlit run app.py
```

Then go to `http://localhost:8501` in your web browser to interact with the application.

## Contributing

If you want to contribute to this project, feel free to fork the repository, make your changes, and submit a pull request. 

## Contact

Created by Parthasarathy Ramamoorthy, Data Scientist at Walmart Global Tech. You can reach out to him via his [LinkedIn](https://www.linkedin.com/in/parthasarathyr97/) profile.

## License

This project is open source and available under the MIT License.
