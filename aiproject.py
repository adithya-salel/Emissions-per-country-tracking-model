# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Function to display welcome message
def display_welcome():
    print("Welcome to the Pollution Tracking and Forecasting Project!")
    print("This tool allows you to analyze pollution data for different countries and predict future pollution levels.")
    print("Follow the instructions below to select your country and pollutant type.\n")

# Load and Inspect the Dataset
def load_data(filepath):
    print("Loading data...")
    try:
        data = pd.read_excel(filepath)
        print("Data loaded successfully!\n")
        return data
    except FileNotFoundError:
        print("Error: File not found. Please check the file path and try again.")
        return None

# Function to display available countries
def display_countries(data):
    print("Available countries for analysis:")
    countries = data["Area"].unique()
    for idx, country in enumerate(countries, 1):
        print(f"{idx}. {country}")
    print()
    return countries

# Function to select country
def select_country(countries):
    while True:
        try:
            choice = int(input("Enter the number corresponding to your chosen country: ")) - 1
            if 0 <= choice < len(countries):
                print(f"Selected Country: {countries[choice]}\n")
                return countries[choice]
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Please enter a number corresponding to a country.")

# Function to display available pollution elements
def display_elements(data, country):
    country_data = data[data["Area"] == country]
    elements = country_data["Element"].unique()
    print("\nAvailable pollution elements for tracking:")
    for idx, element in enumerate(elements, 1):
        print(f"{idx}. {element}")
    return elements

# Function to select pollution element
def select_element(elements):
    while True:
        try:
            choice = int(input("\nEnter the number corresponding to your chosen element: ")) - 1
            if 0 <= choice < len(elements):
                print(f"Selected Element: {elements[choice]}\n")
                return elements[choice]
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Please enter a number corresponding to an element.")

# Function to process and prepare time series data
def prepare_data(data, country, element):
    country_data = data[(data["Area"] == country) & (data["Element"] == element)]
    pollution_series = country_data.iloc[:, 4:].sum().to_frame(name="pollution_level")
    pollution_series.index = pd.to_datetime(pollution_series.index, format='%Y').to_period('Y')
    return pollution_series

# Function to plot pollution trends
def plot_trends(pollution_series, country, element):
    plt.style.use('ggplot')  # Set ggplot style
    # Convert PeriodIndex to DatetimeIndex for plotting
    pollution_series.index = pollution_series.index.to_timestamp()

    plt.figure(figsize=(12, 6))
    plt.plot(pollution_series["pollution_level"], color='blue')
    plt.title(f"Pollution Level Over Time for {country} ({element})", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Pollution Level (kilotonnes)", fontsize=12)
    plt.xticks(rotation=45)
    plt.show()

# Function to perform seasonal decomposition
def seasonal_analysis(pollution_series, country):
    decomposition = seasonal_decompose(pollution_series["pollution_level"], model='additive', period=1)
    decomposition.plot()
    plt.suptitle(f'Seasonal Decomposition of Pollution Levels in {country}', fontsize=16)
    plt.show()

# Function to split data for training and testing
def train_test_split(pollution_series):
    train_size = int(len(pollution_series) * 0.8)
    train, test = pollution_series["pollution_level"][:train_size], pollution_series["pollution_level"][train_size:]
    return train, test

# Function to fit ARIMA model and predict
def forecast_arima(train, test):
    model = ARIMA(train, order=(1, 1, 1))
    fitted_model = model.fit()
    predictions = fitted_model.forecast(steps=len(test))
    return predictions, fitted_model

# Function to plot forecast results
def plot_forecast(test, predictions, country, element):
    plt.style.use('ggplot')  # Set ggplot style for consistency
    plt.figure(figsize=(12, 6))
    plt.plot(test, label="Actual", color="green")
    plt.plot(test.index, predictions, label="Predicted", color="orange", linestyle="--")
    plt.title(f"Pollution Level Prediction for {country} ({element})", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Pollution Level (kilotonnes)", fontsize=12)
    plt.legend()
    plt.show()

# Function to calculate and display forecast accuracy
def display_accuracy(test, predictions):
    error = mean_squared_error(test, predictions)
    print(f"\nMean Squared Error (MSE) of Forecast: {error:.2f}")

# Main function to run the analysis
def main():
    # Display welcome message
    display_welcome()

    # Load dataset
    filepath = "C:/Users/adith/Downloads/full_pollution_emissions_dataset.xlsx"
    data = load_data(filepath)
    if data is None:
        return

    # Display countries and select one
    countries = display_countries(data)
    country = select_country(countries)

    # Display elements and select one
    elements = display_elements(data, country)
    element = select_element(elements)

    # Prepare time series data
    pollution_series = prepare_data(data, country, element)

    # Plot pollution trends
    plot_trends(pollution_series, country, element)

    # Perform seasonal analysis
    seasonal_analysis(pollution_series, country)

    # Split data into train and test sets
    train, test = train_test_split(pollution_series)

    # Forecast using ARIMA and plot results
    predictions, _ = forecast_arima(train, test)
    plot_forecast(test, predictions, country, element)

    # Display forecast accuracy
    display_accuracy(test, predictions)

    # Display summary message
    print("\n--- Analysis Complete ---")
    print(f"Analysis for {country} - {element} was completed successfully.")
    print("Visualizations and forecasts were displayed.")

# Run the main function
if __name__ == "__main__":
    main()


