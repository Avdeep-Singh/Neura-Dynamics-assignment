from unittest.mock import patch, MagicMock
from src.tool import get_weather_info

# We need to provide the full path to the class we want to mock
WRAPPER_PATH = 'src.tool.OpenWeatherMapAPIWrapper'

@patch(WRAPPER_PATH)
def test_get_weather_info_success(MockWeatherWrapper):
    """Tests the weather tool with a successful API call by mocking the wrapper."""
    # 1. Create an instance of the mock wrapper
    mock_instance = MockWeatherWrapper.return_value
    
    # 2. Configure the 'run' method of that instance to return a predictable string
    mock_instance.run.return_value = "The weather in Paris is sunny with a temperature of 20°C."

    # 3. Call the tool
    city = "Paris"
    result = get_weather_info.invoke(city)

    # 4. Assertions
    assert "Paris" in result
    assert "sunny" in result
    
    # Ensure the wrapper was initialized and its 'run' method was called with the correct city
    MockWeatherWrapper.assert_called_once()
    mock_instance.run.assert_called_once_with(city)

@patch(WRAPPER_PATH)
def test_get_weather_info_error(MockWeatherWrapper):
    """Tests the weather tool's error handling by mocking the wrapper."""
    # 1. Create an instance of the mock wrapper
    mock_instance = MockWeatherWrapper.return_value
    
    # 2. Configure the 'run' method to raise an exception
    mock_instance.run.side_effect = Exception("API limit reached")

    # 3. Call the tool
    city = "London"
    result = get_weather_info.invoke(city)

    # 4. Assertions
    # The try/except block in our tool should catch the exception and format it
    assert "An error occurred while fetching weather data: API limit reached" in result