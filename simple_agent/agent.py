import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime
from google import genai
from google.adk.agents.llm_agent import Agent


def get_current_time() -> str:
    """Returns the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_weather(city: str) -> dict:
    """Fetches real-time weather information for a given city.

    Args:
        city: Name of the city (e.g. 'London', 'Tokyo', 'San Francisco').

    Returns:
        A dictionary with current temperature (Celsius and Fahrenheit), condition,
        humidity, and wind speed.
    """
    clean_city = city.strip()
    encoded_city = urllib.parse.quote(clean_city)
    url = f"https://wttr.in/{encoded_city}?format=j1"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/7.88.1"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            current = data.get("current_condition", [{}])[0]
            condition = current.get("weatherDesc", [{}])[0].get("value", "Unknown")
            return {
                "city": clean_city,
                "temp_c": current.get("temp_C"),
                "temp_f": current.get("temp_F"),
                "condition": condition,
                "humidity_percent": current.get("humidity"),
                "wind_kmph": current.get("windspeedKmph"),
            }
    except Exception as exc:
        return {
            "city": clean_city,
            "error": f"Could not retrieve live weather: {str(exc)}",
        }


def generate_weather_meme(city: str, condition: str, joke_theme: str) -> str:
    """Generates a funny cartoon weather meme image matching the city weather and joke theme.

    Args:
        city: The name of the city.
        condition: The current weather condition (e.g., 'Heavy Rain', 'Sunny', 'Snow').
        joke_theme: A short humorous idea or character scenario to draw (e.g., 'A cat in yellow raincoat holding an umbrella', 'A melting snowman drinking iced tea').

    Returns:
        A confirmation message with the generated image file path.
    """
    try:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-04-bb71fffb09ef")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        client = genai.Client(vertexai=True, project=project, location=location)

        prompt = (
            f"A vibrant, funny cartoon meme illustration about weather in {city}. "
            f"Weather condition: {condition}. Scenario: {joke_theme}. "
            f"Comical, cute, expressive character, bright digital art style, high quality."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
        )

        static_dir = os.path.join(os.path.dirname(__file__), "..", "generated_memes")
        os.makedirs(static_dir, exist_ok=True)

        filename = f"weather_meme_{int(time.time())}.png"
        file_path = os.path.join(static_dir, filename)

        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                with open(file_path, "wb") as f:
                    f.write(part.inline_data.data)
                return f"Successfully generated weather meme! Saved to {file_path} (filename: {filename})."

        return "Could not extract image from model response."
    except Exception as exc:
        return f"Image generation skipped: {str(exc)}"


root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='A witty weather assistant delivering fast reports with an interactive riddle joke.',
    instruction=(
        'You are a witty weather reporter. '
        'When the user asks about the weather: '
        '1. If no city is provided, briefly ask them for the city. '
        '2. Use get_weather to fetch the live weather conditions for that city. '
        '3. ALWAYS structure your reply using these EXACT labels on separate lines: '
        '   [FORECAST]: 1-2 short sentences with the temperature (°C and °F) and sky conditions. '
        '   [JOKE_SETUP]: A funny riddle question or joke setup about the weather or city (e.g. "Why did the cloud stay home?"). '
        '   [JOKE_PUNCHLINE]: The punchline to the joke (e.g. "Because it was feeling under the weather!"). '
        'Make sure all three sections are present and clearly labeled. '
        'Keep the text concise and do not wait for image generation.'
    ),
    tools=[get_weather, get_current_time],
)



