import os
import json
import requests
from dotenv import load_dotenv
from .llm_client import LLMClient

# Load environment variables from .env file
load_dotenv()


class GeminiClient(LLMClient):
    """Google Gemini API client implementation."""

    def __init__(self):
        """Initialize Gemini client with API key from environment."""
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Please set it with your Google AI Studio API key."
            )
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    def generate_brief(self, prompt: str) -> str:
        """
        Generate a retention brief using Google Gemini 1.5 Flash.

        Args:
            prompt: The input prompt for the LLM

        Returns:
            Generated text response from Gemini

        Raises:
            Exception: If API call fails or response format is unexpected
        """
        try:
            # Construct request URL with API key
            url = f"{self.base_url}?key={self.api_key}"

            # Prepare request body
            request_body = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ]
            }

            # Make API request
            response = requests.post(url, json=request_body, timeout=30)

            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = f"Gemini API error (HTTP {response.status_code}): {response.text}"
                raise Exception(error_msg)

            # Parse response
            response_data = response.json()

            # Extract generated text
            if 'candidates' not in response_data or len(response_data['candidates']) == 0:
                raise Exception(f"Unexpected API response format: {response_data}")

            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
            return generated_text

        except requests.exceptions.Timeout:
            raise Exception("Gemini API request timed out after 30 seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error calling Gemini API: {str(e)}")
        except KeyError as e:
            raise Exception(f"Failed to extract text from Gemini response - missing key: {str(e)}")
        except Exception as e:
            raise Exception(f"Error generating brief with Gemini: {str(e)}")
