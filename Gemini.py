import requests
from enum import Enum
import requests , os
from dotenv import load_dotenv


load_dotenv() 

class AnalysisType(Enum):
    OFDM = "ofdm"
    COMM_SYSTEM = "communication system"
    CELLULAR = "cellular system"
    LINK_BUDGET = "link budget"

class GeminiAPIClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        self.last_result = None

    def forge_prompt(self, data: dict, results: dict, analysis: AnalysisType) -> str:
        if analysis == AnalysisType.OFDM:
            prompt = (
                "You are an expert in wireless communication. Based on the following simulation results, "
                "analyze the performance of an OFDM system here is the data :\n"
                f"{data}\n"
                 "here are the results:\n"
                f"{results}\n"
                "Include discussion on spectral efficiency, subcarrier allocation, and BER if available."
            )
        elif analysis == AnalysisType.COMM_SYSTEM:
            prompt = (
                    "You’re a communication system engineer. based on the following inputs:\n"
                    f"{data}\n\n"
                    "Analyze the following system results:\n"
                    f"{results}\n\n"
                    "Explain what each parameter and number means and how it affects overall system performance\n" 
                    "Include a discussion on:\n"
                        "- Modulation and its efficiency\n"
                        "- Sampling frequency and quantization levels\n"
                        "- Source and channel encoding rates\n"
                        "- Interleaving and burst formatting\n\n"
                    "\nSummarize the significance of these metrics in terms of data rate, reliability, and bandwidth efficiency. "
                    "\nEnsure the explanation is easy to understand for someone familiar with basic digital communication."
                )

        elif analysis == AnalysisType.CELLULAR:
            prompt = (
                "You are a cellular system engineer. Based on the following data and results, analyze the performance of a cellular system:\n data:\n"
                f"{data}\n"
                "results:\n"
                f"{results}\n"
                "Include insights on cell coverage, handoff behavior, interference, and capacity. and explain how the results where obtained."
            )
        elif analysis == AnalysisType.LINK_BUDGET:
         prompt = (
            "You are a professional radio frequency engineer tasked with analyzing a wireless communication link.\n\n"
            "Link parameters:\n"
            f"{data}\n\n"
            "Link budget results:\n"
            f"{results}\n\n"
            "Please provide a clear, concise explanation of the following:\n"
            "- What the received and transmitted power values indicate\n"
            "- The impact of path loss, antenna gains, amplifier gains, and other losses\n"
            "- Whether the link is considered strong, weak, or borderline\n"
            "- Any potential improvements or recommendations to optimize the link\n\n"
            "Make sure the explanation is understandable to someone with basic RF knowledge."
        )

        else:
            raise ValueError("Unsupported analysis type.")

        return prompt

    def generate_response(self, prompt: str):
        headers = {
            "Content-Type": "application/json"
        }
        body = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        response = requests.post(self.endpoint, headers=headers, json=body)

        if response.status_code == 200:
            data = response.json()
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                self.last_result = text
                return text
            except (KeyError, IndexError):
                raise ValueError("Unexpected response format.")
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")