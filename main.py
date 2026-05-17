import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_conversation(filepath):
    with open(filepath, "r") as f:
        return f.read()

def analyze_conversation(conversation_text):
    prompt = f"""
You are an expert AI conversation quality evaluator.

Analyze this conversation and return ONLY valid JSON, nothing else:

{{
  "tone_score": <number 0-10>,
  "resolution_score": <number 0-10>,
  "hallucination_risk": "<low/medium/high>",
  "missed_opportunities": ["<string>", "<string>"],
  "overall_summary": "<2 sentence summary>",
  "recommendation": "<one actionable improvement>"
}}

Conversation to analyze:
{conversation_text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an AI conversation quality evaluator. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

def save_report(report_data, output_path):
    os.makedirs("reports", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Report saved to {output_path}")

def print_summary(report_data):
    print("\n" + "="*50)
    print("AGENTLENS — CONVERSATION ANALYSIS REPORT")
    print("="*50)
    print(f"Tone Score:         {report_data['tone_score']}/10")
    print(f"Resolution Score:   {report_data['resolution_score']}/10")
    print(f"Hallucination Risk: {report_data['hallucination_risk']}")
    print(f"\nSummary: {report_data['overall_summary']}")
    print(f"\nRecommendation: {report_data['recommendation']}")
    print(f"\nMissed Opportunities:")
    for item in report_data['missed_opportunities']:
        print(f"  - {item}")
    print("="*50 + "\n")

def main():
    conversation_file = "sample_conversations/conversation1.txt"

    print("Loading conversation...")
    conversation = load_conversation(conversation_file)

    print("Analyzing with AI...")
    raw_response = analyze_conversation(conversation)

    print("Parsing results...")
    try:
        report = json.loads(raw_response)
    except json.JSONDecodeError:
        print("Raw response:", raw_response)
        print("Error parsing JSON. Check raw response above.")
        return

    print_summary(report)
    save_report(report, "reports/report1.json")

if __name__ == "__main__":
    main()