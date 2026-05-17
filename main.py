from debugger import run_debugger
import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_conversation(filepath):
    """Reads a conversation text file and returns its content."""
    with open(filepath, "r") as f:
        return f.read()

def analyze_conversation(conversation_text):
    """Sends conversation to Groq AI and gets back a structured quality report."""
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
    """Saves individual JSON report to reports/ folder."""
    os.makedirs("reports", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Report saved → {output_path}")

def print_summary(filename, report_data):
    """Prints a clean readable summary to the terminal."""
    print("\n" + "="*55)
    print(f"AGENTLENS — REPORT: {filename}")
    print("="*55)
    print(f"  Tone Score:         {report_data['tone_score']}/10")
    print(f"  Resolution Score:   {report_data['resolution_score']}/10")
    print(f"  Hallucination Risk: {report_data['hallucination_risk']}")
    print(f"\n  Summary: {report_data['overall_summary']}")
    print(f"\n  Recommendation: {report_data['recommendation']}")
    print(f"\n  Missed Opportunities:")
    for item in report_data['missed_opportunities']:
        print(f"    - {item}")
    print("="*55)

def generate_comparison(all_reports):
    """
    Takes all individual reports and asks AI to compare them.
    Returns a final comparison summary.
    """
    print("\nGenerating comparison report...")

    reports_text = json.dumps(all_reports, indent=2)

    prompt = f"""
You are an AI quality analyst. 

Here are quality reports for multiple customer service conversations:

{reports_text}

Compare all conversations and return ONLY valid JSON:

{{
  "best_conversation": "<filename>",
  "worst_conversation": "<filename>",
  "best_reason": "<why it scored highest>",
  "worst_reason": "<why it scored lowest>",
  "overall_insights": ["<insight 1>", "<insight 2>", "<insight 3>"],
  "team_recommendation": "<one recommendation for the whole support team>"
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an AI conversation quality analyst. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

def print_comparison(comparison):
    """Prints the final comparison report."""
    print("\n" + "*"*55)
    print("AGENTLENS — FINAL COMPARISON REPORT")
    print("*"*55)
    print(f"  Best Conversation:  {comparison['best_conversation']}")
    print(f"  Why:                {comparison['best_reason']}")
    print(f"\n  Worst Conversation: {comparison['worst_conversation']}")
    print(f"  Why:                {comparison['worst_reason']}")
    print(f"\n  Key Insights:")
    for insight in comparison['overall_insights']:
        print(f"    → {insight}")
    print(f"\n  Team Recommendation: {comparison['team_recommendation']}")
    print("*"*55 + "\n")

def main():
    conversations_folder = "sample_conversations"
    all_reports = {}

    # Get all .txt files in the folder
    conversation_files = [
        f for f in os.listdir(conversations_folder)
        if f.endswith(".txt")
    ]

    print(f"\nFound {len(conversation_files)} conversations to analyze.")
    print("="*55)

    # Analyze each conversation one by one
    for filename in conversation_files:
        filepath = os.path.join(conversations_folder, filename)
        print(f"\nAnalyzing: {filename}")

        conversation = load_conversation(filepath)
        raw_response = analyze_conversation(conversation)

        try:
            cleaned = raw_response.strip().strip("```json").strip("```").strip()
            report = json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"  Error parsing response for {filename}")
            print(f"  Raw response: {raw_response}")
            continue

        print_summary(filename, report)

        report_name = filename.replace(".txt", "_report.json")
        save_report(report, f"reports/{report_name}")

        all_reports[filename] = report

    # Generate final comparison across all conversations
    if len(all_reports) > 1:
        raw_comparison = generate_comparison(all_reports)

        try:
            cleaned = raw_comparison.strip().strip("```json").strip("```").strip()
            comparison = json.loads(cleaned)
            print_comparison(comparison)
            save_report(comparison, "reports/final_comparison.json")
            print("Final comparison saved → reports/final_comparison.json")
        except json.JSONDecodeError:
            print("Error parsing comparison response.")
            print("Raw:", raw_comparison)
            
    print("\n" + "="*60)
    print("RUNNING TURN-BY-TURN DEBUGGER ON ALL CONVERSATIONS")
    print("="*60)
    for filename in conversation_files:
        filepath = os.path.join(conversations_folder, filename)
        run_debugger(filepath)

if __name__ == "__main__":
    main()