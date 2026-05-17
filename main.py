from debugger import run_debugger
from scenario_generator import run_scenario_generator
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
    print(f"  Report saved → {output_path}")

def print_summary(filename, report_data):
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

def run_analysis_pipeline(conversations_folder="sample_conversations"):
    """Runs batch analysis + comparison on all conversations in the folder."""
    all_reports = {}

    if not os.path.exists(conversations_folder):
        print(f"Folder '{conversations_folder}' not found. Skipping analysis.")
        return all_reports

    conversation_files = [
        f for f in os.listdir(conversations_folder)
        if f.endswith(".txt")
    ]

    if not conversation_files:
        print("No .txt conversation files found. Skipping analysis.")
        return all_reports

    print(f"\nFound {len(conversation_files)} conversations to analyze.")
    print("="*55)

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

    return all_reports

def run_debugger_pipeline(conversations_folder="sample_conversations"):
    """Runs turn-by-turn debugger on all conversations."""
    if not os.path.exists(conversations_folder):
        print(f"Folder '{conversations_folder}' not found. Skipping debugger.")
        return

    conversation_files = [
        f for f in os.listdir(conversations_folder)
        if f.endswith(".txt")
    ]

    if not conversation_files:
        print("No .txt files found. Skipping debugger.")
        return

    print("\n" + "="*55)
    print("RUNNING TURN-BY-TURN DEBUGGER")
    print("="*55)

    for filename in conversation_files:
        filepath = os.path.join(conversations_folder, filename)
        run_debugger(filepath)

def run_scenario_pipeline():
    """Asks user for agent description and generates test scenarios."""
    print("\n" + "="*55)
    print("SCENARIO GENERATOR")
    print("="*55)
    print("Describe the AI agent you want to test.")
    print("Example: A customer support agent for an e-commerce store")
    print()
    agent_description = input("Agent description: ").strip()

    if not agent_description:
        print("No description provided. Skipping scenario generation.")
        return

    run_scenario_generator(agent_description)

def print_menu():
    print("\n" + "="*55)
    print("  AGENTLENS — AI Agent Testing & Evaluation Tool")
    print("="*55)
    print("  What do you want to do?\n")
    print("  [1] Analyze conversations (batch + comparison)")
    print("  [2] Debug conversations turn-by-turn")
    print("  [3] Generate test scenarios for an AI agent")
    print("  [4] Run everything (full pipeline)")
    print("  [0] Exit")
    print("="*55)

def main():
    while True:
        print_menu()
        choice = input("\nEnter choice (0-4): ").strip()

        if choice == "1":
            run_analysis_pipeline()

        elif choice == "2":
            run_debugger_pipeline()

        elif choice == "3":
            run_scenario_pipeline()

        elif choice == "4":
            print("\nRunning full AgentLens pipeline...\n")
            run_analysis_pipeline()
            run_debugger_pipeline()
            run_scenario_pipeline()

        elif choice == "0":
            print("\nExiting AgentLens. Goodbye!\n")
            break

        else:
            print("\nInvalid choice. Please enter 0, 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()