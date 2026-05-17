import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_conversation(filepath):
    with open(filepath, "r") as f:
        return f.read()

def parse_turns(conversation_text):
    lines = conversation_text.strip().split("\n")
    turns = []
    for line in lines:
        line = line.strip()
        if line.lower().startswith("user:"):
            turns.append({"role": "user", "message": line[5:].strip()})
        elif line.lower().startswith("agent:"):
            turns.append({"role": "agent", "message": line[6:].strip()})
    return turns

def debug_turn(turn_number, user_msg, agent_msg):
    prompt = f"""You are an AI conversation quality expert.

Analyze this single exchange between a user and an AI agent.

Turn {turn_number}:
User: {user_msg}
Agent: {agent_msg}

Evaluate the agent's response and return ONLY a JSON object like this:
{{
  "turn": {turn_number},
  "user_message": "{user_msg}",
  "agent_message": "{agent_msg}",
  "issue_detected": true or false,
  "severity": "NONE" or "LOW" or "MEDIUM" or "HIGH",
  "problem": "describe the problem or write 'None'",
  "better_response": "write a better agent response or write 'Response was good'"
}}

Return ONLY the JSON. No extra text."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def print_debug_report(filename, turns_analysis):
    print("\n" + "="*60)
    print(f"TURN-BY-TURN DEBUG REPORT: {filename}")
    print("="*60)

    issues_found = 0

    for item in turns_analysis:
        print(f"\nTurn {item['turn']}")
        print(f"  USER:  {item['user_message']}")
        print(f"  AGENT: {item['agent_message']}")

        if item.get("issue_detected"):
            issues_found += 1
            severity = item.get("severity", "UNKNOWN")
            print(f"  ⚠️  ISSUE DETECTED — Severity: {severity}")
            print(f"  Problem:        {item.get('problem')}")
            print(f"  Better Response: {item.get('better_response')}")
        else:
            print(f"  ✅ No issue detected")

    print("\n" + "-"*60)
    print(f"Total turns: {len(turns_analysis)}")
    print(f"Issues found: {issues_found}")
    print("="*60)

def save_debug_report(filename, turns_analysis):
    os.makedirs("reports", exist_ok=True)
    report_name = filename.replace(".txt", "_debug.json")
    with open(f"reports/{report_name}", "w") as f:
        json.dump(turns_analysis, f, indent=2)
    print(f"Debug report saved → reports/{report_name}")

def run_debugger(filepath):
    filename = os.path.basename(filepath)
    print(f"\nDebugging: {filename}")

    conversation_text = load_conversation(filepath)
    turns = parse_turns(conversation_text)

    # Pair up user + agent turns
    paired_turns = []
    i = 0
    turn_number = 1
    while i < len(turns) - 1:
        if turns[i]["role"] == "user" and turns[i+1]["role"] == "agent":
            paired_turns.append((turn_number, turns[i]["message"], turns[i+1]["message"]))
            turn_number += 1
            i += 2
        else:
            i += 1

    turns_analysis = []
    for turn_num, user_msg, agent_msg in paired_turns:
        print(f"  Analyzing turn {turn_num}...")
        raw = debug_turn(turn_num, user_msg, agent_msg)
        try:
            cleaned = raw.strip().strip("```json").strip("```").strip()
            result = json.loads(cleaned)
            turns_analysis.append(result)
        except json.JSONDecodeError:
            print(f"  Could not parse turn {turn_num}")

    print_debug_report(filename, turns_analysis)
    save_debug_report(filename, turns_analysis)
    return turns_analysis