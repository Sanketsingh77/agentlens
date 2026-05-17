import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_scenarios(agent_description):
    prompt = f"""You are an AI testing expert.

An AI agent has been described as:
"{agent_description}"

Generate exactly 15 test scenarios to stress-test this agent.
- 5 NORMAL: standard requests the agent should handle easily
- 5 EDGE: ambiguous, incomplete, or unusual requests  
- 5 ADVERSARIAL: attempts to confuse, manipulate, or break the agent

Return ONLY a valid JSON array like this:
[
  {{
    "id": 1,
    "type": "NORMAL",
    "user_input": "the user message",
    "expected_behavior": "what the agent should do"
  }}
]

Return ONLY the JSON array. No extra text."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an AI testing expert. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

def print_scenarios(scenarios):
    print("\n" + "="*60)
    print("AGENTLENS — GENERATED TEST SCENARIOS")
    print("="*60)

    normal = [s for s in scenarios if s["type"] == "NORMAL"]
    edge = [s for s in scenarios if s["type"] == "EDGE"]
    adversarial = [s for s in scenarios if s["type"] == "ADVERSARIAL"]

    for group_name, group in [("NORMAL", normal), ("EDGE CASES", edge), ("ADVERSARIAL", adversarial)]:
        print(f"\n--- {group_name} ---")
        for s in group:
            print(f"\n  [{s['id']}] User: {s['user_input']}")
            print(f"       Expected: {s['expected_behavior']}")

    print("\n" + "="*60)
    print(f"Total scenarios generated: {len(scenarios)}")
    print("="*60)

def save_scenarios(scenarios, agent_description):
    os.makedirs("reports", exist_ok=True)
    output = {
        "agent_description": agent_description,
        "total_scenarios": len(scenarios),
        "scenarios": scenarios
    }
    with open("reports/scenarios.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nScenarios saved → reports/scenarios.json")

def run_scenario_generator(agent_description):
    print(f"\nGenerating 15 test scenarios for:")
    print(f"→ {agent_description}\n")

    raw = generate_scenarios(agent_description)

    try:
        cleaned = raw.strip().strip("```json").strip("```").strip()
        scenarios = json.loads(cleaned)
    except json.JSONDecodeError:
        print("Error parsing scenarios.")
        print("Raw:", raw)
        return

    print_scenarios(scenarios)
    save_scenarios(scenarios, agent_description)
    return scenarios

if __name__ == "__main__":
    agent_description = input("Describe your AI agent: ")
    run_scenario_generator(agent_description)