# AgentLens

AgentLens is an AI Agent Evaluation Platform built to test, debug, and evaluate customer-support AI agents across policy compliance, hallucination risk, tone quality, escalation handling, and security behavior.

The platform allows users to upload conversations, generate different test scenarios, simulate agent responses, and evaluate them with automated PASS/WARN/FAIL verdicts using multiple LLM providers with fallback handling.

---

## Live Demo

🔗 https://agentlens-ujyf.onrender.com

---

## Features

### Conversation Analyzer
- Upload customer-support conversations
- Analyze:
  - Tone quality
  - Policy compliance
  - Hallucination risk
  - Security handling
  - Resolution quality
- Generate structured evaluation reports

---

### Turn-by-Turn Debugger
- Debug AI responses message-by-message
- Identify:
  - Incorrect responses
  - Policy violations
  - Escalation failures
  - Unsafe behavior
- Suggest corrected responses and improvements

---

### Scenario Generator
- Auto-generates categorized AI evaluation scenarios:
  - Normal
  - Edge Cases
  - Adversarial
- Includes:
  - Severity tagging
  - Policy area classification
  - Failure modes
  - Escalation indicators
  - Expected behavior

---

### Scenario Evaluation Engine
- Run evaluation on generated scenarios
- Generates:
  - PASS / WARN / FAIL verdicts
  - Policy compliance checks
  - Tone evaluation
  - Security analysis
  - Hallucination analysis
  - Escalation correctness

---

### Multi-LLM Fallback Architecture
Supports multiple AI providers with automatic fallback handling:

- Groq
- Cerebras
- SambaNova
- OpenRouter

If one provider fails or rate-limits, the system automatically switches to another provider.

---

## Tech Stack

### Backend
- Python
- FastAPI

### Frontend
- HTML
- CSS
- JavaScript

### APIs / AI Providers
- Groq API
- Cerebras API
- SambaNova API
- OpenRouter API

### Deployment
- Render

### Tools
- Git
- GitHub

---

# Project Structure

```bash
agentlens/
│
├── static/
│   └── index.html
│
├── sample_conversations/
│
├── api.py
├── main.py
├── debugger.py
├── scenario_generator.py
├── llm_client.py
├── test_fallback.py
├── requirements.txt
└── README.md
