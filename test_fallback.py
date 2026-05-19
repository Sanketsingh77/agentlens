# test_fallback.py
from llm_client import call_llm

result = call_llm(
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    temperature=0.3
)
print(result)