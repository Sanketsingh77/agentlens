import os
import re
from dotenv import load_dotenv

load_dotenv()

# ─── FALLBACK CHAIN ───────────────────────────────────────
# Tries each provider in order. If one hits rate limit, moves to next.
# Order: Groq → Cerebras → SambaNova → OpenRouter

def call_llm(messages, temperature=0.3, max_tokens=4000):
    """
    Call LLM with automatic fallback across providers.
    Tries Groq first (best quality + speed), then falls back
    to Cerebras, SambaNova, and finally OpenRouter.
    """
    providers = [
        _call_groq,
        _call_sambanova,
        _call_cerebras,        
        _call_openrouter,
    ]

    last_error = None
    for provider in providers:
        try:
            result = provider(messages, temperature, max_tokens)
            if result:
                return result
        except Exception as e:
            err = str(e)
            if _is_rate_limit(err):
                print(f"  ⚠ Rate limit on {provider.__name__}, trying next provider...")
                last_error = e
                continue
            # Non-rate-limit error — raise immediately
            raise e

    raise Exception(
        f"All LLM providers rate limited or unavailable. "
        f"Last error: {last_error}. Please wait and try again."
    )


def _is_rate_limit(err: str) -> bool:
    """Detect rate limit and auth errors across all providers."""
    markers = [
        "429", "rate_limit", "rate limit", "quota",
        "exhausted", "too many", "rate_limit:",
        "401", "invalid_api_key", "authentication"
    ]
    return any(m.lower() in err.lower() for m in markers)


def _call_groq(messages, temperature, max_tokens):
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        print("  ✓ Using Groq (llama-3.3-70b)")
        return response.choices[0].message.content
    except Exception as e:
        err = str(e)
        if _is_rate_limit(err) or "401" in err or "invalid_api_key" in err or "Authentication" in err:
            raise Exception(f"rate_limit: {err}")
        raise e


def _call_sambanova(messages, temperature, max_tokens):
    sambanova_key = os.getenv("SAMBANOVA_API_KEY")
    if not sambanova_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://api.sambanova.ai/v1",
            api_key=sambanova_key
        )
        response = client.chat.completions.create(
            model="Meta-Llama-3.3-70B-Instruct",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        print("  ✓ Using SambaNova (llama-3.3-70b)")
        return response.choices[0].message.content
    except Exception as e:
        if _is_rate_limit(str(e)):
            raise e
        raise e
    
def _call_cerebras(messages, temperature, max_tokens):
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    if not cerebras_key:
        return None
    # Try multiple model names — Cerebras naming is inconsistent
    cerebras_models = [
    "llama-3.3-70b",
    "llama3.3-70b",
    "llama-3.1-8b",
    "llama3.1-8b",
    "llama-3.1-70b",
    ]
    from openai import OpenAI
    client = OpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=cerebras_key
    )
    for model_name in cerebras_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            print(f"  ✓ Using Cerebras ({model_name})")
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "404" in err or "not_found" in err or "does not exist" in err:
                print(f"  ⚠ Cerebras model {model_name} not found, trying next...")
                continue
            if _is_rate_limit(err):
                raise e
            raise e
    # All model names failed — skip to next provider
    print("  ⚠ No working Cerebras model found, skipping...")
    return None


def _call_openrouter(messages, temperature, max_tokens):
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key
        )
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b:free",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        print("  ✓ Using OpenRouter (gpt-oss-120b)")
        return response.choices[0].message.content
    except Exception as e:
        if _is_rate_limit(str(e)):
            raise e
        raise e


def extract_json_payload(raw_text: str) -> str:
    """Pull JSON out of markdown fences if present."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text