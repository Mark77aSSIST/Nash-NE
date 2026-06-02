# 3. LLM Model Rapper; Supports system_prompt delivery by dataset

import time
import openai
import anthropic
from google import genai
from google.genai import types as genai_types

if "gpt"    in ACTIVE_MODELS: oai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
if "claude" in ACTIVE_MODELS: ant_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
if "gemini" in ACTIVE_MODELS: gem_client = genai.Client(api_key=GEMINI_API_KEY)

# Each model function: Add a system_prompt factor
def call_gpt(question, temperature, max_tokens,
             system_prompt="You are a helpful assistant."):
    safe_max = max(max_tokens, 512)   # Reasoning models consume thinking tokens first, thus guaranteeing a minimum of 512

    
    token_ladder = [safe_max, 1024, 2048, 4096]   # Token up phase (including current value, attempt sequential if exceeded)
    token_ladder = sorted(set(token_ladder))  

    for token_limit in token_ladder:
        for attempt in range(2):  
            try:
                resp = oai_client.chat.completions.create(
                    model="gpt-5-mini-2025-08-07",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": question},
                    ],
                    max_completion_tokens=token_limit,
                )
                content = resp.choices[0].message.content

                if content is None:
                    print(f"  gpt content=None "
                          f"(token_limit={token_limit}, "
                          f"usage={resp.usage})")
                    break

                return content.strip()

            except Exception as e:
                err_str = str(e)

                # Output token excess error detection → retry after token up
                if ("max_tokens" in err_str or
                    "output limit" in err_str or
                    "finish_reason" in err_str) and \
                        token_limit < token_ladder[-1]:
                    print(f"  gpt Token exceeded "
                          f"(token_limit={token_limit}) "
                          f"→ {min(t for t in token_ladder if t > token_limit)} retrial")
                    break  

                # Other errors (network, rate limit, etc.) → Retry with the same token
                print(f"  gpt Error "
                      f"(token_limit={token_limit}, "
                      f"Try {attempt+1}/2): {err_str[:80]}")
                time.sleep(2)

    print(f"  Failed with gpt maximum token({token_ladder[-1]}) — return empty response") , 
    return ""

def call_claude(question: str, temperature: float,
                max_tokens: int, system_prompt: str) -> str:
    resp = ant_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text.strip()

def call_gemini(question, temperature, max_tokens, system_prompt="..."):
    safe_max_tokens = max(max_tokens, 512)
    for attempt in range(3):
        try:
            resp = gem_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{system_prompt}\n\n{question}",
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=safe_max_tokens,
                ),
            )
            
            if not resp.candidates:
                print(f"  No gemini candidates ")
                return ""
            candidate = resp.candidates[0]
            if not candidate.content or not candidate.content.parts:
                print(f"  No gemini content "
                      f"(finish_reason={candidate.finish_reason})")
                return ""
            return resp.text.strip()
        except Exception as e:
            print(f"  gemini Error (Try {attempt+1}/3): {e}")
            time.sleep(2)
    return ""

MODEL_FN_MAP = {"gpt": call_gpt, "claude": call_claude, "gemini": call_gemini}

def call_model(model_name: str, question: str,
               temperature: float = None,
               max_tokens: int = None,
               system_prompt: str = None,
               retry: int = 3) -> str:

    temperature   = temperature   or GEN_CFG["temperature"]
    max_tokens    = max_tokens    or GEN_CFG["max_tokens"]
    system_prompt = system_prompt or "You are a helpful assistant."
    fn = MODEL_FN_MAP[model_name]

    for attempt in range(retry):
        try:
            result = fn(question, temperature, max_tokens, system_prompt)
            time.sleep(0.3)
            return result
        except Exception as e:
            print(f"  {model_name} Error (Try {attempt+1}/{retry}): {e}")
            time.sleep(2 ** attempt)
    return ""

print(f" {len(ACTIVE_MODELS)}ea Model Rapper Ready: {ACTIVE_MODELS}")