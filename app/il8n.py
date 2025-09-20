from fastapi import APIRouter
import subprocess

router = APIRouter()

# M2M-100 translation via HuggingFace pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")


def translate_text_m2m(text: str, src: str = 'auto', target: str = 'en') -> str:
    m2m_tokenizer.src_lang = src
    encoded = m2m_tokenizer(text, return_tensors="pt")
    generated_tokens = m2m_model.generate(
        **encoded,
        forced_bos_token_id=m2m_tokenizer.get_lang_id(target)
    )
    return m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


# Ollama summarization
OLLAMA_MODEL = "ggml/assistant"  # change to your offline LLM


def summarize_with_ollama(text: str) -> str:
    prompt = f"Summarize the following groundwater data for a user-friendly answer:\n{text}"
    try:
        proc = subprocess.run(
            ["ollama", "generate", OLLAMA_MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
        return text
    except Exception:
        return text


@router.post('/translate')
async def translate_endpoint(payload: dict):
    text = payload.get('text')
    src = payload.get('src', 'auto')
    target = payload.get('target', 'en')
    res = translate_text_m2m(text, src=src, target=target)
    return {"translation": res}
