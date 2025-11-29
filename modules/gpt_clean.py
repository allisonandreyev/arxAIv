from textwrap import dedent

from openai import OpenAI

client = OpenAI()

def gpt_clean_text(raw_ocr: str) -> str:
    """Lightly repair OCR output without inventing new content."""
    prompt = dedent(
        f"""
        You are repairing OCR output from a scientific machine learning paper.

        RULES:
        - Do NOT add new content.
        - Do NOT invent citations, institutions, datasets, or values.
        - Only fix: OCR errors, spacing, punctuation, broken words, missing accents, and grammar.
        - Preserve the scientific meaning and structure.
        - If a word is unreadable, choose the closest valid English or ML term.
        - If a sentence is nonsense, rewrite it cleanly but preserve the nonsense.

        Clean this OCR text EXACTLY:
        ----------------------------
        {raw_ocr}
        ----------------------------
        """
    )
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()
