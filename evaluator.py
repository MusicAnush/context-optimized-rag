from openai import OpenAI

class RAGEvaluator:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)

    def evaluate_faithfulness(self, query: str, context: str, response: str) -> float:
        """
        Measures deterministic faithfulness (groundedness).
        Returns a score from 0.0 to 1.0 indicating if the response hallucinated.
        """
        prompt = (
            "Analyze the following Response against the provided Context. Determine if every claim made in "
            "the Response is fully supported by the Context. Ignore external knowledge.\n\n"
            f"Context: {context}\n"
            f"Response: {response}\n\n"
            "Output exactly a single floating point score between 0.0 (completely hallucinated or unsupported) "
            "and 1.0 (perfectly faithful and entirely supported by the context). Do not include text or markdown."
        )
        
        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return float(res.choices[0].message.content.strip())
        except Exception:
            return 0.0
