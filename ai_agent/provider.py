import os


class LLMProvider:
    def __init__(self):
        self._provider = os.getenv("AI_PROVIDER", "ollama")
        self._ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

    def call(self, prompt: str) -> str:
        if self._provider == "ollama":
            return self._call_ollama(prompt)
        elif self._provider == "claude":
            return self._call_claude(prompt)
        elif self._provider == "openai":
            return self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown AI provider: {self._provider}")

    def _call_ollama(self, prompt: str) -> str:
        import ollama
        response = ollama.chat(
            model=self._ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def _call_claude(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _call_openai(self, prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
