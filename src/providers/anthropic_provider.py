"""Anthropic Claude LLM provider implementation."""
import os
from anthropic import AsyncAnthropic
from .base import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    """Anthropic provider for Claude models."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20240620", temperature: float = 1.0, max_tokens: int = 2000):
        super().__init__(model, temperature, max_tokens)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = AsyncAnthropic(api_key=api_key)
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using Anthropic API."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are an expert CV analyst with deep knowledge of recruitment and talent assessment.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract content (Anthropic returns a list of content blocks)
        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text
                elif isinstance(block, dict) and 'text' in block:
                    content += block['text']
        
        return LLMResponse(
            content=content,
            model=self.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            metadata={"stop_reason": response.stop_reason}
        )
    
    def get_provider_name(self) -> str:
        return "anthropic"

