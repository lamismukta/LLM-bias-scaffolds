"""Google Gemini LLM provider implementation."""
import os
import google.generativeai as genai
from .base import LLMProvider, LLMResponse


class GeminiProvider(LLMProvider):
    """Google Gemini provider for Gemini models."""
    
    def __init__(self, model: str = "gemini-1.5-pro-latest", temperature: float = 1.0, max_tokens: int = 2000):
        super().__init__(model, temperature, max_tokens)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.client = genai
        
        # Validate model exists and get correct name
        self._validate_model()
    
    def _validate_model(self):
        """Validate model exists and get correct model name."""
        try:
            available_models = list(genai.list_models())
            model_names = [m.name.replace('models/', '') for m in available_models 
                          if 'generateContent' in m.supported_generation_methods]
            
            if not model_names:
                print("Warning: No Gemini models found with generateContent support")
                return
            
            # Print available models for debugging
            print(f"  Available Gemini models: {', '.join(model_names[:5])}")
            
            # Try to find matching model
            model_found = False
            requested_base = self.model.replace('-latest', '').replace('models/', '')
            
            for available_name in model_names:
                available_base = available_name.replace('models/', '')
                # Check exact match
                if self.model == available_name or self.model == f"models/{available_name}":
                    self.model = available_name
                    model_found = True
                    break
                # Check if base names match (e.g., "gemini-1.5-pro" matches "gemini-1.5-pro-001")
                if requested_base in available_base or available_base.startswith(requested_base):
                    self.model = available_name
                    model_found = True
                    print(f"  Matched '{self.model}' to available model '{available_name}'")
                    break
            
            if not model_found and model_names:
                # Use first available model as fallback
                original_model = self.model
                self.model = model_names[0]
                print(f"Warning: Model '{original_model}' not found. Using '{self.model}' instead.")
        except Exception as e:
            print(f"Warning: Could not validate Gemini model: {e}")
            # Continue with original model name
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using Gemini API."""
        import asyncio
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Create model instance
        # Handle models/ prefix if present
        model_name = self.model.replace("models/", "") if self.model.startswith("models/") else self.model
        
        # Try to create model, with fallback to first available model
        try:
            model_instance = genai.GenerativeModel(model_name)
        except Exception as e:
            # If model not found, try to get first available model
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                try:
                    print(f"  Model '{model_name}' not found, searching for alternatives...")
                    available_models = list(genai.list_models())
                    for m in available_models:
                        if 'generateContent' in m.supported_generation_methods:
                            fallback_name = m.name.replace('models/', '')
                            model_instance = genai.GenerativeModel(fallback_name)
                            print(f"  Using available Gemini model: {fallback_name}")
                            self.model = fallback_name  # Update for future calls
                            break
                    else:
                        raise ValueError("No Gemini models available for generateContent")
                except Exception as e2:
                    raise ValueError(f"Could not find Gemini model '{self.model}' and no alternatives available: {e2}")
            else:
                raise  # Re-raise if it's a different error
        
        # Build the full prompt with system message
        full_prompt = f"""You are an expert CV analyst with deep knowledge of recruitment and talent assessment.

{prompt}"""
        
        # Generate content
        # Note: Gemini uses generation_config for parameters
        # Run in executor since Gemini SDK may not be fully async
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model_instance.generate_content(
                full_prompt,
                generation_config=generation_config
            )
        )
        
        # Extract content - handle different response formats
        content = ""
        if hasattr(response, 'text'):
            content = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            content = response.candidates[0].content.parts[0].text if hasattr(response.candidates[0].content, 'parts') else str(response.candidates[0])
        
        # Extract usage information if available
        usage = None
        if hasattr(response, 'usage_metadata'):
            usage_meta = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(usage_meta, 'prompt_token_count', 0),
                "completion_tokens": getattr(usage_meta, 'completion_token_count', 0),
                "total_tokens": getattr(usage_meta, 'total_token_count', 0)
            }
        
        # Extract finish reason
        finish_reason = None
        if hasattr(response, 'candidates') and response.candidates:
            finish_reason = getattr(response.candidates[0], 'finish_reason', None)
        
        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
            metadata={"finish_reason": finish_reason}
        )
    
    def get_provider_name(self) -> str:
        return "gemini"

