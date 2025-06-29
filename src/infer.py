"""
J.A.R.V.I.S. Inference Module
Provides conversational AI capabilities for the trained J.A.R.V.I.S. model.
"""

import torch
import logging
from typing import List, Dict, Optional, Generator
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import json
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JARVISInference:
    """J.A.R.V.I.S. inference engine for conversational AI."""
    
    def __init__(self, model_path: str = "models/JARVIS", 
                 max_length: int = 2048, temperature: float = 0.7):
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.temperature = temperature
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the trained J.A.R.V.I.S. model."""
        logger.info(f"Loading J.A.R.V.I.S. model from: {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path.as_posix()))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path.as_posix()),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Check if LoRA adapter exists
            adapter_path = self.model_path / "adapter_config.json"
            if adapter_path.exists():
                logger.info("Loading LoRA adapter...")
                self.model = PeftModel.from_pretrained(base_model, str(self.model_path.as_posix()))
            else:
                self.model = base_model
            
            self.model.eval()
            logger.info("J.A.R.V.I.S. model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256, 
                         temperature: Optional[float] = None) -> str:
        """Generate a response to the given prompt."""
        
        if temperature is None:
            temperature = self.temperature
        
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                               max_length=self.max_length - max_new_tokens)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generation configuration
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
            **inputs, 
                generation_config=generation_config,
                use_cache=True
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                        skip_special_tokens=True)
        return response.strip()
    
    def chat(self, message: str, conversation_history: Optional[List[Dict]] = None,
             max_new_tokens: int = 256) -> Dict:
        """Engage in a conversation with J.A.R.V.I.S."""
        
        if conversation_history is None:
            conversation_history = []
        
        # Format conversation context
        context = self._format_conversation_context(conversation_history, message)
        
        # Generate response
        response = self.generate_response(context, max_new_tokens)
        
        # Update conversation history
        conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": time.time()
        })
        conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": time.time()
        })
        
        return {
            "response": response,
            "conversation_history": conversation_history,
            "model_info": {
                "name": "J.A.R.V.I.S.",
                "model_path": str(self.model_path),
                "generation_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": self.temperature
                }
            }
        }
    
    def _format_conversation_context(self, history: List[Dict], current_message: str) -> str:
        """Format conversation history into a prompt with proper structure."""
        
        # Start with system prompt
        context_parts = ["J.A.R.VIS is an AI assistant. Be helpful, friendly, and concise."]
        context_parts.append("")
        
        if not history:
            # First message - simple format
            context_parts.append(f"User: {current_message}")
            context_parts.append("Assistant:")
            return "\n".join(context_parts)
        
        # Add recent conversation history (last 4 exchanges to manage context length)
        recent_history = history[-8:]  # Last 8 messages (4 exchanges)
        
        for msg in recent_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")
        
        # Add current message
        context_parts.append(f"User: {current_message}")
        context_parts.append("Assistant:")
        
        return "\n".join(context_parts)
    
    def stream_response(self, prompt: str, max_new_tokens: int = 256,
                       temperature: Optional[float] = None) -> Generator[str, None, None]:
        """Stream response tokens for real-time interaction."""
        
        if temperature is None:
            temperature = self.temperature
        
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=self.max_length - max_new_tokens)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generation configuration for streaming
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Stream generation
        with torch.no_grad():
            streamer = self.model.generate(
                **inputs,
                generation_config=generation_config,
                use_cache=True,
                streamer=None,  # We'll handle streaming manually
                return_dict_in_generate=True,
                output_scores=False
            )
            
            # Stream tokens
            for i in range(len(streamer.sequences[0]) - inputs['input_ids'].shape[1]):
                token = streamer.sequences[0][inputs['input_ids'].shape[1] + i]
                decoded_token = self.tokenizer.decode([token], skip_special_tokens=True)
                yield decoded_token
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": "J.A.R.V.I.S.",
            "model_path": str(self.model_path),
            "device": str(self.device),
            "max_length": self.max_length,
            "temperature": self.temperature,
            "parameters": self.model.num_parameters() if self.model else 0,
            "model_type": "LoRA" if hasattr(self.model, 'peft_config') else "Full"
        }

def interactive_chat():
    """Interactive chat interface with J.A.R.V.I.S."""
    
    print("🤖 J.A.R.V.I.S. AI Assistant")
    print("=" * 50)
    print("Type 'quit' to exit, 'info' for model information")
    print()
    
    # Initialize J.A.R.V.I.S.
    try:
        jarvis = JARVISInference()
        print("✅ J.A.R.V.I.S. is ready!")
        print()
    except Exception as e:
        print(f"❌ Failed to load J.A.R.V.I.S.: {e}")
        return
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("J.A.R.V.I.S.: Goodbye! It was a pleasure assisting you.")
                break
            
            elif user_input.lower() == 'info':
                info = jarvis.get_model_info()
                print(f"\n📊 Model Information:")
                print(f"   Name: {info['model_name']}")
                print(f"   Parameters: {info['parameters']:,}")
                print(f"   Device: {info['device']}")
                print(f"   Model Type: {info['model_type']}")
                print()
                continue
            
            elif not user_input:
                continue
            
            # Generate response
            print("J.A.R.V.I.S.: ", end="", flush=True)
            
            # Stream response for better UX
            response_parts = []
            for token in jarvis.stream_response(
                jarvis._format_conversation_context(conversation_history, user_input)
            ):
                print(token, end="", flush=True)
                response_parts.append(token)
            
            print()  # New line after response
            
            # Update conversation history
            response = "".join(response_parts)
            conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time()
            })
            conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": time.time()
            })
            
            print()  # Empty line for readability
            
        except KeyboardInterrupt:
            print("\n\nJ.A.R.V.I.S.: Goodbye! Feel free to return anytime.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

def main():
    """Main function for inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. Inference")
    parser.add_argument("--model-path", default="models/JARVIS/pretrained", 
                       help="Path to the trained model")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive chat mode")
    parser.add_argument("--prompt", type=str, help="Single prompt to process")
    parser.add_argument("--max-tokens", type=int, default=256,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_chat()
    elif args.prompt:
        jarvis = JARVISInference(args.model_path, temperature=args.temperature)
        response = jarvis.generate_response(args.prompt, args.max_tokens)
        print(f"Prompt: {args.prompt}")
        print(f"Response: {response}")
    else:
        print("Use --interactive for chat mode or --prompt for single response")

if __name__ == "__main__":
    main() 