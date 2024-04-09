from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
class HuggingFaceLLM(LLM):
    model_data : dict
    max_new_tokens : int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        # run_manager: Optional[CallbackManagerForLLMRun] = None,
        # **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        '''
        messages = [{"role": "user", "content": prompt}]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=self.max_new_tokens, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens = True)
        return decoded[0]
        '''
        device = self.model_data['model'].device
        
        inputs = self.model_data['tokenizer'](prompt, return_tensors="pt").to(device)
        outputs = self.model_data['model'].generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.model_data['tokenizer'].decode(outputs[0], skip_special_tokens=True)
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_id": self.model_data['model'].name_or_path, 
                'device' : self.model_data['model'].device}