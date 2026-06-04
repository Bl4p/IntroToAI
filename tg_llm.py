import torch
import math
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TG_LLM:
    def __init__(self, model_name="distilroberta-base"):
        # If you have fine-tuned the model locally, load that. Otherwise, pull the base LLM.
        local_model_path = "./tg-roberta-finetuned"
        if os.path.exists(local_model_path):
            print(f"[TG-LLM] Loading your fine-tuned LLM from {local_model_path}...")
            model_to_load = local_model_path
        else:
            print(f"[TG-LLM] Loading base LLM from Hugging Face: {model_name}...")
            model_to_load = model_name

        # Load Tokenizer and the LLM (num_labels=1 replaces the word-generation head with a continuous float output)
        self.tokenizer = AutoTokenizer.from_pretrained(model_to_load)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_to_load, num_labels=1)
        self.model.eval() 
        print("[TG-LLM] Risk Engine initialized and ready.")

    def _haversine_distance(self, geo1, geo2):
        lat1, lon1 = geo1
        lat2, lon2 = geo2
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return 2 * 6371 * math.asin(math.sqrt(a))

    def _create_llm_prompt(self, current_telemetry, history):
        """
        Translates raw database numbers into a natural language prompt for the LLM.
        """
        prompt = "Analyze the following user login sequence for security threats:\n"
        
        if not history:
            prompt += "[CONTEXT] This is the user's first login attempt. No history exists.\n"
        else:
            prompt += "[HISTORY]\n"
            for i, log in enumerate(history[:5]): # Limit to last 5 logs for context window limits
                prompt += f"Log {i+1}: IP address {log['ip']}, Location coordinates {log['geo'][0]},{log['geo'][1]}, Hardware Device {log['device_hash'][:8]}.\n"
            
            last_geo = history[0]['geo']
            distance = self._haversine_distance(current_telemetry['geo'], last_geo)
            prompt += f"[ANALYSIS] The new login is {distance:.2f} kilometers away from the previous location.\n"

        prompt += f"\n[CURRENT ATTEMPT] IP address {current_telemetry['ip']}, Location coordinates {current_telemetry['geo'][0]},{current_telemetry['geo'][1]}, Hardware Device {current_telemetry['device_hash'][:8]}."
        
        return prompt

    def evaluate_risk(self, current_telemetry, history):
        if not history:
             return 0.05, 1.0

        # 1. Generate the English text sequence
        text_sequence = self._create_llm_prompt(current_telemetry, history)
        
        # 2. Tokenize the words for the LLM
        inputs = self.tokenizer(text_sequence, return_tensors="pt", truncation=True, max_length=512)
        
        # 3. Forward pass through the LLM
        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_logit = outputs.logits.item()
            
        # 4. Map the LLM's raw output to S_anomaly using a Sigmoid activation (bounds between 0.0 and 1.0)
        s_anomaly = 1.0 / (1.0 + math.exp(-raw_logit))
        
        # 5. Dynamic Cryptographic Threshold Modulation
        alpha = 1.0 - (s_anomaly * 0.7)
        alpha = max(0.3, min(alpha, 1.0)) # Enforce architectural limits
        
        return s_anomaly, alpha