import sys
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set the environment variable to use UTF-8 encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

def generate_text(prompt, model, tokenizer, max_length=600, temperature=0.7, top_k=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = input_ids.ne(tokenizer.eos_token_id).long()
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    model_path = './DraculaFlowGPT_1.0'
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    prompt = "Charged at her like a linebacker, pussy"
    generated_text = generate_text(prompt, model, tokenizer)
    
    print("Generated Text:")
    print(generated_text.encode('utf-8').decode('utf-8'))

if __name__ == "__main__":
    main()
