# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 2024
This file contains the code for the tutorial on stance detection using LLMs.
Author: Mao Li
Last updated on 2025-03-22
"""

import pandas as pd
from tqdm import tqdm  # For progress bar
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Read the token from the file
with open("/home/felbasa/token.txt", "r") as token_file:
    token = token_file.read().strip()

# Log in to Hugging Face using the token
login(token=token)

device = "cuda"  # or "cpu" if you don't have GPU
# Load the data
stance = pd.read_excel("data/comments_to_code/merged_codes.xlsx")
# Remove #SemST from the tweet
stance["comment"] = stance["comment"].str.replace("#SemST", "", regex=False)

# Load the model
model_name = "google/gemma-3-12b-it"  # 8x7B MoE model
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto"  # Let the model decide where to load itself
)

# Define prompt elements from your R script
instructions = (
    "Instruction: You have assumed the role of a stakeholder that is presented "
    "with a reddit comment from likely federal workers related to the current policies "
    "on reducing the federal workforce. Please determine the author of the comment's stance "
    "on this topic, and only provide the answer."
)

prompt_template = (
    "Is this comment in 'favor', 'neutral', or 'oppose' the reduction in federal workforce? "
    "Provide one word answer only!\n\nComment: {comment}"
)

# Run inference
tqdm.pandas()
stance["LLM_stance"] = ""

for i, row in stance.iterrows():
    comment = row["comment"]
    
    # Prepare the prompt
    prompt = [
        {
            "role": "system",
            "content": [{"type": "text", "text": instructions}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_template.format(comment=comment)}
            ],
        },
    ]
    
    inputs = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(inputs, max_length=50, num_return_sequences=1)

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the last word (could be improved with regex if needed)
    stance.at[i, "LLM_stance"] = result.strip().split()[-1].lower()

# Save the result
stance.to_excel("outputs/reddit_comments_LLM_analysis.xlsx", index=False)
