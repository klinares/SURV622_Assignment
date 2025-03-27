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

device = "cuda"
# Load the data
stance = pd.read_csv("../data/SemEval2016-testdata-taskA-all-annotations.csv")
# Remove #SemST from the tweet
stance["Tweet"] = stance["Tweet"].str.replace("#SemST", "")

# Load the model
model_name = "google/gemma-3-12b-it"  # 8x7B MoE model\
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map=device
)  # Load model

# Loop through the tweets and generate the stance
for i, tweet in enumerate(tqdm(stance["Tweet"])):
    # TODO: Define the prompt
    prompt = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Topic: {stance.loc[i, 'Target']} Tweet: {tweet}",
                }
            ],
        },
    ]  # It's a example, you will need to change it
    inputs = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    output = model.generate(input, max_length=20, num_return_sequences=1)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    stance.loc[i, "LLM_stance"] = result

# Save the result
stance.to_csv("../data/stance_with_LLM.csv", index=False)
