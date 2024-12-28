import os
import time

import os
import re
import ast
import pandas as pd
import torch
import transformers
from nltk.corpus import wordnet as wn
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder
from torch.nn import CosineSimilarity
import re
import pandas as pd
import ast

import os
import openai
# import dotenv
from openai import OpenAI
import json

def get_llm_score(article, pip_line):

    pipeline = pip_line


    system_prompt = "You are a psychologist specializing in personality analysis. Your task is to assess the Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) as reflected in the text provided."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "For each trait, please determine whether the personality trait in the article appears higher or lower than the average human level. Provide a brief explanation for your judgment." + article}
    ]
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    outputs = pipeline(
        messages,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    generated_text = outputs[0]["generated_text"][-1]["content"]


    return generated_text




model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
def analysis_article(base_dir):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    for i in range(1, 2):  # 假设有 5 个迭代
        folder = f"result_iteration_{i}"
        folder_path = os.path.join(base_dir, folder)


        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        # 遍历文件夹中的所有文件
        for file in os.listdir(folder_path):
            if file.startswith("result-generate") and file.endswith(".txt"):

                file_path = os.path.join(folder_path, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()


                        # 提取 "Iteration 1 answer:" 后面的内容
                        start_key = "Iteration 1 answer:"
                        start_index = content.find(start_key)

                        if start_index != -1:
                            start_index += len(start_key)
                            # 从 start_index 开始提取所有内容
                            answer = content[start_index:].strip()  # 读取从 start_key 之后的所有内容
                            score = get_llm_score(pip_line=pipeline,article= answer)

                            print(f"File: {file}\nAnswer: {answer}\n")
                            print(f"LLM SCORE: \n CONTENT: {score}")
                        else:
                            print(f"Key '{start_key}' not found in file: {file}")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")


# 使用示例
base_directory = "/home/hmsun/wordnet/1223/all_results/meta-llama/Meta-Llama-3.1-8B-Instruct/vanilla"
analysis_article(base_directory)

