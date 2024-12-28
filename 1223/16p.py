from fastapi import requests
from numpy.ma import copy
from random import random
from streamlit import json
import copy
import requests
import os

import json

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

#model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#model_id = "meta-llama/Llama-3.2-3B-Instruct"
model_id = "Qwen/Qwen2.5-3B-Instruct"
# model_id = "qwen2.5-72b-instruct"

words_num = 10
iteration = 5


def getQwenClient():
    openai_api_key = "qwen2.5-72b-instruct-8eeac2dad9cc4155af49b58c6bca953f"

    openai_api_base = "https://its-tyk1.polyu.edu.hk:8080/llm/qwen2.5-72b-instruct"

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client


test_template={
    "combinations":[
        {
            'target_labels':['E', 'N','F', 'J', 'ENFJ'],
            'opposite_labels' : ['I', 'S', 'T','P', 'ISTP'],
            'label_type':"['E','N','F', 'J']",
            'label_prompt':"You are an ENFJ person.",
            'prompt':"You are an extrovert, intuitive, feeling, and judging person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels':['E', 'N', 'F', 'P','ENFP'],
            'opposite_labels' : ['I', 'S', 'T', 'J','ISTJ'],
            'label_type':"['E', 'N', 'F','P']",
            'label_prompt':"You are an ENFP person.",
            'prompt':"You are an extrovert, intuitive, feeling, and prospecting person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        },{
            'target_labels':['E', 'N','T', 'J', 'ENTJ'],
            'opposite_labels' : ['I', 'S', 'F','P', 'ISFP'],
            'label_type':"['E', 'N','T', 'J']",
            'label_prompt':"You are an ENTJ person.",
            'prompt':"You are an extrovert, intuitive, thinking, and judging person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels':['E', 'N', 'T', 'P','ENTP'],
            'opposite_labels' : ['I', 'S', 'F', 'J','ISFJ'],
            'label_type':"['E', 'N', 'T','P']",
            'label_prompt':"You are an ENTP person.",
            'prompt':"You are an extrovert, intuitive, thinking, and prospecting person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        },{
            'target_labels':['E', 'S','F', 'J', 'ESFJ'],
            'opposite_labels' : ['I', 'N', 'T','P', 'INTP'],
            'label_type':"['E', 'S','F', 'J']",
            'label_prompt':"You are an ESFJ person.",
            'prompt':"You are an extrovert, observant, feeling, and judging person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels':['E', 'S', 'F', 'P','ESFP'],
            'opposite_labels' : ['I', 'N', 'T', 'J','INTJ'],
            'label_type':"['E', 'S', 'F','P']",
            'label_prompt':"You are an ESFP person.",
            'prompt':"You are an extrovert, observant, feeling, and prospecting person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        },{
            'target_labels':['E', 'S','T', 'J', 'ESTJ'],
            'opposite_labels' : ['I', 'N', 'F','P', 'INFP'],
            'label_type':"['E', 'S','T', 'J']",
            'label_prompt':"You are an ESTJ person.",
            'prompt':"You are an extrovert, observant, thinking, and judging person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels':['E', 'S', 'T', 'P','ESTP'],
            'opposite_labels' : ['I', 'N', 'F', 'J','INFJ'],
            'label_type':"['E', 'S', 'T','P']",
            'label_prompt':"You are an ESTP person.",
            'prompt':"You are an extrovert, observant, thinking, and prospecting person. You prefer group activities and get energized by social interaction. You tend to be more enthusiastic and more easily excited. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        },{
            'target_labels':['I', 'N','F', 'J', 'INFJ'],
            'opposite_labels' : ['E', 'S', 'T','P', 'ESTP'],
            'label_type':"['I', 'N','F', 'J']",
            'label_prompt':"You are an INFJ person.",
            'prompt':"You are an introvert, intuitive, feeling, and judging person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels':['I', 'N', 'F', 'P','INFP'],
            'opposite_labels' : ['E', 'S', 'T', 'J','ESTJ'],
            'label_type':"['I', 'N', 'F','P']",
            'label_prompt':"You are an INFP person.",
            'prompt':"You are an introvert, intuitive, feeling, and prospecting person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        },{
            'target_labels':['I', 'N','T', 'J', 'INTJ'],
            'opposite_labels' : ['E', 'S', 'F','P', 'ESFP'],
            'label_type':"['I', 'N','T', 'J']",
            'label_prompt':"You are an INTJ person.",
            'prompt':"You are an introvert, intuitive, thinking, and judging person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels':['I', 'N', 'T', 'P','INTP'],
            'opposite_labels' : ['E', 'S', 'F', 'J','ESFJ'],
            'label_type':"['I', 'N', 'T','P']",
            'label_prompt':"You are an INTP person.",
            'prompt':"You are an introvert, intuitive, thinking, and prospecting person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are very imaginative, open-minded and curious. You prefer novelty over stability and focus on hidden meanings and future possibilities. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        },{
            'target_labels':['I', 'S','F', 'J', 'ISFJ'],
            'opposite_labels' : ['E', 'N', 'T','P', 'ENTP'],
            'label_type':"['I', 'S','F', 'J']",
            'label_prompt':"You are an ISFJ person.",
            'prompt':"You are an introvert, observant, feeling, and judging person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels':['I', 'S', 'F', 'P','ISFP'],
            'opposite_labels' : ['E', 'N', 'T', 'J','ENTJ'],
            'label_type':"['I', 'S', 'F','P']",
            'label_prompt':"You are an ISFP person.",
            'prompt':"You are an introvert, observant, feeling, and prospecting person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You are sensitive and emotionally expressive. You are more empathic and less competitive, and focus on social harmony and cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        },{
            'target_labels':['I', 'S','T', 'J', 'ISTJ'],
            'opposite_labels' : ['E', 'N', 'F','P', 'ENFP'],
            'label_type':"['I', 'S','T', 'J']",
            'label_prompt':"You are an ISTJ person.",
            'prompt':"You are an introvert, observant, thinking, and judging person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are decisive, thorough and highly organized. You value clarity, predictability and closure, preferring structure and planning to spontaneity."
        },{
            'target_labels':['I', 'S', 'T', 'P','ISTP'],
            'opposite_labels' : ['E', 'N', 'F', 'J','ENFJ'],
            'label_type':"['I', 'S', 'T','P']",
            'label_prompt':"You are an ISTP person.",
            'prompt':"You are an introvert, observant, thinking, and prospecting person. You prefer solitary activities and get exhausted by social interaction. You tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general. You are highly practical, pragmatic and down-to-earth. You tend to have strong habits and focus on what is happening or has already happened. You focus on objectivity and rationality, prioritizing logic over emotions. You tend to hide your feelings and see efficiency as more important than cooperation. You are very good at improvising and spotting opportunities. You tend to be flexible, relaxed nonconformists and prefer keeping your options open."
        }
    ]
}

single_trait_prompts = {
    "mbti_prompt": [
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are extrovert. They prefer group activities and get energized by social interaction. They tend to be more enthusiastic and more easily excited.",
            "label": "E"
        },
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are introvert. They prefer solitary activities and get exhausted by social interaction. They tend to be quite sensitive to external stimulation (e.g. sound, sight or smell) in general.",
            "label": "I"
        },
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are observant. They are highly practical, pragmatic and down-to-earth. They tend to have strong habits and focus on what is happening or has already happened.",
            "label": "S"
        },
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are intuitive. They are very imaginative, open-minded and curious. They prefer novelty over stability and focus on hidden meanings and future possibilities.",
            "label": "N"
        },
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are thinking. They focus on objectivity and rationality, prioritizing logic over emotions. They tend to hide their feelings and see efficiency as more important than cooperation.",
            "label": "T"
        },
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are feeling. They are sensitive and emotionally expressive. They are more empathic and less competitive, and focus on social harmony and cooperation.",
            "label": "F"
        },
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are judging. They are decisive, thorough and highly organized. They value clarity, predictability and closure, preferring structure and planning to spontaneity.",
            "label": "J"
        },
        {
            "prompt": f"Please generate {words_num} descriptive adjectives of people who are prospecting. They are very good at improvising and spotting opportunities. They tend to be flexible, relaxed nonconformists and prefer keeping their options open.",
            "label": "P"
        }


    ],

}


def llm_generate_adjectives(iteration):
    if model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct" or model_id == "meta-llama/Llama-3.2-3B-Instruct":
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    if model_id == "Qwen/Qwen2.5-3B-Instruct":
        model_name = "Qwen/Qwen2.5-3B-Instruct"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_id == "qwen2.5-72b-instruct":
        client = getQwenClient()

    for prompt in single_trait_prompts["mbti_prompt"]:
        prompt_content = prompt["prompt"]
        label = prompt["label"]

        output_folder = f'16p_all_results/{model_id}/our_method_any/single_result_iteration_{iteration}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_file_name = os.path.join(output_folder, f'{label}-bfi44-output.txt')

        with open(output_file_name, 'w', encoding='utf-8') as f:
            messages = [{"role": "user", "content": prompt_content}]

            if model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct" or model_id == "meta-llama/Llama-3.2-3B-Instruct":
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

            if model_id == "Qwen/Qwen2.5-3B-Instruct":
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True)

                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if model_id == "qwen2.5-72b-instruct":
                chat_response = client.chat.completions.create(

                    model="Qwen2.5-72B-Instruct",

                    # max_tokens=800,

                    temperature=0.7,

                    stop="<|im_end|>",

                    stream=True,

                    messages=[{"role": "user", "content": prompt_content}]

                )

                # Stream the response to console
                generated_text = ""
                for chunk in chat_response:

                    if chunk.choices[0].delta.content:
                        generated_text += chunk.choices[0].delta.content
                        # print(chunk.choices[0].delta.content, end="", flush=True)

            f.write(f"Iteration {iteration} prompting: {prompt_content}\n")
            f.write(f"Iteration {iteration} generated_text: {generated_text}\n")
            answer = generated_text  # [-1]["content"]

            f.write(f"Iteration {iteration} answer: {answer}\n\n")


def txt_to_csv(directory, output_file):
    # 存储数据的列表
    data = []

    # 遍历目录下所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # 提取标签
            label = '-'.join(filename.split('-')[:1])  # 取文件名的前部分作为标签
            # 读取文件内容
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                # 使用正则表达式匹配形容词（既可以带 ** 也可以不带 **）
                matches = re.findall(r'^\d+\.\s+(\*{0,2}[\w-]+\*{0,2})', content, re.MULTILINE)

                # 为每个形容词添加标签和编号
                for i, adjective in enumerate(matches, start=1):
                    # 去掉星号并 strip 形容词
                    cleaned_adjective = adjective.replace('*', '').strip().lower()
                    data.append({'Label': label, 'Num': i, 'Adjectives': cleaned_adjective})

    # 创建DataFrame并保存为CSV文件
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        df = pd.DataFrame(data)
        # 合并现有数据和新数据
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = pd.DataFrame(data)

    # 保存合并后的DataFrame为CSV文件
    combined_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f'Data has been saved to {output_file}.')


def get_synonyms(word, top_n):
    """获取给定单词的前N个最接近的同义词列表（小写且无重复）。"""
    synonyms = set()  # 使用集合来存储唯一的同义词
    synsets_of_word = wn.synsets(word, pos=wn.ADJ)  # 获取指定词性（形容词）的同义词集合
    top_synonyms = []  # 临时存储同义词及其相似度分数

    if not synsets_of_word:
        return []  # 如果没有同义词集，返回空列表

    for syn in synsets_of_word:
        for lemma in syn.lemmas():
            synonym = lemma.name().lower()  # 转换为小写
            similarity = wn.path_similarity(syn, synsets_of_word[0])  # 计算路径相似度
            if similarity is not None:  # 避免无效的相似度
                top_synonyms.append((synonym, similarity))

    # 根据路径相似度对同义词进行排序，并选择前N个
    top_synonyms.sort(key=lambda x: x[1], reverse=True)
    top_synonyms = [syn[0] for syn in top_synonyms[:top_n]]  # 提取最接近的同义词

    # 使用集合去重并返回列表
    synonyms.update(top_synonyms)
    return list(synonyms)


def get_antonyms(word):
    """获取给定单词的反义词。"""

    antonyms = set()  # 创建一个集合来存储反义词

    synsets_of_word = wn.synsets(word, pos=wn.ADJ)
    for syn in synsets_of_word:  # 获取指定词性（形容词）的同义词集合
        for lemma in syn.lemmas():  # 遍历每个同义词的词条
            if lemma.antonyms():  # 检查是否有反义词
                for ant in lemma.antonyms():
                    antonyms.add(ant.name())  # 将第一个反义词添加到集合中

    return list(antonyms)  # 返回反义词列表


def word_net(llm_adjectives_path, aug_llm_adjectives_path):
    """主函数，处理输入文件并生成输出文件。"""
    df = pd.read_csv(llm_adjectives_path)  # 读取输入的CSV文件

    df_syn_list = []
    for i, r in df.iterrows():
        word = df['Adjectives'][i]
        synonyms = get_synonyms(word, 5)
        for w in synonyms:
            df_syn_list.append([df['Label'][i], -1, w])
    df_tmp = pd.DataFrame(df_syn_list, columns=['Label', 'Num', 'Adjectives'])
    df = pd.concat([df, df_tmp]).drop_duplicates().reset_index()
    df = df[['Label', 'Num', 'Adjectives']]

    df_ant_list = []
    for i, r in df.iterrows():
        word = df['Adjectives'][i]
        antonyms = get_antonyms(word)
        for w in antonyms:
            label = df['Label'][i]
            if label.split('-')[-1] == 'Low':
                label = label.split('-')[0] + '-High'
            else:
                label = label.split('-')[0] + '-Low'
            df_ant_list.append([label, -2, w])

    df_tmp = pd.DataFrame(df_ant_list, columns=['Label', 'Num', 'Adjectives'])
    df = pd.concat([df, df_tmp]).drop_duplicates().reset_index()
    df = df[['Label', 'Num', 'Adjectives']]

    df.to_csv(aug_llm_adjectives_path, index=False)


def get_llm_response(pip_line=None, label_prompt=None, prompt=None, model_name=None, client=None,
                     tokenizer=None, model=None):
    if pip_line:
        pipeline = pip_line
    if client:
        client = client
    if model_name:
        model_name = model_name
        tokenizer = tokenizer
        model = model

    messages = [
        {"role": "system",
         "content": "Imagine you are a human, you are specified with the following personality: " + label_prompt + ' ' + prompt},
        {"role": "user",
         "content": "Please share a personal personal story in 800 words. Do not explicitly mention your personality traits in the story."}
    ]

    if model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct" or model_id == "meta-llama/Llama-3.2-3B-Instruct":
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

    if model_id == "Qwen/Qwen2.5-3B-Instruct":
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if model_id == "qwen2.5-72b-instruct":

        chat_response = client.chat.completions.create(

            model="Qwen2.5-72B-Instruct",

            # max_tokens=800,

            temperature=0.7,

            stop="<|im_end|>",

            stream=True,

            messages=[
                {"role": "system",
                 "content": "Imagine you are a human, you are specified with the following personality: " + label_prompt + ' ' + prompt},
                {"role": "user",
                 "content": "Please share a personal personal story in 800 words. Do not explicitly mention your personality traits in the story."}
            ])

        # Stream the response to console
        generated_text = ""
        for chunk in chat_response:

            if chunk.choices[0].delta.content:
                generated_text += chunk.choices[0].delta.content

    #     print('generated_text', generated_text)
    return generated_text


def get_entailment_words(question, words):
    # Preprocess input
    sentences = nltk.sent_tokenize(question)
    sentence_pairs = [(sent, f"You are {word}.") for sent in sentences for word in words]

    #sentence = question
    #sentences_with_words = [f"You are {word}." for word in words]
    # Generate sentence pairs
    #sentence_pairs = [(sentence, sent) for sent in sentences_with_words]

    # Load NLI model
    model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    scores = model.predict(sentence_pairs)

    # Convert scores to probabilities using softmax
    label_mapping = ["contradiction", "entailment", "neutral"]
    scores = torch.tensor(scores, dtype=torch.float32)
    scores = torch.nn.functional.softmax(scores, dim=1)

    # Determine labels
    labels = [label_mapping[score.argmax()] for score in scores]
    results = [{"word": word, "label": label_mapping[score.argmax()], "scores": score} for word, score in
               zip(words, scores)]

    # Filter words with label "entailment"
    entailment_words = [result['word'] for result in results if result['label'] == "entailment"]

    # Return the entailment words as a comma-separated string
    return entailment_words


def get_response(pip_line=None, label_prompt=None, prompt=None, prompt_by_words=None, model_name=None,
                 client=None, tokenizer=None, model=None):
    if pip_line:
        pipeline = pip_line
    if client:
        client = client
    if model_name:
        model_name = model_name
        tokenizer = tokenizer
        model = model

    if prompt_by_words == "":  # matched no words
        system_prompt = "Imagine you are a human, you are specified with the following personality: " + label_prompt + ' ' + prompt
    else:
        system_prompt = "Imagine you are a human, you are specified with the following personality: " + label_prompt + ' ' + prompt + " You are " + prompt_by_words

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": "Please share a personal personal story in 800 words. Do not explicitly mention your personality traits in the story."}
    ]

    if model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct" or model_id == "meta-llama/Llama-3.2-3B-Instruct":
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

    if model_id == "Qwen/Qwen2.5-3B-Instruct":
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if model_id == "qwen2.5-72b-instruct":
        chat_response = client.chat.completions.create(

            model="Qwen2.5-72B-Instruct",

            # max_tokens=800,

            temperature=0.7,

            stop="<|im_end|>",

            stream=True,

            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",
                 "content": "Please share a personal personal story in 800 words. Do not explicitly mention your personality traits in the story."}])

        # Stream the response to console
        generated_text = ""
        for chunk in chat_response:

            if chunk.choices[0].delta.content:
                generated_text += chunk.choices[0].delta.content

    return generated_text



def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result

    return func_wrapper


def get_response_vanilla(pip_line=None, label_prompt=None, model_name=None, client=None, tokenizer=None,
                         model=None):
    if pip_line:
        pipeline = pip_line
    if client:
        client = client
    if model_name:
        model_name = model_name
        tokenizer = tokenizer
        model = model

    system_prompt = "Imagine you are a human, you are specified with the following personality: " + label_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": "Please share a personal personal story in 800 words. Do not explicitly mention your personality traits in the story."}]

    if model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct" or model_id == "meta-llama/Llama-3.2-3B-Instruct":
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

    if model_id == "Qwen/Qwen2.5-3B-Instruct":
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if model_id == "qwen2.5-72b-instruct":
        chat_response = client.chat.completions.create(

            model="Qwen2.5-72B-Instruct",

            # max_tokens=800,

            temperature=0.7,

            stop="<|im_end|>",

            stream=True,

            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",
                 "content": "Please share a personal personal story in 800 words. Do not explicitly mention your personality traits in the story."}])

        # Stream the response to console
        generated_text = ""
        for chunk in chat_response:

            if chunk.choices[0].delta.content:
                generated_text += chunk.choices[0].delta.content

    return generated_text


@timer
def vanilla():
    target_dir = 'vanilla'
    # if not os.path.exists(target_dir):
    #     os.makedirs(target_dir)

    for itr in range(iteration):
        for item in test_template["combinations"]:

            label_content = ast.literal_eval(item["label_type"])
            label_content_str = '-'.join(label_content)

            output_file_name = f'16p_all_results/{model_id}/{target_dir}/result_iteration_{itr + 1}/result-generate-{label_content_str}-bfi44-output.txt'

            if not os.path.exists(os.path.dirname(output_file_name)):
                os.makedirs(os.path.dirname(output_file_name))

            with open(output_file_name, 'a', encoding='utf-8') as f:

                if model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct" or model_id == "meta-llama/Llama-3.2-3B-Instruct":
                    pipeline = transformers.pipeline(
                        "text-generation",
                        model=model_id,
                        model_kwargs={"torch_dtype": torch.bfloat16},
                        device_map="auto",
                    )
                    answer = get_response_vanilla(pip_line=pipeline,
                                                  label_prompt=item["label_prompt"])

                if model_id == "Qwen/Qwen2.5-3B-Instruct":
                    model_name = "Qwen/Qwen2.5-3B-Instruct"

                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype="auto",
                        device_map="auto"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    answer = get_response_vanilla(label_prompt=item["label_prompt"],
                                                  model_name=model_name, tokenizer=tokenizer, model=model)

                if model_id == "qwen2.5-72b-instruct":
                    client = getQwenClient()
                    answer = get_response_vanilla(label_prompt=item["label_prompt"],
                                                  client=client)

                f.write(f"Iteration {itr + 1} answer: {answer}\n")



def get_response_combine(pip_line=None, prompt=None, model_name=None, client=None, tokenizer=None,
                         model=None):
    if pip_line:
        pipeline = pip_line
    if client:
        client = client
    if model_name:
        model_name = model_name
        tokenizer = tokenizer
        model = model

    system_prompt = "Imagine you are a human, you are specified with the following personality: " + prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": "Please share a personal personal story in 800 words. Do not explicitly mention your personality traits in the story."}]

    if model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct" or model_id == "meta-llama/Llama-3.2-3B-Instruct":
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

    if model_id == "Qwen/Qwen2.5-3B-Instruct":
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if model_id == "qwen2.5-72b-instruct":
        chat_response = client.chat.completions.create(

            model="Qwen2.5-72B-Instruct",

            # max_tokens=800,

            temperature=0.7,

            stop="<|im_end|>",

            stream=True,

            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",
                 "content": "Please share a personal personal story in 800 words. Do not explicitly mention your personality traits in the story."}])

        # Stream the response to console
        generated_text = ""
        for chunk in chat_response:

            if chunk.choices[0].delta.content:
                generated_text += chunk.choices[0].delta.content

    return generated_text


@timer
def combine():
    target_dir = 'combine'
    # if not os.path.exists(target_dir):
    #     os.makedirs(target_dir)

    for itr in range(iteration):
        for item in test_template["combinations"]:

            label_content = ast.literal_eval(item["label_type"])
            label_content_str = '-'.join(label_content)

            output_file_name = f'16p_all_results/{model_id}/{target_dir}/result_iteration_{itr + 1}/result-generate-{label_content_str}-bfi44-output.txt'

            if not os.path.exists(os.path.dirname(output_file_name)):
                os.makedirs(os.path.dirname(output_file_name))

            with open(output_file_name, 'a', encoding='utf-8') as f:

                if model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct" or model_id == "meta-llama/Llama-3.2-3B-Instruct":
                    # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

                    pipeline = transformers.pipeline(
                        "text-generation",
                        model=model_id,
                        model_kwargs={"torch_dtype": torch.bfloat16},
                        device_map="auto",
                    )
                    answer = get_response_combine(pip_line=pipeline, prompt=item["prompt"])
                    print(answer)

                if model_id == "Qwen/Qwen2.5-3B-Instruct":
                    model_name = "Qwen/Qwen2.5-3B-Instruct"

                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype="auto",
                        device_map="auto"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    answer = get_response_combine(prompt=item["prompt"], model_name=model_name,
                                                  tokenizer=tokenizer, model=model)
                    print(answer)

                if model_id == "qwen2.5-72b-instruct":
                    client = getQwenClient()
                    answer = get_response_combine(prompt=item["prompt"], client=client)
                    print(answer)

                f.write(f"Iteration {itr + 1} answer: {answer}\n")


@timer
def main_run():
    for itr in range(iteration):

        directory = f'16p_all_results/{model_id}/our_method_any/single_result_iteration_{itr + 1}'

        # llm产生单个维度的词语
        llm_adjectives_path = f'16p_all_results/{model_id}/our_method_any/result_iteration_{itr + 1}/gen_words.csv'
        llm_generate_adjectives(itr + 1)
        if not os.path.exists(os.path.dirname(llm_adjectives_path)):
            os.makedirs(os.path.dirname(llm_adjectives_path))
        txt_to_csv(directory, llm_adjectives_path)

        # 在提取的csv文件基础上生成同义词反义词
        aug_llm_adjectives_path = f'16p_all_results/{model_id}/our_method_any/result_iteration_{itr + 1}/aug_gen_words.csv'
        word_net(llm_adjectives_path, aug_llm_adjectives_path)

        for item in test_template["combinations"]:
            target_labels = item["target_labels"]
            opposite_labels = item["opposite_labels"]

            print("当前的标签是：")

            print(target_labels, ' ...')

            df_full_adj = pd.read_csv(aug_llm_adjectives_path)
            pos_set = set()
            neg_set = set()
            for i, r in df_full_adj.iterrows():
                if r['Label'] in target_labels:
                    pos_set.add(r['Adjectives'])
                if r['Label'] in opposite_labels:
                    neg_set.add(r['Adjectives'])

            words_modified = list(pos_set - neg_set)  # 差集
            print(words_modified)

            # 把差集存起来
            label_content = ast.literal_eval(item["label_type"])
            label_content_str = '-'.join(label_content)
            output_file_name = f'16p_all_results/{model_id}/our_method_any/result_iteration_{itr + 1}/{label_content_str}-words.txt'
            with open(output_file_name, 'w', encoding='utf-8') as file:
                for word in words_modified:
                    file.write(word + '\n')

            # 开始prompting
            output_file_name = f'16p_all_results/{model_id}/our_method_any/result_iteration_{itr + 1}/result-generate-{label_content_str}-bfi44-output.txt'

            with open(output_file_name, 'a', encoding='utf-8') as f:
                print("加载模型")


                if model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct" or model_id == "meta-llama/Llama-3.2-3B-Instruct":
                    pipeline = transformers.pipeline(
                        "text-generation",
                        model=model_id,
                        model_kwargs={"torch_dtype": torch.bfloat16},
                        device_map="auto",
                    )
                    llm_init_response = get_llm_response(pip_line=pipeline,
                                                         label_prompt=item["label_prompt"], prompt=item["prompt"])
                    f.write(f"Iteration {itr + 1} llm_init_response: {llm_init_response}\n")


                if model_id == "Qwen/Qwen2.5-3B-Instruct":
                    model_name = "Qwen/Qwen2.5-3B-Instruct"

                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype="auto",
                        device_map="auto"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    llm_init_response = get_llm_response(label_prompt=item["label_prompt"],
                                                         prompt=item["prompt"], model_name=model_name,
                                                         tokenizer=tokenizer, model=model)
                    f.write(f"Iteration {itr + 1} llm_init_response: {llm_init_response}\n")

                if model_id == "qwen2.5-72b-instruct":
                    client = getQwenClient()
                    llm_init_response = get_llm_response(label_prompt=item["label_prompt"],
                                                         prompt=item["prompt"], client=client)
                    f.write(f"Iteration {itr + 1} llm_init_response: {llm_init_response}\n")


                print("生成初始描述")

                print(llm_init_response)

                # 初始化模型
                entailment_words = get_entailment_words(llm_init_response, words_modified)
                # print(f"Entailment words: {entailment_words}")

                # for word in entailment_words: # print label distribution
                #     i = df_full_adj.loc[df_full_adj['Adjectives'] == word].index
                #     print(word, df_full_adj.iloc[i]['Label'])

                if len(entailment_words) == 0:
                    prompt_by_words = ""
                else:
                    words = ""
                    length = len(entailment_words)
                    for i in range(length):
                        if i != length - 1:
                            words += entailment_words[i] + ', '
                        else:
                            words += 'and ' + entailment_words[i] + '.'

                prompt_by_words = words
                print('---')
                print(prompt_by_words)
                print('---')

                if model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct" or model_id == "meta-llama/Llama-3.2-3B-Instruct":
                    answer = get_response(pip_line=pipeline, label_prompt=item["label_prompt"],
                                          prompt=item["prompt"], prompt_by_words=prompt_by_words)

                if model_id == "Qwen/Qwen2.5-3B-Instruct":
                    answer = get_response(label_prompt=item["label_prompt"],
                                          prompt=item["prompt"],
                                          prompt_by_words=prompt_by_words, model_name=model_name,
                                          tokenizer=tokenizer, model=model)

                if model_id == "qwen2.5-72b-instruct":
                    answer = get_response(label_prompt=item["label_prompt"],
                                          prompt=item["prompt"],
                                          prompt_by_words=prompt_by_words, client=client)

                f.write(f"Iteration{itr + 1} {item['prompt']} + prompt_by_words + {prompt_by_words}\n ")
                f.write(f"Iteration {itr + 1} answer: {answer}\n")





if __name__ == '__main__':
    vanilla()
    combine()
    main_run()
