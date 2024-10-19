
import re
import pandas as pd
import ast
from openai import OpenAI
client = OpenAI()
import time
import os




def get_final_scores(columns, dim):
    score = 0
    if dim == 'EXT':
        score += columns[0]
        score += (6 - columns[1])
        score += columns[2]
        score += columns[3]
        score += (6 - columns[4])
        score += columns[5]
        score += (6 - columns[6])
        score += columns[7]
        score = score/8
    if dim == 'EST':
        score += columns[0]
        score += (6 - columns[1])
        score += columns[2]
        score += columns[3]
        score += (6 - columns[4])
        score += columns[5]
        score += (6 - columns[6])
        score += columns[7]
        score = score / 8
    if dim == 'AGR':
        score += (6 - columns[0])
        score += columns[1]
        score += (6 - columns[2])
        score += columns[3]
        score += columns[4]
        score += (6 - columns[5])
        score += columns[6]
        score += (6 - columns[7])
        score += columns[8]
        score = score / 9
    if dim == 'CSN':
        score += columns[0]
        score += (6 - columns[1])
        score += columns[2]
        score += (6 - columns[3])
        score += (6 - columns[4])
        score += columns[5]
        score += columns[6]
        score += columns[7]
        score += (6 - columns[8])
        score = score / 9
    if dim == 'OPN':
        score += columns[0]
        score += columns[1]
        score += columns[2]
        score += columns[3]
        score += columns[4]
        score += columns[5]
        score += (6 - columns[6])
        score += columns[7]
        score += (6 - columns[8])
        score += columns[9]
        score = score / 10
    return score

prompt_template = {
    "ipip50_prompt": [
        {'prompt': 'You are low on extraversion, low on emotional stability, low on agreeableness, low on conscientiousness, and low on openness to experience. ',
        'label': "['E-Low', 'N-Low', 'A-Low', 'C-Low', 'O-Low']"},
        {'prompt': 'You are low on extraversion, low on emotional stability, low on agreeableness, low on conscientiousness, and high on openness to experience. ',
        'label': "['E-Low', 'N-Low', 'A-Low', 'C-Low', 'O-High']"},
        {'prompt': 'You are low on extraversion, low on emotional stability, low on agreeableness, high on conscientiousness, and low on openness to experience. ',
        'label': "['E-Low', 'N-Low', 'A-Low', 'C-High', 'O-Low']"},
        {'prompt': 'You are low on extraversion, low on emotional stability, low on agreeableness, high on conscientiousness, and high on openness to experience. ',
        'label': "['E-Low', 'N-Low', 'A-Low', 'C-High', 'O-High']"},
        {'prompt': 'You are low on extraversion, low on emotional stability, high on agreeableness, low on conscientiousness, and low on openness to experience. ',
        'label': "['E-Low', 'N-Low', 'A-High', 'C-Low', 'O-Low']"},
        {'prompt': 'You are low on extraversion, low on emotional stability, high on agreeableness, low on conscientiousness, and high on openness to experience. ',
        'label': "['E-Low', 'N-Low', 'A-High', 'C-Low', 'O-High']"},
        {'prompt': 'You are low on extraversion, low on emotional stability, high on agreeableness, high on conscientiousness, and low on openness to experience. ',
        'label': "['E-Low', 'N-Low', 'A-High', 'C-High', 'O-Low']"},
        {'prompt': 'You are low on extraversion, low on emotional stability, high on agreeableness, high on conscientiousness, and high on openness to experience. ',
        'label': "['E-Low', 'N-Low', 'A-High', 'C-High', 'O-High']"},
        {'prompt': 'You are low on extraversion, high on emotional stability, low on agreeableness, low on conscientiousness, and low on openness to experience. ',
        'label': "['E-Low', 'N-High', 'A-Low', 'C-Low', 'O-Low']"},
        {'prompt': 'You are low on extraversion, high on emotional stability, low on agreeableness, low on conscientiousness, and high on openness to experience. ',
        'label': "['E-Low', 'N-High', 'A-Low', 'C-Low', 'O-High']"},
        {'prompt': 'You are low on extraversion, high on emotional stability, low on agreeableness, high on conscientiousness, and low on openness to experience. ',
        'label': "['E-Low', 'N-High', 'A-Low', 'C-High', 'O-Low']"},
        {'prompt': 'You are low on extraversion, high on emotional stability, low on agreeableness, high on conscientiousness, and high on openness to experience. ',
        'label': "['E-Low', 'N-High', 'A-Low', 'C-High', 'O-High']"},
        {'prompt': 'You are low on extraversion, high on emotional stability, high on agreeableness, low on conscientiousness, and low on openness to experience. ',
        'label': "['E-Low', 'N-High', 'A-High', 'C-Low', 'O-Low']"},
        {'prompt': 'You are low on extraversion, high on emotional stability, high on agreeableness, low on conscientiousness, and high on openness to experience. ',
        'label': "['E-Low', 'N-High', 'A-High', 'C-Low', 'O-High']"},
        {'prompt': 'You are low on extraversion, high on emotional stability, high on agreeableness, high on conscientiousness, and low on openness to experience. ',
        'label': "['E-Low', 'N-High', 'A-High', 'C-High', 'O-Low']"},
        {'prompt': 'You are low on extraversion, high on emotional stability, high on agreeableness, high on conscientiousness, and high on openness to experience. ',
        'label': "['E-Low', 'N-High', 'A-High', 'C-High', 'O-High']"},
        {'prompt': 'You are high on extraversion, low on emotional stability, low on agreeableness, low on conscientiousness, and low on openness to experience. ',
        'label': "['E-High', 'N-Low', 'A-Low', 'C-Low', 'O-Low']"},
        {'prompt': 'You are high on extraversion, low on emotional stability, low on agreeableness, low on conscientiousness, and high on openness to experience. ',
        'label': "['E-High', 'N-Low', 'A-Low', 'C-Low', 'O-High']"},
        {'prompt': 'You are high on extraversion, low on emotional stability, low on agreeableness, high on conscientiousness, and low on openness to experience. ',
        'label': "['E-High', 'N-Low', 'A-Low', 'C-High', 'O-Low']"},
        {'prompt': 'You are high on extraversion, low on emotional stability, low on agreeableness, high on conscientiousness, and high on openness to experience. ',
        'label': "['E-High', 'N-Low', 'A-Low', 'C-High', 'O-High']"},
        {'prompt': 'You are high on extraversion, low on emotional stability, high on agreeableness, low on conscientiousness, and low on openness to experience. ',
        'label': "['E-High', 'N-Low', 'A-High', 'C-Low', 'O-Low']"},
        {'prompt': 'You are high on extraversion, low on emotional stability, high on agreeableness, low on conscientiousness, and high on openness to experience. ',
        'label': "['E-High', 'N-Low', 'A-High', 'C-Low', 'O-High']"},
        {'prompt': 'You are high on extraversion, low on emotional stability, high on agreeableness, high on conscientiousness, and low on openness to experience. ',
        'label': "['E-High', 'N-Low', 'A-High', 'C-High', 'O-Low']"},
        {'prompt': 'You are high on extraversion, low on emotional stability, high on agreeableness, high on conscientiousness, and high on openness to experience. ',
        'label': "['E-High', 'N-Low', 'A-High', 'C-High', 'O-High']"},
        {'prompt': 'You are high on extraversion, high on emotional stability, low on agreeableness, low on conscientiousness, and low on openness to experience. ',
        'label': "['E-High', 'N-High', 'A-Low', 'C-Low', 'O-Low']"},
        {'prompt': 'You are high on extraversion, high on emotional stability, low on agreeableness, low on conscientiousness, and high on openness to experience. ',
        'label': "['E-High', 'N-High', 'A-Low', 'C-Low', 'O-High']"},
        {'prompt': 'You are high on extraversion, high on emotional stability, low on agreeableness, high on conscientiousness, and low on openness to experience. ',
        'label': "['E-High', 'N-High', 'A-Low', 'C-High', 'O-Low']"},
        {'prompt': 'You are high on extraversion, high on emotional stability, low on agreeableness, high on conscientiousness, and high on openness to experience. ',
        'label': "['E-High', 'N-High', 'A-Low', 'C-High', 'O-High']"},
        {'prompt': 'You are high on extraversion, high on emotional stability, high on agreeableness, low on conscientiousness, and low on openness to experience. ',
        'label': "['E-High', 'N-High', 'A-High', 'C-Low', 'O-Low']"},
        {'prompt': 'You are high on extraversion, high on emotional stability, high on agreeableness, low on conscientiousness, and high on openness to experience. ',
        'label': "['E-High', 'N-High', 'A-High', 'C-Low', 'O-High']"},
        {'prompt': 'You are high on extraversion, high on emotional stability, high on agreeableness, high on conscientiousness, and low on openness to experience. ',
        'label': "['E-High', 'N-High', 'A-High', 'C-High', 'O-Low']"},
        {'prompt': 'You are high on extraversion, high on emotional stability, high on agreeableness, high on conscientiousness, and high on openness to experience. ',
        'label': "['E-High', 'N-High', 'A-High', 'C-High', 'O-High']"}
    ]
}

# 创建列名列表
column_names = ['EXT1', 'AGR1', 'CSN1', 'EST1', 'OPN1',
                'EXT2', 'AGR2', 'CSN2', 'EST2', 'OPN2',
                'EXT3', 'AGR3', 'CSN3', 'EST3', 'OPN3',
                'EXT4', 'AGR4', 'CSN4', 'EST4', 'OPN4',
                'EXT5', 'AGR5', 'CSN5', 'EST5', 'OPN5',
                'EXT6', 'AGR6', 'CSN6', 'EST6', 'OPN6',
                'EXT7', 'AGR7', 'CSN7', 'EST7', 'OPN7',
                'EXT8', 'AGR8', 'CSN8', 'EST8', 'OPN8',
                'OPN9', 'AGR9', 'CSN9', 'OPN10']


# 创建 DataFrame
df = pd.DataFrame(columns=column_names)

def extract_first_number(answer):
    match = re.search(r'^\d+', answer)
    if match:
        return int(match.group())
    else:
        return None


def generateResponse(prompt, question):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Imagine you are a human. "+prompt
            },
            {
                "role": "user",
                "content": '''Given a statement of you. Please choose from the following options to identify how accurately this statement describes you. 
                                1. Very Inaccurate
                                2. Moderately Inaccurate 
                                3. Neither Accurate Nor Inaccurate
                                4. Moderately Accurate
                                5. Very Accurate
                                Please only answer with the option number. \nHere is the statement: ''' + question}
        ],
        # temperature=0.6,
        # max_tokens=256,
        # top_p=0.9
    )
    print(response.choices[0].message.content)

    generated_text = response.choices[0].message.content
    time.sleep(5)
    return generated_text


if __name__ == '__main__':
    for ipip_item in prompt_template["ipip50_prompt"]:
        ipip_prompt = ipip_item["prompt"]
        #ipip_label_content = ipip_item["label"]
        ipip_label_content = ast.literal_eval(ipip_item["label"])  # 转换为列表
        ipip_label_content_str = '-'.join(ipip_label_content)

        output_file_name = f'vanilla-result/{ipip_label_content_str}-vanilla-bfi44-gpt3.5-turbo-output.txt'
        result_file_name = f'vanilla-result/{ipip_label_content_str}-vanilla-bfi44-gpt3.5-turbo-result.csv'

        if not os.path.isfile(result_file_name):
            df = pd.DataFrame(columns=
               ['EXT1', 'AGR1', 'CSN1', 'EST1', 'OPN1',
                'EXT2', 'AGR2', 'CSN2', 'EST2', 'OPN2',
                'EXT3', 'AGR3', 'CSN3', 'EST3', 'OPN3',
                'EXT4', 'AGR4', 'CSN4', 'EST4', 'OPN4',
                'EXT5', 'AGR5', 'CSN5', 'EST5', 'OPN5',
                'EXT6', 'AGR6', 'CSN6', 'EST6', 'OPN6',
                'EXT7', 'AGR7', 'CSN7', 'EST7', 'OPN7',
                'EXT8', 'AGR8', 'CSN8', 'EST8', 'OPN8',
                'OPN9', 'AGR9', 'CSN9', 'OPN10'])
            df.to_csv(result_file_name, index=False)

        with open(output_file_name, 'a', encoding='utf-8') as f, open(result_file_name, 'a', encoding='utf-8') as r:
            with open('../bfi44.txt', 'r') as f2:
                question_list = f2.readlines()
                answer_list = []
                extracted_numbers = []
                all_results = []

                for run in range(20):  # 运行100次
                    extracted_numbers = []

                    for q in question_list:
                        generated_text = generateResponse(ipip_prompt, q)

                        extracted_number = extract_first_number(generated_text)
                        extracted_numbers.append(extracted_number)
                        print(f"Cycle {run+1} extracted numbers:")
                        f.write(f"Cycle {run+1} extracted numbers:")
                        print(extracted_numbers)
                        f.write(', '.join(map(str, extracted_numbers)) + '\n')
                        #all_results.append(extracted_numbers)

                        f.write(f"cycle: {run+1}\n")
                        print(f"cycle: {run+1}\n")
                        f.write(f"prompting: Imagine you are a human. {ipip_prompt}\n")
                        print(f"prompting: Imagine you are a human. {ipip_prompt}\n")
                        f.write(
                            '''Given a statement of you. Please choose from the following options to identify how accurately this statement describes you. 
                                1. Very Inaccurate
                                2. Moderately Inaccurate 
                                3. Neither Accurate Nor Inaccurate
                                4. Moderately Accurate
                                5. Very Accurate
                                Please only answer with the option number. \nHere is the statement: ''' + q)
                        print(
                            '''Given a statement of you. Please choose from the following options to identify how accurately this statement describes you. 
                                1. Very Inaccurate
                                2. Moderately Inaccurate 
                                3. Neither Accurate Nor Inaccurate
                                4. Moderately Accurate
                                5. Very Accurate
                                Please only answer with the option number. \nHere is the statement: ''' + q)
                        f.write(generated_text + '\n')
                        print(generated_text + '\n')

                    print(f"Run {run + 1} extracted numbers:")
                    print(extracted_numbers)

                    all_results.append(extracted_numbers)

                    # 将结果转换为 DataFrame
                result_df = pd.DataFrame(all_results, columns=column_names)

                # 保存结果到 CSV 文件
                result_df.to_csv(result_file_name, index=False)

            df = pd.read_csv(result_file_name, sep=',')

            dims = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']
            # 生成列名
            columns = [i + str(j) for j in range(1, 11) for i in dims]
            # 只保留存在的列
            existing_columns = [col for col in columns if col in df.columns]
            df = df[existing_columns]

            # 计算每个维度的最终得分
            for i in dims:
                relevant_columns = [col for col in existing_columns if col.startswith(i)]
                df[i + '_all'] = df.apply(
                    lambda r: get_final_scores(columns=[r[col] for col in relevant_columns], dim=i),
                    axis=1
                )

            # 打印每个维度的得分
            for i in dims:
                print(f"{i}_all:")
                print(df[i + '_all'])
                print()

            # 获取最终得分
            final_scores = [df[i + '_all'][0] for i in dims]
            print(final_scores)

            # 计算每个维度的得分
            for i in dims:
                relevant_columns = [col for col in existing_columns if col.startswith(i)]
                df[i + '_Score'] = df.apply(
                    lambda r: get_final_scores(columns=[r[col] for col in relevant_columns], dim=i),
                    axis=1
                )

            # 读取原始数据
            original_df = pd.read_csv(result_file_name, sep=',')

            # 合并新旧数据
            result_df = pd.concat([original_df, df[[f"{i}_Score" for i in dims]]], axis=1)

            # 保存结果到 CSV 文件
            result_df.to_csv(result_file_name, index=False)









