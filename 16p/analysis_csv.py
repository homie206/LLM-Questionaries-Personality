import os
import pandas as pd

# 指定文件夹路径
folder_path = '/Users/haoming/Desktop/gpt_16p/LLM-Personality-Questionnaires/16p/qwen2.5-72b-instruct/combined-result'

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        csv_path = os.path.join(folder_path, filename)

        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 统计第二列的种类计数
        if df.shape[1] > 1:  # 确保有至少两列
            counts = df.iloc[:, 1].value_counts()

            # 生成同名的txt文件
            txt_filename = filename.replace('.csv', '.txt')
            txt_path = os.path.join(folder_path, txt_filename)

            # 将计数结果写入TXT文件
            with open(txt_path, 'w') as f:
                for value, count in counts.items():
                    f.write(f"{value}: {count}\n")

print("统计完成！")