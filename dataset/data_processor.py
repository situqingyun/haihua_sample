import json
import math
import random


def preprocess_data(input_file, split_eval=0.05):
    """
        切分训练集和验证集
    """
    with open(input_file, "r", encoding="utf-8-sig") as f:
        data_list = json.load(f)
    split_index = math.floor(len(data_list) * (1 - split_eval))

    train_list = data_list[:split_index]
    eval_list = data_list[split_index:]

    with open('dataset/haihua/train.json', "w", encoding="utf-8-sig") as f:
        random.shuffle(train_list)
        json.dump(train_list, f)
    with open('dataset/haihua/dev.json', "w", encoding="utf-8-sig") as f:
        json.dump(eval_list, f)

def debug_data(input_file):
    with open(input_file, "r", encoding="utf-8-sig") as f:
        data_list = json.load(f)
    train_list = data_list[:10]
    with open('haihua/train.json', "w", encoding="utf-8-sig") as f:
        json.dump(train_list, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # preprocess_data('dataset/haihua/raw/train.json')
    debug_data('haihua/raw/train.json')