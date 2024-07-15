import json
import pandas as pd
from pprint import pprint
import hashlib
# from collections import defaultdict

system_message = "You are Melina, a professional text optimizer. You are to read carefully from each of the sentences provided, then re-write those sentences without changing their meaning."


def create_user_message(type, row):
    return f"""Fading {type}:\n {row}"""


def create_assistant_message(row):
    return row


def prepare_conversation(data):
    messages = []
    messages.append({"role": "system", "content": system_message})

    messages.append(
        {"role": "user", "content": create_user_message("essay", data[1])})

    assistant_message = create_assistant_message(data[0])
    messages.append({"role": "assistant", "content": assistant_message})

    return {"messages": messages}

# def merge_dicts(row, *dict):
#     # merged_dict = defaultdict(list)
#     hash_object = hashlib.sha256(row['title'].encode('utf-8'))
#     hash_digest = hash_object.hexdigest()

#     for key, value in dict.items():
#         dict[hash_digest].append(row['abstract'])

#     # # 将defaultdict转换为普通字典
#     # merged_dict = dict(merged_dict)

#     return dict


def prepare_dict(row, data_dict):
    hash_object = hashlib.sha256(row['title'].encode('utf-8'))
    hash_digest = hash_object.hexdigest()

    # target = data_dict[hash_digest].values()
    target = data_dict.get(hash_digest)
    pair = {}
    if target:
        pair = target
    else:
        data_dict[hash_digest] = pair

    pair[row['label']] = row['abstract']
    data_dict[hash_digest].update(pair)

    return data_dict


def write_jsonl(data_list: list, filename: str) -> None:
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)


training_df = pd.read_csv("datasets/training.csv")
validation_df = pd.read_csv("datasets/test.csv")

training_data_dict = {}
validation_data_dict = {}

# training_data = training_df.apply(prepare_conversation, axis=1, args=(training_data_dict,)).tolist()
training_df.apply(prepare_dict, axis=1, args=(training_data_dict,))
validation_df.apply(prepare_dict, axis=1, args=(validation_data_dict,))

training_data = [prepare_conversation(v) for v in training_data_dict.values()]
validation_data = [prepare_conversation(v) for v in validation_data_dict.values()]
training_file_name = "finetune_training_t8.jsonl"
write_jsonl(training_data, training_file_name)

validation_file_name = "finetune_validation_t8.jsonl"
write_jsonl(validation_data, validation_file_name)
# for example in training_data[:5]:
#     print(example)

# print(len(json.dumps(prepare_conversation(training_df.iloc[0]))))
# pprint(prepare_conversation(recipe_df.iloc[0]))
