import json
import pandas as pd
from pprint import pprint

system_message = "You are Melina, a professional text paraphraser. You are to read carefully from text provided, then re-write text with same meaning and length."

def create_user_message(row):
    # return f"""Fading {type}:\n {row['generated_intro']}"""
    return row['paraphrase']

def create_assistant_message(row):
    return row['original']


def prepare_conversation(row):
    messages = []
    messages.append({"role": "system", "content": system_message})

    messages.append({"role": "user", "content": create_user_message(row)})

    # assistant_message = create_assistant_message(row)
    messages.append({"role": "assistant", "content": create_assistant_message(row)})

    return {"messages": messages}


def write_jsonl(data_list: list, filename: str) -> None:
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)


training_df = pd.read_json("datasets/train.jsonl", lines=True).loc[1200:1300]
validation_df = pd.read_json("datasets/train.jsonl", lines=True).loc[300:320]

training_data = training_df.apply(prepare_conversation, axis=1).tolist()
validation_data = validation_df.apply(prepare_conversation, axis=1).tolist()

training_file_name = "finetune_training_t9.jsonl"
write_jsonl(training_data, training_file_name)

validation_file_name = "finetune_validation_t9.jsonl"
write_jsonl(validation_data, validation_file_name)
# for example in training_data[:5]:
#     print(example)

# print(len(json.dumps(prepare_conversation(training_df.iloc[0]))))
# pprint(training_df)
# pprint(validation_df)



