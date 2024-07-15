import json
import pandas as pd
from pprint import pprint

system_message = "You are Melina, a professional text optimizer. You are to read carefully from each of the sentences provided, then re-write those sentences without changing their meaning."

def create_user_message(type, row):
    return f"""Fading {type}:\n {row['generated_intro']}"""

def create_assistant_message(row):
    return row['wiki_intro']


def prepare_conversation(row):
    messages = []
    messages.append({"role": "system", "content": system_message})

    messages.append({"role": "user", "content": create_user_message("essay", row)})

    assistant_message = create_assistant_message(row)
    messages.append({"role": "assistant", "content": assistant_message})

    return {"messages": messages}


def write_jsonl(data_list: list, filename: str) -> None:
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)


training_df = pd.read_csv("datasets/training.csv")
validation_df = pd.read_csv("datasets/test.csv")

training_data = training_df.apply(prepare_conversation, axis=1).tolist()
validation_data = validation_df.apply(prepare_conversation, axis=1).tolist()

training_file_name = "finetune_training_t7.jsonl"
write_jsonl(training_data, training_file_name)

validation_file_name = "finetune_validation_t7.jsonl"
write_jsonl(validation_data, validation_file_name)
# for example in training_data[:5]:
#     print(example)

# print(len(json.dumps(prepare_conversation(training_df.iloc[0]))))
# pprint(prepare_conversation(recipe_df.iloc[0]))
