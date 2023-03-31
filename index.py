import json
import re
import glob
import os

import pandas as pd
import numpy as np

from huggingface_hub import HfApi, HfFolder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

# 1. Load dataset

input_tasks_path = "alpaca_data_cleaned.json"

with open(input_tasks_path, "rb") as f:
    json_data = json.loads(f.read())
    df = pd.DataFrame(json_data)


def write_json_file(blob, file_path):
    with open(file_path, "w") as file:
        json.dump(blob, file, sort_keys=True)


# 2.Utils
def matches_regex(regex, text):
    return bool(re.compile(regex).search(text))


def contains_code(text):
    ## filter based on keywords that indicate code
    code_blacklist = ["&&", "||", "<html>", ";\n", "SELECT"]

    return (
        any(code_keyword in text for code_keyword in code_blacklist)
        | matches_regex(r"\w+\(\w*\) \{", text)
        | matches_regex(r"def \w+\(", text)  # e.g. myFunc() {
        | matches_regex(r"\[A-z]+\.[A-z]+", text)  # e.g. def parse_list(
        | matches_regex(r": [\w\.#]{1,12};", text)  # e.g. this.language
        | matches_regex(r"<\/\w+>", text)  # e.g. font-size: 1.3em;  # e.g. </html>
    )


def contains_words(text):
    return matches_regex(r"[A-z]{3,}", text)  # words with at least three characters


def is_translatable(text):
    if text == "":
        return True
    return (contains_code(text) is False) & contains_words(text)


# 3. Translate
def translate_and_update_series(text_series):
    # memorize whether and where the list contains non-translatable content
    is_translatable_index = text_series.apply(lambda x: is_translatable(x) is False)
    text_list_source_language = text_series.tolist()

    # replace non-translatable content with an empty string
    text_series[is_translatable_index] = ""

    # translate list
    text_list = text_series.tolist()
    translated_list = translate_list(text_list)

    # if list contains non-translatable content, replace accordingly
    # This part is broken, hours wasted: 1
    if is_translatable_index.sum() > 0:
        for index, text_is_translatable in enumerate(is_translatable_index.tolist()):
            if text_is_translatable:
                translated_list[index] = text_list_source_language[index]

    return translated_list


def login_hugging_face(token: str) -> None:
    """
    Loging to Hugging Face portal with a given token.
    """
    if not token:
        return

    api = HfApi()
    api.set_access_token(token)
    folder = HfFolder()
    folder.save_token(token)

    return None


def nllb_translate(text_list):
    inputs = tokenizer(text_list, return_tensors="pt", padding=True).to("cuda")

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["khm_Khmr"],
        max_length=2048,
    )
    res_nllb = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True).to("cpu")
    return res_nllb


def translate_list(text_list):
    # combined_response = translator.translate_text(text_list, source_lang="EN", target_lang=TARGET_LANG, formality=FORMALITY)
    combined_response = nllb_translate(text_list)

    # return [response.text for response in combined_response]
    return combined_response


def translate_dataframe(df):
    os.makedirs(output_dir, exist_ok=True)

    # Check for dataframe rows that are not translatable
    print("Checking for non-translatable rows")
    df['instruction'] = df.instruction.apply(lambda x: x if is_translatable(x) else None)
    df['input'] = df.input.apply(lambda x: x if is_translatable(x) else None)
    df['output'] = df.output.apply(lambda x: x if is_translatable(x) else None)

    # Drop rows with None
    print("Dropping rows with None")
    df = df.dropna()

    # Split into chunks
    print("Splitting into chunks")
    number_of_chunks = df.shape[0] // chunk_size
    chunked_df_list = np.array_split(df, number_of_chunks)

    start_index = 1
    print('Translate..')
    for index, chunk_df in enumerate(tqdm(chunked_df_list[start_index:])):
        instruction_list_translated = translate_list(chunk_df.instruction.to_list())
        input_list_translated = translate_list(chunk_df.input.to_list())
        output_list_translated = translate_list(chunk_df.output.to_list())

        translated_df = pd.DataFrame(
            {
                "instruction": instruction_list_translated,
                "input": input_list_translated,
                "output": output_list_translated,
            }
        )
        translated_dict = translated_df.to_dict("records")

        write_json_file(translated_dict, f"{output_dir}chunk{start_index+index}.json")


chunk_size = 5
output_dir = "./data/output/"

login_hugging_face(None)

# Load tokenizer and model
print('Loading tokenizer and model')
tokenizer = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="eng_Latn"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-600M", use_auth_token=True
).to("cuda")
print('Tokenizer and model loaded')

# Translate
translate_dataframe(df)


def combine_chunks():
    translated_tasks_list = []
    for index in range(0, len(glob.glob(f"{output_dir}*.json"))):
        with open(f"{output_dir}chunk{index}.json", "rb") as f:
            translated_tasks_list += json.loads(f.read())
    write_json_file(translated_tasks_list, f"./translated_tasks_km_nllb.json")


combine_chunks()
