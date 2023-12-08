import pandas as pd
import json
import math


def main():
    df = pd.read_csv("data.csv", sep=";")
    df["input"] = df["input"].fillna("")
    text_col = []
    text_col_mistral = []
    jsonl = ""
    guanacojsonl = ""
    for _, row in df.iterrows():
        prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        instruction = str(row["instruction"])
        input = str(row["input"])
        output = str(row["output"])
        if len(input.strip()) == 0:
            text = prompt + "### Instruction:\n" + instruction + "\n### Response:\n" + output
        else:
            text = prompt + "### Instruction:\n" + instruction + "\n### Input:\n" + input + "\n### Response:\n" + output
        text_mistral = f"""<s>[INST] {instruction} [/INST] \\n {output} </s>"""
        text_col.append(text)
        text_col_mistral.append(text_mistral)
        jsondict = {"instruction": instruction, "output": output}
        guanacodict = {"text": f"### Human: {instruction}### Assistant: {output}"}
        if len(input.strip()) > 0:
            jsondict["input"] = input
        jsonl = jsonl + json.dumps(jsondict) + "\n"
        guanacojsonl = guanacojsonl + json.dumps(guanacodict) + "\n"
    jsonl = jsonl.strip()
    guanacojsonl = guanacojsonl.strip()
    with open("train.jsonl", "w") as f:
        f.write(jsonl)

    # Guanaco:
    with open("train_guanaco.jsonl", "w") as f:
        f.write(guanacojsonl)

    df_mistral = df.copy()
    df_mistral.loc[:,"text"] = text_col_mistral
    df.to_csv("train_mistral.csv", index=False)

    df.loc[:,"text"] = text_col
    df.to_csv("train.csv", index=False)
    # if the train file is less than 40, then copy-paste it:
    if len(df) <= 40:
        multiplicator_factor = math.ceil(40/len(df))
        dflist = []
        dflist_mistral = []
        for i in range(multiplicator_factor):
            dflist.append(df)
            dflist_mistral.append(df_mistral)
        df_extended = pd.concat(dflist)
        df_extended.to_csv("train_extended.csv", index=False)

        df_extended_mistral = pd.concat(dflist_mistral)
        df_extended_mistral.to_csv("train_extended_mistral.csv", index=False)

if __name__ == '__main__':
    main()


