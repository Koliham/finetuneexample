import pandas as pd
import json


def main():
    df = pd.read_csv("data.csv", sep=";")
    df["input"] = df["input"].fillna("")
    text_col = []
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
        text_col.append(text)
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

    df.loc[:,"text"] = text_col
    df.to_csv("train.csv", index=False)

if __name__ == '__main__':
    main()


