import pandas as pd
from copy import copy
from json import dump

train_data = pd.read_csv(".\\smilestyle_dataset.tsv", sep="	")
tags = [
    "informal",
    "android",
    "azae",
    "chat",
    "choding",
    "emoticon",
    "enfp",
    "gentle",
    "halbae",
    "halmae",
    "joongding",
    "king",
    "naruto",
    "seonbi",
    "sosim",
    "translator"
]

cleaned = []

for index, formal in enumerate(train_data['formal']):
    data = {"formal":formal,
            "styles":{
                tag:train_data[tag][index] for tag in tags if train_data[tag][index]
            }
            }
    
    cleaned.append(data)

with open(".\\dataset.json", "w", encoding='utf8') as file:
    dump(cleaned, file, ensure_ascii=False, indent="\t")

with open(".\\dataset.json", "r", encoding='utf8') as file:
    data = file.read()

data = data.splitlines()

result = []

for i, line in enumerate(data[:-1]):
    if ('NaN' in line): continue

    if (line.replace("	", "") in '()[],{},'):result.append(line); continue

    if ('"' in data[i+1]):

        if (line[-1] == ","):
            if '"' in data[i+1]: result.append(line)
            else:
                result.append(line[:-1])
        else:
            result.append(line)

result.append("]")
result = eval("\n".join(result))

with open(".\\dataset.json", "w", encoding='utf8') as file:
    dump(result, file, ensure_ascii=False, indent="\t")