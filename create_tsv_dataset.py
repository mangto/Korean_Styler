from json import load

with open(".\\dataset.json", "r", encoding='utf8') as file:
    data = load(file)

sep = "	"
result = f"original{sep}styled"

for obj in data:
    formal = obj.get("formal", "")
    styles = obj.get("styles", {})
    for style in styles:
        result += "\n" + "[" + style + "] " + formal + sep + styles.get(style, "")

with open(".\\dataset.tsv", "w", encoding='utf8') as file:
    file.write(result)