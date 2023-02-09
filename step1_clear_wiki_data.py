input_path = "./example_data/text/wiki_corpus.txt"
output_path = "./example_data/text/wiki_corpus_part.txt"

result = []
count = 0
with open(input_path, encoding="utf-8") as rfile:
    for line in rfile.readlines():
        line = line.strip().replace("=", "").replace("*", "").replace("#", "").strip()
        if len(line) < 4:
            continue
        result.append(line)
        count += 1
        if count == 100000:
            break

with open(output_path, 'w', encoding="utf-8") as wfile:
    for line in result:
        wfile.writelines(line + "\n")
