# puts into texts_check.txt the first 1000 lines of texts.txt
with open("texts.txt", "r") as f:
    lines = f.readlines()
    with open("data/texts_check.txt", "w") as out_f:
        for line in lines[:1000]:
            out_f.write(line)