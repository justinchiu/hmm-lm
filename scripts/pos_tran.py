
from pathlib import Path
import re

def is_num(x):
    return re.match(r'^-?\d+(?:\.\d+)?$', x)

txt = Path(".data/wsj/wsj.raw")
tag = Path(".data/wsj/wsj.tag")

txt_proc = Path(".data/wsj/wsj.txt")

# preprocess text, as in Tran et al 2016

lines = []
for line in txt.read_text().split("\n"):
    lines.append(" ".join(
        x if not is_num(x) else "0" * len(x)
        for x in line.strip().split()
    ))
txt_proc.write_text("\n".join(lines))
