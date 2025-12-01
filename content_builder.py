import re
import pandas as pd

regex = re.compile(r"\{id([0-9]+)\}")

def build(df, threshold=0.8):
    df_fix = df[df["pred"] > threshold]
    d = {str(idx): content for idx, content in zip(df_fix["id"], df_fix["contents"])}
    for k in sorted(list(d.keys())):
        if k not in d:
            continue
        if not isinstance(d[k], str):
            del d[k]
            continue
        repids = re.findall(regex, d[k])
        while repids:
            for idx in repids:
                idx = str(idx)
                if idx in d:
                    if isinstance(d, str):
                        d[k] = d[k].replace(f"{{id{idx}}}", d[idx])
                    del d[idx]
                else:
                    d[k] = d[k].replace(f"{{id{idx}}}", "")
            repids = re.findall(regex, d[k])
    return ' '.join([d[k] for k in sorted(list(d.keys()))]).replace("[ID_NOT_FOUND]", "")


if __name__ == "__main__":
    df = pd.read_csv("example.csv")
    print(build(df))
