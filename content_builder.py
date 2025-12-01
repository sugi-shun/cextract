import re
import pandas as pd

# コンパイル済みの正規表現を再利用
regex = re.compile(r"\{id([0-9]+)\}")

def build(df: pd.DataFrame, threshold: float = 0.8) -> str:
    df_fix = df[df["pred"] > threshold]
    d = {str(idx): content for idx, content in zip(df_fix["id"], df_fix["contents"]) if isinstance(content, str)}
    sorted_ids = sorted([int(k) for k in d.keys()])
    resolved_contents = {} # 解決済みのコンテンツを保持する辞書
    for target_id_num in sorted_ids:
        target_id = str(target_id_num)
        if target_id not in d:
            continue
        current_contents = d[target_id]
        def replace_match(match):
            ref_id = match.group(1) 
            if ref_id in resolved_contents:
                return resolved_contents[ref_id]

            if ref_id in d:
                return d[ref_id] 

            return ""

        new_contents = regex.sub(replace_match, current_contents)
        resolved_contents[target_id] = new_contents
        
        del d[target_id]

    final_keys = sorted([int(k) for k in resolved_contents.keys()])
    final_output = ' '.join([resolved_contents[str(k)] for k in final_keys])
    return final_output.replace("[ID_NOT_FOUND]", "") 


if __name__ == "__main__":
    # サンプルのDataFrameを作成
    data = {
        'id': [100, 200, 300, 400],
        'contents': ["Start {id200} End. iddle {id300} {id500}.", "ho", "Deepest A {id100}.", "Ignored"],
        'pred': [0.95, 0.85, 0.75, 0.99]
    }
    df_simple = pd.DataFrame(data)
    
    # 200 -> C D
    # 100 -> A C D B
    print("\n### 最適化された関数による実行結果 ###")
    print(f"Result: {build_optimized(df_simple, threshold=0.8)}")
