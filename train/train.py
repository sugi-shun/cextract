import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import (LSTM, BatchNormalization, Dense, Dropout,
                                     Embedding, Input, concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

CONFIG = {
    "MAX_XPATH_LEN": 15,
    "MAX_ATTR_KEY_LEN": 5,
    "MAX_ATTR_VAL_LEN": 20,
    "VOCAB_SIZE_TAG": 100,
    "VOCAB_SIZE_ATTR": 1000,
    "EMBED_DIM": 32,
    "LSTM_UNITS": 64,
    "DROPOUT_RATE": 0.3,
}


class DOMPreprocessor:
    def __init__(self):
        self.tag_tokenizer = Tokenizer(num_words=CONFIG["VOCAB_SIZE_TAG"], oov_token="<UNK>")
        self.attr_tokenizer = Tokenizer(num_words=CONFIG["VOCAB_SIZE_ATTR"], oov_token="<UNK>")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _clean_xpath(self, xpath: str) -> List[str]:
        clean_path = re.sub(r'\[\d+\]', '', xpath)
        return clean_path.split('/')

    def _parse_attributes(self, attrs_json: str) -> Tuple[List[str], List[str]]:
        try:
            attrs = json.loads(attrs_json)
        except:
            return [], []

        keys = []
        values = []
        for k, v in attrs.items():
            keys.append(str(k).lower())
            if isinstance(v, list):
                values.extend([str(x).lower() for x in v])
            else:
                values.append(str(v).lower())
        return keys, values

    def _extract_numeric_features(self, row: pd.Series) -> List[float]:
        content_str = str(row.get('contents', ''))
        text_only = re.sub(r'\{id\d+\}', '', content_str)
        text_len = len(text_only)
        log_text_len = np.log1p(text_len)
        child_count = len(re.findall(r'\{id\d+\}', content_str))
        depth = len(row['xpath'].split('/'))
        match = re.search(r'\[(\d+)\]$', row['xpath'])
        sibling_index = int(match.group(1)) if match else 0
        return [log_text_len, float(child_count), float(depth), float(sibling_index)]

    def fit(self, df: pd.DataFrame):
        all_xpaths = [self._clean_xpath(x) for x in df['xpath']]
        self.tag_tokenizer.fit_on_texts(all_xpaths)
        all_attr_texts = []
        for a_json in df['attributes']:
            k, v = self._parse_attributes(a_json)
            all_attr_texts.append(k + v)
        self.attr_tokenizer.fit_on_texts(all_attr_texts)
        numeric_data = [self._extract_numeric_features(row) for _, row in df.iterrows()]
        self.scaler.fit(numeric_data)
        self.is_fitted = True
        print("✅ Preprocessor fitted.")

    def transform_single(self, row_dict: Dict) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise Exception("Preprocessor not fitted!")
        xpath_tokens = self._clean_xpath(row_dict['xpath'])
        xpath_seq = self.tag_tokenizer.texts_to_sequences([xpath_tokens])
        xpath_pad = pad_sequences(xpath_seq, maxlen=CONFIG["MAX_XPATH_LEN"], padding='post')
        keys, values = self._parse_attributes(row_dict['attributes'])
        key_seq = self.attr_tokenizer.texts_to_sequences([keys])
        key_pad = pad_sequences(key_seq, maxlen=CONFIG["MAX_ATTR_KEY_LEN"], padding='post')
        val_seq = self.attr_tokenizer.texts_to_sequences([values])
        val_pad = pad_sequences(val_seq, maxlen=CONFIG["MAX_ATTR_VAL_LEN"], padding='post')
        row_series = pd.Series(row_dict)
        nums = np.array([self._extract_numeric_features(row_series)])
        nums_scaled = self.scaler.transform(nums)
        return {
            "xpath_in": xpath_pad,
            "attr_key_in": key_pad,
            "attr_val_in": val_pad,
            "numeric_in": nums_scaled
        }


def build_model():
    xpath_in = Input(shape=(CONFIG["MAX_XPATH_LEN"],), name="xpath_in")
    x1 = Embedding(CONFIG["VOCAB_SIZE_TAG"] + 1, CONFIG["EMBED_DIM"])(xpath_in)
    x1 = LSTM(CONFIG["LSTM_UNITS"], return_sequences=False)(x1)

    attr_key_in = Input(shape=(CONFIG["MAX_ATTR_KEY_LEN"],), name="attr_key_in")
    x2 = Embedding(CONFIG["VOCAB_SIZE_ATTR"] + 1, CONFIG["EMBED_DIM"])(attr_key_in)
    x2 = LSTM(CONFIG["LSTM_UNITS"] // 2, return_sequences=False)(x2)

    attr_val_in = Input(shape=(CONFIG["MAX_ATTR_VAL_LEN"],), name="attr_val_in")
    x3 = Embedding(CONFIG["VOCAB_SIZE_ATTR"] + 1, CONFIG["EMBED_DIM"])(attr_val_in)
    x3 = LSTM(CONFIG["LSTM_UNITS"], return_sequences=False)(x3)

    numeric_in = Input(shape=(4,), name="numeric_in")
    x4 = Dense(16, activation="relu")(numeric_in)

    merged = concatenate([x1, x2, x3, x4])
    m = BatchNormalization()(merged)  # 安定化
    m = Dense(64, activation="relu")(m)
    m = Dropout(CONFIG["DROPOUT_RATE"])(m)  # 汎化性能向上

    output = Dense(1, activation="sigmoid")(m)

    model = Model(inputs=[xpath_in, attr_key_in, attr_val_in, numeric_in], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "precision", "recall"])
    return model


def clean_structure_garbage(df, xpath_col='xpath'):
    initial_count = len(df)
    FORBIDDEN_ANCESTORS = {
        'head', 'script', 'style', 'noscript', 'template',
        'nav', 'aside', 'footer', 'menu',
        'applet', 'object', 'embed', "table", "meta",
        "figure", "figcaption", "img", "form" 
    }
    FORBIDDEN_LEAFS = {
        'svg', 'path', 'g'
        'button',
        'input', 'select', 'optgroup', 'option', 'textarea',
        'iframe', 'canvas'
        'sup', 'sub', "html", "body"
    }

    def is_garbage_structure(xpath_str):
        if not isinstance(xpath_str, str):
            return True  # XPathがないなら除外
        parts = xpath_str.split('/')

        tags = [p.split('[')[0] for p in parts if p]

        if not tags:
            return True

        if not set(tags).isdisjoint(FORBIDDEN_ANCESTORS):
            return True

        leaf_tag = tags[-1]
        if leaf_tag in FORBIDDEN_LEAFS:
            return True

        return False

    df = df[~df[xpath_col].apply(is_garbage_structure)]
    removed_count = initial_count - len(df)
    print(f"構造フィルタで削除: {removed_count} / {initial_count}")

    return df


def load_data_and_labels(csv_paths, txt_paths):
    all_data = []
    for csv_path, txt_path in zip(csv_paths, txt_paths):
        df = pd.read_csv(csv_path)
        df = clean_structure_garbage(df, xpath_col='xpath')
        labels_map = {}  # id -> 1 (content)
        with open(txt_path, 'r', encoding='utf-8') as f:
            content_ids = []
            for line in f:
                if line.startswith("contents:"):
                    ids_str = line.split(":")[1].strip()
                    if ids_str:
                        content_ids = [int(x.strip()) for x in ids_str.split(",")]
                    break
            df['target'] = df['id'].apply(lambda x: 1 if x in content_ids else 0)
            all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def save_preprocessor(preprocessor, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(preprocessor, f)


if __name__ == "__main__":
    import pickle

    # 1. ダミーデータの生成 (実際のファイルがある想定)
    csv_files = [f"cextract_data/orig/data/{i}.csv" for i in range(30)]
    txt_files = [f"cextract_data/annotation/{i}.txt" for i in range(30)]
    df_train = load_data_and_labels(csv_files, txt_files)

    # 2. 前処理 (Fitting)
    preprocessor = DOMPreprocessor()
    preprocessor.fit(df_train)

    # 3. 学習用データセットの作成 (バッチ処理)
    # 実際はメモリ節約のためジェネレータを使うのが良いが、ここではシンプルに変換
    X_inputs = {
        "xpath_in": [], "attr_key_in": [], "attr_val_in": [], "numeric_in": []
    }
    y_train = df_train['target'].values

    print("学習データ変換中...")
    for _, row in df_train.iterrows():
        # transform_singleはバッチ次元(1, N)を持っているので[0]で取り出す
        inputs = preprocessor.transform_single(row.to_dict())
        X_inputs["xpath_in"].append(inputs["xpath_in"][0])
        X_inputs["attr_key_in"].append(inputs["attr_key_in"][0])
        X_inputs["attr_val_in"].append(inputs["attr_val_in"][0])
        X_inputs["numeric_in"].append(inputs["numeric_in"][0])

    # numpy配列化
    for k in X_inputs:
        X_inputs[k] = np.array(X_inputs[k])

    # 4. モデル構築と学習
    model = build_model()

    # 不均衡データ対策: クラス重みの計算
    # 記事本文(1)は圧倒的に少ないため、間違えたときのペナルティを大きくする
    neg, pos = np.bincount(y_train)
    total = neg + pos
    class_weight = {0: 1.0, 1: (neg / pos)}  # 正例を強めに

    print("学習開始...")
    model.fit(
        X_inputs, y_train,
        epochs=10,
        batch_size=32,
        class_weight=class_weight,
        verbose=1
    )

    print("\n=== 推論テスト (1行ごとの予測) ===")

    # 5. 未知の1行データに対する予測フロー
    test_row = {
        "id": 999,
        "xpath": "html/body/div[0]/article/div/p[2]",
        "attributes": json.dumps({"class": "article-content", "data-type": "main"}),
        "contents": "AIモデルの汎化性能を高めるためには、ドロップアウトが有効です。"
    }

    # 前処理 -> 辞書形式のnumpy配列(1, N)が返る
    input_tensor = preprocessor.transform_single(test_row)

    # 予測
    pred_prob = model.predict(input_tensor, verbose=0)[0][0]

    print(f"入力データ: {test_row['xpath']}")
    print(f"属性: {test_row['attributes']}")
    print(f"予測確率 (コンテンツである確率): {pred_prob:.4f}")

    if pred_prob > 0.5:
        print("判定: 記事本文です")
    else:
        print("判定: その他要素です")

    MODEL_FILENAME = 'dom_predictor_model.h5'
    PREPROCESSOR_FILENAME = 'dom_preprocessor.pkl'

    # 1. Kerasモデルの保存
    model.save(MODEL_FILENAME)
    print(f"✅ Kerasモデルを '{MODEL_FILENAME}' に保存しました。")

    # 2. Preprocessorの保存
    try:
        save_preprocessor(preprocessor, PREPROCESSOR_FILENAME)
        print(f"Preprocessorを '{PREPROCESSOR_FILENAME}' に保存しました。")
    except Exception as e:
        print(f"Preprocessorの保存中にエラーが発生しました: {e}")
