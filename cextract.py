import json
import os
import pickle
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import content_builder
from download_html import download
from html2csv import extract_dom_data

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

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise Exception("Preprocessor not fitted!")
        all_xpaths = [self._clean_xpath(x) for x in df['xpath']]
        xpath_seq = self.tag_tokenizer.texts_to_sequences(all_xpaths)
        xpath_pad = pad_sequences(xpath_seq, maxlen=CONFIG["MAX_XPATH_LEN"], padding='post')

        all_keys, all_values = [], []
        for a_json in df['attributes']:
            keys, values = self._parse_attributes(a_json)
            all_keys.append(keys)
            all_values.append(values)

        key_seq = self.attr_tokenizer.texts_to_sequences(all_keys)
        key_pad = pad_sequences(key_seq, maxlen=CONFIG["MAX_ATTR_KEY_LEN"], padding='post')

        val_seq = self.attr_tokenizer.texts_to_sequences(all_values)
        val_pad = pad_sequences(val_seq, maxlen=CONFIG["MAX_ATTR_VAL_LEN"], padding='post')

        numeric_data = np.array(df.apply(self._extract_numeric_features, axis=1).tolist())
        nums_scaled = self.scaler.transform(numeric_data)

        return {
            "xpath_in": xpath_pad,
            "attr_key_in": key_pad,
            "attr_val_in": val_pad,
            "numeric_in": nums_scaled
        }


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
    return df


def predict(model, preprocessor, df):
    df_fix = clean_structure_garbage(df, xpath_col='xpath')
    if df_fix.empty:
        df_fix["pred"] = []
        return df_fix
    input_tensor_batch = preprocessor.transform(df_fix)
    pred_probs = model.predict(input_tensor_batch, verbose=0)
    df_fix["pred"] = pred_probs.flatten()

    return df_fix


if __name__ == "__main__":
    import sys
    url = sys.argv[1]
    print("loading...")
    model = tf.keras.models.load_model('models/dom_predictor_model.h5')
    with open("models/dom_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    print("downloading...")
    html = download(url)
    print("extracting...")
    dom_array = extract_dom_data(html)
    print("predicting...")
    df = pd.DataFrame(dom_array)
    df_pred = predict(model, preprocessor, df)
    print("building...")
    content = content_builder.build(df_pred, 0.9)
    print(content)
    df_pred.to_csv("example.csv", index=False)
