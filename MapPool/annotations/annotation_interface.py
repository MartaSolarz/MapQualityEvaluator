#!/usr/bin/env python3
"""
Interface do ręcznej adnotacji map statystycznych
Użycie: streamlit run annotation_interface.py -- --data_path train_data_200k.parquet --target 3000

Testowo: streamlit run ./annotations/annotation_interface.py -- --data_path ./data_all/above_value_minus1/sample_200k.parquet --target 3000
"""

import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import json
from pathlib import Path
import argparse
import sys
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Map annotation interface')
    parser.add_argument('--data_path', type=str, required=True, help='Path to parquet file with data')
    parser.add_argument('--target', type=int, default=3000, help='Target number of valid annotations (default: 3000)')

    args = parser.parse_args(sys.argv[1:])
    return args


if 'args_loaded' not in st.session_state:
    try:
        args = parse_args()
        st.session_state.data_path = args.data_path
        st.session_state.target_annotations = args.target
        st.session_state.args_loaded = True
    except SystemExit:
        st.error(
            "⚠️ Uruchom skrypt z parametrami: streamlit run annotation_interface.py -- --data_path <path> --target <number>")
        st.stop()

ANNOTATIONS_FILE = Path("annotations_progress.jsonl")


def load_data(data_path):
    try:
        df = pd.read_parquet(data_path)
        return df
    except Exception as e:
        st.error(f"Błąd wczytywania danych: {e}")
        return None


def load_annotations():
    if ANNOTATIONS_FILE.exists():
        annotations = {}
        with open(ANNOTATIONS_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    annotations[record['uid']] = record
        return annotations
    return {}


def save_annotation(uid, label):
    record = {
        'uid': uid,
        'label': label,
        'timestamp': datetime.now().isoformat()
    }
    with open(ANNOTATIONS_FILE, 'a') as f:
        f.write(json.dumps(record) + '\n')


def check_url_valid(url):
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        return response.status_code == 200
    except:
        return False


def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        return None
    except:
        return None


def get_annotation_stats(annotations):
    stats = {'TAK': 0, 'NIE': 0, 'INVALID': 0}
    for ann in annotations.values():
        label = ann['label']
        if label in stats:
            stats[label] += 1
    return stats


import random

def get_next_sample(df, annotations):
    annotated_uids = set(annotations.keys())
    remaining = df[~df['uid'].isin(annotated_uids)]

    if len(remaining) == 0:
        return None

    available_bins = [b for b, remaining_count in st.session_state.bin_remaining.items() if remaining_count > 0]

    if not available_bins:
        return None

    selected_bin = random.choice(available_bins)

    bin_candidates = remaining[remaining['score_bin'] == selected_bin]
    if len(bin_candidates) == 0:
        st.session_state.bin_remaining[selected_bin] = 0
        return get_next_sample(df, annotations)

    sample = bin_candidates.sample(n=1).iloc[0]

    st.session_state.bin_remaining[selected_bin] -= 1

    return sample



if 'annotations' not in st.session_state:
    st.session_state.annotations = load_annotations()

if 'data' not in st.session_state:
    st.session_state.data = load_data(st.session_state.data_path)
    if st.session_state.data is None:
        st.stop()

if 'current_sample' not in st.session_state:
    st.session_state.current_sample = None
    st.session_state.current_image = None
    st.session_state.url_checked = False

st.title("🗺️ Interface do adnotacji map statystycznych")
st.markdown("---")

bin_counts = st.session_state.data['score_bin'].value_counts().to_dict()

total_samples_to_annotate = st.session_state.target_annotations
bin_proportions = {b: c / sum(bin_counts.values()) for b, c in bin_counts.items()}

if 'bin_remaining' not in st.session_state:
    st.session_state.bin_remaining = {b: int(total_samples_to_annotate * p) for b, p in bin_proportions.items()}


stats = get_annotation_stats(st.session_state.annotations)
valid_annotations = stats['TAK'] + stats['NIE']
target = st.session_state.target_annotations

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("✅ TAK", stats['TAK'])
with col2:
    st.metric("❌ NIE", stats['NIE'])
with col3:
    st.metric("🚫 INVALID", stats['INVALID'])
with col4:
    st.metric("🎯 Postęp", f"{valid_annotations}/{target}")

progress = min(valid_annotations / target, 1.0)
st.progress(progress)

if valid_annotations >= target:
    st.success(f"🎉 Gratulacje! Osiągnięto cel {target} valid annotations!")
    st.balloons()
    st.stop()

st.markdown("---")

if st.session_state.current_sample is None:
    sample = get_next_sample(st.session_state.data, st.session_state.annotations)
    if sample is None:
        st.warning("Brak więcej próbek do adnotacji!")
        st.stop()
    st.session_state.current_sample = sample
    st.session_state.url_checked = False
    st.session_state.current_image = None

sample = st.session_state.current_sample

st.subheader("Informacje o próbce")
info_col1, info_col2 = st.columns(2)
with info_col1:
    st.write(f"**UID:** `{sample['uid']}`")
    st.write(f"**Score słownika:** `{sample['score']}`")
with info_col2:
    if sample['pos_keys']:
        st.write(f"**Pozytywne słowa:** {sample['pos_keys']}")
    if sample['neg_keys']:
        st.write(f"**Negatywne słowa:** {sample['neg_keys']}")

if not st.session_state.url_checked:
    with st.spinner("Sprawdzanie URL..."):
        print(sample['url'])
        url_valid = check_url_valid(sample['url'])
        st.session_state.url_checked = True

        if not url_valid:
            st.warning("⚠️ URL nie odpowiada (status != 200) - automatycznie oznaczam jako INVALID")
            save_annotation(sample['uid'], 'INVALID')
            st.session_state.annotations[sample['uid']] = {
                'uid': sample['uid'],
                'label': 'INVALID',
            }
            st.session_state.current_sample = None
            st.rerun()
        else:
            with st.spinner("Ładowanie obrazka..."):
                img = load_image_from_url(sample['url'])
                if img is None:
                    st.warning("⚠️ Nie udało się wczytać obrazka - automatycznie oznaczam jako INVALID")
                    save_annotation(sample['uid'], 'INVALID')
                    st.session_state.annotations[sample['uid']] = {
                        'uid': sample['uid'],
                        'label': 'INVALID',
                    }
                    st.session_state.current_sample = None
                    st.rerun()
                else:
                    st.session_state.current_image = img

if st.session_state.current_image is not None:
    st.image(st.session_state.current_image, caption=f"Score: {sample['score']}", use_container_width=True)

    st.markdown("---")

    st.subheader("Czy to mapa statystyczna?")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ TAK (Y)", key="yes_btn", use_container_width=True, type="primary"):
            save_annotation(sample['uid'], 'YES')
            st.session_state.annotations[sample['uid']] = {
                'uid': sample['uid'],
                'label': 'TAK',
            }
            st.session_state.current_sample = None
            st.rerun()

    with col2:
        if st.button("❌ NIE (N)", key="no_btn", use_container_width=True):
            save_annotation(sample['uid'], 'NO')
            st.session_state.annotations[sample['uid']] = {
                'uid': sample['uid'],
                'label': 'NIE',
            }
            st.session_state.current_sample = None
            st.rerun()

    with col3:
        if st.button("🚫 INVALID URL (I)", key="invalid_btn", use_container_width=True):
            save_annotation(sample['uid'], 'INVALID')
            st.session_state.annotations[sample['uid']] = {
                'uid': sample['uid'],
                'label': 'INVALID',
            }
            st.session_state.current_sample = None
            st.rerun()

    st.markdown("---")
    st.caption("⌨️ Skróty klawiszowe: Y = TAK | N = NIE | I = INVALID")

with st.sidebar:
    st.header("📊 Statystyki")
    st.write(f"**Plik danych:** `{st.session_state.data_path}`")
    st.write(f"**Cel:** {target} valid annotations")
    st.write(f"**Pozostało:** {target - valid_annotations}")

    total_annotations = len(st.session_state.annotations)
    st.write(f"**Łącznie oznaczono:** {total_annotations}")

    if total_annotations > 0:
        st.write(f"**% INVALID:** {stats['INVALID'] / total_annotations * 100:.1f}%")

    st.markdown("---")
    st.write(f"**Plik zapisu:** `{ANNOTATIONS_FILE}`")

    if st.button("🔄 Reset session (nie usuwa pliku)"):
        for key in list(st.session_state.keys()):
            if key != 'args_loaded':
                del st.session_state[key]
        st.rerun()