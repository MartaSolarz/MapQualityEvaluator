#!/usr/bin/env python3
"""
Interface do adnotacji próbek wybranych przez Active Learning

Użycie:
# Iteration mode (standard AL workflow)
streamlit run annotation_interface.py -- --iteration 1 --target 600

# File mode (custom file, np. test set)
streamlit run annotation_interface.py -- --file data/test_mini_200_to_annotate.parquet --target 200

Automatyczne funkcje:
- HEAD request sprawdza URL przed wyświetleniem
- Automatycznie pomija invalid URLs (nie liczą się do targetu)
- Zapisuje progress po każdej adnotacji
- Resume: można przerwać i wrócić później
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
    parser = argparse.ArgumentParser(description='Active Learning annotation interface')
    parser.add_argument('--iteration', type=int, help='Numer iteracji (np. 1)')
    parser.add_argument('--file', type=str, help='Ścieżka do pliku do adnotacji (alternatywa dla --iteration)')
    parser.add_argument('--target', type=int, default=600, help='Target liczba VALID annotations (default: 600)')
    
    args = parser.parse_args(sys.argv[1:])
    
    # Validation: musi być --iteration XOR --file
    if args.iteration is None and args.file is None:
        parser.error("Podaj --iteration LUB --file")
    
    if args.iteration is not None and args.file is not None:
        parser.error("Podaj TYLKO --iteration LUB --file (nie oba)")
    
    return args


if 'args_loaded' not in st.session_state:
    try:
        args = parse_args()
        st.session_state.target_annotations = args.target
        
        BASE_DIR = Path(__file__).parent
        
        # MODE 1: Iteration (standard AL workflow)
        if args.iteration is not None:
            st.session_state.mode = 'iteration'
            st.session_state.iteration_num = args.iteration
            
            ITERATION_DIR = BASE_DIR / "iterations" / f"iteration_{args.iteration}"
            
            if not ITERATION_DIR.exists():
                st.error(f"❌ Katalog iteracji nie istnieje: {ITERATION_DIR}")
                st.stop()
            
            st.session_state.iteration_dir = ITERATION_DIR
            st.session_state.data_path = ITERATION_DIR / "to_annotate.parquet"
            st.session_state.annotations_file = ITERATION_DIR / "annotations_progress.jsonl"
            st.session_state.output_file = ITERATION_DIR / "annotated.parquet"
            st.session_state.partial_output_file = ITERATION_DIR / "annotated_partial.parquet"
        
        # MODE 2: Custom file (e.g., test set)
        else:
            st.session_state.mode = 'file'
            st.session_state.iteration_num = None
            
            # Parse file path
            file_path = Path(args.file)
            if not file_path.is_absolute():
                file_path = BASE_DIR / file_path
            
            if not file_path.exists():
                st.error(f"❌ Plik nie istnieje: {file_path}")
                st.stop()
            
            # Output: ten sam katalog + zmiana nazwy
            output_dir = file_path.parent
            base_name = file_path.stem.replace('_to_annotate', '')
            
            st.session_state.data_path = file_path
            st.session_state.annotations_file = output_dir / f"{base_name}_annotations_progress.jsonl"
            st.session_state.output_file = output_dir / f"{base_name}.parquet"
            st.session_state.partial_output_file = output_dir / f"{base_name}_partial.parquet"
            st.session_state.iteration_dir = output_dir
        
        st.session_state.args_loaded = True
        
    except SystemExit as e:
        st.error("⚠️ Uruchom: streamlit run annotation_interface.py -- --iteration <N> LUB --file <path>")
        st.stop()


def load_data():
    """Wczytaj dane do adnotacji"""
    try:
        df = pd.read_parquet(st.session_state.data_path)
        return df
    except Exception as e:
        st.error(f"Błąd wczytywania danych: {e}")
        return None


def load_baseline_uids():
    """Wczytaj UIDs już zaadnotowane w baseline
    
    UWAGA: To jest opcjonalne - próbki z baseline są już wykluczone z pool.
    Ta funkcja służy tylko do double-check.
    """
    baseline_path = Path(__file__).parent / "data" / "baseline" / "baseline.parquet"
    
    if baseline_path.exists():
        try:
            df_baseline = pd.read_parquet(baseline_path)
            return set(df_baseline['uid'].values)
        except Exception as e:
            # Nie jest to krytyczne - próbki są już wykluczone
            return set()
    else:
        # Nie jest to problem - próbki są już wykluczone z pool
        return set()


def load_annotations():
    """Wczytaj zapisane adnotacje"""
    annotations_file = st.session_state.annotations_file
    
    if annotations_file.exists():
        annotations = {}
        with open(annotations_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    annotations[record['uid']] = record
        return annotations
    return {}


def save_annotation(uid, url, label):
    """Zapisz adnotację do JSONL"""
    record = {
        'uid': uid,
        'url': url,
        'label': label,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(st.session_state.annotations_file, 'a') as f:
        f.write(json.dumps(record) + '\n')


def check_url_valid(url):
    """Sprawdź czy URL działa (HEAD request)"""
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        return response.status_code == 200
    except:
        return False


def load_image_from_url(url):
    """Wczytaj obrazek z URL"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        return None
    except:
        return None


def get_annotation_stats(annotations):
    """Oblicz statystyki adnotacji"""
    stats = {'YES': 0, 'NO': 0, 'INVALID': 0}
    
    for ann in annotations.values():
        label = ann['label'].upper()
        if label in stats:
            stats[label] += 1
    
    return stats


def get_next_sample(df, annotations, baseline_uids=None):
    """Pobierz kolejną próbkę do adnotacji (losowo z remaining)
    
    Args:
        df: DataFrame z próbkami
        annotations: dict z już zaadnotowanymi próbkami
        baseline_uids: set z UIDs z baseline (do wykluczenia)
    """
    annotated_uids = set(annotations.keys())
    
    # Wykluczamy: już zaadnotowane w tej sesji + z baseline
    exclude_uids = annotated_uids
    if baseline_uids is not None:
        exclude_uids = annotated_uids | baseline_uids
    
    remaining = df[~df['uid'].isin(exclude_uids)]
    
    if len(remaining) == 0:
        return None
    
    # Losowa próbka z remaining
    sample = remaining.sample(n=1).iloc[0]
    return sample


# ============================================================================
# MAIN APP
# ============================================================================

# Initialize session state
if 'annotations' not in st.session_state:
    st.session_state.annotations = load_annotations()

if 'data' not in st.session_state:
    st.session_state.data = load_data()
    if st.session_state.data is None:
        st.stop()

if 'baseline_uids' not in st.session_state:
    st.session_state.baseline_uids = load_baseline_uids()

if 'current_sample' not in st.session_state:
    st.session_state.current_sample = None
    st.session_state.current_image = None
    st.session_state.url_checked = False

# Header
st.title("🎯 Active Learning - Adnotacja")

if st.session_state.mode == 'iteration':
    st.markdown(f"**Iteracja:** {st.session_state.iteration_num}")
else:
    st.markdown(f"**Plik:** `{st.session_state.data_path.name}`")

st.markdown("---")

# Statistics
stats = get_annotation_stats(st.session_state.annotations)
valid_annotations = stats['YES'] + stats['NO']
target = st.session_state.target_annotations

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("✅ YES", stats['YES'])
with col2:
    st.metric("❌ NO", stats['NO'])
with col3:
    st.metric("🚫 INVALID", stats['INVALID'])
with col4:
    st.metric("🎯 Postęp", f"{valid_annotations}/{target}")

# Progress bar
progress = min(valid_annotations / target, 1.0)
st.progress(progress)

# Check if target reached
if valid_annotations >= target:
    st.success(f"🎉 Gratulacje! Osiągnięto cel: {target} valid annotations!")
    st.balloons()
    
    # Zapisz finalne wyniki
    output_file = st.session_state.output_file
    
    if not output_file.exists():
        st.info("💾 Zapisywanie finalnych wyników...")
        
        # Wczytaj wszystkie adnotacje i merge z original data
        annotations_list = []
        for uid, ann in st.session_state.annotations.items():
            annotations_list.append({
                'uid': ann['uid'],
                'url': ann['url'],
                'label': ann['label']
            })
        
        df_annotated = pd.DataFrame(annotations_list)
        df_annotated.to_parquet(output_file, index=False)
        
        st.success(f"✅ Wyniki zapisane: {output_file}")
        st.info(f"📊 Total annotations: {len(df_annotated)} (YES: {stats['YES']}, NO: {stats['NO']}, INVALID: {stats['INVALID']})")
    
    st.stop()

st.markdown("---")

# Get next sample
if st.session_state.current_sample is None:
    sample = get_next_sample(
        st.session_state.data, 
        st.session_state.annotations,
        st.session_state.baseline_uids
    )
    
    if sample is None:
        st.warning("⚠️ Brak więcej próbek do adnotacji!")
        
        # Zapisz to co mamy
        output_file = st.session_state.partial_output_file
        annotations_list = []
        for uid, ann in st.session_state.annotations.items():
            annotations_list.append({
                'uid': ann['uid'],
                'url': ann['url'],
                'label': ann['label']
            })
        
        df_annotated = pd.DataFrame(annotations_list)
        df_annotated.to_parquet(output_file, index=False)
        
        st.info(f"💾 Zapisano częściowe wyniki: {output_file}")
        st.stop()
    
    st.session_state.current_sample = sample
    st.session_state.url_checked = False
    st.session_state.current_image = None

sample = st.session_state.current_sample

# Sample info
st.subheader("📋 Informacje o próbce")
info_col1, info_col2 = st.columns(2)

with info_col1:
    st.write(f"**UID:** `{sample['uid']}`")
    
    # Pred proba jeśli dostępne
    if 'pred_proba' in sample:
        st.write(f"**Model prediction:** `{sample['pred_proba']:.3f}`")
        
        # Hint na podstawie predykcji
        if sample['pred_proba'] > 0.7:
            st.info("💡 Model myśli że to **mapa** (high confidence)")
        elif sample['pred_proba'] < 0.3:
            st.info("💡 Model myśli że to **NIE mapa** (high confidence)")
        else:
            st.warning("💡 Model jest **niepewny** - case graniczny!")

with info_col2:
    total_samples = len(st.session_state.data)
    already_annotated = len(st.session_state.annotations)
    
    # Oblicz ile próbek z datasetu jest w baseline
    baseline_overlap = len(set(st.session_state.data['uid']) & st.session_state.baseline_uids)
    
    # Pozostałe = wszystkie - już zaadnotowane - z baseline (overlap)
    exclude_uids = set(st.session_state.annotations.keys()) | st.session_state.baseline_uids
    remaining = len(st.session_state.data[~st.session_state.data['uid'].isin(exclude_uids)])
    
    st.write(f"**Pozostałe próbki:** {remaining}/{total_samples}")
    
    if baseline_overlap > 0:
        st.write(f"**Wykluczone (baseline):** {baseline_overlap}")
    
    if stats['INVALID'] > 0:
        invalid_rate = stats['INVALID'] / (stats['INVALID'] + valid_annotations)
        st.write(f"**% INVALID:** {invalid_rate:.1%}")

# Check URL validity
if not st.session_state.url_checked:
    with st.spinner("🔍 Sprawdzanie URL..."):
        url_valid = check_url_valid(sample['url'])
        st.session_state.url_checked = True
        
        if not url_valid:
            st.warning("⚠️ URL nie odpowiada (status != 200) - automatycznie oznaczam jako INVALID")
            save_annotation(sample['uid'], sample['url'], 'INVALID')
            st.session_state.annotations[sample['uid']] = {
                'uid': sample['uid'],
                'url': sample['url'],
                'label': 'INVALID',
            }
            st.session_state.current_sample = None
            st.rerun()
        else:
            # Try to load image
            with st.spinner("📥 Ładowanie obrazka..."):
                img = load_image_from_url(sample['url'])
                
                if img is None:
                    st.warning("⚠️ Nie udało się wczytać obrazka - automatycznie oznaczam jako INVALID")
                    save_annotation(sample['uid'], sample['url'], 'INVALID')
                    st.session_state.annotations[sample['uid']] = {
                        'uid': sample['uid'],
                        'url': sample['url'],
                        'label': 'INVALID',
                    }
                    st.session_state.current_sample = None
                    st.rerun()
                else:
                    st.session_state.current_image = img

# Display image and annotation buttons
if st.session_state.current_image is not None:
    st.image(st.session_state.current_image, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🤔 Czy to mapa statystyczna?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ YES (Y)", key="yes_btn", use_container_width=True, type="primary"):
            save_annotation(sample['uid'], sample['url'], 'YES')
            st.session_state.annotations[sample['uid']] = {
                'uid': sample['uid'],
                'url': sample['url'],
                'label': 'YES',
            }
            st.session_state.current_sample = None
            st.rerun()
    
    with col2:
        if st.button("❌ NO (N)", key="no_btn", use_container_width=True):
            save_annotation(sample['uid'], sample['url'], 'NO')
            st.session_state.annotations[sample['uid']] = {
                'uid': sample['uid'],
                'url': sample['url'],
                'label': 'NO',
            }
            st.session_state.current_sample = None
            st.rerun()
    
    with col3:
        if st.button("🚫 INVALID (I)", key="invalid_btn", use_container_width=True):
            save_annotation(sample['uid'], sample['url'], 'INVALID')
            st.session_state.annotations[sample['uid']] = {
                'uid': sample['uid'],
                'url': sample['url'],
                'label': 'INVALID',
            }
            st.session_state.current_sample = None
            st.rerun()
    
    st.markdown("---")
    st.caption("⌨️ Skróty klawiszowe: Y = YES | N = NO | I = INVALID")

# Sidebar
with st.sidebar:
    st.header("📊 Statystyki")
    
    if st.session_state.mode == 'iteration':
        st.write(f"**Iteracja:** {st.session_state.iteration_num}")
        st.write(f"**Katalog:** `{st.session_state.iteration_dir.name}`")
    else:
        st.write(f"**Mode:** Custom file")
        st.write(f"**Plik:** `{st.session_state.data_path.name}`")
    
    st.write(f"**Target:** {target} valid annotations")
    st.write(f"**Pozostało:** {target - valid_annotations}")
    
    total_annotations = len(st.session_state.annotations)
    st.write(f"**Łącznie oznaczono:** {total_annotations}")
    
    baseline_count = len(st.session_state.baseline_uids)
    if baseline_count > 0:
        st.write(f"**Wykluczone (baseline):** {baseline_count}")
    
    if total_annotations > 0:
        st.write(f"**% INVALID:** {stats['INVALID'] / total_annotations * 100:.1f}%")
        
        if stats['INVALID'] + valid_annotations > 0:
            st.write(f"**% YES (z valid):** {stats['YES'] / (stats['YES'] + stats['NO']) * 100:.1f}%")
    
    st.markdown("---")
    st.write(f"**Plik zapisu:**")
    st.code(f"{st.session_state.annotations_file.name}", language="text")
    
    st.markdown("---")
    
    if st.button("🔄 Reset session", help="Nie usuwa pliku, tylko resetuje interfejs"):
        for key in list(st.session_state.keys()):
            if key != 'args_loaded':
                del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    st.caption("💡 **Tip:** Interface automatycznie pomija invalid URLs, więc nie liczą się do targetu!")
