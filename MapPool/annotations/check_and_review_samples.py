#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generuje prosty HTML do review próbek z możliwością zmiany etykiety

Użycie:
python3 generate_review_html.py --file data/train.parquet --review-label NO --change-to YES
python3 generate_review_html.py --file data/train.parquet --review-label YES --change-to NO
python3 generate_review_html.py --file data/train.parquet  # Pokazuje WSZYSTKIE próbki

Output:
- Otwiera HTML w przeglądarce
- Klik na obrazek = zaznaczenie (kolorowy obrys)
- Przycisk "Export Changes" = pobiera CSV z zaznaczonymi UIDs
- Uruchom apply_definition_changes.py z tym CSV
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import webbrowser
import tempfile
import json


def load_data_file(file_path):
    """
    Wczytaj dane z pliku parquet lub JSONL

    Returns:
        pd.DataFrame z kolumnami: uid, url, label (i opcjonalnie score)
    """
    file_path = Path(file_path)

    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
        return df

    elif file_path.suffix == '.jsonl':
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not records:
            raise ValueError("❌ Plik JSONL jest pusty")

        df = pd.DataFrame(records)

        # Walidacja kolumn
        required_cols = ['uid']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Brak wymaganych kolumn: {missing_cols}")

        if 'url' not in df.columns:
            df['url'] = None

        if 'label' not in df.columns:
            df['label'] = None

        return df

    else:
        raise ValueError(f"❌ Nieobsługiwany format: {file_path.suffix}")


def generate_html(df_samples, output_path, review_label=None, change_to='YES'):
    """
    Generuj HTML z grid view i clickable selection

    Args:
        df_samples: DataFrame z próbkami do przeglądu
        output_path: Ścieżka do pliku wyjściowego HTML
        review_label: Etykieta próbek które przeglądamy (NO, YES, INVALID, None=wszystkie)
        change_to: Na jaką etykietę zmieniamy (YES, NO, INVALID)
    """

    # Tytuł w zależności od review_label
    if review_label is None:
        title_text = f"Review ALL Samples - Change to {change_to}"
        instruction_text = f"Przeglądasz <strong>WSZYSTKIE próbki</strong>. Kliknij na <strong>tło karty</strong> żeby zaznaczyć do zmiany na <strong>{change_to}</strong>."
        stats_prefix = "Total"
    else:
        title_text = f"Review {review_label} Samples - Change to {change_to}"
        instruction_text = f"Przeglądasz próbki z etykietą <strong>{review_label}</strong>. Kliknij na <strong>tło karty</strong> żeby zaznaczyć do zmiany na <strong>{change_to}</strong>."
        stats_prefix = f"Total {review_label}"

    # Kolory w zależności od change_to
    if change_to == 'YES':
        select_color = '#4CAF50'  # Zielony
        select_bg = '#e8f5e9'
        select_text = '✓ YES'
    elif change_to == 'NO':
        select_color = '#f44336'  # Czerwony
        select_bg = '#ffebee'
        select_text = '✗ NO'
    else:  # INVALID
        select_color = '#FF9800'  # Pomarańczowy
        select_bg = '#fff3e0'
        select_text = '⚠ INVALID'

    html = f"""
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title_text}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .instructions {{
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background: #e3f2fd;
            border-radius: 8px;
            font-size: 16px;
        }}
        .stats {{
            text-align: center;
            margin: 20px 0;
            font-size: 18px;
            color: #666;
        }}
        .controls {{
            text-align: center;
            margin: 20px 0;
        }}
        .btn {{
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            margin: 0 10px;
            transition: all 0.3s;
        }}
        .btn-primary {{
            background: {select_color};
            color: white;
        }}
        .btn-primary:hover {{
            background: {select_color}dd;
        }}
        .btn-secondary {{
            background: #2196F3;
            color: white;
        }}
        .btn-secondary:hover {{
            background: #0b7dda;
        }}
        .btn-danger {{
            background: #9E9E9E;
            color: white;
        }}
        .btn-danger:hover {{
            background: #757575;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px auto;
            max-width: 100%;
        }}
        .item {{
            background: white;
            border: 3px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            cursor: pointer;
            transition: all 0.2s;
            position: relative;
        }}
        .item:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        .item.selected {{
            border-color: {select_color};
            background: {select_bg};
            box-shadow: 0 0 15px {select_color}80;
        }}
        .item.selected::after {{
            content: "{select_text}";
            position: absolute;
            top: 10px;
            right: 10px;
            background: {select_color};
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
        }}
        .item img {{
            width: 100%;
            height: 200px;
            object-fit: contain;
            background: #f9f9f9;
            border-radius: 4px;
        }}
        .item .uid {{
            text-align: center;
            margin-top: 8px;
            font-size: 11px;
            color: #666;
            word-break: break-all;
        }}
        .item .label {{
            text-align: center;
            margin-top: 4px;
            font-size: 12px;
            font-weight: bold;
            padding: 3px 8px;
            border-radius: 4px;
            display: inline-block;
        }}
        .label-yes {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        .label-no {{
            background: #ffebee;
            color: #c62828;
        }}
        .label-invalid {{
            background: #fff3e0;
            color: #e65100;
        }}
        .label-none {{
            background: #f5f5f5;
            color: #757575;
        }}
        .item .score {{
            text-align: center;
            margin-top: 4px;
            font-size: 10px;
            color: #999;
        }}
        .item .proba {{
            text-align: center;
            margin-top: 4px;
            font-size: 11px;
            font-weight: bold;
            color: #1976D2;
        }}
        .error-img {{
            background: #ffebee !important;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #c62828;
            font-size: 14px;
        }}

        /* Modal/Lightbox */
        .modal {{
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.95);
            overflow: auto;
        }}
        .modal.active {{
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .modal-content {{
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
            animation: zoom 0.3s;
        }}
        @keyframes zoom {{
            from {{transform: scale(0.5)}}
            to {{transform: scale(1)}}
        }}
        .modal-close {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: #fff;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
            z-index: 2001;
        }}
        .modal-close:hover {{
            color: #bbb;
        }}
        .modal-info {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px 30px;
            border-radius: 8px;
            text-align: center;
            max-width: 80%;
            z-index: 2001;
        }}
        .modal-info .uid {{
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .modal-info .url {{
            font-size: 12px;
            color: #ccc;
            word-break: break-all;
        }}
        .modal-actions {{
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 2001;
        }}
        .modal-btn {{
            background: {select_color};
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 6px;
            cursor: pointer;
            margin: 5px;
        }}
        .modal-btn:hover {{
            background: {select_color}dd;
        }}
    </style>
</head>
<body>
    <!-- Modal dla powiększonych obrazów -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <div class="modal-actions" onclick="event.stopPropagation()">
            <button class="modal-btn" onclick="toggleModalSelection()">Toggle → {change_to}</button>
        </div>
        <img class="modal-content" id="modalImage" onclick="event.stopPropagation()">
        <div class="modal-info" onclick="event.stopPropagation()">
            <div class="uid" id="modalUid"></div>
            <div class="url" id="modalUrl"></div>
        </div>
    </div>

    <h1>🔍 {title_text}</h1>

    <div class="instructions">
        <strong>Instrukcja:</strong> {instruction_text}
        Kliknij na <strong>obrazek</strong> żeby powiększyć. W modalu: SPACE/ENTER = toggle, ESC = zamknij.
    </div>

    <div class="stats">
        <strong>{stats_prefix}:</strong> <span id="total">{len(df_samples)}</span> | 
        <strong>Zaznaczone:</strong> <span id="selected-count">0</span> |
        <strong>%:</strong> <span id="percentage">0.0%</span>
    </div>

    <div class="controls">
        <button class="btn btn-primary" onclick="exportChanges()">💾 Export Changes to {change_to}</button>
        <button class="btn btn-secondary" onclick="selectAll()">✅ Select All</button>
        <button class="btn btn-danger" onclick="clearAll()">❌ Clear All</button>
    </div>

    <div class="gallery">
"""

    # Generate items
    for _, row in df_samples.iterrows():
        uid = row['uid']
        url = row.get('url', '')
        label = row.get('label', None)

        # Handle missing URLs
        if pd.isna(url) or url == '' or url is None:
            url = 'placeholder'
            img_html = '<div class="error-img">Brak URL</div>'
        else:
            img_html = f'<img src="{url}" alt="{uid}" onerror="handleImageError(this)" loading="lazy" onclick="event.stopPropagation(); openModal(\'{url}\', \'{uid}\', this.closest(\'.item\'))">'

        # Label badge
        if pd.isna(label) or label is None or label == '':
            label_class = 'label-none'
            label_text = 'NO LABEL'
        elif label == 'YES' or label == 1:
            label_class = 'label-yes'
            label_text = 'YES'
        elif label == 'NO' or label == 0:
            label_class = 'label-no'
            label_text = 'NO'
        elif label == 'INVALID' or label == -1:
            label_class = 'label-invalid'
            label_text = 'INVALID'
        else:
            label_class = 'label-none'
            label_text = str(label)

        label_html = f'<div class="label {label_class}">{label_text}</div>'

        # Score
        score = row.get('score', '')
        score_text = f'<div class="score">Score: {score:.3f}</div>' if score != '' and not pd.isna(score) else ''

        # Pred proba (if available)
        pred_proba = row.get('pred_proba', '')
        proba_text = f'<div class="proba">P(YES): {pred_proba:.3f}</div>' if pred_proba != '' and not pd.isna(
            pred_proba) else ''

        html += f"""
        <div class="item" data-uid="{uid}" data-url="{url}" onclick="toggleSelection(this)">
            {img_html}
            <div class="uid">{uid[:20]}...</div>
            {label_html}
            {score_text}
            {proba_text}
        </div>
"""

    review_label_str = review_label if review_label else 'ALL'

    html += f"""
    </div>

    <div class="controls">
        <button class="btn btn-primary" onclick="exportChanges()">💾 Export Changes (CSV)</button>
    </div>

    <script>
        let selectedUIDs = new Set();
        let currentModalElement = null;
        const CHANGE_TO_LABEL = '{change_to}';
        const REVIEW_LABEL = '{review_label_str}';

        function openModal(imgSrc, uid, itemElement) {{
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            const modalUid = document.getElementById('modalUid');
            const modalUrl = document.getElementById('modalUrl');

            currentModalElement = itemElement;

            modal.classList.add('active');
            modalImg.src = imgSrc;
            modalUid.textContent = uid;
            modalUrl.textContent = itemElement.getAttribute('data-url');
        }}

        function closeModal() {{
            const modal = document.getElementById('imageModal');
            modal.classList.remove('active');
            currentModalElement = null;
        }}

        function toggleModalSelection() {{
            if (currentModalElement) {{
                toggleSelection(currentModalElement);
            }}
        }}

        function toggleSelection(element) {{
            const uid = element.getAttribute('data-uid');

            if (selectedUIDs.has(uid)) {{
                selectedUIDs.delete(uid);
                element.classList.remove('selected');
            }} else {{
                selectedUIDs.add(uid);
                element.classList.add('selected');
            }}

            updateStats();
        }}

        function selectAll() {{
            const items = document.querySelectorAll('.item');
            items.forEach(item => {{
                const uid = item.getAttribute('data-uid');
                selectedUIDs.add(uid);
                item.classList.add('selected');
            }});
            updateStats();
        }}

        function clearAll() {{
            const items = document.querySelectorAll('.item');
            items.forEach(item => {{
                item.classList.remove('selected');
            }});
            selectedUIDs.clear();
            updateStats();
        }}

        function updateStats() {{
            const total = document.querySelectorAll('.item').length;
            const selected = selectedUIDs.size;
            const percentage = total > 0 ? (selected / total * 100).toFixed(1) : 0;

            document.getElementById('selected-count').textContent = selected;
            document.getElementById('percentage').textContent = percentage + '%';
        }}

        function exportChanges() {{
            if (selectedUIDs.size === 0) {{
                alert('⚠️ Nie zaznaczono żadnych próbek!');
                return;
            }}

            // Generate CSV
            let csv = 'uid,new_label,timestamp\\n';
            const timestamp = new Date().toISOString();

            selectedUIDs.forEach(uid => {{
                csv += `${{uid}},${{CHANGE_TO_LABEL}},${{timestamp}}\\n`;
            }});

            // Download
            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `definition_changes_${{REVIEW_LABEL}}_to_${{CHANGE_TO_LABEL}}.csv`;
            a.click();
            URL.revokeObjectURL(url);

            alert(`✅ Exported ${{selectedUIDs.size}} changes (${{REVIEW_LABEL}} → ${{CHANGE_TO_LABEL}})!\\n\\nNastępne kroki:\\n1. Skopiuj CSV do odpowiedniego folderu\\n2. Uruchom: python3 apply_definition_changes.py --changes <plik.csv>`);
        }}

        function handleImageError(img) {{
            img.style.display = 'none';
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-img';
            errorDiv.style.width = '100%';
            errorDiv.style.height = '200px';
            errorDiv.textContent = '❌ Nie można załadować';
            img.parentNode.insertBefore(errorDiv, img);
        }}

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            // Modal shortcuts
            if (document.getElementById('imageModal').classList.contains('active')) {{
                if (e.key === 'Escape') {{
                    closeModal();
                }}
                if (e.key === ' ' || e.key === 'Enter') {{
                    e.preventDefault();
                    toggleModalSelection();
                }}
                return;
            }}

            // Global shortcuts
            if (e.key === 'a' && e.ctrlKey) {{
                e.preventDefault();
                selectAll();
            }}
            if (e.key === 'c' && e.ctrlKey && e.shiftKey) {{
                e.preventDefault();
                clearAll();
            }}
            if (e.key === 's' && e.ctrlKey) {{
                e.preventDefault();
                exportChanges();
            }}
        }});

        // Close modal on click outside
        document.getElementById('imageModal').addEventListener('click', function(e) {{
            if (e.target === this) {{
                closeModal();
            }}
        }});
    </script>
</body>
</html>
"""

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate review HTML for samples with label change',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  # Przegląd WSZYSTKICH próbek → zmiana na YES:
  python3 generate_review_html.py --file data/pool_with_predictions.parquet --change-to YES

  # Przegląd NO → zmiana na YES:
  python3 generate_review_html.py --file data/train.parquet --review-label NO --change-to YES

  # Przegląd YES → zmiana na NO:
  python3 generate_review_html.py --file data/train.parquet --review-label YES --change-to NO

  # Przegląd z JSONL + uzupełnienie URLs:
  python3 generate_review_html.py --file annotations.jsonl --parquet-source data/full.parquet
        """
    )
    parser.add_argument('--file', type=str, required=True,
                        help='Plik z adnotacjami (.parquet lub .jsonl)')
    parser.add_argument('--review-label', type=str, default=None,
                        choices=['NO', 'YES', 'INVALID', None],
                        help='Która etykieta do przeglądu (default: None = wszystkie)')
    parser.add_argument('--change-to', type=str, default='YES',
                        choices=['YES', 'NO', 'INVALID'],
                        help='Na jaką etykietę zmieniamy (default: YES)')
    parser.add_argument('--output', type=str,
                        help='Output HTML file (default: temp file)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Nie otwieraj przeglądarki automatycznie')
    parser.add_argument('--parquet-source', type=str,
                        help='Plik parquet ze źródłowymi danymi (do uzupełnienia URLs dla JSONL)')
    parser.add_argument('--limit', type=int,
                        help='Ogranicz liczbę wyświetlanych próbek (np. 100)')

    args = parser.parse_args()

    # Load data
    input_path = Path(args.file)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    if not input_path.exists():
        print(f"❌ Plik nie istnieje: {input_path}")
        return

    print(f"📂 Wczytywanie: {input_path}")
    if args.review_label:
        print(f"   Review: {args.review_label} → Change to: {args.change_to}")
    else:
        print(f"   Review: ALL → Change to: {args.change_to}")

    try:
        df = load_data_file(input_path)
    except Exception as e:
        print(f"❌ Błąd wczytywania: {e}")
        return

    # Jeśli to JSONL i brak URLs, spróbuj uzupełnić z parquet
    if input_path.suffix == '.jsonl':
        if 'url' not in df.columns or df['url'].isna().all():
            if args.parquet_source:
                print(f"   📦 Uzupełnianie URLs z: {args.parquet_source}")
                try:
                    df_source = pd.read_parquet(args.parquet_source)
                    if 'uid' in df_source.columns and 'url' in df_source.columns:
                        if 'url' in df.columns:
                            df = df.drop('url', axis=1)
                        df = df.merge(df_source[['uid', 'url']], on='uid', how='left')
                        print(f"   ✅ URLs uzupełnione: {df['url'].notna().sum()} / {len(df)}")
                except Exception as e:
                    print(f"   ⚠️  Nie udało się uzupełnić URLs: {e}")

    # Filter by review_label (if specified)
    if args.review_label is not None:
        if 'label' not in df.columns:
            print(f"❌ Brak kolumny 'label' w pliku!")
            return

        if df['label'].dtype == 'object':
            df_filtered = df[df['label'] == args.review_label].copy()
        else:
            # Mapowanie dla integer labels
            label_map = {'NO': 0, 'YES': 1, 'INVALID': -1}
            if args.review_label in label_map:
                df_filtered = df[df['label'] == label_map[args.review_label]].copy()
            else:
                print(f"❌ Nieznana etykieta: {args.review_label}")
                return
    else:
        # Show all samples
        df_filtered = df.copy()

    print(f"\n📊 Statystyki:")
    print(f"   Total próbek w pliku: {len(df)}")

    if args.review_label is not None:
        print(f"   Próbek {args.review_label}: {len(df_filtered)}")

        if len(df_filtered) == 0:
            print(f"\n❌ Brak próbek z etykietą {args.review_label}!")
            if 'label' in df.columns:
                print(f"   Dostępne etykiety: {df['label'].unique()}")
            return
    else:
        print(f"   Showing ALL samples: {len(df_filtered)}")
        if 'label' in df.columns:
            print(f"   Label distribution: {df['label'].value_counts().to_dict()}")

    # Sort by pred_proba if available (high to low), else by score
    if 'pred_proba' in df_filtered.columns:
        df_filtered = df_filtered.sort_values('pred_proba', ascending=False)
        print(f"   ✅ Posortowano po pred_proba (high to low)")
    elif 'score' in df_filtered.columns:
        df_filtered = df_filtered.sort_values('score', ascending=False)
        print(f"   ✅ Posortowano po score (high to low)")

    # Apply limit if specified
    if args.limit and args.limit < len(df_filtered):
        print(f"   ⚠️  Ograniczono do {args.limit} próbek (z {len(df_filtered)})")
        df_filtered = df_filtered.head(args.limit)

    # Generate HTML
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(tempfile.mktemp(suffix='.html'))

    print(f"\n🎨 Generowanie HTML...")
    generate_html(df_filtered, output_path, review_label=args.review_label, change_to=args.change_to)

    print(f"✅ Wygenerowano: {output_path}")
    print(f"   Próbek do przeglądu: {len(df_filtered)}")

    # Open in browser
    if not args.no_browser:
        print("\n🌐 Otwieranie w przeglądarce...")
        webbrowser.open(f'file://{output_path.absolute()}')
    else:
        print(f"\n💡 Otwórz w przeglądarce:")
        print(f"   open {output_path}")

    print("\n" + "=" * 80)
    print("📋 INSTRUKCJA:")
    print("=" * 80)
    if args.review_label:
        print(f"1. Przeglądasz próbki z etykietą: {args.review_label}")
    else:
        print(f"1. Przeglądasz WSZYSTKIE próbki")
    print(f"2. Kliknij obrazki które chcesz zmienić na {args.change_to}")
    print(f"3. Kliknij 'Export Changes' → pobierze CSV")
    print(f"4. Użyj CSV do aktualizacji adnotacji")
    print("=" * 80)


if __name__ == '__main__':
    main()