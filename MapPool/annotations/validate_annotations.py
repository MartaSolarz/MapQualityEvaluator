#!/usr/bin/env python3
"""
Skrypt do walidacji adnotacji z pliku annotations_to_check.parquet.
Wyświetla obrazy w przeglądarce po 25 obok siebie z podpisami uid.
Dla trybu INVALID sprawdza czy URLe faktycznie nie odpowiadają.
"""

import pandas as pd
import argparse
import requests
from pathlib import Path
import tempfile
import webbrowser
from typing import List, Tuple
from tqdm import tqdm
import time


def check_url_responds(url: str, timeout: int = 10) -> bool:
    """
    Sprawdza czy URL odpowiada i zwraca obraz.
    
    Args:
        url: URL do sprawdzenia
        timeout: Timeout w sekundach
        
    Returns:
        True jeśli URL odpowiada i zwraca obraz, False w przeciwnym razie
    """
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            # Sprawdź czy to obraz
            content_type = response.headers.get('Content-Type', '')
            if 'image' in content_type:
                return True
        
        # Jeśli HEAD nie działa, spróbuj GET z małym zakresem
        response = requests.get(url, timeout=timeout, stream=True, allow_redirects=True)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            if 'image' in content_type:
                return True
                
        return False
    except Exception as e:
        return False


def validate_invalid_urls(df: pd.DataFrame) -> List[str]:
    """
    Sprawdza URLe oznaczone jako INVALID czy faktycznie nie odpowiadają.
    
    Args:
        df: DataFrame z kolumnami uid, url, label
        
    Returns:
        Lista uid, które są oznaczone jako INVALID ale URLe odpowiadają
    """
    invalid_df = df[df['label'] == 'INVALID'].copy()
    responding_uids = []
    
    print(f"\nSprawdzanie {len(invalid_df)} URLi oznaczonych jako INVALID...")
    
    for idx, row in tqdm(invalid_df.iterrows(), total=len(invalid_df)):
        uid = row['uid']
        url = row['url']
        
        if check_url_responds(url):
            responding_uids.append(uid)
            print(f"\n⚠️  URL odpowiada mimo oznaczenia INVALID: {uid}")
            print(f"   URL: {url}")
        
        # Małe opóźnienie żeby nie przeciążać serwerów
        time.sleep(0.1)
    
    return responding_uids


def generate_html_gallery(df: pd.DataFrame, label: str) -> str:
    """
    Generuje HTML z galerią obrazów.
    
    Args:
        df: DataFrame z kolumnami uid, url, label
        label: Label do wyświetlenia (YES/NO/INVALID)
        
    Returns:
        Ścieżka do wygenerowanego pliku HTML
    """
    filtered_df = df[df['label'] == label].copy()
    
    html_content = f"""
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Walidacja adnotacji - {label}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            overflow-x: hidden;
            max-width: 100vw;
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin: 0 0 10px 0;
        }}
        .stats {{
            text-align: center;
            margin-bottom: 20px;
            font-size: 18px;
            color: #666;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin: 0 auto;
            max-width: 100%;
            width: 100%;
        }}
        .item {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
            min-width: 0;
            overflow: hidden;
        }}
        .item:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .item img {{
            width: 100%;
            height: 200px;
            object-fit: contain;
            background: #f9f9f9;
            border-radius: 4px;
            cursor: pointer;
        }}
        .item .uid {{
            text-align: center;
            margin-top: 8px;
            font-size: 11px;
            color: #666;
            word-break: break-all;
            overflow-wrap: break-word;
        }}
        .item .url {{
            text-align: center;
            margin-top: 4px;
            font-size: 9px;
            color: #999;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 100%;
        }}
        .error-img {{
            background: #ffebee !important;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #c62828;
            font-size: 14px;
        }}
        
        /* Modal/Lightbox styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
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
        }}
        .modal-close:hover,
        .modal-close:focus {{
            color: #bbb;
        }}
        .modal-info {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 15px 30px;
            border-radius: 8px;
            text-align: center;
            max-width: 80%;
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
    </style>
    <script>
        function handleImageError(img) {{
            img.style.display = 'none';
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-img';
            errorDiv.style.width = '100%';
            errorDiv.style.height = '200px';
            errorDiv.textContent = '❌ Nie można załadować obrazu';
            img.parentNode.insertBefore(errorDiv, img);
        }}
        
        function openModal(imgSrc, uid, url) {{
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            const modalUid = document.getElementById('modalUid');
            const modalUrl = document.getElementById('modalUrl');
            
            modal.classList.add('active');
            modalImg.src = imgSrc;
            modalUid.textContent = uid;
            modalUrl.textContent = url;
        }}
        
        function closeModal() {{
            const modal = document.getElementById('imageModal');
            modal.classList.remove('active');
        }}
        
        // Zamknij modal po kliknięciu ESC
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeModal();
            }}
        }});
        
        // Zamknij modal po kliknięciu poza obrazem
        document.addEventListener('click', function(event) {{
            const modal = document.getElementById('imageModal');
            if (event.target === modal) {{
                closeModal();
            }}
        }});
    </script>
</head>
<body>
    <!-- Modal dla powiększonych obrazów -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage" onclick="event.stopPropagation()">
        <div class="modal-info" onclick="event.stopPropagation()">
            <div class="uid" id="modalUid"></div>
            <div class="url" id="modalUrl"></div>
        </div>
    </div>
    
    <h1>Walidacja adnotacji - Label: {label}</h1>
    <div class="stats">
        Łącznie: {len(filtered_df)} obrazów
    </div>
    <div class="gallery">
"""
    
    # Wyświetl wszystkie obrazy w jednej galerii (bez podziału na strony)
    for _, row in filtered_df.iterrows():
        uid = row['uid']
        url = row['url']
        
        html_content += f"""
        <div class="item">
            <img src="{url}" alt="{uid}" onerror="handleImageError(this)" loading="lazy" onclick="openModal('{url}', '{uid}', '{url}')">
            <div class="uid" title="{uid}">{uid[:16]}...</div>
            <div class="url" title="{url}">{url}</div>
        </div>
"""
    
    html_content += '    </div>\n'
    
    html_content += """
</body>
</html>
"""
    
    # Zapisz do pliku tymczasowego
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8')
    temp_file.write(html_content)
    temp_file.close()
    
    return temp_file.name


def main():
    parser = argparse.ArgumentParser(
        description='Walidacja adnotacji z pliku annotations_to_check.parquet'
    )
    parser.add_argument(
        'label',
        choices=['YES', 'NO', 'INVALID'],
        help='Label do wyświetlenia (YES/NO/INVALID)'
    )
    parser.add_argument(
        '--parquet-path',
        default='../data_all/above_value_minus1/annotations_to_check.parquet',
        help='Ścieżka do pliku parquet (domyślnie: ../data_all/above_value_minus1/annotations_to_check.parquet)'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Nie otwieraj przeglądarki automatycznie'
    )
    
    args = parser.parse_args()
    
    # Wczytaj dane
    parquet_path = Path(__file__).parent / args.parquet_path
    print(f"Wczytywanie danych z: {parquet_path}")
    
    if not parquet_path.exists():
        print(f"❌ Plik nie istnieje: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    print(f"Wczytano {len(df)} wierszy")
    print(f"\nRozkład labelów:")
    print(df['label'].value_counts())
    
    # Dla INVALID - sprawdź czy URLe faktycznie nie odpowiadają
    if args.label == 'INVALID':
        print("\n" + "="*60)
        print("TRYB INVALID - Sprawdzanie czy URLe faktycznie nie odpowiadają")
        print("="*60)
        
        responding_uids = validate_invalid_urls(df)
        
        if responding_uids:
            print(f"\n⚠️  UWAGA! Znaleziono {len(responding_uids)} URLi, które odpowiadają mimo oznaczenia INVALID:")
            for uid in responding_uids:
                print(f"  - {uid}")
            
            # Zapisz do pliku
            output_file = Path(__file__).parent / 'invalid_but_responding.txt'
            with open(output_file, 'w') as f:
                f.write('\n'.join(responding_uids))
            print(f"\n✅ Lista zapisana do: {output_file}")
        else:
            print("\n✅ Wszystkie URLe oznaczone jako INVALID faktycznie nie odpowiadają")
    
    # Generuj HTML
    print(f"\nGenerowanie galerii dla label: {args.label}")
    html_path = generate_html_gallery(df, args.label)
    print(f"✅ Wygenerowano HTML: {html_path}")
    
    # Otwórz w przeglądarce
    if not args.no_browser:
        print("Otwieranie w przeglądarce...")
        webbrowser.open(f'file://{html_path}')
    else:
        print(f"\nAby otworzyć w przeglądarce, użyj:")
        print(f"open {html_path}")


if __name__ == '__main__':
    main()
