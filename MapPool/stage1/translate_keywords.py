# import json
# import time
# from googletrans import Translator
# from tqdm import tqdm
#
# positive_keywords = {
#     # === (5 pkt) ===
#     'choropleth': 5,
#     'cartogram': 5,
#     'proportional symbol': 5,
#     'graduated symbol': 5,
#     'graduated circles': 5,
#     'statistical map': 5,
# }
#
# LANGUAGES = ['en', 'fr', 'de', 'ko', 'ru', 'es', 'pl']
# DO_NOT_TRANSLATE = {'GDP', 'R&D', 'NUTS'}
# OUTPUT_FILE = "positive_keywords.json"
# SLEEP_BETWEEN = 0.4
#
# translator = Translator()
#
# def translate_phrase(phrase, lang):
#     if phrase in DO_NOT_TRANSLATE:
#         return phrase
#     if lang == "en":
#         return phrase
#
#     try:
#         return translator.translate(phrase, dest=lang).text
#     except Exception as e:
#         print(f"❌ Błąd tłumaczenia: {phrase} -> {lang}: {e}")
#         return None
#
#
# def main():
#     result = {}
#
#     for word, score in tqdm(positive_keywords.items()):
#         translations = {}
#
#         for lang in LANGUAGES:
#             translated = translate_phrase(word, lang)
#             translations[lang] = [translated] if translated else []
#
#             time.sleep(SLEEP_BETWEEN)
#
#         result[word] = {
#             "score": score,
#             "translations": translations
#         }
#
#     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#         json.dump(result, f, ensure_ascii=False, indent=2)
#
#     print(f"\n✅ Zapisano: {OUTPUT_FILE}")
#     print(f"📦 Liczba haseł: {len(result)}")
#     print(f"🌍 Liczba języków: {len(LANGUAGES)}")
#
#
# if __name__ == "__main__":
#     main()

import json
import time
from googletrans import Translator
from tqdm import tqdm
import os

# ====== TWÓJ SŁOWNIK ======
positive_keywords = {
    'choropleth': 5,
    'cartogram': 5,
    'proportional symbol': 5,
    'graduated symbol': 5,
    'graduated circles': 5,
    'statistical map': 5,
    'per capita': 4,
    'share of': 4,
    # ... reszta słownika
}

# ====== KONFIG ======
LANGUAGES = ['en', 'fr', 'de', 'ko', 'ru', 'es', 'pl']
DO_NOT_TRANSLATE = {'GDP', 'R&D', 'NUTS'}
OUTPUT_FILE = "positive_keywords.json"
CACHE_FILE = "translation_cache.json"
BATCH_SIZE = 10
SLEEP_BETWEEN_BATCH = 0.3  # sekundy

translator = Translator()

# ====== Wczytaj cache jeśli istnieje ======
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        CACHE = json.load(f)
else:
    CACHE = {}

def translate_batch(phrases, lang):
    """Batch translation dla listy fraz."""
    results = []
    try:
        translations = translator.translate(phrases, dest=lang)
        # translator.translate zwraca listę obiektów Translation
        for t in translations:
            results.append(t.text)
    except Exception as e:
        print(f"❌ Błąd batchowego tłumaczenia {phrases} -> {lang}: {e}")
        results = [None] * len(phrases)
    return results

def main():
    result = {}

    for word, score in tqdm(positive_keywords.items(), desc="Hasła"):
        translations = {}

        for lang in LANGUAGES:
            key = f"{word}||{lang}"
            if key in CACHE:
                translations[lang] = [CACHE[key]]
            else:
                if word in DO_NOT_TRANSLATE or lang == "en":
                    translated = word
                else:
                    translated = translate_batch([word], lang)[0]
                    time.sleep(SLEEP_BETWEEN_BATCH)
                translations[lang] = [translated]
                CACHE[key] = translated

        result[word] = {
            "score": score,
            "translations": translations
        }

    # Zapis cache (po zakończeniu)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(CACHE, f, ensure_ascii=False, indent=2)

    # Zapis wynikowego słownika
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Zapisano: {OUTPUT_FILE}")
    print(f"📦 Liczba haseł: {len(result)}")
    print(f"🌍 Liczba języków: {len(LANGUAGES)}")

if __name__ == "__main__":
    main()

