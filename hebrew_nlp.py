"""
Hebrew NLP helpers.

Provides optional DictaBERT-based lemmatization and heuristic Hebrew lemmatization.
"""

import re

from utils.utils import get_logger

logger = get_logger(__name__)

from transformers import AutoModel, AutoTokenizer



DICTABERT_LEX_MODEL = "dicta-il/dictabert-lex"
_LEMMATIZER = None
_HEURISTIC_LEMMATIZER = None


class HeuristicHebrewLemmatizer:
    """Lightweight heuristic normalizer/lemmatizer for Hebrew matching."""

    def normalize(self, text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[\u0591-\u05C7]", "", text)
        text = (
            text.replace("ך", "כ")
            .replace("ם", "מ")
            .replace("ן", "נ")
            .replace("ף", "פ")
            .replace("ץ", "צ")
        )
        return text

    def lemmaish(self, word: str) -> str:
        w = self.normalize(word)
        w = re.sub(r"[^a-z0-9\u05d0-\u05ea]", "", w)
        if len(w) <= 2:
            return w

        prefixes = set("והבלכמש")
        while len(w) > 3 and w[0] in prefixes:
            w = w[1:]

        suffixes = [
            "יהם", "היו", "ות", "ים", "נו", "כם", "כן", "יה", "יו", "תי", "ת", "ה", "ו", "י", "ך",
        ]
        for sfx in suffixes:
            if len(w) - len(sfx) >= 3 and w.endswith(sfx):
                w = w[:-len(sfx)]
                break
        return w

    def keyword_forms(self, keyword: str) -> set[str]:
        k = self.lemmaish(keyword)
        if not k:
            return set()
        forms = {k}
        for p in ["", "ו", "ה", "ב", "ל", "כ", "מ", "ש"]:
            forms.add(f"{p}{k}")
        return {f for f in forms if len(f) >= 2}

    def line_tokens(self, line_text: str) -> list[str]:
        norm = self.normalize(line_text)
        words = re.findall(r"[a-z0-9\u05d0-\u05ea]+", norm)
        tokens = []
        for w in words:
            lw = self.lemmaish(w)
            if lw:
                tokens.append(lw)
        return tokens

    def normalize_for_surface_search(self, text: str) -> str:
        text = self.normalize(text)
        return re.sub(r"[^a-z0-9\u05d0-\u05ea\s]+", " ", text)

    def line_matches_keyword(self, line_text: str, keyword: str) -> bool:
        k_forms = self.keyword_forms(keyword)
        if not k_forms:
            return False
        tokens = self.line_tokens(line_text)
        for tok in tokens:
            for kf in k_forms:
                if tok == kf or tok.startswith(kf) or kf.startswith(tok):
                    return True
        return False


class DictaLemmatizer:
    """Lazy Hebrew lemmatizer wrapper based on DictaBERT-lex."""

    def __init__(self, model_name: str = DICTABERT_LEX_MODEL):
        if AutoTokenizer is None or AutoModel is None:
            raise ImportError("transformers is not installed")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

    def lemmatize_text(self, text: str) -> list[str]:
        text = str(text or "").strip()
        if not text:
            return []
        pairs = self.model.predict([text], self.tokenizer)
        if not pairs:
            return []
        first = pairs[0] if isinstance(pairs[0], list) else []
        lemmas = []
        for p in first:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                lemmas.append(str(p[1]))
        return lemmas


def get_dicta_lemmatizer():
    """
    Return a cached lemmatizer instance.
    Returns False if unavailable (dependency/model load failure).
    """
    global _LEMMATIZER
    if _LEMMATIZER is not None:
        return _LEMMATIZER
    try:
        _LEMMATIZER = DictaLemmatizer()
        logger.info(f"Loaded Hebrew lemmatizer model: {DICTABERT_LEX_MODEL}")
    except Exception as e:
        logger.warning(f"Could not load Dicta lemmatizer, using heuristic fallback: {e}")
        _LEMMATIZER = False
    return _LEMMATIZER


def get_heuristic_lemmatizer() -> HeuristicHebrewLemmatizer:
    global _HEURISTIC_LEMMATIZER
    if _HEURISTIC_LEMMATIZER is None:
        _HEURISTIC_LEMMATIZER = HeuristicHebrewLemmatizer()
    return _HEURISTIC_LEMMATIZER

