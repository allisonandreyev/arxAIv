"""OCR-based gibberish scoring for figures.

Runs OCR on a figure and reports the fraction of readable word tokens that do
not look like real language. This is deliberately independent of
``gpt_clean.gpt_clean_text``: the raw OCR output is scored as-is, so a figure
whose text renders as garbage is not repaired into something plausible before
being measured.
"""

import math
import random
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import nan
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Words that are legitimate in ML figures but missing from a system dictionary.
DOMAIN_WORDS = {
    # Architectures, methods and datasets.
    "adam", "adamw", "arxiv", "backbone", "baseline", "bert", "cifar", "clip",
    "cnn", "coco", "convnext", "cutmix", "deberta", "diffusion", "dropout",
    "encoder", "decoder", "embedding", "embeddings", "gan", "gelu", "gpt",
    "gradcam", "imagenet", "latent", "layernorm", "llm", "llms", "logits",
    "lora", "lstm", "mixup", "mlp", "mnist", "multimodal", "nerf", "nlp",
    "relu", "resnet", "roberta", "sgd", "softmax", "swin", "tokenizer",
    "transformer", "unet", "vae", "vit", "zeroshot",
    # Metrics and training vocabulary.
    "acc", "auroc", "batch", "bleu", "epoch", "epochs", "eval", "fid",
    "finetune", "finetuned", "finetuning", "flops", "gpu", "gpus", "iou",
    "iter", "iters", "mae", "mse", "params", "perplexity", "pretrain",
    "pretrained", "pretraining", "psnr", "rmse", "rouge", "runtime", "sota",
    "ssim", "throughput", "timestep", "timesteps", "tokens", "wallclock",
    # Surnames that appear in named quantities and citations.
    "bhattacharyya", "frobenius", "hellinger", "jaccard", "kullback",
    "leibler", "mahalanobis", "minkowski", "wasserstein",
    # Greek letters spelled out, common in axis labels.
    "alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta", "lambda",
    "sigma", "tau", "phi", "psi", "omega",
    # Figure furniture.
    "avg", "fig", "std", "stddev", "val",
    # Ordinary words missing from Webster's 2nd, which the dictionary predates
    # or simply omits.
    "benchmark", "box", "dataset", "datasets", "pipeline", "workflow",
}

SYSTEM_DICTIONARIES = (
    Path("/usr/share/dict/words"),
    Path("/usr/dict/words"),
)

VOWELS = set("aeiouy")

# Tuning constants for the character-plausibility rules.
MIN_TOKEN_LEN = 3          # shorter tokens ("n", "of", "x") are not scored
MAX_ACRONYM_LEN = 5        # ALLCAPS tokens up to this length are treated as acronyms
MIN_VOWEL_RATIO = 0.15
MAX_VOWEL_RATIO = 0.80
MAX_CONSONANT_RUN = 4      # a run this long or longer is implausible
MAX_CHAR_REPEAT = 3        # "aaa" and longer
MAX_CASE_SWITCHES = 2      # "tExTuRe"-style OCR garbling

# A token whose average bigram log-probability falls below the Nth percentile of
# real dictionary words is treated as implausible. 5 is a deliberate trade: it
# catches shape-plausible garble like "vahdatoin" at the cost of flagging the
# rarest genuine words, which are exempt anyway when the lexicon contains them.
PLAUSIBILITY_PERCENTILE = 5
FALLBACK_THRESHOLD = -3.08  # used when no system dictionary is installed
SMOOTHING = 0.5
ALPHABET_SIZE = 28          # 26 letters plus the two word-boundary markers
CALIBRATION_SAMPLE = 20000

# The bigram test needs enough characters to mean anything: a 3-letter token has
# only four transitions, so a single unusual pair condemns it. Short tokens are
# judged on shape alone.
MIN_BIGRAM_LEN = 5

# The system dictionary lists headwords only -- no plurals, no inflections -- so
# "images" and "networks" are absent while "image" and "network" are present.
# A token is accepted when stripping one of these suffixes leaves a known word.
SUFFIXES = ("s", "es", "ed", "ing", "er", "ers", "est", "ly", "ness")

# A token is only considered word-like if it is letters, optionally with an
# internal hyphen or apostrophe. Anything containing a digit is a number, a
# unit or a tick label, not a word, and is skipped.
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
HAS_DIGIT_RE = re.compile(r"\d")


@dataclass(frozen=True)
class Lexicon:
    """Known words plus a character bigram model built from the same words."""

    words: frozenset
    bigram_logp: Dict[Tuple[str, str], float]
    unseen_logp: float
    threshold: float

    def plausibility(self, token: str) -> float:
        """Mean log P(next char | previous char), with word-boundary markers.

        Real words score near -2.5; OCR garble scores well below, because its
        character transitions are ones English rarely makes.
        """
        padded = "^" + token + "$"
        pairs = list(zip(padded, padded[1:]))
        if not pairs:
            return 0.0
        total = sum(self.bigram_logp.get(pair, self.unseen_logp) for pair in pairs)
        return total / len(pairs)


@lru_cache(maxsize=1)
def load_lexicon() -> Lexicon:
    """Build the word set and bigram model from the first dictionary found.

    Without a system dictionary the model is built from ``DOMAIN_WORDS`` alone
    and the threshold falls back to a fixed constant, so scoring still works but
    is coarser.
    """
    words: Set[str] = set(DOMAIN_WORDS)

    for path in SYSTEM_DICTIONARIES:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                word = line.strip().lower()
                if word.isalpha() and len(word) >= MIN_TOKEN_LEN:
                    words.add(word)
        break

    bigrams: Counter = Counter()
    prefixes: Counter = Counter()
    for word in words:
        padded = "^" + word + "$"
        for pair in zip(padded, padded[1:]):
            bigrams[pair] += 1
            prefixes[pair[0]] += 1

    denominator = SMOOTHING * ALPHABET_SIZE
    bigram_logp = {
        pair: math.log((count + SMOOTHING) / (prefixes[pair[0]] + denominator))
        for pair, count in bigrams.items()
    }
    unseen_logp = math.log(SMOOTHING / denominator)

    lexicon = Lexicon(frozenset(words), bigram_logp, unseen_logp, FALLBACK_THRESHOLD)

    # Calibrate the threshold against the dictionary the model was built from.
    if len(words) > CALIBRATION_SAMPLE:
        sample = random.Random(0).sample(sorted(words), CALIBRATION_SAMPLE)
        scores = sorted(lexicon.plausibility(word) for word in sample)
        threshold = scores[len(scores) * PLAUSIBILITY_PERCENTILE // 100]
        lexicon = Lexicon(lexicon.words, bigram_logp, unseen_logp, threshold)

    return lexicon


# Figures are scattered labels and axis text, not paragraphs, so sparse-text
# segmentation reads them far better than the default page model.
PSM_SPARSE_TEXT = "11"


def ocr_text(path: Path, lang: str = "eng", psm: str = PSM_SPARSE_TEXT) -> str:
    """Return raw OCR text for an image. No cleaning, no model repair."""
    try:
        import pytesseract  # optional; the CLI fallback below is equivalent
        from PIL import Image

        return pytesseract.image_to_string(
            Image.open(path), lang=lang, config=f"--psm {psm}"
        )
    except ImportError:
        pass

    binary = shutil.which("tesseract")
    if binary is None:
        raise RuntimeError(
            "OCR requires either the 'pytesseract' package or the 'tesseract' binary."
        )

    with tempfile.TemporaryDirectory() as tmp:
        out_base = Path(tmp) / "ocr"
        subprocess.run(
            [binary, str(path), str(out_base), "-l", lang, "--psm", psm],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return (out_base.with_suffix(".txt")).read_text(encoding="utf-8", errors="ignore")


def tokenize(text: str) -> List[str]:
    """Extract scorable word tokens from raw OCR text.

    Skips anything containing a digit, tokens shorter than ``MIN_TOKEN_LEN``,
    and short ALLCAPS tokens, which are nearly always acronyms (CNN, RGB, IoU).
    """
    tokens = []
    for raw in text.split():
        if HAS_DIGIT_RE.search(raw):
            continue
        for match in TOKEN_RE.finditer(raw):
            token = match.group(0).strip("-'")
            if len(token) < MIN_TOKEN_LEN:
                continue
            if token.isupper() and len(token) <= MAX_ACRONYM_LEN:
                continue
            tokens.append(token)
    return tokens


def known_word(word: str, words: frozenset) -> bool:
    """True if the word is in the lexicon, or is an inflection of one."""
    if word in words:
        return True
    for suffix in SUFFIXES:
        if not word.endswith(suffix) or len(word) <= len(suffix) + 1:
            continue
        stem = word[: -len(suffix)]
        if stem in words:
            return True
        if stem + "e" in words:  # "sampled" -> "sample"
            return True
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[:-1] in words:
            return True  # "running" -> "run"
        if stem.endswith("i") and stem[:-1] + "y" in words:
            return True  # "densities" -> "density"
    return False


def is_gibberish(token: str, lexicon: Lexicon) -> bool:
    """Decide whether a single OCR token looks like language.

    A token in the lexicon is always accepted. Everything else is judged on
    character shape -- vowel balance, consonant runs, repeated characters,
    erratic capitalisation -- and then on how plausible its character
    transitions are under the bigram model.
    """
    letters = [c for c in token.lower() if c.isalpha()]
    if len(letters) < MIN_TOKEN_LEN:
        return False

    lowered = "".join(letters)
    if known_word(lowered, lexicon.words):
        return False

    case_switches = sum(
        1
        for a, b in zip(token, token[1:])
        if a.isalpha() and b.isalpha() and a.isupper() != b.isupper()
    )
    if case_switches > MAX_CASE_SWITCHES:
        return True

    vowel_ratio = sum(1 for c in lowered if c in VOWELS) / len(lowered)
    if vowel_ratio < MIN_VOWEL_RATIO or vowel_ratio > MAX_VOWEL_RATIO:
        return True

    run = 0
    for c in lowered:
        run = run + 1 if c not in VOWELS else 0
        if run >= MAX_CONSONANT_RUN:
            return True

    repeat = 1
    for a, b in zip(lowered, lowered[1:]):
        repeat = repeat + 1 if a == b else 1
        if repeat >= MAX_CHAR_REPEAT:
            return True

    if len(lowered) < MIN_BIGRAM_LEN:
        return False

    return lexicon.plausibility(lowered) < lexicon.threshold


def gibberish_ratio(path: Path, text: Optional[str] = None) -> Tuple[float, int]:
    """Return (ratio, token_count) for a figure.

    ``ratio`` is the share of scorable OCR tokens flagged as gibberish, and is
    ``nan`` when the figure yields no scorable text, since a figure with no
    words is not the same as a figure with no gibberish. Pass ``text`` to score
    OCR output you already have instead of re-running OCR.
    """
    if text is None:
        text = ocr_text(path)

    tokens = tokenize(text)
    if not tokens:
        return nan, 0

    lexicon = load_lexicon()
    flagged = sum(1 for token in tokens if is_gibberish(token, lexicon))
    return flagged / len(tokens), len(tokens)
