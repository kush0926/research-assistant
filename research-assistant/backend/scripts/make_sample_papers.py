"""Generate a SYNTHETIC demo corpus (16 PDFs, voice-based Parkinson's detection) for pipeline testing.

These documents are NOT real publications and their numbers are invented. They exist only so the full pipeline
(ingest -> RAG -> compare -> gap detection) can be demonstrated offline. A gap is deliberately planted:
Transformer-family models are never evaluated on PC-GITA, though PC-GITA is common and transformers are used
elsewhere; one paper explicitly proposes that pair and two provide broader future-work context. For your real evaluation, replace these with
15-25 real papers (see eval/README).
"""
import json
import sys
import textwrap
from pathlib import Path

import pymupdf

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_papers")
OUT.mkdir(parents=True, exist_ok=True)

# (title, authors, year, method, dataset, acc, proposed, extra_limit, future)
P = [
    ("SVM-Based Detection of Parkinson's Disease from Sustained Vowel Phonations", "A. Rao, B. Iyer", 2017, "SVM", "UCI Parkinson's", 88.4, False, None, None),
    ("Random Forest Classification of Parkinson's Disease Using Voice Features", "C. Mehta, D. Shah", 2018, "Random Forest", "UCI Parkinson's", 90.1, False, None, None),
    ("MFCC and SVM for Parkinson's Disease Detection on PC-GITA", "E. Gomez, F. Ruiz", 2018, "SVM", "PC-GITA", 84.7, False, "The study uses a small number of speakers.", None),
    ("k-NN and SVM Comparison for Parkinson's Disease Detection on Telemonitoring Data", "G. Nair, H. Das", 2019, "k-NN", "UCI Telemonitoring", 86.2, False, None, None),
    ("Convolutional Neural Networks on Spectrograms for Parkinson's Disease Detection", "I. Kumar, J. Patel", 2019, "CNN", "PC-GITA", 91.3, True, "The model was only evaluated on a single dataset.", "Future work should evaluate more robust models on PC-GITA with larger speaker groups."),
    ("CNN-Based Parkinson's Disease Detection on the Italian Voice Dataset", "K. Rossi, L. Bianchi", 2020, "CNN", "Italian Parkinson's Voice and Speech", 92.8, True, None, "Future work should explore stronger architectures on additional corpora."),
    ("LSTM Networks for Parkinson's Disease Detection from Continuous Speech", "M. Silva, N. Costa", 2020, "LSTM", "PC-GITA", 89.5, True, "Training data is limited and class imbalance remains a problem.", "In the future, more robust models on PC-GITA should be investigated."),
    ("GRU and MFCC Features for Parkinson's Disease Detection on mPower", "O. Brown, P. Green", 2020, "GRU", "mPower", 87.9, True, None, None),
    ("Transfer Learning with ResNet for Parkinson's Disease Detection from Spectrograms", "Q. Lee, R. Kim", 2021, "ResNet", "mPower", 93.4, True, "The approach has not been evaluated on clinical-grade recordings.", None),
    ("Ensemble of CNN and LSTM for Parkinson's Disease Detection on UCI Parkinson's Data", "S. Wang, T. Chen", 2021, "CNN", "UCI Parkinson's", 95.2, True, None, None),
    ("Wav2vec 2.0 Embeddings for Parkinson's Disease Detection on the Italian Dataset", "U. Ferrari, V. Conti", 2022, "wav2vec 2.0", "Italian Parkinson's Voice and Speech", 94.1, True, "Only a single language is covered.", None),
    ("Transformer-Based Audio Models for Parkinson's Disease Detection on mPower", "W. Davis, X. Moore", 2022, "Transformer", "mPower", 94.6, True, "Evaluation is restricted to smartphone recordings.", "Future work should test transformer models on additional clinical corpora such as PC-GITA."),
    ("HuBERT Representations for Parkinson's Disease Detection on UCI Parkinson's", "Y. Taylor, Z. Wilson", 2023, "HuBERT", "UCI Parkinson's", 96.0, True, None, None),
    ("Self-Supervised Transformer for Severity Estimation of Parkinson's Disease on mPower", "A. Novak, B. Horvat", 2023, "Transformer", "mPower", 90.2, True, "Severity labels are noisy.", None),
    ("CNN Parkinson's Disease Detection Revisited on PC-GITA", "C. Diaz, D. Ortiz", 2023, "CNN", "PC-GITA", 82.1, False, "Results are lower than previously reported and the discrepancy is unexplained.", "Further work should resolve differences in preprocessing."),
    ("Data Augmentation with SpecAugment for Parkinson's Disease Detection on mPower", "E. Petrov, F. Ivanov", 2024, "CNN", "mPower", 92.0, True, "Few participants were available for external validation.", None),
]


def write_pdf(path: Path, title, authors, year, method, dataset, acc, proposed, limit, future):
    doc = pymupdf.open()
    page = doc.new_page(); y = 60
    W, H = page.rect.width, page.rect.height

    def newline(h):
        nonlocal page, y
        if y + h > H - 60:
            page = doc.new_page(); y = 60

    def para(text, size=10, bold=False):
        nonlocal y
        for ln in textwrap.wrap(text, 95):
            newline(size + 4)
            page.insert_text((60, y), ln, fontsize=size, fontname="hebo" if bold else "helv"); y += size + 4
        y += 4

    def head(text):
        nonlocal y
        newline(30); y += 6
        page.insert_text((60, y), text, fontsize=12.5, fontname="hebo"); y += 18

    page.insert_text((60, y), "SYNTHETIC DEMO DOCUMENT - not a real publication", fontsize=7, fontname="helv"); y += 16
    for ln in textwrap.wrap(title, 62):
        page.insert_text((60, y), ln, fontsize=17, fontname="hebo"); y += 22
    y += 2
    page.insert_text((60, y), authors, fontsize=10.5, fontname="helv"); y += 14
    page.insert_text((60, y), f"Proceedings of the Demo Workshop on Speech Health, {year}", fontsize=9, fontname="helv"); y += 22
    pd = "Parkinson's disease detection"
    dm = "deep" if method in ("CNN", "LSTM", "GRU", "ResNet", "Transformer", "wav2vec 2.0", "HuBERT") else "classical"
    pr = "We propose" if proposed else "We evaluate"
    head("Abstract")
    para(f"This paper addresses {pd} from voice recordings. {pr} a {method}-based pipeline evaluated on the {dataset} dataset. "
         f"Our approach achieves {acc}% accuracy on {dataset}. Keywords: Parkinson's disease, speech, {method}, voice biomarkers.")
    head("1. Introduction")
    para(f"Early {pd} from speech is an important clinical problem. The aim of this study is to investigate {dm} models for {pd}. "
         f"{pr} {method} on {dataset} and report accuracy and F1.")
    head("2. Related Work")
    para("Prior work applied SVM, Random Forest, and CNN models to voice-based screening. Several studies use MFCC features.")
    head("3. Methodology")
    para(f"We use MFCC features and a {method} classifier. {pr} the {method} architecture with standard preprocessing. "
         f"The training set uses an 80/20 split with 5-fold cross-validation, batch size 32 and the Adam optimizer.")
    head("4. Experiments")
    para(f"Experiments use the {dataset} dataset. Evaluation metrics are accuracy, precision, recall and F1.")
    head("5. Results")
    para(f"The {method} model achieved {acc}% accuracy on {dataset}. F1 reached {round(acc - 1.3, 1)}% and recall was {round(acc - 0.6, 1)}%.")
    head("6. Discussion")
    para("The results suggest voice is a reliable biomarker. " + (limit or "A limitation is the controlled recording conditions."))
    head("7. Conclusion")
    para(f"We demonstrated {method} for {pd} on {dataset}. " + (future or ""))
    head("References")
    para("[1] Little M. et al. Suitability of dysphonia measurements for telemonitoring of Parkinson's disease. 2009.")
    doc.set_metadata({"title": title, "author": authors})
    doc.save(path)


if __name__ == "__main__":
    for i, row in enumerate(P, 1):
        write_pdf(OUT / f"paper_{i:02d}.pdf", *row)
    (OUT / "gold_gaps.json").write_text(json.dumps([
        {"problem": "Parkinson's disease detection", "method": "Transformer", "dataset": "PC-GITA",
         "note": "planted: transformers never evaluated on PC-GITA; one explicit proposal plus two broader future-work contexts"},
    ], indent=2))
    (OUT / "ie_gold.json").write_text(json.dumps(
        [{"title": r[0], "methods": [r[3]], "datasets": [r[4]]} for r in P], indent=1))
    (OUT / "retrieval_queries.json").write_text(json.dumps(
        [{"q": f"Which study evaluates {r[3]} on the {r[4]} dataset?",
          "relevant_files": [f"paper_{i:02d}.pdf"]} for i, r in enumerate(P, 1)], indent=1))
    print(f"wrote {len(P)} PDFs to {OUT}")
