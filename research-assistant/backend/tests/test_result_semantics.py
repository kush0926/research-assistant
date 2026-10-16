from types import SimpleNamespace
import pytest

from app.services.extraction import extract_results, LIMIT_RE


@pytest.mark.parametrize("text", [
    "These features exceeded 99% of maximum AUC with a small number of features.",
    "The method delivered a 10% AUC improvement over the baseline.",
])
def test_relative_percentages_are_not_absolute_results(text):
    chunk = SimpleNamespace(id=1, section="results", text=text, page_number=1)
    assert extract_results([chunk], "PC-GITA", "SVM") == []


def test_absolute_auc_is_normalized_and_features_are_not_result_models():
    chunk = SimpleNamespace(id=1, section="results", text="MFCC features achieved 0.75 AUC on PC-GITA.", page_number=1)
    result = extract_results([chunk], None, "XGBoost")
    assert result[0]["value"] == 75
    assert result[0]["method"] == "XGBoost"


def test_explicit_small_percentage_is_not_scaled_to_a_fraction():
    chunk = SimpleNamespace(id=1, section="results", text="The model achieved 0.5% accuracy.", page_number=1)
    assert extract_results([chunk], None, None)[0]["value"] == 0.5


def test_multiple_models_are_not_attributed_to_the_first_mentioned_name():
    chunk = SimpleNamespace(id=1, section="results", text="SVM and CNN achieved 0.75 AUC on PC-GITA.", page_number=1)
    assert extract_results([chunk], None, None)[0]["method"] is None


def test_small_feature_set_is_not_a_small_cohort_limitation():
    assert not LIMIT_RE.search("The model worked well with a small number of features.")
    assert LIMIT_RE.search("The study was limited by a small number of participants.")
