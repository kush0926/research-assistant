from types import SimpleNamespace
import pytest

from app.services.extraction import canonical, find_entities, grounded_llm_entities


@pytest.mark.parametrize("name", ["LightGBM", "PASE", "WavLM", "OpenL3"])
def test_real_speech_model_names_are_recognized(name):
    assert name in find_entities(f"We evaluated {name} on speech data.", "methods")


def test_italianpvs_alias_is_canonicalized():
    assert canonical("datasets", "ItalianPVS") == "Italian Parkinson's Voice and Speech"


def test_llm_entities_require_matching_name_and_complete_source_quote():
    sentence = "We evaluated LightGBM using the ItalianPVS dataset."
    chunks = [SimpleNamespace(id=7, text=sentence)]
    result = grounded_llm_entities([{"name": "LightGBM", "role": "baseline", "quote": sentence}], "methods", chunks)
    assert result[0]["chunk_id"] == 7
    assert result[0]["category"] == "classical"
    assert grounded_llm_entities([{"name": "LSTM", "quote": sentence}], "methods", chunks) is None
    assert grounded_llm_entities([{"name": "LightGBM", "quote": "We evaluated LightGBM and achieved 99%."}], "methods", chunks) is None


def test_generic_names_and_preprocessing_entities_are_not_accepted():
    chunks = [SimpleNamespace(id=7, text="Each dataset was normalized with average pooling.")]
    assert grounded_llm_entities([{"name": "Each", "quote": chunks[0].text}], "datasets", chunks) is None
    assert grounded_llm_entities([{"name": "average pooling", "category": "technique", "quote": chunks[0].text}], "methods", chunks) is None


def test_missing_malformed_and_explicit_empty_outputs_are_distinct():
    assert grounded_llm_entities(None, "methods", []) is None
    assert grounded_llm_entities("bad", "methods", []) is None
    assert grounded_llm_entities([], "methods", []) == []
