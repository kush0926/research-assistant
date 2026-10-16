import pytest

from app.services.extraction import novel_datasets


@pytest.mark.parametrize("word", ["Each", "Every", "Both", "These", "Those", "Another", "Several", "Multiple"])
def test_determiners_are_not_novel_dataset_names(word):
    assert not novel_datasets(f"{word} dataset was split into speaker-disjoint folds.", set())


def test_named_dataset_is_retained_while_determiners_are_removed():
    result = novel_datasets("We used the SpanishNeuro corpus. Each dataset was normalized.", set())
    assert result == {"SpanishNeuro": 1}
