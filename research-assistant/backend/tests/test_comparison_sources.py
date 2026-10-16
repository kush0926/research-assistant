from types import SimpleNamespace
from app.services import comparison


def test_comparison_cells_retain_each_supporting_chunk_and_exclude_features(monkeypatch):
    values = {"methods": [{"name": "MFCC", "category": "feature", "role": "used", "chunk_id": 1},
                          {"name": "LightGBM", "category": "classical", "role": "used", "chunk_id": 2}],
              "datasets": [{"name": "A", "chunk_id": 3}, {"name": "B", "chunk_id": 4}],
              "result_records": [{"value": 90, "metric": "Accuracy", "chunk_id": 5},
                                 {"value": 85, "metric": "F1", "chunk_id": 6}]}
    monkeypatch.setattr(comparison, "get_fields", lambda *args: {key: {"value": value} for key, value in values.items()})
    row = comparison.build_table(None, [SimpleNamespace(id=1, title="Study", year=2025)])["rows"][0]
    assert row["algorithm"]["text"] == "LightGBM"
    assert row["algorithm"]["chunk_ids"] == [2]
    assert row["dataset"]["chunk_ids"] == [3, 4]
    assert row["results"]["chunk_ids"] == [5, 6]


def test_cell_deduplicates_source_ids_and_preserves_legacy_pointer():
    cell = comparison._cell("Evidence", 3, 1, chunk_ids=[3, 4, None])
    assert cell["chunk_id"] == 3
    assert cell["chunk_ids"] == [3, 4]
