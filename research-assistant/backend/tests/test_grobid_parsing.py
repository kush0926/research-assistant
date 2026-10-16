import pytest

from app.config import settings
from app.services import pdf_processing as pdf


TEI = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
<teiHeader><fileDesc>
<titleStmt><title>Speech study</title></titleStmt>
<publicationStmt><date when="2026">2026</date></publicationStmt>
<sourceDesc><biblStruct><analytic><author><persName><forename>Ada</forename> <surname>Example</surname></persName></author>
<idno type="DOI">10.1234/example</idno></analytic><monogr><title level="j">Speech Journal</title>
<imprint><date type="published" when="2021-06-01"/></imprint></monogr></biblStruct></sourceDesc>
</fileDesc><profileDesc><abstract><p coords="1,0,0,10,10">First abstract paragraph.</p>
<p coords="1,0,20,10,10">Second abstract paragraph.</p></abstract></profileDesc></teiHeader>
<text><body><div><head>Introduction</head><p coords="2,0,0,10,10">An introduction on page two.</p>
<div><head>Methods</head><p coords="2,0,20,10,10;3,0,0,10,10">
<s coords="2,0,20,10,10">The method starts on page two.</s>
<s coords="3,0,0,10,10">The method continues on page three.</s></p>
<figure coords="3,0,20,10,10"><table><row><cell>Accuracy</cell><cell>95%</cell></row></table></figure>
</div></div></body><back><listBibl>
<biblStruct coords="4,0,0,10,10"><title>First reference.</title></biblStruct>
<biblStruct coords="5,0,0,10,10"><title>Second reference.</title></biblStruct>
</listBibl></back></text></TEI>"""


def test_grobid_preserves_nested_sections_sentence_pages_and_published_metadata():
    result = pdf._grobid_tei_sections(TEI)
    sections = result["sections"]

    assert result["meta"] == {"title": "Speech study", "authors": ["Ada Example"], "year": 2021,
                              "doi": "10.1234/example", "venue": "Speech Journal"}
    assert next(s for s in sections if "starts on page two" in s["text"])["page"] == 2
    page_three = next(s for s in sections if "continues on page three" in s["text"])
    assert page_three["page"] == 3 and page_three["section"] == "method"
    assert "Accuracy 95%" in page_three["text"]
    assert [s["page"] for s in sections if s["section"] == "references"] == [4, 5]
    assert "Second abstract paragraph" in sections[0]["text"]


def test_grobid_respects_the_same_page_limit_as_the_fallback_parser(monkeypatch):
    monkeypatch.setattr(settings, "MAX_PDF_PAGES", 2)
    sections = pdf._grobid_tei_sections(TEI)["sections"]

    assert sections and all(s["page"] <= 2 for s in sections)
    assert all("continues on page three" not in s["text"] for s in sections)


def test_grobid_does_not_invent_page_one_when_coordinates_are_missing():
    with pytest.raises(ValueError, match="source-page coordinates"):
        pdf._grobid_tei_sections(TEI.replace('coords="2,0,0,10,10"', ""))


def test_process_pdf_validates_before_calling_grobid(monkeypatch):
    calls = []

    def invalid_pdf(_path):
        raise pdf.PDFError("Encrypted PDF")

    monkeypatch.setattr(pdf, "parse_sandboxed", invalid_pdf)
    monkeypatch.setattr(pdf, "grobid_sections", lambda _path: calls.append(True))
    with pytest.raises(pdf.PDFError, match="Encrypted PDF"):
        pdf.process_pdf("unused.pdf")
    assert calls == []


def test_process_pdf_keeps_all_abstract_and_reference_pages(monkeypatch):
    sections = [{"section": "abstract", "page": 1, "text": "First abstract."},
                {"section": "abstract", "page": 2, "text": "Second abstract."},
                {"section": "method", "page": 3, "text": "A detailed method is described here. " * 10},
                {"section": "references", "page": 4, "text": "First reference."},
                {"section": "references", "page": 5, "text": "Second reference."}]
    monkeypatch.setattr(pdf, "parse_sandboxed", lambda _path: {"page_count": 5, "pages": [], "pdf_meta": {}})
    monkeypatch.setattr(pdf, "grobid_sections", lambda _path: {"sections": sections, "meta": {"title": "Study"}})
    monkeypatch.setattr(pdf, "crossref_lookup", lambda _title: {})

    result = pdf.process_pdf("unused.pdf")

    assert result["page_count"] == 5
    assert result["meta"]["abstract"] == "First abstract. Second abstract."
    assert result["references_text"] == "First reference.\nSecond reference."
    assert all(chunk["section"] != "references" for chunk in result["chunks"])
