from app.services.pdf_processing import chunk_sections, detect_sections, extract_metadata


def line(text, page=1, size=10, bold=False):
    return {"text": text, "page": page, "size": size, "bold": bold, "y": 0}


def test_arxiv_watermark_is_not_the_title_or_publication_year():
    pages = [[line("arXiv:2005.14647v2 [eess.AS] 25 Aug 2020", size=20),
              line("Assessment of Parkinson's Disease Medication State", size=14),
              line("through Automatic Speech Analysis", size=14),
              line("Ada Example and Ben Author", size=12),
              line("Abstract", bold=True), line("We compare speech models.")]]

    metadata = extract_metadata(pages, {}, detect_sections(pages, 10))

    assert metadata["title"] == "Assessment of Parkinson's Disease Medication State through Automatic Speech Analysis"
    assert metadata["year"] == 2020
    assert metadata["venue"] == "arXiv"


def test_valid_pdf_title_is_preserved_when_a_heading_uses_a_larger_font():
    pages = [[line("A large publisher banner", size=24), line("Actual study title", size=14)]]

    metadata = extract_metadata(pages, {"title": "Actual study title"}, [])

    assert metadata["title"] == "Actual study title"


def test_year_in_abstract_citation_does_not_override_document_date():
    pages = [[line("A speech classification study", size=14), line("Abstract", bold=True),
              line("Prior work from 2010 evaluated these speakers.")]]

    metadata = extract_metadata(pages, {"creationDate": "D:20220301000000Z"}, [])

    assert metadata["year"] == 2022


def test_sections_and_chunks_keep_the_page_when_a_section_continues():
    pages = [[line("Introduction", bold=True), line("This passage is on the first page.")],
             [line("This passage continues on the second page.", page=2),
              line("References", page=2, bold=True), line("First reference.", page=2)],
             [line("Second reference.", page=3)]]

    sections = detect_sections(pages, 10)
    chunks = chunk_sections(sections)

    assert [(section["section"], section["page"]) for section in sections] == [
        ("introduction", 1), ("introduction", 2), ("references", 2), ("references", 3)]
    assert [(chunk["page"], chunk["text"]) for chunk in chunks] == [
        (1, "This passage is on the first page."), (2, "This passage continues on the second page.")]
