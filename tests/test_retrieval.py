import pytest

from foi_agent.retrieval import SOURCES, _as_trace_documents, load_records, search

REQUIRED_FIELDS = {
    "regime_guidance": {"id", "title", "section", "test", "key_points"},
    "exemption_guidance": {"id", "section", "title", "test", "key_points"},
    "publication_scheme": {"id", "title", "description", "period", "frequency", "url"},
    "disclosure_log": {"id", "date", "regime", "title", "summary", "outcome", "exemptions", "url"},
}
OUTCOMES = {"released", "part released", "refused", "not held"}


def test_senior_officer_salaries_finds_the_publication_scheme_record():
    results = search("publication_scheme", "senior officer salaries")
    assert results[0]["id"] == "PS-003"
    assert results[0]["source"] == "publication_scheme"
    assert results[0]["score"] > 0


def test_hits_are_traced_as_retriever_documents():
    hits = search("publication_scheme", "senior officer salaries", k=1)
    [document] = _as_trace_documents(hits)["documents"]

    assert document["type"] == "Document"
    assert document["metadata"] == {
        "id": "PS-003",
        "source": "publication_scheme",
        "score": hits[0]["score"],
        "url": hits[0]["url"],
    }
    assert "Senior officer remuneration" in document["page_content"]
    assert "publication_scheme" not in document["page_content"]


def test_results_are_capped_at_k():
    query = "information council published annual report 2025/26"
    assert len(search("publication_scheme", query, k=10)) > 2
    assert len(search("publication_scheme", query, k=2)) == 2


def test_query_with_no_matching_terms_returns_nothing():
    assert search("publication_scheme", "zzqx") == []


@pytest.mark.parametrize("source", SOURCES)
def test_records_have_required_fields(source):
    records = load_records(source)
    assert records
    for record in records:
        missing = REQUIRED_FIELDS[source] - record.keys()
        assert not missing, f"{record.get('id')} is missing {missing}"


def test_disclosure_log_outcomes_are_valid():
    for record in load_records("disclosure_log"):
        assert record["outcome"] in OUTCOMES, record["id"]


def test_ids_are_unique_across_sources():
    ids = [record["id"] for source in SOURCES for record in load_records(source)]
    assert len(ids) == len(set(ids))
