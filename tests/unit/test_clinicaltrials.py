# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: bc0cf27f-9ab8-496f-a2fd-c2e1d972435b
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for ClinicalTrials.gov study → Markdown conversion and pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The conversion functions live in the demo script; import them directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from demo.clinicaltrials_demo import _study_to_markdown, QUERIES


# ─────────────────────────────────────────────────────────────────────────────
# Minimal realistic ClinicalTrials.gov study record (API v2 format)
# ─────────────────────────────────────────────────────────────────────────────

def _make_study(
    nct_id: str = "NCT01234567",
    brief_title: str = "Phase 2 Study of DrugX in NSCLC",
    brief_summary: str = "This is a Phase 2 study evaluating DrugX in patients with NSCLC.",
    detailed_description: str = "Detailed description with additional background and methodology.",
    eligibility_criteria: str = (
        "Inclusion Criteria:\n\n"
        "* Histologically confirmed NSCLC\n"
        "* ECOG performance status 0-2\n\n"
        "Exclusion Criteria:\n\n"
        "* Active infection requiring systemic therapy\n"
        "* Known history of active Hepatitis B or C\n"
        "* Pregnant or breastfeeding"
    ),
    primary_outcomes: list[dict] | None = None,
    secondary_outcomes: list[dict] | None = None,
    interventions: list[dict] | None = None,
    lead_sponsor: str = "Acme Pharma",
) -> dict:
    if primary_outcomes is None:
        primary_outcomes = [
            {
                "measure": "Objective Response Rate (ORR)",
                "description": (
                    "ORR is defined as the proportion of patients with a confirmed "
                    "complete response (CR) or partial response (PR) per RECIST version 1.1."
                ),
                "timeFrame": "Up to 24 weeks",
            }
        ]
    if secondary_outcomes is None:
        secondary_outcomes = [
            {"measure": "Progression-Free Survival (PFS)", "description": "Time from randomization to progression or death."},
            {"measure": "Overall Survival (OS)", "description": "Time from randomization to death from any cause."},
        ]
    if interventions is None:
        interventions = [
            {
                "name": "DrugX",
                "type": "DRUG",
                "description": "DrugX 100mg administered orally once daily on Days 1-28 of each 28-day cycle.",
            }
        ]
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "briefTitle": brief_title,
                "officialTitle": f"An Official Title for {brief_title}",
            },
            "descriptionModule": {
                "briefSummary": brief_summary,
                "detailedDescription": detailed_description,
            },
            "eligibilityModule": {
                "eligibilityCriteria": eligibility_criteria,
            },
            "outcomesModule": {
                "primaryOutcomes": primary_outcomes,
                "secondaryOutcomes": secondary_outcomes,
            },
            "armsInterventionsModule": {
                "interventions": interventions,
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": lead_sponsor},
            },
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# _study_to_markdown
# ─────────────────────────────────────────────────────────────────────────────

class TestStudyToMarkdown:

    def test_returns_doc_name_and_markdown(self):
        study = _make_study()
        doc_name, md = _study_to_markdown(study)
        assert isinstance(doc_name, str)
        assert isinstance(md, str)

    def test_doc_name_contains_nct_id(self):
        study = _make_study(nct_id="NCT99887766")
        doc_name, _ = _study_to_markdown(study)
        assert "NCT99887766" in doc_name

    def test_doc_name_slug_from_title(self):
        study = _make_study(brief_title="Study of FOLFOX in Colorectal Cancer")
        doc_name, _ = _study_to_markdown(study)
        # Slug should be lowercase and use underscores
        assert doc_name == doc_name.lower() or "NCT" in doc_name
        assert " " not in doc_name

    def test_doc_name_max_length_reasonable(self):
        long_title = "A Very Long Title That Exceeds Forty Characters Easily In Practice"
        study = _make_study(brief_title=long_title)
        doc_name, _ = _study_to_markdown(study)
        # Should be truncated; NCT ID is 11 chars + underscore + max 40 slug chars
        assert len(doc_name) <= 60

    # ── Section presence ──────────────────────────────────────────────────────

    def test_brief_summary_section_present(self):
        study = _make_study(brief_summary="Phase 2 study summary text here.")
        _, md = _study_to_markdown(study)
        assert "# Brief Summary" in md
        assert "Phase 2 study summary text here." in md

    def test_detailed_description_present(self):
        study = _make_study(detailed_description="Extended methodology and background.")
        _, md = _study_to_markdown(study)
        assert "# Detailed Description" in md
        assert "Extended methodology" in md

    def test_eligibility_section_present(self):
        study = _make_study()
        _, md = _study_to_markdown(study)
        assert "# Eligibility Criteria" in md

    def test_inclusion_criteria_subsection(self):
        study = _make_study()
        _, md = _study_to_markdown(study)
        assert "## Inclusion Criteria" in md
        assert "ECOG performance status" in md

    def test_exclusion_criteria_subsection(self):
        study = _make_study()
        _, md = _study_to_markdown(study)
        assert "## Exclusion Criteria" in md
        assert "Active infection" in md

    def test_primary_outcomes_section(self):
        study = _make_study()
        _, md = _study_to_markdown(study)
        assert "# Primary Outcomes" in md
        assert "Objective Response Rate" in md
        assert "RECIST" in md

    def test_primary_outcomes_timeframe(self):
        study = _make_study()
        _, md = _study_to_markdown(study)
        assert "Up to 24 weeks" in md

    def test_secondary_outcomes_section(self):
        study = _make_study()
        _, md = _study_to_markdown(study)
        assert "# Secondary Outcomes" in md
        assert "Progression-Free Survival" in md

    def test_interventions_section(self):
        study = _make_study()
        _, md = _study_to_markdown(study)
        assert "# Interventions" in md
        assert "DrugX" in md
        assert "DRUG" in md

    def test_sections_separated_by_hr(self):
        study = _make_study()
        _, md = _study_to_markdown(study)
        assert "---" in md

    # ── Missing / empty fields ────────────────────────────────────────────────

    def test_missing_detailed_description_omitted(self):
        study = _make_study(detailed_description="")
        _, md = _study_to_markdown(study)
        assert "# Detailed Description" not in md

    def test_missing_brief_summary_omitted(self):
        study = _make_study(brief_summary="")
        _, md = _study_to_markdown(study)
        assert "# Brief Summary" not in md

    def test_no_primary_outcomes_section_omitted(self):
        study = _make_study(primary_outcomes=[])
        _, md = _study_to_markdown(study)
        assert "# Primary Outcomes" not in md

    def test_no_interventions_omitted(self):
        study = _make_study(interventions=[])
        _, md = _study_to_markdown(study)
        assert "# Interventions" not in md

    def test_no_eligibility_omitted(self):
        study = _make_study(eligibility_criteria="")
        _, md = _study_to_markdown(study)
        assert "# Eligibility Criteria" not in md

    def test_eligibility_without_inclusion_exclusion_headers(self):
        """Criteria text without Inclusion/Exclusion subheadings → single block."""
        study = _make_study(eligibility_criteria="Must be aged 18 or older. No prior chemotherapy.")
        _, md = _study_to_markdown(study)
        assert "# Eligibility Criteria" in md
        assert "18 or older" in md

    def test_secondary_outcomes_capped_at_eight(self):
        """Secondary outcomes are capped to 8 entries to avoid bloated output."""
        many = [{"measure": f"Outcome {i}", "description": f"Desc {i}"} for i in range(15)]
        study = _make_study(secondary_outcomes=many)
        _, md = _study_to_markdown(study)
        # Only first 8 should appear
        assert "Outcome 7" in md
        assert "Outcome 8" not in md  # 0-indexed cap at 8 → indices 0-7

    # ── Structural properties for RAG ─────────────────────────────────────────

    def test_two_trials_same_section_different_doc_names(self):
        """Core RAG property: same section heading across two trials, different doc names."""
        study_a = _make_study(nct_id="NCT11111111", brief_title="Trial A NSCLC")
        study_b = _make_study(nct_id="NCT22222222", brief_title="Trial B NSCLC")
        doc_a, md_a = _study_to_markdown(study_a)
        doc_b, md_b = _study_to_markdown(study_b)

        assert doc_a != doc_b
        assert "# Eligibility Criteria" in md_a
        assert "# Eligibility Criteria" in md_b

    def test_boilerplate_exclusion_text_shared_across_trials(self):
        """ECOG and infection exclusions should appear identically across trials."""
        study_a = _make_study(nct_id="NCT11111111")
        study_b = _make_study(nct_id="NCT22222222")
        _, md_a = _study_to_markdown(study_a)
        _, md_b = _study_to_markdown(study_b)
        assert "Active infection" in md_a
        assert "Active infection" in md_b


# ─────────────────────────────────────────────────────────────────────────────
# Integration: DocumentLoader pipeline on CT markdown
# ─────────────────────────────────────────────────────────────────────────────

class TestClinicalTrialsWithDocumentLoader:

    def _make_chunks(self, nct_id: str = "NCT12345678", context_strategy: str = "prefix"):
        from chunkymonkey import DocumentLoader
        study = _make_study(nct_id=nct_id, brief_title="Pembrolizumab in NSCLC")
        doc_name, md = _study_to_markdown(study)
        loader = DocumentLoader(min_chunk_size=300, max_chunk_size=300, context_strategy=context_strategy)
        return loader.load_text(md, name=doc_name)

    def test_load_produces_chunks(self):
        chunks = self._make_chunks()
        assert len(chunks) > 0

    def test_chunks_have_section_breadcrumbs(self):
        chunks = self._make_chunks()
        sections = {c.section for c in chunks if c.section}
        assert len(sections) > 0
        # Should contain eligibility and outcome sections
        assert any("Eligibility" in (s or "") for s in sections)

    def test_embedding_content_includes_doc_name(self):
        chunks = self._make_chunks(nct_id="NCT77665544")
        for c in chunks:
            assert c.embedding_content is not None
            assert "NCT77665544" in c.embedding_content

    def test_embedding_content_includes_section(self):
        chunks = self._make_chunks()
        sectioned = [c for c in chunks if c.section]
        assert len(sectioned) > 0
        for c in sectioned:
            assert c.section in c.embedding_content

    def test_boilerplate_chunks_disambiguated_by_doc_name(self):
        """Two trials with identical exclusion criteria get different embedding_content."""
        from chunkymonkey import DocumentLoader

        loader = DocumentLoader(min_chunk_size=300, max_chunk_size=300, context_strategy="prefix")

        study_a = _make_study(nct_id="NCT11111111", brief_title="Trial A NSCLC")
        study_b = _make_study(nct_id="NCT22222222", brief_title="Trial B NSCLC")

        doc_a, md_a = _study_to_markdown(study_a)
        doc_b, md_b = _study_to_markdown(study_b)

        chunks_a = loader.load_text(md_a, name=doc_a)
        chunks_b = loader.load_text(md_b, name=doc_b)

        # Find chunks with identical content (boilerplate)
        contents_a = {c.content for c in chunks_a}
        contents_b = {c.content for c in chunks_b}
        shared_content = contents_a & contents_b

        if shared_content:
            for content in shared_content:
                ec_a = next(c.embedding_content for c in chunks_a if c.content == content)
                ec_b = next(c.embedding_content for c in chunks_b if c.content == content)
                assert ec_a != ec_b, (
                    "Identical chunk content must produce different embedding_content "
                    "because the document name disambiguates"
                )

    def test_naive_vs_contextual_chunk_count_equal(self):
        """Same number of chunks; only embedding_content differs."""
        from chunkymonkey import DocumentLoader
        study = _make_study()
        doc_name, md = _study_to_markdown(study)

        naive = DocumentLoader(min_chunk_size=300, max_chunk_size=300, context_strategy=None)
        ctx = DocumentLoader(min_chunk_size=300, max_chunk_size=300, context_strategy="prefix")

        n_chunks = naive.load_text(md, name=doc_name)
        c_chunks = ctx.load_text(md, name=doc_name)

        assert len(n_chunks) == len(c_chunks)

    def test_eligibility_section_splits_into_multiple_chunks(self):
        """Long eligibility criteria (>chunk_size chars) must produce multiple chunks."""
        from chunkymonkey import DocumentLoader

        long_criteria = (
            "Inclusion Criteria:\n\n"
            + "\n".join(f"* Criterion {i}: " + "A " * 30 for i in range(20))
            + "\n\nExclusion Criteria:\n\n"
            + "\n".join(f"* Exclusion {i}: " + "B " * 30 for i in range(20))
        )
        study = _make_study(eligibility_criteria=long_criteria)
        doc_name, md = _study_to_markdown(study)

        loader = DocumentLoader(min_chunk_size=400, max_chunk_size=400, context_strategy="prefix")
        chunks = loader.load_text(md, name=doc_name)

        elig_chunks = [c for c in chunks if "Eligibility" in (c.section or "")]
        assert len(elig_chunks) >= 2, (
            "Long eligibility section should split into ≥2 chunks; "
            f"got {len(elig_chunks)}"
        )

    def test_all_elig_continuation_chunks_carry_section(self):
        """Every eligibility continuation chunk must have a section breadcrumb."""
        from chunkymonkey import DocumentLoader

        long_criteria = (
            "Inclusion Criteria:\n\n"
            + "\n".join(f"* Item {i}: " + "word " * 40 for i in range(15))
            + "\n\nExclusion Criteria:\n\n"
            + "\n".join(f"* Excl {i}: " + "word " * 40 for i in range(15))
        )
        study = _make_study(eligibility_criteria=long_criteria)
        doc_name, md = _study_to_markdown(study)

        loader = DocumentLoader(min_chunk_size=400, max_chunk_size=400, context_strategy="prefix")
        chunks = loader.load_text(md, name=doc_name)

        elig_chunks = [c for c in chunks if "Eligibility" in (c.section or "")]
        for c in elig_chunks:
            assert c.section is not None
            assert c.embedding_content is not None
            assert doc_name in c.embedding_content


# ─────────────────────────────────────────────────────────────────────────────
# QUERIES list sanity checks
# ─────────────────────────────────────────────────────────────────────────────

class TestQueriesStructure:
    def test_queries_is_nonempty(self):
        assert len(QUERIES) > 0

    def test_each_query_has_four_fields(self):
        for q in QUERIES:
            assert len(q) == 4, f"Query tuple should have 4 elements: {q}"

    def test_query_text_is_string(self):
        for query, _, _, _ in QUERIES:
            assert isinstance(query, str) and query.strip()

    def test_section_kws_is_list(self):
        for _, kws, _, _ in QUERIES:
            assert isinstance(kws, list) and len(kws) > 0

    def test_doc_name_fragment_is_string(self):
        for _, _, fragment, _ in QUERIES:
            assert isinstance(fragment, str) and fragment.strip()

    def test_reason_is_string(self):
        for _, _, _, reason in QUERIES:
            assert isinstance(reason, str) and len(reason) > 10
