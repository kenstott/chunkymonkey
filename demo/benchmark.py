# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: c33488c3-1577-4e5d-9554-e852ac7cbbe6
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""
Chunky Monkey — Two-phase benchmark.

Build phase: fetch raw content once, extract text to library.
Run phase:   sample N docs per type, evaluate chunking strategies, print RQI table.

Usage:
    python demo/benchmark.py build --library /tmp/cm_bench
    python demo/benchmark.py run   --library /tmp/cm_bench
    python demo/benchmark.py run   --library /tmp/cm_bench --n-per-type 200 --seed 42
    python demo/benchmark.py run   --library /tmp/cm_bench --n-edgar 38 --n-ct 500
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import random
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

sys.path.insert(0, str(Path(__file__).parent.parent))

from chunkymonkey.chunking import chunk_document
from chunkymonkey.extractors import EdgarExtractor
from chunkymonkey.models import DocumentChunk

# ─────────────────────────────────────────────────────────────────────────────
# HTTP utility
# ─────────────────────────────────────────────────────────────────────────────

_UA = "chunkymonkey-benchmark/1.0 (research; contact@example.com)"


def _http_get(url: str, headers: dict | None = None, retries: int = 3,
              timeout: int = 30, max_bytes: int | None = None) -> bytes:
    h = headers or {"User-Agent": _UA}
    req = urllib.request.Request(url, headers=h)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, context=_SSL_CTX, timeout=timeout) as r:
                data = r.read(max_bytes) if max_bytes else r.read()
                return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError(f"fetch failed {url}: {exc}") from exc
    raise RuntimeError("unreachable")


# ─────────────────────────────────────────────────────────────────────────────
# EDGAR — 40 companies
# ─────────────────────────────────────────────────────────────────────────────

_EDGAR_FILINGS = [
    # Hospital / health systems (10)
    {"name": "hca",   "co": "HCA Healthcare",           "cik": "860730"},
    {"name": "thc",   "co": "Tenet Healthcare",          "cik": "70858"},
    {"name": "uhs",   "co": "Universal Health Services", "cik": "352915"},
    {"name": "cyh",   "co": "Community Health Systems",  "cik": "1108109"},
    {"name": "dva",   "co": "DaVita",                    "cik": "927066"},
    {"name": "ehc",   "co": "Encompass Health",          "cik": "785161"},
    {"name": "sgry",  "co": "Surgery Partners",          "cik": "1638833"},
    {"name": "achc",  "co": "Acadia Healthcare",         "cik": "1520697"},
    {"name": "amed",  "co": "Amedisys",                  "cik": "896429"},
    {"name": "lhcg",  "co": "LHC Group",                 "cik": "1259524"},
    # Pharmaceutical / biotech (10)
    {"name": "pfe",   "co": "Pfizer",                    "cik": "78003"},
    {"name": "mrk",   "co": "Merck",                     "cik": "310158"},
    {"name": "jnj",   "co": "Johnson & Johnson",         "cik": "200406"},
    {"name": "abbv",  "co": "AbbVie",                    "cik": "1551152"},
    {"name": "bmy",   "co": "Bristol-Myers Squibb",      "cik": "14272"},
    {"name": "lly",   "co": "Eli Lilly",                 "cik": "59478"},
    {"name": "amgn",  "co": "Amgen",                     "cik": "318154"},
    {"name": "gild",  "co": "Gilead Sciences",           "cik": "882095"},
    {"name": "regn",  "co": "Regeneron",                 "cik": "872589"},
    {"name": "vrtx",  "co": "Vertex Pharmaceuticals",    "cik": "875320"},
    # Health insurance / managed care (6)
    {"name": "unh",   "co": "UnitedHealth Group",        "cik": "731766"},
    {"name": "elv",   "co": "Elevance Health",           "cik": "1156039"},
    {"name": "ci",    "co": "Cigna",                     "cik": "1739940"},
    {"name": "hum",   "co": "Humana",                    "cik": "49071"},
    {"name": "cnc",   "co": "Centene",                   "cik": "1071739"},
    {"name": "moh",   "co": "Molina Healthcare",         "cik": "928954"},
    # Technology (6)
    {"name": "aapl",  "co": "Apple",                     "cik": "320193"},
    {"name": "msft",  "co": "Microsoft",                 "cik": "789019"},
    {"name": "googl", "co": "Alphabet",                  "cik": "1652044"},
    {"name": "amzn",  "co": "Amazon",                    "cik": "1018724"},
    {"name": "crm",   "co": "Salesforce",                "cik": "1108524"},
    {"name": "nvda",  "co": "NVIDIA",                    "cik": "1045810"},
    # Financial services (4)
    {"name": "jpm",   "co": "JPMorgan Chase",            "cik": "19617"},
    {"name": "gs",    "co": "Goldman Sachs",             "cik": "886982"},
    {"name": "wfc",   "co": "Wells Fargo",               "cik": "72971"},
    {"name": "ms",    "co": "Morgan Stanley",            "cik": "895421"},
    # Retail / consumer / diversified (4)
    {"name": "wmt",   "co": "Walmart",                   "cik": "104169"},
    {"name": "cvs",   "co": "CVS Health",                "cik": "1067294"},
    {"name": "abc",   "co": "AmerisourceBergen",         "cik": "1140859"},
    {"name": "mck",   "co": "McKesson",                  "cik": "927653"},
]

_EDGAR_HDR = {
    "User-Agent": _UA,
    "Accept-Encoding": "gzip, deflate",
}
_EDGAR_MAX_BYTES = 1_500_000


def _get_latest_10k_url(cik: str) -> str:
    index_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    data = json.loads(_http_get(index_url, _EDGAR_HDR))
    filings = data.get("filings", {}).get("recent", {})
    cik_int = int(data.get("cik", cik))
    for form, acc_no, doc in zip(
        filings.get("form", []),
        filings.get("accessionNumber", []),
        filings.get("primaryDocument", []),
    ):
        if form == "10-K":
            acc = acc_no.replace("-", "")
            return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}"
    raise RuntimeError(f"No 10-K for CIK {cik}")


def _build_edgar(lib_dir: Path) -> None:
    raw_dir   = lib_dir / "_raw"
    out_dir   = lib_dir / "edgar"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = EdgarExtractor(infer_bold_headings=True)
    for f in _EDGAR_FILINGS:
        txt_file = out_dir / f"{f['name']}.txt"
        if txt_file.exists():
            continue
        raw_file = raw_dir / f"edgar_{f['name']}.htm"
        if raw_file.exists():
            raw = raw_file.read_bytes()
        else:
            try:
                url = _get_latest_10k_url(f["cik"])
                raw = _http_get(url, _EDGAR_HDR, max_bytes=_EDGAR_MAX_BYTES)
                raw_file.write_bytes(raw)
                time.sleep(0.5)
            except Exception as exc:
                print(f"    SKIP {f['co']}: {exc}")
                continue
        try:
            text = extractor.extract(raw, f"{f['name']}.htm")
            txt_file.write_text(text, encoding="utf-8")
        except Exception as exc:
            print(f"    SKIP extract {f['co']}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# ClinicalTrials — 14 conditions
# ─────────────────────────────────────────────────────────────────────────────

_CT_CONDITIONS = [
    "non-small cell lung cancer",
    "colorectal cancer",
    "breast cancer",
    "ovarian cancer",
    "pancreatic cancer",
    "prostate cancer",
    "melanoma",
    "hepatocellular carcinoma",
    "multiple myeloma",
    "renal cell carcinoma",
    "glioblastoma",
    "type 2 diabetes",
    "rheumatoid arthritis",
    "heart failure",
]


def _fetch_trials(condition: str, n: int) -> list[dict]:
    params = "&".join([
        f"query.cond={urllib.parse.quote(condition)}",
        "filter.overallStatus=COMPLETED",
        f"pageSize={min(n * 3, 200)}",
        "format=json",
    ])
    data = json.loads(_http_get(f"https://clinicaltrials.gov/api/v2/studies?{params}"))
    studies = data.get("studies", [])

    def _is_p23(s: dict) -> bool:
        phases = (s.get("protocolSection", {})
                   .get("designModule", {})
                   .get("phases", []))
        return any(p in ("PHASE2", "PHASE3") for p in phases)

    return [s for s in studies if _is_p23(s)][:n]


def _study_to_markdown(study: dict) -> tuple[str, str]:
    ps     = study.get("protocolSection", {})
    id_mod = ps.get("identificationModule", {})
    nct_id = id_mod.get("nctId", "NCTUNKNOWN")
    title  = id_mod.get("briefTitle", "Untitled")
    parts: list[str] = []

    desc = ps.get("descriptionModule", {})
    if desc.get("briefSummary"):
        parts.append(f"# Brief Summary\n\n{desc['briefSummary'].strip()}")
    if desc.get("detailedDescription"):
        parts.append(f"# Detailed Description\n\n{desc['detailedDescription'].strip()}")

    elig_raw = ps.get("eligibilityModule", {}).get("eligibilityCriteria", "").strip()
    if elig_raw:
        incl = re.search(r"inclusion criteria[:\s]*(.*?)(?:exclusion criteria|$)",
                         elig_raw, re.I | re.S)
        excl = re.search(r"exclusion criteria[:\s]*(.*?)$", elig_raw, re.I | re.S)
        e = ["# Eligibility Criteria"]
        if incl:
            e.append(f"\n## Inclusion Criteria\n\n{incl.group(1).strip()}")
        if excl:
            e.append(f"\n## Exclusion Criteria\n\n{excl.group(1).strip()}")
        if not incl and not excl:
            e.append(f"\n{elig_raw}")
        parts.append("\n".join(e))

    outcomes = ps.get("outcomesModule", {})
    primary  = outcomes.get("primaryOutcomes", [])
    if primary:
        lines = ["# Primary Outcomes"]
        for o in primary:
            m, d, t = o.get("measure",""), o.get("description",""), o.get("timeFrame","")
            lines.append(f"\n- **{m}**")
            if d: lines.append(f"  {d}")
            if t: lines.append(f"  Time frame: {t}")
        parts.append("\n".join(lines))

    secondary = outcomes.get("secondaryOutcomes", [])
    if secondary:
        parts.append("# Secondary Outcomes\n\n" +
                     "\n".join(f"- {o.get('measure','')}" for o in secondary[:6]))

    arms = ps.get("armsInterventionsModule", {})
    for iv in arms.get("interventions", []):
        lines = [f"# Interventions\n\n**{iv.get('type','')}: {iv.get('name','')}**"]
        if iv.get("description"):
            lines.append(iv["description"].strip())
        parts.append("\n".join(lines))
        break

    return nct_id, "\n\n".join(parts)


def _build_ct(lib_dir: Path, n_trials: int = 100) -> None:
    raw_dir = lib_dir / "_raw"
    out_dir = lib_dir / "ct"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for condition in _CT_CONDITIONS:
        slug       = condition.replace(" ", "_")
        cache_file = raw_dir / f"ct_{slug}.json"
        if cache_file.exists():
            studies = json.loads(cache_file.read_text())
        else:
            studies = _fetch_trials(condition, n_trials)
            cache_file.write_text(json.dumps(studies))
            time.sleep(0.4)
        for s in studies:
            try:
                nct_id, text = _study_to_markdown(s)
                txt_file = out_dir / f"{nct_id}.txt"
                if not txt_file.exists() and text.strip():
                    txt_file.write_text(text, encoding="utf-8")
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Federal Register — 5 searches
# ─────────────────────────────────────────────────────────────────────────────

_FED_REG_SEARCHES = [
    {"name": "cms_hospital",  "agency": "centers-for-medicare-medicaid-services",
     "search": "hospital inpatient payment prospective system"},
    {"name": "cms_physician", "agency": "centers-for-medicare-medicaid-services",
     "search": "physician fee schedule quality payment"},
    {"name": "fda_drugs",     "agency": "food-and-drug-administration",
     "search": "drug approval safety oncology cancer"},
    {"name": "oig_health",    "agency": "office-of-inspector-general-department-of-health-and-human-services",
     "search": "hospital compliance fraud waste abuse"},
    {"name": "hhs_privacy",   "agency": "health-and-human-services-department",
     "search": "HIPAA privacy security healthcare"},
]
_FR_API    = "https://www.federalregister.gov/api/v1/documents.json"
_FR_FIELDS = ["title","abstract","document_number","publication_date",
              "agencies","topics","cfr_references","type","body_html_url"]


def _fetch_fr_page(cfg: dict, page: int) -> list[dict]:
    field_str = "&".join(f"fields%5B%5D={f}" for f in _FR_FIELDS)
    params = "&".join([
        f"q={urllib.parse.quote(cfg['search'])}",
        f"agencies%5B%5D={urllib.parse.quote(cfg['agency'])}",
        f"per_page=20&page={page}&order=newest",
        field_str,
    ])
    try:
        data = json.loads(_http_get(f"{_FR_API}?{params}"))
        return data.get("results", [])
    except Exception as exc:
        print(f"      WARNING FR {cfg['name']} p{page}: {exc}")
        return []


def _fetch_fr_body(url: str) -> str:
    """Fetch and strip HTML body for a single FR document. max_bytes=32_000."""
    try:
        raw   = _http_get(url, retries=2, timeout=15, max_bytes=32_000)
        html  = raw.decode("utf-8", errors="replace")
        plain = re.sub(r"<[^>]+>", " ", html)
        plain = re.sub(r"\s{3,}", "\n\n", plain).strip()
        return plain[:8000]
    except Exception:
        return ""


def _fed_reg_to_markdown(doc: dict, body_text: str) -> str:
    title    = doc.get("title", "Untitled").strip()
    pub_date = doc.get("publication_date", "")
    doc_no   = doc.get("document_number", "")
    abstract = (doc.get("abstract") or "").strip()
    agencies = ", ".join(a.get("name","") for a in (doc.get("agencies") or []))
    topics   = ", ".join(doc.get("topics") or [])
    cfr      = ", ".join(f"{c.get('title','')} CFR {c.get('part','')}"
                         for c in (doc.get("cfr_references") or []))
    lines = [f"# {title}",
             f"\n**Agency:** {agencies}",
             f"**Date:** {pub_date}  **Doc#:** {doc_no}"]
    if cfr:     lines.append(f"**CFR:** {cfr}")
    if topics:  lines.append(f"**Topics:** {topics}")
    if abstract: lines.append(f"\n## Summary\n\n{abstract}")
    if body_text:
        lines.append(f"\n## Rule Text (excerpt)\n\n{body_text}")
    return "\n".join(lines)


def _build_fr(lib_dir: Path, n_pages: int = 100) -> None:
    raw_dir = lib_dir / "_raw"
    out_dir = lib_dir / "fr"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg in _FED_REG_SEARCHES:
        meta_file = raw_dir / f"fr_{cfg['name']}.json"
        if meta_file.exists():
            docs = json.loads(meta_file.read_text())
        else:
            docs = []
            for page in range(1, n_pages + 1):
                batch = _fetch_fr_page(cfg, page)
                docs.extend(batch)
                if len(batch) < 20:
                    break
                time.sleep(0.25)
            meta_file.write_text(json.dumps(docs))

        for docno, doc in enumerate(docs):
            doc_no_raw = doc.get("document_number", f"doc{docno}").replace("/", "-")
            txt_file   = out_dir / f"fr_{cfg['name']}_{doc_no_raw}.txt"
            if txt_file.exists():
                continue
            body_text = doc.get("_body_text", "")
            if not body_text:
                body_url = doc.get("body_html_url", "")
                if body_url:
                    body_text = _fetch_fr_body(body_url)
                    time.sleep(0.1)
            text = _fed_reg_to_markdown(doc, body_text)
            if text.strip():
                txt_file.write_text(text, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# PubMed — 40 topics
# ─────────────────────────────────────────────────────────────────────────────

_PUBMED_TOPICS = [
    # Oncology (15)
    "pembrolizumab non-small cell lung cancer PD-L1 overall survival",
    "FOLFOX oxaliplatin colorectal cancer adjuvant chemotherapy",
    "trastuzumab HER2 breast cancer pathological complete response",
    "olaparib PARP inhibitor BRCA ovarian cancer maintenance",
    "pancreatic cancer gemcitabine nab-paclitaxel survival outcomes",
    "prostate cancer enzalutamide abiraterone castration resistant",
    "melanoma ipilimumab nivolumab combination checkpoint inhibitor",
    "hepatocellular carcinoma sorafenib atezolizumab bevacizumab",
    "multiple myeloma bortezomib lenalidomide daratumumab",
    "renal cell carcinoma sunitinib nivolumab cabozantinib",
    "tumor microenvironment checkpoint inhibitor T cell resistance",
    "CAR-T cell therapy hematologic malignancy efficacy",
    "RECIST response criteria solid tumors imaging assessment",
    "cancer precision medicine biomarker targeted therapy",
    "immunotherapy toxicity adverse events management irAE",
    # Healthcare operations (10)
    "hospital readmission reduction quality improvement CMS",
    "patient safety medication error prevention culture",
    "sepsis management bundle compliance mortality outcomes",
    "electronic health records clinical decision support physician",
    "nursing shortage healthcare workforce retention burnout",
    "hospital acquired infection prevention bundle compliance",
    "value-based care bundled payment episode outcomes",
    "telehealth telemedicine patient outcomes chronic disease",
    "emergency department length of stay throughput efficiency",
    "surgical site infection perioperative care prevention",
    # Health economics (8)
    "Medicare DRG hospital inpatient payment rate reimbursement",
    "pharmaceutical drug pricing cost-effectiveness healthcare",
    "insurance coverage disparities access healthcare outcomes",
    "hospital operating margin financial performance efficiency",
    "biosimilar biologic drug competition cost savings",
    "EHR implementation cost benefit healthcare organization",
    "prescription drug adherence outcomes cost-effectiveness",
    "Medicaid managed care outcomes quality performance",
    # Regulatory / safety / methods (7)
    "FDA accelerated approval breakthrough oncology regulation",
    "pharmacovigilance adverse event post-market drug safety",
    "randomized controlled trial design endpoint primary",
    "HIPAA data breach healthcare privacy security",
    "hospital accreditation Joint Commission quality standards",
    "drug-drug interaction pharmacokinetics CYP450",
    "systematic review meta-analysis evidence-based medicine",
    # Cardiology (16)
    "atrial fibrillation ablation antiarrhythmic stroke prevention",
    "heart failure reduced ejection fraction HFrEF sacubitril valsartan",
    "heart failure preserved ejection fraction HFpEF spironolactone empagliflozin",
    "acute myocardial infarction STEMI primary PCI reperfusion",
    "cardiogenic shock mechanical circulatory support IABP Impella outcomes",
    "atrial fibrillation anticoagulation direct oral anticoagulant DOAC",
    "heart failure hospitalization readmission guideline directed therapy",
    "coronary artery disease revascularization CABG PCI outcomes",
    "cardiac biomarkers troponin BNP heart failure diagnosis prognosis",
    "implantable cardioverter defibrillator ICD sudden cardiac death prevention",
    "hypertension antihypertensive therapy cardiovascular risk reduction",
    "valvular heart disease transcatheter aortic valve TAVR SAVR outcomes",
    "pulmonary arterial hypertension endothelin prostacyclin PDE5 inhibitor",
    "heart failure with mildly reduced ejection fraction HFmrEF treatment",
    "acute heart failure decompensated diuresis inotrope vasodilator",
    "ventricular tachycardia catheter ablation antiarrhythmic sudden death",
    # Neurology (16)
    "Parkinson disease levodopa dopaminergic therapy motor fluctuations",
    "Alzheimer disease amyloid beta tau biomarker clinical trial",
    "multiple sclerosis natalizumab ocrelizumab relapse remission",
    "epilepsy drug resistant seizure levetiracetam lamotrigine",
    "ischemic stroke thrombolysis thrombectomy reperfusion outcomes",
    "glioblastoma temozolomide bevacizumab MGMT methylation survival",
    "Parkinson disease deep brain stimulation quality of life",
    "Alzheimer dementia cognitive decline prevention lifestyle intervention",
    "multiple sclerosis progressive disability neurodegeneration neuroprotection",
    "epilepsy ketogenic diet vagus nerve stimulation refractory seizure",
    "stroke rehabilitation functional recovery motor cortex neuroplasticity",
    "migraine CGRP monoclonal antibody preventive treatment efficacy",
    "amyotrophic lateral sclerosis ALS riluzole edaravone survival",
    "Alzheimer disease lecanemab donanemab amyloid plaque removal",
    "multiple sclerosis ofatumumab cladribine oral disease modifying",
    "Parkinson disease alpha-synuclein neuroprotection disease modification",
    # Infectious Disease (14)
    "HIV antiretroviral therapy ART viral suppression CD4 outcomes",
    "hepatitis C direct acting antiviral DAA sofosbuvir ledipasvir cure",
    "sepsis antibiotic therapy blood culture organ dysfunction mortality",
    "Clostridioides difficile fecal microbiota transplant recurrence treatment",
    "COVID-19 antiviral nirmatrelvir paxlovid hospitalization reduction",
    "HIV PrEP pre-exposure prophylaxis tenofovir emtricitabine prevention",
    "hepatitis C genotype treatment duration sustained virologic response",
    "sepsis surviving sepsis campaign bundle fluid resuscitation vasopressor",
    "COVID-19 vaccine mRNA effectiveness hospitalization severe disease",
    "antimicrobial resistance carbapenem ESKAPE pathogen clinical outcomes",
    "influenza antiviral oseltamivir zanamivir treatment hospitalization",
    "COVID-19 long COVID post-acute sequelae PASC symptoms outcomes",
    "HIV integrase inhibitor bictegravir dolutegravir switch regimen",
    "C difficile fidaxomicin bezlotoxumab recurrence prevention",
    # Endocrinology (12)
    "type 1 diabetes closed loop insulin pump continuous glucose monitor",
    "obesity GLP-1 receptor agonist semaglutide weight loss cardiovascular",
    "thyroid cancer papillary differentiated radioiodine thyroidectomy",
    "Cushing syndrome adrenal hypercortisolism diagnosis treatment outcomes",
    "type 2 diabetes SGLT2 inhibitor empagliflozin cardiovascular renal",
    "obesity bariatric surgery metabolic outcomes diabetes remission",
    "thyroid nodule fine needle aspiration biopsy malignancy risk",
    "adrenal insufficiency glucocorticoid replacement therapy outcomes",
    "hyperparathyroidism cinacalcet parathyroidectomy calcium PTH",
    "polycystic ovary syndrome PCOS metformin letrozole ovulation induction",
    "non-alcoholic fatty liver disease NAFLD NASH fibrosis treatment",
    "diabetes ketoacidosis DKA management insulin infusion bicarbonate",
    # Hematology (12)
    "acute myeloid leukemia AML venetoclax azacitidine induction remission",
    "chronic lymphocytic leukemia CLL ibrutinib venetoclax BTK inhibitor",
    "diffuse large B-cell lymphoma DLBCL R-CHOP rituximab survival",
    "sickle cell disease hydroxyurea voxelotor crizanlizumab outcomes",
    "AML FLT3 mutation midostaurin gilteritinib targeted therapy",
    "CLL minimal residual disease MRD fixed duration venetoclax obinutuzumab",
    "NHL CAR-T axicabtagene ciloleucel refractory large B-cell lymphoma",
    "sickle cell gene therapy lentiviral vector stem cell transplant",
    "myelodysplastic syndrome MDS lenalidomide azacitidine transfusion",
    "acute lymphoblastic leukemia ALL blinatumomab inotuzumab CAR-T",
    "polycythemia vera ruxolitinib phlebotomy thrombosis hydroxyurea",
    "venous thromboembolism DVT pulmonary embolism anticoagulation DOAC",
    # Rare Disease (12)
    "cystic fibrosis CFTR modulator elexacaftor tezacaftor ivacaftor lung function",
    "Gaucher disease enzyme replacement therapy imiglucerase substrate reduction",
    "Fabry disease alpha-galactosidase agalsidase renal cardiac outcomes",
    "phenylketonuria PKU phenylalanine hydroxylase sapropterin dietary management",
    "cystic fibrosis pulmonary exacerbation antibiotic hospitalization outcomes",
    "Gaucher type 1 eliglustat substrate reduction therapy bone outcomes",
    "Fabry disease migalastat pharmacological chaperone quality of life",
    "rare genetic disorder natural history registry real-world evidence",
    "spinal muscular atrophy SMA nusinersen risdiplam gene therapy outcomes",
    "Duchenne muscular dystrophy exon skipping antisense oligonucleotide",
    "hereditary transthyretin amyloidosis tafamidis patisiran gene silencing",
    "Pompe disease acid alpha-glucosidase enzyme replacement respiratory",
    # Health Services Research (14)
    "hospital length of stay discharge planning care coordination reduction",
    "30-day hospital readmission risk prediction machine learning EHR",
    "care transitions post-acute skilled nursing facility outcomes",
    "patient experience HCAHPS scores hospital quality satisfaction",
    "health disparities racial ethnic minority access quality outcomes",
    "accountable care organization ACO shared savings quality performance",
    "social determinants of health food insecurity housing outcomes",
    "primary care patient centered medical home chronic disease management",
    "hospital volume outcome relationship surgery mortality quality",
    "nurse staffing ratios patient safety mortality hospital outcomes",
    "emergency department boarding overcrowding patient safety throughput",
    "hospital at home acute care substitution outcomes readmission",
    "care coordination fragmented care transitions chronic disease elderly",
    "HCAHPS domain communication nurse physician overall hospital rating",
    # Pharmacoeconomics (14)
    "cost-effectiveness analysis QALY ICER threshold willingness to pay",
    "budget impact model pharmaceutical formulary payer decision",
    "real-world evidence claims database treatment effectiveness outcomes",
    "medication adherence persistence chronic disease outcomes cost",
    "indirect comparison network meta-analysis relative effectiveness",
    "patient reported outcomes PRO quality of life oncology clinical trial",
    "value-based pricing outcomes-based contract pharmaceutical",
    "healthcare resource utilization hospitalization costs chronic disease",
    "pharmacoeconomic model Markov cohort simulation lifetime horizon",
    "generic drug substitution cost savings adherence therapeutic equivalence",
    "population-level health economic model burden of disease productivity",
    "cost utility analysis willingness to pay threshold oncology specialty",
    "claims analysis administrative database adherence medication possession ratio",
    "comparative effectiveness research active comparator new user design",
    # Clinical Informatics (12)
    "natural language processing NLP clinical notes extraction information",
    "ICD-10 coding automation machine learning clinical documentation",
    "predictive modeling EHR readmission sepsis early warning alert",
    "clinical decision support interruptive alert medication safety override",
    "electronic health record interoperability FHIR HL7 data exchange",
    "machine learning imaging radiology pathology diagnosis AI",
    "clinical data warehouse analytics population health management",
    "NLP named entity recognition medication adverse event extraction",
    "federated learning clinical AI model privacy preserving health data",
    "large language model clinical text summarization discharge note",
    "patient matching record linkage probabilistic deduplication EHR",
    "precision oncology tumor mutational burden microsatellite instability biomarker",
    # Regulatory Science (14)
    "FDA breakthrough therapy designation accelerated approval clinical benefit",
    "EMA CHMP assessment report marketing authorisation conditional approval",
    "real-world evidence regulatory submission FDA guidance post-approval",
    "post-market surveillance safety signal detection pharmacovigilance AI",
    "FDA adaptive clinical trial design Bayesian platform master protocol",
    "biosimilar approval totality of evidence immunogenicity clinical study",
    "pediatric drug development extrapolation FDA PREA written request",
    "digital health software medical device FDA De Novo regulatory pathway",
    "expedited drug approval pathway orphan drug rare disease designation",
    "risk evaluation mitigation strategy REMS opioid safe use prescriber",
    "FDA real-world data evidence framework post-market study drug",
    "EMA adaptive pathways iterative development access medicine",
    "drug product quality manufacturing GMP compliance FDA inspection",
    "combination product drug device FDA regulatory pathway premarket approval",
    # Oncology – additional (6)
    "bladder cancer checkpoint inhibitor atezolizumab erdafitinib FGFR",
    "gastric gastroesophageal junction cancer HER2 trastuzumab nivolumab",
    "endometrial cancer mismatch repair deficient pembrolizumab dostarlimab",
    "cervical cancer pembrolizumab bevacizumab chemotherapy survival",
    "ovarian cancer VEGF bevacizumab anti-angiogenic chemotherapy maintenance",
    "thyroid cancer lenvatinib sorafenib differentiated medullary outcomes",
    # Pulmonology (6)
    "COPD exacerbation triple inhaler therapy ICS LABA LAMA outcomes",
    "idiopathic pulmonary fibrosis nintedanib pirfenidone antifibrotic progression",
    "asthma biologic dupilumab mepolizumab type 2 inflammation eosinophil",
    "pulmonary embolism direct oral anticoagulant rivaroxaban apixaban",
    "obstructive sleep apnea CPAP cardiovascular risk reduction adherence",
    "lung transplant chronic rejection bronchiolitis obliterans immunosuppression",
    # Gastroenterology (6)
    "Crohn disease vedolizumab ustekinumab anti-TNF biologic remission",
    "ulcerative colitis tofacitinib ozanimod filgotinib JAK inhibitor",
    "hepatocellular carcinoma Barcelona staging TACE systemic therapy",
    "primary biliary cholangitis obeticholic acid ursodeoxycholic acid",
    "inflammatory bowel disease treat-to-target mucosal healing outcomes",
    "colorectal cancer screening colonoscopy FIT stool test adherence",
    # Nephrology / Rheumatology (6)
    "chronic kidney disease progression SGLT2 inhibitor finerenone outcomes",
    "lupus nephritis voclosporin belimumab anifrolumab renal outcomes",
    "rheumatoid arthritis JAK inhibitor upadacitinib baricitinib tofacitinib",
    "IgA nephropathy sparsentan atrasentan RAS blockade proteinuria",
    "ANCA vasculitis avacopan cyclophosphamide rituximab remission",
    "psoriatic arthritis secukinumab ixekizumab biologic disease control",
]
_NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _fetch_pubmed_ids(query: str, n: int) -> list[str]:
    url = (f"{_NCBI_BASE}/esearch.fcgi?db=pubmed"
           f"&term={urllib.parse.quote(query)}"
           f"&retmax={n}&retmode=json&sort=relevance")
    data = json.loads(_http_get(url))
    return data.get("esearchresult", {}).get("idlist", [])


def _fetch_pubmed_xml(ids: list[str]) -> str:
    url = (f"{_NCBI_BASE}/efetch.fcgi?db=pubmed"
           f"&id={','.join(ids)}&rettype=abstract&retmode=xml")
    return _http_get(url).decode("utf-8", errors="replace")


def _parse_pubmed_xml(xml: str) -> list[dict]:
    articles = []
    for block in re.finditer(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.S):
        b = block.group(1)
        def _tag(pattern: str) -> str:
            m = re.search(pattern, b, re.S)
            return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else ""
        pmid    = _tag(r"<PMID[^>]*>(\d+)</PMID>")
        title   = _tag(r"<ArticleTitle>(.*?)</ArticleTitle>")
        year    = _tag(r"<PubDate>.*?<Year>(\d+)</Year>.*?</PubDate>")
        journal = _tag(r"<Title>(.*?)</Title>")
        ab_parts = re.findall(
            r'<AbstractText(?:\s+Label="([^"]*)")?[^>]*>(.*?)</AbstractText>', b, re.S)
        abstract = "\n\n".join(
            (f"**{lbl}**\n{re.sub(r'<[^>]+>',' ',txt).strip()}" if lbl
             else re.sub(r"<[^>]+>", " ", txt).strip())
            for lbl, txt in ab_parts
        )
        if pmid and abstract:
            articles.append({"pmid": pmid, "title": title, "year": year,
                              "journal": journal, "abstract": abstract})
    return articles


def _build_pubmed(lib_dir: Path, n_per_topic: int = 200) -> None:
    raw_dir = lib_dir / "_raw"
    out_dir = lib_dir / "pubmed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for topic in _PUBMED_TOPICS:
        slug       = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:40]
        cache_file = raw_dir / f"pm_{slug}.json"
        if cache_file.exists():
            articles = json.loads(cache_file.read_text())
        else:
            ids = _fetch_pubmed_ids(topic, n_per_topic)
            if not ids:
                continue
            xml      = _fetch_pubmed_xml(ids)
            articles = _parse_pubmed_xml(xml)
            cache_file.write_text(json.dumps(articles))
            time.sleep(0.35)
        for art in articles:
            pmid     = art["pmid"]
            txt_file = out_dir / f"pubmed_{pmid}.txt"
            if txt_file.exists():
                continue
            title    = art.get("title", "Untitled")
            year     = art.get("year", "")
            journal  = art.get("journal", "")
            abstract = art.get("abstract", "")
            if not abstract:
                continue
            meta = f"*{journal}, {year}*" if journal else f"*{year}*"
            text = f"# {title}\n\n{meta}  (PMID: {pmid})\n\n## Abstract\n\n{abstract}"
            txt_file.write_text(text, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# FDA drug labels
# ─────────────────────────────────────────────────────────────────────────────

_OPENFDA_URL = "https://api.fda.gov/drug/label.json"
_LABEL_SECTIONS = [
    ("indications_and_usage",     "Indications and Usage"),
    ("warnings_and_cautions",     "Warnings and Precautions"),
    ("adverse_reactions",         "Adverse Reactions"),
    ("dosage_and_administration", "Dosage and Administration"),
    ("clinical_pharmacology",     "Clinical Pharmacology"),
    ("clinical_studies",          "Clinical Studies"),
    ("mechanism_of_action",       "Mechanism of Action"),
    ("contraindications",         "Contraindications"),
]


def _fetch_fda_page(skip: int, limit: int = 100) -> list[dict]:
    url = (f"{_OPENFDA_URL}?search=openfda.product_type:PRESCRIPTION"
           f"&limit={limit}&skip={skip}")
    try:
        data = json.loads(_http_get(url, timeout=20))
        return data.get("results", [])
    except Exception as exc:
        print(f"      WARNING FDA skip={skip}: {exc}")
        return []


def _fda_label_to_markdown(label: dict) -> tuple[str, str] | None:
    openfda = label.get("openfda", {})
    generic = (openfda.get("generic_name") or [None])[0]
    brand   = (openfda.get("brand_name")   or [None])[0]
    if not generic and not brand:
        return None
    name_slug = re.sub(r"[^a-z0-9]+", "_", (generic or brand).lower())[:32]
    app_no    = (openfda.get("application_number") or [""])[0].replace("/","_")
    doc_name  = f"fda_{name_slug}_{app_no}" if app_no else f"fda_{name_slug}"
    lines = [f"# {(generic or '').title()} ({brand}) — FDA Drug Label",
             f"\n**Generic:** {generic}  **Brand:** {brand}"]
    for field_key, section_title in _LABEL_SECTIONS:
        content = label.get(field_key, [])
        if content:
            text = re.sub(r"<[^>]+>", " ", " ".join(content))
            text = re.sub(r"\s{3,}", "\n\n", text).strip()
            if text:
                lines.append(f"\n## {section_title}\n\n{text[:2500]}")
    if len(lines) <= 2:
        return None
    return doc_name, "\n".join(lines)


def _build_fda(lib_dir: Path, n_pages: int = 40) -> None:
    raw_dir    = lib_dir / "_raw"
    out_dir    = lib_dir / "fda"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_names: set[str] = set()
    # Pre-populate seen_names from existing txt files to avoid duplicates
    for f in out_dir.glob("*.txt"):
        seen_names.add(f.stem)
    for page in range(n_pages):
        skip       = page * 100
        cache_file = raw_dir / f"fda_page_{page}.json"
        if cache_file.exists():
            labels = json.loads(cache_file.read_text())
        else:
            labels = _fetch_fda_page(skip)
            if not labels:
                break
            cache_file.write_text(json.dumps(labels))
            time.sleep(0.4)
        for label in labels:
            parsed = _fda_label_to_markdown(label)
            if parsed is None:
                continue
            doc_name, text = parsed
            if doc_name in seen_names:
                continue
            seen_names.add(doc_name)
            txt_file = out_dir / f"{doc_name}.txt"
            if not txt_file.exists():
                txt_file.write_text(text, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Python stdlib docs — 134 modules
# ─────────────────────────────────────────────────────────────────────────────

_PY_BASE = "https://docs.python.org/3/library"
_PYTHON_MODULES = [
    # Logging
    "logging","logging.handlers","logging.config",
    # File system
    "pathlib","os.path","shutil","glob","filecmp","fileinput","fnmatch","linecache",
    # Data / typing
    "dataclasses","typing","functools","itertools","operator","copy","pprint","reprlib",
    # Concurrency
    "asyncio","threading","multiprocessing","concurrent.futures","queue",
    "subprocess","signal","contextvars","selectors",
    # Storage
    "sqlite3","csv","json","pickle","shelve","dbm","struct","codecs",
    # Network / web
    "http.client","http.server","urllib.request","urllib.parse","urllib.error",
    "socket","ssl","socketserver","xmlrpc.client","xmlrpc.server",
    # Encoding / crypto
    "hashlib","hmac","secrets","base64","binascii","quopri",
    # Date / time
    "datetime","time","calendar","zoneinfo","timeit",
    # Collections / algorithms
    "collections","collections.abc","heapq","bisect","array","weakref","gc",
    # Text
    "re","textwrap","string","difflib","unicodedata","readline",
    # CLI / config
    "argparse","configparser","getopt","getpass","cmd",
    # Testing / debugging
    "unittest","unittest.mock","doctest","pdb","trace","cProfile","profile","timeit","faulthandler",
    # OO / introspection
    "abc","io","contextlib","inspect","ast","dis","importlib","pkgutil","importlib.metadata",
    # OS / platform
    "os","sys","platform","site","sysconfig","tempfile","zipfile","tarfile","gzip","bz2","lzma",
    # Misc stdlib
    "enum","uuid","random","statistics","decimal","math","fractions","numbers","cmath",
    "traceback","warnings","keyword","token","tokenize","compileall",
    # Email / internet
    "email","smtplib","imaplib","mailbox","mimetypes","html.parser",
    "xml.etree.ElementTree","urllib.robotparser","ipaddress",
    # Compression / archive
    "zipimport","zipapp","venv",
    # Numeric extras
    "fractions","array","struct",
]


def _build_python(lib_dir: Path) -> None:
    from chunkymonkey.extractors._html import HtmlExtractor
    raw_dir   = lib_dir / "_raw"
    out_dir   = lib_dir / "python"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = HtmlExtractor()
    for mod in _PYTHON_MODULES:
        safe_name = mod.replace(".", "_")
        txt_file  = out_dir / f"pydoc_{safe_name}.txt"
        if txt_file.exists():
            continue
        raw_file = raw_dir / f"py_{safe_name}.html"
        url      = f"{_PY_BASE}/{mod}.html"
        if raw_file.exists():
            raw = raw_file.read_bytes()
        else:
            try:
                raw = _http_get(url)
                raw_file.write_bytes(raw)
                time.sleep(0.2)
            except Exception as exc:
                print(f"      SKIP py/{mod}: {exc}")
                continue
        try:
            text = extractor.extract(raw, url)
            if text.strip():
                txt_file.write_text(text, encoding="utf-8")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Build command
# ─────────────────────────────────────────────────────────────────────────────

def cmd_build(args: argparse.Namespace) -> None:
    lib_dir = Path(args.library)
    lib_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building library at {lib_dir}")

    print("  EDGAR 10-K filings...")
    _build_edgar(lib_dir)

    print("  ClinicalTrials protocols...")
    _build_ct(lib_dir, n_trials=args.n_ct_trials)

    print("  Federal Register...")
    _build_fr(lib_dir, n_pages=args.n_fr_pages)

    print("  PubMed abstracts...")
    _build_pubmed(lib_dir, n_per_topic=args.n_pubmed)

    print("  FDA drug labels...")
    _build_fda(lib_dir, n_pages=args.n_fda_pages)

    print("  Python stdlib docs...")
    _build_python(lib_dir)

    # Print summary
    print("\nLibrary summary:")
    for subdir in ("edgar", "ct", "fr", "pubmed", "fda", "python"):
        d = lib_dir / subdir
        n = len(list(d.glob("*.txt"))) if d.exists() else 0
        print(f"  {subdir:<10} {n:>5} docs")


# ─────────────────────────────────────────────────────────────────────────────
# Query set — 38 queries
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Query:
    text: str
    category: str
    target_docs: list[str]
    target_sections: list[str]
    note: str = ""


QUERIES: list[Query] = [
    # ── SCOPED (14) ──────────────────────────────────────────────────────────
    Query("HCA Healthcare net revenue growth admissions same-facility hospitals",
          "scoped", ["hca"], ["Revenue","7","Admissions"], "financial-scoped"),
    Query("Tenet Healthcare uncompensated care charity bad debt provision",
          "scoped", ["thc"], ["Charity","Uncompensated","Bad Debt","7"], "financial-scoped"),
    Query("Universal Health Services behavioral health psychiatric inpatient",
          "scoped", ["uhs"], ["Behavioral","Psychiatric","Segment","1"], "financial-scoped"),
    Query("Community Health Systems rural hospital acute care market competition",
          "scoped", ["cyh"], ["Competition","Rural","Market","1"], "financial-scoped"),
    Query("NSCLC non-small cell lung cancer pembrolizumab PD-L1 TPS eligibility",
          "scoped", ["nsclc","lung"], ["Eligibility","Inclusion"], "clinical-scoped"),
    Query("breast cancer HER2 trastuzumab pertuzumab pathological complete response",
          "scoped", ["breast"], ["Primary Outcome","Brief Summary"], "clinical-scoped"),
    Query("ovarian cancer BRCA mutation PARP inhibitor olaparib maintenance",
          "scoped", ["ovarian","ovar"], ["Eligibility","Intervention","Brief Summary"], "clinical-scoped"),
    Query("CMS inpatient hospital Medicare DRG payment rate wage index IPPS",
          "scoped", ["fr_cms_hospital"], ["DRG","Payment","Rate","Summary"], "fr-scoped"),
    Query("OIG hospital billing upcoding false claims corporate integrity",
          "scoped", ["fr_oig"], ["Compliance","Fraud","False Claims"], "fr-scoped"),
    Query("pembrolizumab KEYTRUDA FDA indications approved cancer checkpoint",
          "scoped", ["fda_pembrolizumab"], ["Indications","Usage"], "fda-scoped"),
    Query("oxaliplatin ELOXATIN adverse reactions peripheral neuropathy toxicity",
          "scoped", ["fda_oxaliplatin"], ["Adverse Reactions","Warnings"], "fda-scoped"),
    Query("Pfizer revenue oncology biopharmaceutical segment product sales",
          "scoped", ["pfe"], ["Revenue","Segment","7"], "pharma-financial-scoped"),
    Query("UnitedHealth Group Optum health services revenue growth margin",
          "scoped", ["unh"], ["Optum","Revenue","7","Segment"], "insurance-scoped"),
    Query("Python logging handlers formatters log levels basicConfig",
          "scoped", ["pydoc_logging"], ["logging","Handler","Formatter"], "tech-scoped"),
    # ── SYNTHESIS (12) ───────────────────────────────────────────────────────
    Query("Medicare reimbursement CMS payment rule hospital margin financial impact",
          "synthesis", [], ["Revenue","Reimbursement","Payment","7","CMS"], "cross-domain"),
    Query("pembrolizumab clinical evidence FDA approval mechanism cancer",
          "synthesis", [], ["Approval","Clinical","Mechanism","Indications"], "cross-domain"),
    Query("HIPAA cybersecurity breach patient data healthcare compliance risk",
          "synthesis", [], ["HIPAA","Cybersecurity","Privacy","Risk","Security"], "cross-domain"),
    Query("cancer drug adverse events patient safety management clinical practice",
          "synthesis", [], ["Adverse","Toxicity","Management","Clinical","Safety"], "cross-domain"),
    Query("hospital operating cost inflation labor nursing shortage workforce",
          "synthesis", [], ["Labor","Cost","Workforce","Operating","7"], "within-financial"),
    Query("oncology eligibility ECOG performance status organ function lab threshold",
          "synthesis", [], ["Eligibility","Inclusion","ECOG","Performance"], "within-clinical"),
    Query("clinical trial primary endpoint RECIST objective response overall survival",
          "synthesis", [], ["Primary Outcome","RECIST","Response","Survival"], "within-clinical"),
    Query("Python async await coroutine event loop concurrency",
          "synthesis", [], ["asyncio","coroutine","event loop","await"], "within-tech"),
    Query("HER2 breast cancer targeted therapy dual blockade survival benefit",
          "synthesis", [], ["HER2","Breast","Survival","Trastuzumab","Pertuzumab"], "research+clinical"),
    Query("FOLFOX colorectal cancer chemotherapy survival toxicity neuropathy",
          "synthesis", [], ["FOLFOX","Colorectal","Survival","Toxicity"], "research+clinical"),
    Query("healthcare data breach notification HIPAA OCR investigation penalty",
          "synthesis", [], ["Breach","Notification","HIPAA","OCR"], "regulatory+operational"),
    Query("FDA pharmacovigilance post-market surveillance drug safety reporting",
          "synthesis", [], ["Safety","Pharmacovigilance","Post-Market","Surveillance"], "fr+fda"),
    # ── BROAD (12) ───────────────────────────────────────────────────────────
    Query("risk factors regulatory compliance government programs operations",
          "broad", [], ["1A","Risk","Regulatory","Compliance"], "general"),
    Query("patient outcomes quality performance measurement healthcare",
          "broad", [], ["Quality","Outcome","Performance","Patient"], "general"),
    Query("insurance payer reimbursement revenue cycle billing coverage",
          "broad", [], ["Insurance","Payer","Revenue","Billing","Coverage"], "general"),
    Query("workforce staffing labor shortage retention culture diversity",
          "broad", [], ["Workforce","Labor","Staffing","Talent","Human Capital"], "general"),
    Query("technology software system data security architecture",
          "broad", [], ["Technology","Software","System","Data","Architecture"], "general"),
    Query("adverse events reporting disclosure material",
          "broad", [], ["Adverse","Event","Safety","Risk"],
          "DISAMBIGUATION: patient safety (CT/FDA) vs financial events (10-K) vs regulatory (FR)"),
    Query("primary outcomes results key performance",
          "broad", [], ["Primary","Outcome","Performance","Result"],
          "DISAMBIGUATION: KPIs (10-K) vs clinical trial endpoints (CT)"),
    Query("sponsor filing disclosure registration",
          "broad", [], ["Sponsor","Filing","Registration","Disclosure"],
          "DISAMBIGUATION: financial sponsor (10-K) vs trial sponsor (CT)"),
    Query("protocol monitoring audit compliance review",
          "broad", [], ["Protocol","Monitoring","Audit","Compliance"],
          "DISAMBIGUATION: internal controls (10-K/9A) vs clinical monitoring (CT) vs regulatory (FR)"),
    Query("intervention treatment therapy mechanism action",
          "broad", [], ["Intervention","Treatment","Mechanism","Action"],
          "DISAMBIGUATION: business interventions (10-K) vs medical interventions (CT/FDA)"),
    Query("logging records documentation data retention storage",
          "broad", [], ["logging","Record","Documentation","Data","Retention"],
          "DISAMBIGUATION: Python logging module vs medical record retention (10-K/FR)"),
    Query("exception handling error failure recovery resilience",
          "broad", [], ["Exception","Error","Failure","Recovery","Resilience"],
          "DISAMBIGUATION: Python exceptions (pydoc) vs operational failures (10-K) vs AE (CT)"),
]

_CAT_WEIGHT = {"scoped": 0.40, "synthesis": 0.30, "broad": 0.30}
_C_WEIGHT   = {"C1": 0.30, "C2": 0.15, "C3": 0.20, "C4": 0.25, "C5": 0.10}

# ─────────────────────────────────────────────────────────────────────────────
# Chunking strategies
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Strategy:
    name: str
    min_size: int
    max_size: int
    breadcrumb: bool


_STRATEGIES = [
    Strategy("ctx-400",    0,   400,  True),
    Strategy("naive-400",  0,   400,  False),
    Strategy("ctx-800",    0,   800,  True),
    Strategy("naive-800",  0,   800,  False),
    Strategy("ctx-1200",   400, 1200, True),
    Strategy("ctx-1600",   0,   1600, True),
    Strategy("naive-1600", 0,   1600, False),
]


def _chunk_corpus(corpus: list[tuple[str, str]], s: Strategy) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for doc_name, text in corpus:
        chunks.extend(chunk_document(doc_name, text,
                                     min_chunk_size=s.min_size,
                                     max_chunk_size=s.max_size,
                                     include_breadcrumb=s.breadcrumb))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Embedding + retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _embed(model, chunks: list[DocumentChunk]) -> np.ndarray:
    print(f"    embedding {len(chunks):,} chunks ...", end=" ", flush=True)
    t0 = time.time()
    vecs = model.encode([c.content for c in chunks], batch_size=128,
                        normalize_embeddings=True, show_progress_bar=False)
    print(f"{time.time()-t0:.0f}s")
    return vecs


def _retrieve(query: str, vecs: np.ndarray, model, k: int) -> list[int]:
    qv = model.encode([query], normalize_embeddings=True)[0]
    return list(np.argsort(vecs @ qv)[::-1][:k])


# ─────────────────────────────────────────────────────────────────────────────
# Relevance + component scorers
# ─────────────────────────────────────────────────────────────────────────────

def _relevance(chunk: DocumentChunk, q: Query) -> int:
    doc     = (chunk.document_name or "").lower()
    sec     = (chunk.section or "").lower()
    content = chunk.content.lower()
    dm = not q.target_docs or any(d.lower() in doc for d in q.target_docs)
    sm = not q.target_sections or any(
        s.lower() in sec or s.lower() in content for s in q.target_sections)
    return (2 if dm and sm else 1 if sm else 0)


def _is_distractor(chunk: DocumentChunk, q: Query) -> bool:
    if not q.target_docs:
        return False
    doc  = (chunk.document_name or "").lower()
    sec  = (chunk.section or "").lower()
    cont = chunk.content.lower()
    dm   = any(d.lower() in doc for d in q.target_docs)
    sm   = q.target_sections and any(
        s.lower() in sec or s.lower() in cont for s in q.target_sections)
    return bool(sm) and not dm


_CONT_RE = re.compile(r"\[(TABLE|LIST|PARA):(cont|start|end)\]", re.I)
_TERM_RE = re.compile(r"[.?!\"')\]>}\d]\s*$")


def c1_ndcg(retrieved: list[DocumentChunk], q: Query, k: int) -> float:
    rels  = [_relevance(c, q) for c in retrieved[:k]]
    ideal = sorted(rels, reverse=True)
    def dcg(s): return sum(r / math.log2(i + 2) for i, r in enumerate(s))
    idcg = dcg(ideal)
    return dcg(rels) / idcg if idcg else 0.0


def c2_coherence(retrieved: list[DocumentChunk], k: int) -> float:
    ok = 0
    for c in retrieved[:k]:
        text = c.content.strip()
        if _CONT_RE.match(text.split("\n")[0] if text else ""):
            continue
        prose = [l for l in text.split("\n") if l.strip() and not l.strip().startswith("[")]
        last  = prose[-1].strip() if prose else ""
        if last and not _TERM_RE.search(last):
            continue
        ok += 1
    return ok / k if k else 0.0


def c3_scope(retrieved: list[DocumentChunk], q: Query, k: int) -> float:
    if not q.target_docs:
        return 1.0
    penalty = sum(1.0 / math.log2(i + 2)
                  for i, c in enumerate(retrieved[:k]) if _is_distractor(c, q))
    return max(0.0, 1.0 - penalty)


def c4_answer_sufficiency(retrieved, q, k, client=None):
    if client is None:
        return None
    context = "\n\n---\n\n".join(c.content[:400] for c in retrieved[:k])
    prompt  = (f"Query: {q.text}\n\nContext:\n{context}\n\n"
               "Rate 2 (correct/complete), 1 (partial), 0 (wrong/unanswerable). Single integer only.")
    try:
        import os
        msg   = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=4,
                                       messages=[{"role":"user","content":prompt}])
        score = int(re.search(r"[012]", msg.content[0].text).group())
        return score / 2.0
    except Exception:
        return None


def c5_efficiency(retrieved: list[DocumentChunk], q: Query, k: int) -> float:
    total    = sum(len(c.content.split()) for c in retrieved[:k])
    relevant = sum(len(c.content.split()) for c in retrieved[:k] if _relevance(c, q) > 0)
    return relevant / total if total else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# RQI
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QR:
    query: Query
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float | None = None
    c5: float = 0.0


def _rqi(results: list[QR], use_c4: bool) -> float:
    w = dict(_C_WEIGHT)
    if not use_c4:
        tot = sum(v for k, v in w.items() if k != "C4")
        w   = {k: (v/tot if k != "C4" else 0.0) for k, v in w.items()}
    cat: dict[str, list[float]] = {c: [] for c in _CAT_WEIGHT}
    for r in results:
        comps = {"C1": r.c1, "C2": r.c2, "C3": r.c3, "C5": r.c5}
        if use_c4 and r.c4 is not None:
            comps["C4"] = r.c4
        cat[r.query.category].append(sum(w.get(k, 0)*v for k, v in comps.items()))
    return sum(_CAT_WEIGHT[c] * (sum(s)/len(s)) for c, s in cat.items() if s)


def _mean(vs): return sum(v for v in vs if v is not None) / max(1, sum(1 for v in vs if v is not None))


# ─────────────────────────────────────────────────────────────────────────────
# Run command
# ─────────────────────────────────────────────────────────────────────────────

_TYPE_DIRS = ["edgar", "ct", "fr", "pubmed", "fda", "python"]


def _load_type(lib_dir: Path, type_name: str, n: int, rng: random.Random) -> list[tuple[str, str]]:
    d = lib_dir / type_name
    if not d.exists():
        return []
    files = sorted(d.glob("*.txt"))
    if n >= 0:
        files = rng.sample(files, min(n, len(files)))
    result = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
            if text.strip():
                result.append((f.stem, text))
        except Exception:
            pass
    return result


def cmd_run(args: argparse.Namespace) -> None:
    lib_dir = Path(args.library)
    rng     = random.Random(args.seed)

    # Resolve per-type N
    n_default = args.n_per_type
    n_by_type = {
        "edgar":  args.n_edgar  if args.n_edgar  is not None else n_default,
        "ct":     args.n_ct     if args.n_ct     is not None else n_default,
        "fr":     args.n_fr     if args.n_fr     is not None else n_default,
        "pubmed": args.n_pubmed if args.n_pubmed is not None else n_default,
        "fda":    args.n_fda    if args.n_fda    is not None else n_default,
        "python": args.n_python if args.n_python is not None else n_default,
    }

    # Load corpus
    corpus: list[tuple[str, str]] = []
    print("Corpus stats:")
    print(f"  {'Type':<10}  {'Available':>10}  {'Sampled':>8}")
    print("  " + "─" * 32)
    for t in _TYPE_DIRS:
        d = lib_dir / t
        available = len(list(d.glob("*.txt"))) if d.exists() else 0
        n         = n_by_type[t]
        docs      = _load_type(lib_dir, t, n, rng)
        sampled   = len(docs)
        print(f"  {t:<10}  {available:>10}  {sampled:>8}")
        corpus.extend(docs)
    print(f"  {'TOTAL':<10}  {'':>10}  {len(corpus):>8}")
    print()

    if not corpus:
        print("No documents loaded. Run 'build' first.")
        return

    llm_client = None
    use_c4     = False

    from sentence_transformers import SentenceTransformer
    model_name = getattr(args, "model", "all-MiniLM-L6-v2")
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name)
    K     = args.top_k
    print(f"Queries: {len(QUERIES)}  Top-k: {K}\n")

    all_results:  dict[str, list[QR]] = {}
    chunk_counts: dict[str, int]      = {}

    for strat in _STRATEGIES:
        print(f"Strategy: {strat.name}  (min={strat.min_size} max={strat.max_size} "
              f"breadcrumb={strat.breadcrumb})")
        chunks = _chunk_corpus(corpus, strat)
        chunk_counts[strat.name] = len(chunks)
        print(f"  {len(chunks):,} chunks")
        vecs = _embed(model, chunks)

        results: list[QR] = []
        for q in QUERIES:
            top_idx    = _retrieve(q.text, vecs, model, K)
            top_chunks = [chunks[i] for i in top_idx]
            r = QR(query=q)
            r.c1 = c1_ndcg(top_chunks, q, K)
            r.c2 = c2_coherence(top_chunks, K)
            r.c3 = c3_scope(top_chunks, q, K) if q.category == "scoped" else 1.0
            r.c4 = c4_answer_sufficiency(top_chunks, q, K, llm_client) if use_c4 else None
            r.c5 = c5_efficiency(top_chunks, q, K)
            results.append(r)

        all_results[strat.name] = results
        print(f"  RQI = {_rqi(results, use_c4):.4f}\n")

    # ── Results table ─────────────────────────────────────────────────────────
    w = dict(_C_WEIGHT)
    if not use_c4:
        tot = sum(v for k, v in w.items() if k != "C4")
        w   = {k: (v/tot if k != "C4" else 0.0) for k, v in w.items()}

    col = 10
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    hdr = (f"  {'Strategy':<14}  {'RQI':>{col}}  {'C1 nDCG':>{col}}"
           f"  {'C2 Coher':>{col}}  {'C3 Scope':>{col}}  {'C5 Effic':>{col}}")
    print(hdr)
    print("  " + "─" * (len(hdr)-2))
    for s in _STRATEGIES:
        res = all_results[s.name]
        rqi = _rqi(res, use_c4)
        print(f"  {s.name:<14}  {rqi:>{col}.4f}"
              f"  {_mean([r.c1 for r in res]):>{col}.4f}"
              f"  {_mean([r.c2 for r in res]):>{col}.4f}"
              f"  {_mean([r.c3 for r in res if r.query.category=='scoped']):>{col}.4f}"
              f"  {_mean([r.c5 for r in res]):>{col}.4f}")
    print()

    # Per-category
    print("─" * 72)
    print("  Per-category RQI:")
    ch = f"  {'Strategy':<14}  {'scoped':>{col}}  {'synthesis':>{col}}  {'broad':>{col}}"
    print(ch)
    print("  " + "─" * (len(ch)-2))
    for s in _STRATEGIES:
        res = all_results[s.name]
        row = f"  {s.name:<14}"
        for cat in ("scoped","synthesis","broad"):
            cr = [r for r in res if r.query.category == cat]
            if not cr:
                row += f"  {'N/A':>{col}}"
                continue
            scores = []
            for r in cr:
                comps = {"C1":r.c1,"C2":r.c2,"C3":r.c3,"C5":r.c5}
                if use_c4 and r.c4 is not None:
                    comps["C4"] = r.c4
                scores.append(sum(w.get(k,0)*v for k,v in comps.items()))
            row += f"  {sum(scores)/len(scores):>{col}.4f}"
        print(row)
    print()

    # Disambiguation
    dis = [r for r in next(iter(all_results.values())) if "DISAMBIGUATION" in r.query.note]
    if dis:
        print("─" * 72)
        print("  Disambiguation queries — nDCG@k  C5 efficiency:")
        dh = f"  {'Strategy':<14}  {'nDCG@k':>{col}}  {'C5 Effic':>{col}}"
        print(dh)
        print("  " + "─" * (len(dh)-2))
        for s in _STRATEGIES:
            dq = [r for r in all_results[s.name] if "DISAMBIGUATION" in r.query.note]
            print(f"  {s.name:<14}  {_mean([r.c1 for r in dq]):>{col}.4f}"
                  f"  {_mean([r.c5 for r in dq]):>{col}.4f}")
        print()

    # Breadcrumb advantage
    print("─" * 72)
    print("  Breadcrumb advantage at matched window sizes:")
    mh = (f"  {'Window':<8}  {'ctx RQI':>10}  {'naive RQI':>10}"
          f"  {'ctx chunks':>12}  {'naive chunks':>13}  {'Δ RQI':>8}")
    print(mh)
    print("  " + "─" * (len(mh) - 2))
    _pairs = [("ctx-400", "naive-400"), ("ctx-800", "naive-800"), ("ctx-1600", "naive-1600")]
    for ctx_name, naive_name in _pairs:
        if ctx_name not in all_results or naive_name not in all_results:
            continue
        cr     = _rqi(all_results[ctx_name], use_c4)
        nr     = _rqi(all_results[naive_name], use_c4)
        delta  = cr - nr
        window = ctx_name.split("-")[1]
        cc     = chunk_counts.get(ctx_name, 0)
        nc     = chunk_counts.get(naive_name, 0)
        print(f"  {window:<8}  {cr:>10.4f}  {nr:>10.4f}  {cc:>12,}  {nc:>13,}  {delta:>+8.4f}")
    print()

    # Window-size effect (naive)
    print("  Window-size effect (naive):")
    naive_pairs = [("naive-400", "naive-800"), ("naive-800", "naive-1600")]
    for a_name, b_name in naive_pairs:
        if a_name not in all_results or b_name not in all_results:
            continue
        ar    = _rqi(all_results[a_name], use_c4)
        br    = _rqi(all_results[b_name], use_c4)
        wa    = a_name.split("-")[1]
        wb    = b_name.split("-")[1]
        delta = br - ar
        print(f"    {wa} → {wb}:   naive {ar:.4f} → {br:.4f}  ({delta:+.4f})")
    print()

    # Summary line
    ctx_rqi   = _rqi(all_results["ctx-1200"], use_c4)
    best_name = max((s for s in _STRATEGIES if not s.breadcrumb),
                    key=lambda s: _rqi(all_results[s.name], use_c4)).name
    best_rqi  = _rqi(all_results[best_name], use_c4)
    pct       = (ctx_rqi - best_rqi) / best_rqi * 100 if best_rqi else float("inf")
    print(f"  ctx-1200 (default): {ctx_rqi:.4f}  "
          f"best naive: {best_rqi:.4f} ({best_name})  "
          f"advantage: {pct:+.1f}%")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Sweep command
# ─────────────────────────────────────────────────────────────────────────────

_SWEEP_SCALES = [500, 2_500, 6_000, 12_000, 24_000, 50_000]


# ─────────────────────────────────────────────────────────────────────────────
# Embedding cache (incremental: only embed new docs)
# ─────────────────────────────────────────────────────────────────────────────

def _cache_dir(lib_dir: Path, strat_name: str) -> Path:
    return lib_dir / "_embcache" / strat_name


def _load_emb_cache(
    lib_dir: Path, strat_name: str, model_name: str
) -> tuple[list[DocumentChunk], np.ndarray | None, set[str]]:
    d        = _cache_dir(lib_dir, strat_name)
    meta_f   = d / "meta.json"
    chunks_f = d / "chunks.pkl"
    vecs_f   = d / "vecs.npy"
    if not (meta_f.exists() and chunks_f.exists() and vecs_f.exists()):
        return [], None, set()
    meta = json.loads(meta_f.read_text())
    if meta.get("model") != model_name:
        print(f"    cache model mismatch ({meta.get('model')} vs {model_name}), ignoring")
        return [], None, set()
    with open(chunks_f, "rb") as f:
        chunks = pickle.load(f)
    vecs      = np.load(str(vecs_f))
    doc_names = set(meta.get("doc_names", []))
    return chunks, vecs, doc_names


def _save_emb_cache(
    lib_dir: Path, strat_name: str, model_name: str,
    chunks: list[DocumentChunk], vecs: np.ndarray, doc_names: set[str],
) -> None:
    d = _cache_dir(lib_dir, strat_name)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    np.save(str(d / "vecs.npy"), vecs)
    (d / "meta.json").write_text(json.dumps({
        "model":     model_name,
        "doc_names": sorted(doc_names),
    }))


def _sample_proportional(
    all_docs: dict[str, list[tuple[str, str]]],
    n: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Sample n docs proportionally from available types; min 1 per type."""
    types     = [t for t in _TYPE_DIRS if all_docs.get(t)]
    available = {t: all_docs[t] for t in types}
    total_avail = sum(len(v) for v in available.values())
    if total_avail == 0:
        return []

    # Proportional allocation floored, then distribute remainder
    shares: dict[str, float] = {t: len(available[t]) / total_avail * n for t in types}
    alloc:  dict[str, int]   = {t: max(1, int(shares[t])) for t in types}

    # Cap each type at what's available
    for t in types:
        alloc[t] = min(alloc[t], len(available[t]))

    # Adjust total toward n by scaling largest types
    current = sum(alloc.values())
    if current < n:
        for t in sorted(types, key=lambda x: -len(available[x])):
            gap = n - current
            if gap <= 0:
                break
            can_add = len(available[t]) - alloc[t]
            add     = min(gap, can_add)
            alloc[t] += add
            current  += add

    result: list[tuple[str, str]] = []
    for t in types:
        k     = alloc[t]
        pool  = available[t]
        taken = rng.sample(pool, min(k, len(pool)))
        result.extend(taken)
    return result


def cmd_sweep(args: argparse.Namespace) -> None:
    lib_dir = Path(args.library)

    # Load ALL docs per type
    all_docs: dict[str, list[tuple[str, str]]] = {}
    for t in _TYPE_DIRS:
        d = lib_dir / t
        docs: list[tuple[str, str]] = []
        if d.exists():
            for f in sorted(d.glob("*.txt")):
                try:
                    text = f.read_text(encoding="utf-8")
                    if text.strip():
                        docs.append((f.stem, text))
                except Exception:
                    pass
        all_docs[t] = docs

    total_avail = sum(len(v) for v in all_docs.values())
    full_corpus: list[tuple[str, str]] = [doc for t in _TYPE_DIRS for doc in all_docs[t]]

    from sentence_transformers import SentenceTransformer
    model_name = args.model
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name)
    K      = args.top_k
    use_c4 = False

    print()
    print("=" * 72)
    print(f"Chunky Monkey — Scale Sweep ({len(_SWEEP_SCALES)} points, seed={args.seed})")
    print("=" * 72)
    print(f"Library: {lib_dir}  (total: {total_avail:,} docs available)")
    print()

    # ── Phase 1: chunk + embed (incremental via cache) ────────────────────────
    print("Phase 1: chunk and embed full corpus per strategy (incremental)")
    print("─" * 72)
    # strat_data: name → (chunks, vecs, doc_name→chunk_indices)
    strat_data: dict[str, tuple[list[DocumentChunk], np.ndarray, dict[str, list[int]]]] = {}

    for strat in _STRATEGIES:
        cached_chunks, cached_vecs, cached_names = _load_emb_cache(
            lib_dir, strat.name, model_name)
        new_docs = [(dn, txt) for dn, txt in full_corpus if dn not in cached_names]

        print(f"  {strat.name}  ({len(cached_names):,} cached, {len(new_docs):,} new docs)")

        if new_docs:
            new_chunks = _chunk_corpus(new_docs, strat)
            print(f"    {len(new_chunks):,} new chunks")
            new_vecs = _embed(model, new_chunks)
            if cached_vecs is not None and len(cached_vecs):
                all_chunks: list[DocumentChunk] = cached_chunks + new_chunks
                all_vecs = np.vstack([cached_vecs, new_vecs])
            else:
                all_chunks = new_chunks
                all_vecs   = new_vecs
            all_names = cached_names | {dn for dn, _ in new_docs}
            _save_emb_cache(lib_dir, strat.name, model_name, all_chunks, all_vecs, all_names)
            print(f"    cache saved ({len(all_chunks):,} total chunks)")
        else:
            all_chunks = cached_chunks
            all_vecs   = cached_vecs
            print(f"    {len(all_chunks):,} chunks (fully cached)")

        doc_to_idx: dict[str, list[int]] = {}
        for i, c in enumerate(all_chunks):
            doc_to_idx.setdefault(c.document_name or "", []).append(i)
        strat_data[strat.name] = (all_chunks, all_vecs, doc_to_idx)

    print()

    # ── Phase 2: filter by scale, evaluate ────────────────────────────────────
    print("Phase 2: evaluate at each scale point (filter, no re-embed)")
    print("─" * 72)
    col_names  = [s.name for s in _STRATEGIES]
    hdr_strats = "  ".join(f"{n:>10}" for n in col_names)
    hdr = f"  {'Scale':>6}  {'Docs':>6}  {hdr_strats}"
    sep = "─" * len(hdr)
    print(hdr)
    print(sep)

    scale_results: dict[int, dict[str, float]] = {}
    scale_chunks:  dict[int, dict[str, int]]   = {}

    # Cumulative sampling: each scale adds to the previous corpus
    cumulative_corpus: list[tuple[str, str]] = []
    remaining: dict[str, list[tuple[str, str]]] = {t: list(docs) for t, docs in all_docs.items()}

    for scale in _SWEEP_SCALES:
        n       = min(scale, total_avail)
        to_add  = n - len(cumulative_corpus)
        if to_add > 0:
            new_docs = _sample_proportional(remaining, to_add, random.Random(args.seed + scale))
            added_names = {dn for dn, _ in new_docs}
            for t in _TYPE_DIRS:
                remaining[t] = [(dn, txt) for dn, txt in remaining[t] if dn not in added_names]
            cumulative_corpus.extend(new_docs)
        actual        = len(cumulative_corpus)
        sampled_names = {doc_name for doc_name, _ in cumulative_corpus}

        strat_rqi:    dict[str, float] = {}
        strat_chunks: dict[str, int]   = {}

        for strat in _STRATEGIES:
            chunks, vecs, doc_to_idx = strat_data[strat.name]
            indices   = sorted(i for dn in sampled_names for i in doc_to_idx.get(dn, []))
            sub_chunks = [chunks[i] for i in indices]
            sub_vecs   = vecs[np.array(indices, dtype=np.intp)] if indices else vecs[:0]

            strat_chunks[strat.name] = len(sub_chunks)

            results: list[QR] = []
            for q in QUERIES:
                top_idx    = _retrieve(q.text, sub_vecs, model, K)
                top_chunks = [sub_chunks[i] for i in top_idx]
                r = QR(query=q)
                r.c1 = c1_ndcg(top_chunks, q, K)
                r.c2 = c2_coherence(top_chunks, K)
                r.c3 = c3_scope(top_chunks, q, K) if q.category == "scoped" else 1.0
                r.c4 = None
                r.c5 = c5_efficiency(top_chunks, q, K)
                results.append(r)
            strat_rqi[strat.name] = _rqi(results, use_c4)

        scale_results[scale] = strat_rqi
        scale_chunks[scale]  = strat_chunks

        rqi_cols = "  ".join(f"{strat_rqi[s.name]:>10.4f}" for s in _STRATEGIES)
        print(f"  {scale:>6,}  {actual:>6,}  {rqi_cols}")

    print()
    print("Breadcrumb advantage (Δ = ctx - naive) by scale:")
    bh = (f"  {'Scale':>6}  {'Δ@400':>8}  {'Δ@800':>8}  {'Δ@1600':>8}"
          f"  {'ctx-1200 vs best-naive':>23}")
    print(bh)
    print("  " + "─" * (len(bh) - 2))

    for scale in _SWEEP_SCALES:
        sr = scale_results[scale]
        d400  = sr.get("ctx-400",  0.0) - sr.get("naive-400",  0.0)
        d800  = sr.get("ctx-800",  0.0) - sr.get("naive-800",  0.0)
        d1600 = sr.get("ctx-1600", 0.0) - sr.get("naive-1600", 0.0)
        ctx1200   = sr.get("ctx-1200", 0.0)
        best_naive = max(
            sr.get("naive-400", 0.0),
            sr.get("naive-800", 0.0),
            sr.get("naive-1600", 0.0),
        )
        pct = (ctx1200 - best_naive) / best_naive * 100 if best_naive else float("inf")
        print(f"  {scale:>6,}  {d400:>+8.4f}  {d800:>+8.4f}  {d1600:>+8.4f}  {pct:>+22.1f}%")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Chunky Monkey two-phase benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # build
    bp = sub.add_parser("build", help="Fetch raw content and extract text to library")
    bp.add_argument("--library",     default="/tmp/cm_bench", metavar="DIR",
                    help="Library directory (default: /tmp/cm_bench)")
    bp.add_argument("--n-pubmed",    type=int, default=200, metavar="N",
                    help="PubMed abstracts per topic (default 200)")
    bp.add_argument("--n-fr-pages",  type=int, default=100, metavar="N",
                    help="Federal Register pages per search (default 100)")
    bp.add_argument("--n-fda-pages", type=int, default=40,  metavar="N",
                    help="FDA label pages to fetch (default 40)")
    bp.add_argument("--n-ct-trials", type=int, default=100, metavar="N",
                    help="ClinicalTrials trials per condition (default 100)")
    bp.set_defaults(func=cmd_build)

    # run
    rp = sub.add_parser("run", help="Sample docs from library and evaluate chunking strategies")
    rp.add_argument("--library",    default="/tmp/cm_bench", metavar="DIR",
                    help="Library directory (default: /tmp/cm_bench)")
    rp.add_argument("--n-per-type", type=int, default=100, metavar="N",
                    help="Docs sampled per corpus type (default 100; -1 = all)")
    rp.add_argument("--seed",       type=int, default=0,
                    help="Random seed for sampling (default 0)")
    rp.add_argument("--top-k",      type=int, default=5,
                    help="Retrieval top-k (default 5)")
    rp.add_argument("--model",      default="all-MiniLM-L6-v2",
                    help="SentenceTransformer model name")
    # Per-type overrides
    rp.add_argument("--n-edgar",  type=int, default=None, metavar="N")
    rp.add_argument("--n-ct",     type=int, default=None, metavar="N")
    rp.add_argument("--n-fr",     type=int, default=None, metavar="N")
    rp.add_argument("--n-pubmed", type=int, default=None, metavar="N")
    rp.add_argument("--n-fda",    type=int, default=None, metavar="N")
    rp.add_argument("--n-python", type=int, default=None, metavar="N")
    rp.set_defaults(func=cmd_run)

    # sweep
    sp = sub.add_parser("sweep", help="Scale-sweep benchmark across 5 corpus sizes")
    sp.add_argument("--library", default="/tmp/cm_bench", metavar="DIR",
                    help="Library directory (default: /tmp/cm_bench)")
    sp.add_argument("--model",   default="all-MiniLM-L6-v2",
                    help="SentenceTransformer model name")
    sp.add_argument("--top-k",   type=int, default=5,
                    help="Retrieval top-k (default 5)")
    sp.add_argument("--seed",    type=int, default=0,
                    help="Random seed for sampling (default 0)")
    sp.set_defaults(func=cmd_sweep)

    return ap


def main() -> None:
    ap   = _make_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
