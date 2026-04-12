"""
MedicalNERTask — Medical Named-Entity Recognition benchmark task for FormatShield.

This task contains 15 hardcoded clinical text snippets annotated with four
entity categories: conditions, medications, dosages, and procedures.

Models must extract these entities into a structured :class:`MedicalEntities`
response.  Because entity spans often require disambiguation (e.g. a drug name
versus a symptom name), this task is classified MEDIUM-HIGH complexity and
is expected to benefit from TTF routing.

Complexity: MEDIUM-HIGH
Expected TTF benefit: True (disambiguation requires reasoning)
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MedicalEntities(BaseModel):
    """Structured schema for medical named-entity extraction."""

    conditions: list[str]
    """Diagnosed or mentioned medical conditions, diseases, or symptoms."""

    medications: list[str]
    """Drug names or medication classes mentioned in the text."""

    dosages: list[str]
    """Dosage information tied to medications (e.g. '500 mg twice daily')."""

    procedures: list[str]
    """Medical procedures, tests, surgeries, or interventions mentioned."""


# ---------------------------------------------------------------------------
# 15 hardcoded clinical text snippets with annotated entities
# ---------------------------------------------------------------------------

_PROBLEMS: list[dict[str, Any]] = [
    {
        "text": (
            "The patient is a 58-year-old male with a history of type 2 diabetes mellitus "
            "and hypertension. He was admitted for an elective coronary artery bypass graft "
            "(CABG) procedure. Current medications include metformin 1000 mg twice daily and "
            "lisinopril 10 mg once daily."
        ),
        "entities": {
            "conditions": ["type 2 diabetes mellitus", "hypertension"],
            "medications": ["metformin", "lisinopril"],
            "dosages": ["1000 mg twice daily", "10 mg once daily"],
            "procedures": ["coronary artery bypass graft", "CABG"],
        },
    },
    {
        "text": (
            "A 34-year-old woman presented to the emergency department with severe asthma "
            "exacerbation. She was treated with albuterol 2.5 mg via nebuliser and "
            "prednisone 40 mg orally. A chest X-ray was performed to rule out pneumonia."
        ),
        "entities": {
            "conditions": ["asthma exacerbation", "pneumonia"],
            "medications": ["albuterol", "prednisone"],
            "dosages": ["2.5 mg via nebuliser", "40 mg orally"],
            "procedures": ["chest X-ray"],
        },
    },
    {
        "text": (
            "Patient has chronic kidney disease stage 3 and anaemia of chronic disease. "
            "Laboratory findings show a haemoglobin of 9.2 g/dL. Erythropoietin-stimulating "
            "agent therapy was initiated with darbepoetin alfa 60 mcg subcutaneously every "
            "two weeks. A renal biopsy was recommended."
        ),
        "entities": {
            "conditions": ["chronic kidney disease stage 3", "anaemia of chronic disease"],
            "medications": ["darbepoetin alfa", "erythropoietin-stimulating agent"],
            "dosages": ["60 mcg subcutaneously every two weeks"],
            "procedures": ["renal biopsy"],
        },
    },
    {
        "text": (
            "A 72-year-old female with known atrial fibrillation and heart failure with "
            "reduced ejection fraction (HFrEF) is maintained on warfarin 5 mg daily, "
            "carvedilol 25 mg twice daily, and furosemide 40 mg once daily. "
            "An echocardiogram performed last month showed an EF of 35%."
        ),
        "entities": {
            "conditions": [
                "atrial fibrillation",
                "heart failure with reduced ejection fraction",
                "HFrEF",
            ],
            "medications": ["warfarin", "carvedilol", "furosemide"],
            "dosages": ["5 mg daily", "25 mg twice daily", "40 mg once daily"],
            "procedures": ["echocardiogram"],
        },
    },
    {
        "text": (
            "Post-operative note: The patient underwent laparoscopic cholecystectomy for "
            "acute cholecystitis. She received cefazolin 1 g IV prophylactically before "
            "the procedure. Pain is managed with ibuprofen 400 mg every 6 hours as needed. "
            "No wound infection noted."
        ),
        "entities": {
            "conditions": ["acute cholecystitis", "wound infection"],
            "medications": ["cefazolin", "ibuprofen"],
            "dosages": ["1 g IV", "400 mg every 6 hours"],
            "procedures": ["laparoscopic cholecystectomy"],
        },
    },
    {
        "text": (
            "Patient presents with newly diagnosed stage IIIA non-small cell lung cancer "
            "(NSCLC). Pathology confirmed adenocarcinoma. CT-guided biopsy and PET scan "
            "were performed for staging. Treatment plan includes concurrent chemoradiotherapy "
            "with carboplatin AUC 6 and paclitaxel 200 mg/m² every three weeks."
        ),
        "entities": {
            "conditions": ["non-small cell lung cancer", "NSCLC", "adenocarcinoma"],
            "medications": ["carboplatin", "paclitaxel"],
            "dosages": ["AUC 6", "200 mg/m² every three weeks"],
            "procedures": ["CT-guided biopsy", "PET scan", "chemoradiotherapy"],
        },
    },
    {
        "text": (
            "A 45-year-old male with rheumatoid arthritis not controlled on methotrexate "
            "15 mg weekly was started on adalimumab 40 mg subcutaneously every other week. "
            "A tuberculosis screening (QuantiFERON-TB Gold) and hepatitis B serology were "
            "ordered prior to initiating biologic therapy."
        ),
        "entities": {
            "conditions": ["rheumatoid arthritis", "tuberculosis"],
            "medications": ["methotrexate", "adalimumab"],
            "dosages": ["15 mg weekly", "40 mg subcutaneously every other week"],
            "procedures": ["QuantiFERON-TB Gold", "hepatitis B serology"],
        },
    },
    {
        "text": (
            "This 29-year-old female with major depressive disorder and generalised anxiety "
            "disorder is prescribed sertraline 100 mg once daily and clonazepam 0.5 mg "
            "twice daily as needed for acute anxiety. Cognitive behavioural therapy (CBT) "
            "sessions are ongoing."
        ),
        "entities": {
            "conditions": ["major depressive disorder", "generalised anxiety disorder"],
            "medications": ["sertraline", "clonazepam"],
            "dosages": ["100 mg once daily", "0.5 mg twice daily"],
            "procedures": ["cognitive behavioural therapy", "CBT"],
        },
    },
    {
        "text": (
            "The patient has end-stage renal disease and is on haemodialysis three times "
            "per week. He is also being treated for secondary hyperparathyroidism with "
            "cinacalcet 60 mg once daily and sevelamer carbonate 800 mg three times daily "
            "with meals. An AV fistula was created surgically last year."
        ),
        "entities": {
            "conditions": ["end-stage renal disease", "secondary hyperparathyroidism"],
            "medications": ["cinacalcet", "sevelamer carbonate"],
            "dosages": ["60 mg once daily", "800 mg three times daily with meals"],
            "procedures": ["haemodialysis", "AV fistula creation"],
        },
    },
    {
        "text": (
            "Emergency notes: 66-year-old male arrived with suspected ST-elevation "
            "myocardial infarction (STEMI). An ECG confirmed anterior STEMI. "
            "He was given aspirin 300 mg and clopidogrel 600 mg loading doses, then "
            "taken for emergency percutaneous coronary intervention (PCI)."
        ),
        "entities": {
            "conditions": ["ST-elevation myocardial infarction", "STEMI", "anterior STEMI"],
            "medications": ["aspirin", "clopidogrel"],
            "dosages": ["300 mg", "600 mg loading dose"],
            "procedures": ["ECG", "percutaneous coronary intervention", "PCI"],
        },
    },
    {
        "text": (
            "Patient is a 50-year-old woman with Hashimoto's thyroiditis and hypothyroidism. "
            "She is maintained on levothyroxine 75 mcg once daily taken on an empty stomach. "
            "Thyroid function tests (TFTs) including TSH and free T4 were drawn and a "
            "thyroid ultrasound was performed."
        ),
        "entities": {
            "conditions": ["Hashimoto's thyroiditis", "hypothyroidism"],
            "medications": ["levothyroxine"],
            "dosages": ["75 mcg once daily"],
            "procedures": ["thyroid function tests", "TFTs", "thyroid ultrasound"],
        },
    },
    {
        "text": (
            "A 7-year-old boy was brought in with a first-time unprovoked generalised "
            "tonic-clonic seizure lasting four minutes. An EEG and MRI brain were ordered. "
            "After discussion of risks and benefits, levetiracetam 250 mg twice daily was "
            "started. Epilepsy was listed as the working diagnosis."
        ),
        "entities": {
            "conditions": ["generalised tonic-clonic seizure", "epilepsy"],
            "medications": ["levetiracetam"],
            "dosages": ["250 mg twice daily"],
            "procedures": ["EEG", "MRI brain"],
        },
    },
    {
        "text": (
            "Discharge summary: Patient with Crohn's disease was hospitalised for an acute "
            "flare. Treatment included IV methylprednisolone 60 mg daily for three days, "
            "followed by a tapering course of oral prednisolone. Colonoscopy with biopsy "
            "was performed. Infliximab induction therapy is planned as an outpatient."
        ),
        "entities": {
            "conditions": ["Crohn's disease"],
            "medications": ["methylprednisolone", "prednisolone", "infliximab"],
            "dosages": ["60 mg daily for three days"],
            "procedures": ["colonoscopy", "biopsy"],
        },
    },
    {
        "text": (
            "The patient is a 78-year-old male with Parkinson's disease and orthostatic "
            "hypotension. He is on levodopa/carbidopa 100/25 mg three times daily and "
            "fludrocortisone 0.1 mg once daily. A tilt table test was performed to "
            "assess autonomic dysfunction."
        ),
        "entities": {
            "conditions": [
                "Parkinson's disease",
                "orthostatic hypotension",
                "autonomic dysfunction",
            ],
            "medications": ["levodopa", "carbidopa", "fludrocortisone"],
            "dosages": ["100/25 mg three times daily", "0.1 mg once daily"],
            "procedures": ["tilt table test"],
        },
    },
    {
        "text": (
            "A 22-year-old athlete presented with right knee pain following a sporting "
            "injury. Physical examination and MRI confirmed an anterior cruciate ligament "
            "(ACL) tear with associated medial meniscus injury. Arthroscopic ACL "
            "reconstruction was scheduled. Naproxen 500 mg twice daily was prescribed "
            "for pain control pending surgery."
        ),
        "entities": {
            "conditions": ["anterior cruciate ligament tear", "ACL tear", "medial meniscus injury"],
            "medications": ["naproxen"],
            "dosages": ["500 mg twice daily"],
            "procedures": ["MRI", "arthroscopic ACL reconstruction"],
        },
    },
]

# Quick-mode uses the first 5 problems only
_QUICK_SLICE = 5


def _token_set(text: str) -> set[str]:
    """Normalise an entity string to a case-folded token set for fuzzy matching."""
    return {t.lower().strip(".,;:") for t in text.split() if t.strip(".,;:")}


def _entity_f1(predicted_list: list[str], truth_list: list[str]) -> float:
    """
    Compute a token-level F1 score between two lists of entity strings.

    Each entity is converted to a lower-cased token set.  An entity is
    considered a match when its token set has non-empty intersection with a
    ground-truth entity's token set.

    Returns
    -------
    float
        F1 score in [0.0, 1.0].
    """
    if not truth_list and not predicted_list:
        return 1.0
    if not truth_list or not predicted_list:
        return 0.0

    truth_sets = [_token_set(e) for e in truth_list]
    pred_sets = [_token_set(e) for e in predicted_list]

    # Precision: fraction of predictions that overlap with any truth entity
    precision_hits = sum(1 for ps in pred_sets if ps and any(ps & ts for ts in truth_sets))
    precision = precision_hits / len(pred_sets) if pred_sets else 0.0

    # Recall: fraction of truth entities covered by any prediction
    recall_hits = sum(1 for ts in truth_sets if ts and any(ps & ts for ps in pred_sets))
    recall = recall_hits / len(truth_sets) if truth_sets else 0.0

    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class MedicalNERTask:
    """
    Medical Named-Entity Recognition benchmark task.

    Contains 15 hardcoded clinical text snippets pre-annotated with four
    entity categories: conditions, medications, dosages, and procedures.

    The model must extract these into a structured :class:`MedicalEntities`
    response.  Scoring uses a token-level F1 metric averaged across all four
    entity categories.

    Attributes
    ----------
    name:
        Stable task identifier used in benchmark result records.
    expected_ttf_benefit:
        ``True`` because disambiguation (e.g. distinguishing drug name from
        symptom) is easier when the model reasons step-by-step.
    schema:
        The Pydantic model class that defines the expected output shape.
    complexity:
        Qualitative complexity label consumed by the harness for reporting.
    """

    name: str = "medical_ner"
    expected_ttf_benefit: bool = True
    schema = MedicalEntities
    complexity: str = "MEDIUM-HIGH"

    def get_problems(self, quick: bool = False) -> list[dict[str, Any]]:
        """
        Return the list of benchmark problems.

        Parameters
        ----------
        quick:
            When ``True`` returns only the first 5 problems.

        Returns
        -------
        list[dict]
            Each element has keys:

            ``"text"`` : str
                Clinical text passed to the model as input.
            ``"entities"`` : dict
                Ground-truth entity dict with keys ``conditions``,
                ``medications``, ``dosages``, ``procedures``.
        """
        problems = _PROBLEMS[:_QUICK_SLICE] if quick else _PROBLEMS
        return [{"text": p["text"], "entities": p["entities"]} for p in problems]

    def score_response(
        self,
        predicted: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> float:
        """
        Score a model response against the annotated ground-truth entities.

        Scoring is a macro-averaged token-level F1 across the four entity
        categories (conditions, medications, dosages, procedures).

        Parameters
        ----------
        predicted:
            A dict representation of a :class:`MedicalEntities` instance
            produced by the model.  Expected keys: ``conditions``,
            ``medications``, ``dosages``, ``procedures``.  Missing keys are
            treated as empty lists.
        ground_truth:
            The annotated entity dict for the clinical text.

        Returns
        -------
        float
            Macro-averaged F1 score in [0.0, 1.0].
        """
        if not isinstance(predicted, dict):
            logger.debug("score_response: predicted is not a dict, got %r", type(predicted))
            return 0.0

        categories = ["conditions", "medications", "dosages", "procedures"]
        category_scores: list[float] = []

        for cat in categories:
            pred_list = predicted.get(cat, []) or []
            truth_list = ground_truth.get(cat, []) or []

            # Ensure both are lists of strings
            if not isinstance(pred_list, list):
                pred_list = []
            if not isinstance(truth_list, list):
                truth_list = []

            pred_strings = [str(e) for e in pred_list if e]
            truth_strings = [str(e) for e in truth_list if e]

            category_scores.append(_entity_f1(pred_strings, truth_strings))

        return sum(category_scores) / len(category_scores)

    def build_prompt(self, text: str) -> str:
        """
        Construct the full prompt string sent to the model.

        Parameters
        ----------
        text:
            The clinical note or medical text snippet.

        Returns
        -------
        str
            A formatted prompt requesting structured entity extraction.
        """
        return (
            "You are a clinical NLP assistant.  Extract all named medical entities "
            "from the following clinical text and return them as structured JSON.\n\n"
            "Extract exactly these four categories:\n"
            "  - conditions: list of medical conditions, diseases, or symptoms\n"
            "  - medications: list of drug names or medication classes\n"
            "  - dosages: list of dosage strings associated with medications\n"
            "  - procedures: list of medical procedures, tests, or surgeries\n\n"
            "Return only the JSON object.  Use empty lists for categories with no "
            "entities.  Do not include explanation text outside the JSON.\n\n"
            f"Clinical text:\n{text}"
        )
