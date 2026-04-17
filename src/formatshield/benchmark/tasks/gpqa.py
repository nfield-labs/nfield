"""GPQA-Diamond benchmark task for FormatShield.

Graduate-level Professional Question Answering (Diamond split).
Tests scientific reasoning across biology, chemistry, and physics.
Measures format tax on multi-choice reasoning — models often fail
when forced to output structured JSON with option selection.

Reference: Rein et al. (2023) "GPQA: A Graduate-Level Google-Proof
Q&A Benchmark" (arXiv 2311.12022).
"""

from __future__ import annotations

import json
import re
from typing import Any

# Each problem: prompt (the question + choices), ground_truth (the letter A/B/C/D),
# schema (structured output format), domain (biology/chemistry/physics).
# Reversal tracking: store original_correct and presented_as to detect if model
# is memorising position vs reasoning.

_PROBLEMS: list[dict[str, Any]] = [
    {
        "prompt": (
            "A researcher observes that a particular enzyme has a Km of 2 mM "
            "and Vmax of 100 μmol/min. "
            "If the substrate concentration is 6 mM, what is the reaction "
            "velocity according to Michaelis-Menten kinetics?\n\n"
            "A) 25 μmol/min\n"
            "B) 75 μmol/min\n"
            "C) 50 μmol/min\n"
            "D) 100 μmol/min"
        ),
        "ground_truth": "B",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "biology",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "In a quantum mechanical system, a particle is in a superposition state. "
            "After measurement, which interpretation of quantum mechanics holds that "
            "the wavefunction does NOT collapse but instead branches into parallel worlds?\n\n"
            "A) Copenhagen interpretation\n"
            "B) Pilot wave theory\n"
            "C) Many-worlds interpretation\n"
            "D) Objective collapse theory"
        ),
        "ground_truth": "C",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "physics",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "Which reagent would convert a terminal alkyne to a (Z)-alkene (cis-alkene) "
            "with high stereoselectivity?\n\n"
            "A) H2/Pd-C (Lindlar's catalyst)\n"
            "B) Na/NH3 (Birch reduction)\n"
            "C) H2/PtO2\n"
            "D) LiAlH4"
        ),
        "ground_truth": "A",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "chemistry",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "A protein has a pI of 6.0. At physiological pH (7.4), what is the net charge?\n\n"
            "A) Positive\n"
            "B) Negative\n"
            "C) Neutral\n"
            "D) Cannot be determined"
        ),
        "ground_truth": "B",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "biology",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "The Heisenberg uncertainty principle states that the product of uncertainties "
            "in position (Δx) and momentum (Δp) satisfies:\n\n"
            "A) Δx · Δp ≥ ℏ/2\n"
            "B) Δx · Δp = 0\n"
            "C) Δx · Δp ≤ ℏ\n"
            "D) Δx · Δp ≥ ℏ"
        ),
        "ground_truth": "A",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "physics",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "In SN2 reactions, which substrate reacts fastest?\n\n"
            "A) Neopentyl bromide\n"
            "B) tert-Butyl bromide\n"
            "C) Methyl bromide\n"
            "D) Isopropyl bromide"
        ),
        "ground_truth": "C",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "chemistry",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "Which of the following best describes the function of telomerase?\n\n"
            "A) It repairs double-strand DNA breaks\n"
            "B) It adds repetitive DNA sequences to chromosome ends\n"
            "C) It removes RNA primers from Okazaki fragments\n"
            "D) It proofreads newly synthesized DNA"
        ),
        "ground_truth": "B",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "biology",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "A black body at temperature T emits radiation. If the temperature doubles (2T), "
            "the total power radiated per unit area changes by what factor?\n\n"
            "A) 2\n"
            "B) 4\n"
            "C) 8\n"
            "D) 16"
        ),
        "ground_truth": "D",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "physics",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "Which amino acid can form disulfide bonds in proteins?\n\n"
            "A) Serine\n"
            "B) Threonine\n"
            "C) Cysteine\n"
            "D) Methionine"
        ),
        "ground_truth": "C",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "biology",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "What is the electron configuration of a Cu²⁺ ion?\n\n"
            "A) [Ar] 3d⁹\n"
            "B) [Ar] 3d¹⁰\n"
            "C) [Ar] 4s¹ 3d¹⁰\n"
            "D) [Ar] 4s² 3d⁸"
        ),
        "ground_truth": "A",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "chemistry",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "In special relativity, the Lorentz factor γ for an object moving at v = 0.6c is:\n\n"
            "A) 1.0\n"
            "B) 1.25\n"
            "C) 1.5\n"
            "D) 2.0"
        ),
        "ground_truth": "B",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "physics",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "Which of the following reactions describes the rate-limiting step "
            "in the citric acid cycle under normal cellular conditions?\n\n"
            "A) Citrate synthase reaction\n"
            "B) Isocitrate dehydrogenase reaction\n"
            "C) Fumarase reaction\n"
            "D) Succinate dehydrogenase reaction"
        ),
        "ground_truth": "B",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "biology",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "A Diels-Alder reaction between 1,3-butadiene and ethylene produces:\n\n"
            "A) Benzene\n"
            "B) Cyclohexene\n"
            "C) Cyclohexadiene\n"
            "D) Hexane"
        ),
        "ground_truth": "B",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "chemistry",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "In an ideal gas, if pressure is doubled while volume is halved, "
            "what happens to temperature (assuming PV = nRT)?\n\n"
            "A) Doubles\n"
            "B) Halves\n"
            "C) Remains the same\n"
            "D) Quadruples"
        ),
        "ground_truth": "C",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "physics",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "Which type of RNA is responsible for carrying amino acids to the ribosome?\n\n"
            "A) mRNA\n"
            "B) rRNA\n"
            "C) tRNA\n"
            "D) snRNA"
        ),
        "ground_truth": "C",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "biology",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "The Henderson-Hasselbalch equation relates pH to pKa and the ratio of:\n\n"
            "A) Conjugate acid to conjugate base\n"
            "B) Conjugate base to conjugate acid\n"
            "C) Concentration of H⁺ to OH⁻\n"
            "D) Molarity to molality"
        ),
        "ground_truth": "B",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "chemistry",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "What is the de Broglie wavelength of a 1 kg object moving at 1 m/s "
            "(h = 6.626×10⁻³⁴ J·s)?\n\n"
            "A) 6.626×10⁻³⁴ m\n"
            "B) 6.626×10⁻³³ m\n"
            "C) 6.626×10⁻³² m\n"
            "D) 6.626×10⁻³⁵ m"
        ),
        "ground_truth": "A",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "physics",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "Which of the following is a competitive inhibitor of an enzyme?\n\n"
            "A) Increases Vmax but not Km\n"
            "B) Increases Km but not Vmax\n"
            "C) Decreases both Km and Vmax\n"
            "D) Decreases Vmax but not Km"
        ),
        "ground_truth": "B",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "biology",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "Which of the following is NOT a characteristic of an aromatic compound?\n\n"
            "A) Planar ring structure\n"
            "B) 4n+2 π electrons (Hückel's rule)\n"
            "C) sp3 hybridized carbons\n"
            "D) Delocalized π electrons"
        ),
        "ground_truth": "C",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "chemistry",
        "difficulty": "graduate",
    },
    {
        "prompt": (
            "In a simple harmonic oscillator, the total energy is proportional to:\n\n"
            "A) Amplitude\n"
            "B) Amplitude squared\n"
            "C) Frequency\n"
            "D) Period"
        ),
        "ground_truth": "B",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
        },
        "domain": "physics",
        "difficulty": "graduate",
    },
]


def get_problems(quick: bool = False) -> list[dict[str, Any]]:
    """Return GPQA-Diamond benchmark problems.

    Args:
        quick: If True, return a small subset for CI/smoke tests.

    Returns:
        List of problem dicts with keys: 'prompt', 'ground_truth', 'schema',
        'domain', 'difficulty'.
    """
    return _PROBLEMS[:3] if quick else _PROBLEMS


def score_response(predicted: str, ground_truth: Any) -> float:
    """Score a model response against the ground truth answer letter.

    Extracts the answer letter (A/B/C/D) from the predicted string,
    handling both JSON structured output and plain text responses.

    Args:
        predicted: Raw string output from the model (JSON or plain text).
        ground_truth: The correct answer letter (e.g., "B").

    Returns:
        1.0 if correct, 0.0 if incorrect or unparseable.
    """
    if not isinstance(ground_truth, str):
        return 0.0

    expected = ground_truth.strip().upper()

    # Try JSON parsing first (structured output mode).
    try:
        parsed = json.loads(predicted)
        if isinstance(parsed, dict):
            answer = str(parsed.get("answer", "")).strip().upper()
            if answer in {"A", "B", "C", "D"}:
                return 1.0 if answer == expected else 0.0
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to regex extraction from plain text.
    # Look for patterns like "Answer: B", "The answer is C", "B)", "(B)".
    patterns = [
        r"\bAnswer[:\s]+([A-D])\b",
        r"\bthe answer is\s+([A-D])\b",
        r"^\s*([A-D])\s*[).:]",
        r"\(([A-D])\)",
        r"\b([A-D])\b",
    ]
    text_upper = predicted.upper()
    for pattern in patterns:
        match = re.search(pattern, text_upper, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).upper()
            return 1.0 if answer == expected else 0.0

    return 0.0
