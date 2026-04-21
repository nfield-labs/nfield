"""
Schema–Prompt Alignment Gate.

Detects schema–prompt misalignment BEFORE routing, preventing schema poisoning.

Core principle: Task intent > schema structure

When schema doesn't match the task, FormatShield should:
  1. Detect misalignment (similarity score)
  2. Downgrade schema authority (strict → loose → ignore)
  3. Fallback appropriately (warn + auto-infer, or reject)

This prevents catastrophic failure: "structurally valid, semantically wrong"

Reference: Cases where email-validation schema was fed to:
  - Drug interaction analysis
  - Legal clause exclusivity reasoning
  - User registration (wrong fields)

All produced valid JSON + task-misaligned content.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SchemaAuthority(Enum):
    """How much schema should enforce the output."""
    
    STRICT = "strict"          # Schema is ground truth; enforce fully
    LOOSE = "loose"            # Schema is formatting hint; allow flexibility
    IGNORE = "ignore"          # Schema ignored; task-driven output only
    FALLBACK = "fallback"      # Reject schema, use unstructured output


@dataclass
class SchemaAlignmentResult:
    """Result of schema–prompt alignment check.
    
    Attributes
    ----------
    alignment_score: float
        Semantic similarity between schema intent and prompt intent (0–1).
        ≥0.7: good alignment
        0.4–0.7: weak alignment
        <0.4: severe misalignment
    
    schema_intent_keywords: list[str]
        Keywords extracted from schema (field names, descriptions, types)
    
    prompt_intent_keywords: list[str]
        Keywords extracted from prompt (task verbs, domain concepts)
    
    authority: SchemaAuthority
        Recommended schema authority level
    
    explanation: str
        Human-readable explanation of alignment assessment
    
    should_enforce_schema: bool
        True if schema should be enforced; False if schema is misaligned
    """
    alignment_score: float
    schema_intent_keywords: list[str]
    prompt_intent_keywords: list[str]
    authority: SchemaAuthority
    explanation: str
    should_enforce_schema: bool


def extract_schema_intent(schema: dict[str, Any]) -> list[str]:
    """Extract semantic intent from JSON schema.
    
    Identifies what the schema is designed to capture by analyzing:
    - Top-level field names
    - Property descriptions
    - Enum values
    - Required fields
    - Type constraints (email, date, etc.)
    
    Example:
        schema = {
            "type": "object",
            "properties": {
                "email": {"type": "string", "format": "email"},
                "phone": {"type": "string"},
                "validation_status": {"enum": ["valid", "invalid"]}
            },
            "required": ["email", "validation_status"]
        }
        → ["email", "phone", "validation", "contact"]
    
    Parameters
    ----------
    schema: dict
        JSON Schema dict
    
    Returns
    -------
    list[str]
        Keywords representing schema intent
    """
    keywords = []
    
    def _extract(obj: Any) -> None:
        """Recursively extract keywords."""
        if isinstance(obj, dict):
            # Field names
            for key in obj.keys():
                if key in ("properties", "items", "additionalProperties"):
                    continue
                keywords.extend(_normalize_keyword(key))
            
            # Descriptions
            if "description" in obj and isinstance(obj["description"], str):
                keywords.extend(_extract_words(obj["description"]))
            
            # Enum values (semantic content)
            if "enum" in obj and isinstance(obj["enum"], list):
                for val in obj["enum"]:
                    if isinstance(val, str):
                        keywords.extend(_normalize_keyword(val))
            
            # Format hints (email, date, etc.)
            if "format" in obj and isinstance(obj["format"], str):
                keywords.append(obj["format"])
            
            # Type constraints (what domain?)
            if "type" in obj and isinstance(obj["type"], str):
                keywords.append(obj["type"])
            
            # Recurse into nested structures
            for key, val in obj.items():
                if key not in ("description", "enum", "format", "type"):
                    _extract(val)
        
        elif isinstance(obj, list):
            for item in obj:
                _extract(item)
    
    _extract(schema)
    
    # Remove duplicates, empty strings, and normalize
    keywords = list(set(kw for kw in keywords if kw and len(kw) > 1))
    return sorted(keywords)


def extract_prompt_intent(prompt: str) -> list[str]:
    """Extract semantic intent from prompt text.
    
    Identifies what task the prompt describes by analyzing:
    - Task verbs (analyze, calculate, classify, etc.)
    - Domain keywords (drug, patient, legal, etc.)
    - Reasoning indicators (explain, compare, evaluate)
    - Nouns (primary concepts)
    
    Example:
        prompt = "Analyze drug interactions between warfarin and ibuprofen"
        → ["analyze", "drug", "interaction", "warfarin", "ibuprofen"]
    
    Parameters
    ----------
    prompt: str
        The task prompt
    
    Returns
    -------
    list[str]
        Keywords representing prompt intent
    """
    # Extract all words
    words = _extract_words(prompt)
    
    # Filter to meaningful content (remove stop words, short words)
    keywords = [
        w for w in words
        if len(w) > 2 and w not in STOP_WORDS
    ]
    
    return sorted(keywords)


def compute_alignment_score(
    schema_keywords: list[str],
    prompt_keywords: list[str],
) -> float:
    """Compute semantic similarity between schema and prompt intent.
    
    Uses Jaccard similarity (intersection / union):
        - 1.0 = perfect overlap
        - 0.5 = moderate overlap
        - 0.0 = no overlap
    
    Parameters
    ----------
    schema_keywords: list[str]
        Keywords from schema intent
    prompt_keywords: list[str]
        Keywords from prompt intent
    
    Returns
    -------
    float
        Similarity score in [0, 1]
    """
    if not schema_keywords or not prompt_keywords:
        return 0.0
    
    schema_set = set(schema_keywords)
    prompt_set = set(prompt_keywords)
    
    intersection = len(schema_set & prompt_set)
    union = len(schema_set | prompt_set)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def assess_schema_alignment(
    schema: dict[str, Any],
    prompt: str,
) -> SchemaAlignmentResult:
    """Assess alignment between schema and prompt task.
    
    Decision logic:
        score ≥ 0.7: STRICT (schema is ground truth)
        score ∈ [0.4, 0.7): LOOSE (schema is hint; allow flexibility)
        score < 0.4: IGNORE or FALLBACK (schema misaligned; warn or reject)
    
    Parameters
    ----------
    schema: dict
        JSON Schema
    prompt: str
        Task prompt
    
    Returns
    -------
    SchemaAlignmentResult
        Alignment assessment + recommended authority level
    """
    schema_keywords = extract_schema_intent(schema)
    prompt_keywords = extract_prompt_intent(prompt)
    
    score = compute_alignment_score(schema_keywords, prompt_keywords)
    
    # Determine authority level
    if score >= 0.70:
        authority = SchemaAuthority.STRICT
        explanation = (
            f"Schema well-aligned with prompt intent (score={score:.2f}). "
            f"Schema structure matches task requirements. "
            f"Enforce schema strictly."
        )
        should_enforce = True
    
    elif score >= 0.40:
        authority = SchemaAuthority.LOOSE
        explanation = (
            f"Schema partially aligned with prompt (score={score:.2f}). "
            f"Some schema fields relevant, but may conflict with task. "
            f"Allow flexibility; validate critical fields only."
        )
        should_enforce = False
    
    else:
        authority = SchemaAuthority.FALLBACK
        explanation = (
            f"Schema misaligned with prompt (score={score:.2f}). "
            f"Schema keywords: {schema_keywords[:5]}. "
            f"Prompt keywords: {prompt_keywords[:5]}. "
            f"Likely schema was copied/wrong. FALLBACK to unstructured."
        )
        should_enforce = False
    
    return SchemaAlignmentResult(
        alignment_score=score,
        schema_intent_keywords=schema_keywords,
        prompt_intent_keywords=prompt_keywords,
        authority=authority,
        explanation=explanation,
        should_enforce_schema=should_enforce,
    )


def _extract_words(text: str) -> list[str]:
    """Extract lowercase words from text, removing punctuation."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return words


def _normalize_keyword(key: str) -> list[str]:
    """Normalize a keyword by splitting camelCase and snake_case."""
    # snake_case: email_validation → [email, validation]
    snake_split = key.split("_")
    
    # camelCase: emailValidation → [email, validation]
    camel_parts = []
    current = ""
    for char in key:
        if char.isupper() and current:
            camel_parts.append(current.lower())
            current = char.lower()
        else:
            current += char.lower()
    if current:
        camel_parts.append(current)
    
    # Combine all variants, remove empty
    all_parts = list(set(snake_split + camel_parts))
    return [p for p in all_parts if p and len(p) > 1]


# Stop words (high-frequency, low-information)
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "may", "might", "can",
    "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "when", "where", "why", "how",
    "not", "no", "yes",
    "as", "if", "else", "then",
}
