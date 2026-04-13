"""
CodeExtractionTask — Code entity extraction benchmark task for FormatShield.

This task contains 15 hardcoded Python function definitions. Models must
extract the function name, argument names with their types, the return type,
and a one-sentence description of what the function does.

Because code parsing requires understanding of Python syntax and type
annotation conventions, this task is MEDIUM complexity. TTF routing
helps avoid hallucinated argument types.

Complexity: MEDIUM
Expected TTF benefit: True
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ArgumentInfo(BaseModel):
    """Information about a single function argument."""

    name: str
    """The argument name."""

    type_annotation: str
    """The type annotation as a string, or 'Any' if unannotated."""


class CodeEntities(BaseModel):
    """Structured schema for code entity extraction."""

    function_name: str
    """The name of the function being extracted."""

    arguments: list[ArgumentInfo]
    """List of arguments with their names and type annotations."""

    return_type: str
    """The return type annotation as a string, or 'None' if not annotated."""

    description: str
    """One-sentence description of what the function does."""


# ---------------------------------------------------------------------------
# 15 hardcoded Python function definitions with ground-truth extractions
# ---------------------------------------------------------------------------

_PROBLEMS: list[dict[str, Any]] = [
    {
        "code": (
            "def calculate_discount(price: float, discount_pct: float) -> float:\n"
            '    """Apply a percentage discount to a price."""\n'
            "    return price * (1 - discount_pct / 100)"
        ),
        "ground_truth": {
            "function_name": "calculate_discount",
            "arguments": [
                {"name": "price", "type_annotation": "float"},
                {"name": "discount_pct", "type_annotation": "float"},
            ],
            "return_type": "float",
        },
    },
    {
        "code": (
            "def merge_sorted_lists(a: list[int], b: list[int]) -> list[int]:\n"
            '    """Merge two sorted lists into a single sorted list."""\n'
            "    result = []\n"
            "    i, j = 0, 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n"
            "            result.append(a[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(b[j])\n"
            "            j += 1\n"
            "    return result + a[i:] + b[j:]"
        ),
        "ground_truth": {
            "function_name": "merge_sorted_lists",
            "arguments": [
                {"name": "a", "type_annotation": "list[int]"},
                {"name": "b", "type_annotation": "list[int]"},
            ],
            "return_type": "list[int]",
        },
    },
    {
        "code": (
            "async def fetch_user_data(\n"
            "    user_id: int,\n"
            "    include_history: bool = False,\n"
            "    max_results: int = 100,\n"
            ") -> dict[str, Any]:\n"
            '    """Fetch user profile and optionally their activity history."""\n'
            "    ..."
        ),
        "ground_truth": {
            "function_name": "fetch_user_data",
            "arguments": [
                {"name": "user_id", "type_annotation": "int"},
                {"name": "include_history", "type_annotation": "bool"},
                {"name": "max_results", "type_annotation": "int"},
            ],
            "return_type": "dict[str, Any]",
        },
    },
    {
        "code": (
            "def tokenize(text: str, lower: bool = True, strip_punct: bool = False) -> list[str]:\n"
            '    """Split text into tokens with optional lowercasing and punctuation removal."""\n'
            "    tokens = text.lower().split() if lower else text.split()\n"
            "    return tokens"
        ),
        "ground_truth": {
            "function_name": "tokenize",
            "arguments": [
                {"name": "text", "type_annotation": "str"},
                {"name": "lower", "type_annotation": "bool"},
                {"name": "strip_punct", "type_annotation": "bool"},
            ],
            "return_type": "list[str]",
        },
    },
    {
        "code": (
            "def compute_iou(box_a: tuple[float, float, float, float],\n"
            "                box_b: tuple[float, float, float, float]) -> float:\n"
            '    """Compute Intersection over Union of two bounding boxes."""\n'
            "    ..."
        ),
        "ground_truth": {
            "function_name": "compute_iou",
            "arguments": [
                {"name": "box_a", "type_annotation": "tuple[float, float, float, float]"},
                {"name": "box_b", "type_annotation": "tuple[float, float, float, float]"},
            ],
            "return_type": "float",
        },
    },
    {
        "code": (
            "def retry(max_attempts: int, delay: float = 1.0, backoff: float = 2.0):\n"
            '    """Decorator that retries a function on exception with exponential backoff."""\n'
            "    def decorator(func):\n"
            "        return func\n"
            "    return decorator"
        ),
        "ground_truth": {
            "function_name": "retry",
            "arguments": [
                {"name": "max_attempts", "type_annotation": "int"},
                {"name": "delay", "type_annotation": "float"},
                {"name": "backoff", "type_annotation": "float"},
            ],
            "return_type": "Any",
        },
    },
    {
        "code": (
            "def batch(iterable: list[Any], size: int) -> list[list[Any]]:\n"
            '    """Split an iterable into batches of the given size."""\n'
            "    return [iterable[i:i + size] for i in range(0, len(iterable), size)]"
        ),
        "ground_truth": {
            "function_name": "batch",
            "arguments": [
                {"name": "iterable", "type_annotation": "list[Any]"},
                {"name": "size", "type_annotation": "int"},
            ],
            "return_type": "list[list[Any]]",
        },
    },
    {
        "code": (
            "def normalize_vector(v: list[float]) -> list[float]:\n"
            '    """L2-normalize a vector to unit length."""\n'
            "    norm = sum(x**2 for x in v) ** 0.5\n"
            "    return [x / norm for x in v] if norm > 0 else v"
        ),
        "ground_truth": {
            "function_name": "normalize_vector",
            "arguments": [
                {"name": "v", "type_annotation": "list[float]"},
            ],
            "return_type": "list[float]",
        },
    },
    {
        "code": (
            "def parse_date_range(\n"
            "    start: str,\n"
            "    end: str,\n"
            "    fmt: str = '%Y-%m-%d',\n"
            ") -> tuple[datetime, datetime]:\n"
            '    """Parse a start/end date string pair into datetime objects."""\n'
            "    from datetime import datetime\n"
            "    return datetime.strptime(start, fmt), datetime.strptime(end, fmt)"
        ),
        "ground_truth": {
            "function_name": "parse_date_range",
            "arguments": [
                {"name": "start", "type_annotation": "str"},
                {"name": "end", "type_annotation": "str"},
                {"name": "fmt", "type_annotation": "str"},
            ],
            "return_type": "tuple[datetime, datetime]",
        },
    },
    {
        "code": (
            "def flatten(nested: list[Any], depth: int = 1) -> list[Any]:\n"
            '    """Flatten a nested list up to the given depth."""\n'
            "    if depth == 0:\n"
            "        return nested\n"
            "    result = []\n"
            "    for item in nested:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(flatten(item, depth - 1))\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result"
        ),
        "ground_truth": {
            "function_name": "flatten",
            "arguments": [
                {"name": "nested", "type_annotation": "list[Any]"},
                {"name": "depth", "type_annotation": "int"},
            ],
            "return_type": "list[Any]",
        },
    },
    {
        "code": (
            "def chunk_text(text: str, max_tokens: int, overlap: int = 0) -> list[str]:\n"
            '    """Split text into overlapping chunks of at most max_tokens words."""\n'
            "    words = text.split()\n"
            "    step = max(1, max_tokens - overlap)\n"
            "    return [' '.join(words[i:i+max_tokens]) for i in range(0, len(words), step)]"
        ),
        "ground_truth": {
            "function_name": "chunk_text",
            "arguments": [
                {"name": "text", "type_annotation": "str"},
                {"name": "max_tokens", "type_annotation": "int"},
                {"name": "overlap", "type_annotation": "int"},
            ],
            "return_type": "list[str]",
        },
    },
    {
        "code": (
            "def cosine_similarity(a: list[float], b: list[float]) -> float:\n"
            '    """Compute cosine similarity between two equal-length vectors."""\n'
            "    dot = sum(x * y for x, y in zip(a, b))\n"
            "    mag_a = sum(x**2 for x in a) ** 0.5\n"
            "    mag_b = sum(x**2 for x in b) ** 0.5\n"
            "    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0"
        ),
        "ground_truth": {
            "function_name": "cosine_similarity",
            "arguments": [
                {"name": "a", "type_annotation": "list[float]"},
                {"name": "b", "type_annotation": "list[float]"},
            ],
            "return_type": "float",
        },
    },
    {
        "code": (
            "class DataPipeline:\n"
            "    def transform(\n"
            "        self,\n"
            "        records: list[dict[str, Any]],\n"
            "        schema: dict[str, str],\n"
            "        strict: bool = False,\n"
            "    ) -> list[dict[str, Any]]:\n"
            '        """Apply schema-based type coercion to a list of records."""\n'
            "        return records"
        ),
        "ground_truth": {
            "function_name": "transform",
            "arguments": [
                {"name": "self", "type_annotation": "Any"},
                {"name": "records", "type_annotation": "list[dict[str, Any]]"},
                {"name": "schema", "type_annotation": "dict[str, str]"},
                {"name": "strict", "type_annotation": "bool"},
            ],
            "return_type": "list[dict[str, Any]]",
        },
    },
    {
        "code": (
            "def validate_config(\n"
            "    config: dict[str, Any],\n"
            "    required_keys: list[str],\n"
            "    defaults: dict[str, Any] | None = None,\n"
            ") -> dict[str, Any]:\n"
            '    """Validate a config dict, fill in defaults, and return the merged config."""\n'
            "    ..."
        ),
        "ground_truth": {
            "function_name": "validate_config",
            "arguments": [
                {"name": "config", "type_annotation": "dict[str, Any]"},
                {"name": "required_keys", "type_annotation": "list[str]"},
                {"name": "defaults", "type_annotation": "dict[str, Any] | None"},
            ],
            "return_type": "dict[str, Any]",
        },
    },
    {
        "code": (
            "def levenshtein_distance(s1: str, s2: str) -> int:\n"
            '    """Compute the Levenshtein edit distance between two strings."""\n'
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0] * (n + 1) for _ in range(m + 1)]\n"
            "    for i in range(m + 1):\n"
            "        dp[i][0] = i\n"
            "    for j in range(n + 1):\n"
            "        dp[0][j] = j\n"
            "    for i in range(1, m + 1):\n"
            "        for j in range(1, n + 1):\n"
            "            cost = 0 if s1[i-1] == s2[j-1] else 1\n"
            "            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)\n"
            "    return dp[m][n]"
        ),
        "ground_truth": {
            "function_name": "levenshtein_distance",
            "arguments": [
                {"name": "s1", "type_annotation": "str"},
                {"name": "s2", "type_annotation": "str"},
            ],
            "return_type": "int",
        },
    },
]


def _score_code_extraction(predicted: str, ground_truth: dict[str, Any]) -> float:
    """Score code extraction response against ground truth.

    Checks function name (50% weight) + argument names (50% weight).

    Returns:
        Float in [0.0, 1.0].
    """
    try:
        parsed = json.loads(predicted)
    except (json.JSONDecodeError, ValueError):
        return 0.0

    if not isinstance(parsed, dict):
        return 0.0

    score = 0.0

    # Function name match (50%)
    if parsed.get("function_name", "").strip() == ground_truth.get("function_name", ""):
        score += 0.5

    # Argument names match (50%)
    gt_args = {a["name"] for a in ground_truth.get("arguments", [])}
    pred_args_raw = parsed.get("arguments", [])
    pred_args: set[str] = set()
    if isinstance(pred_args_raw, list):
        for arg in pred_args_raw:
            if isinstance(arg, dict) and "name" in arg:
                pred_args.add(str(arg["name"]).strip())

    if gt_args:
        arg_score = len(gt_args & pred_args) / len(gt_args)
        score += 0.5 * arg_score

    return score


class CodeExtractionTask:
    """Code entity extraction benchmark task.

    Measures FormatShield's routing accuracy on Python code parsing,
    a MEDIUM complexity task expected to benefit from TTF routing.
    """

    name = "code_extraction"
    complexity = "MEDIUM"
    expected_ttf_benefit = True

    def get_problems(self, quick: bool = False) -> list[dict[str, Any]]:
        """Return benchmark problems.

        Args:
            quick: If True, return a small subset for CI/smoke tests.

        Returns:
            List of dicts with keys: 'prompt', 'ground_truth', 'schema'.
        """
        problems = []
        for p in _PROBLEMS:
            schema = CodeEntities.model_json_schema()
            problems.append(
                {
                    "prompt": (
                        "Extract the function name, arguments with types, return type, "
                        "and a one-sentence description from this code:\n\n"
                        f"```python\n{p['code']}\n```"
                    ),
                    "ground_truth": p["ground_truth"],
                    "schema": schema,
                }
            )
        return problems[:3] if quick else problems

    def score_response(self, predicted: str, ground_truth: Any) -> float:
        """Score a model response against ground truth.

        Args:
            predicted: Raw string output from the model (expected JSON).
            ground_truth: Ground truth dict with function_name, arguments, return_type.

        Returns:
            Float in [0.0, 1.0] where 1.0 = perfect match.
        """
        return _score_code_extraction(predicted, ground_truth)
