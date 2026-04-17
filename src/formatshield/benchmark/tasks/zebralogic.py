"""ZebraLogic benchmark task for FormatShield.

Logic puzzles requiring multi-step constraint satisfaction. Tests whether
constrained decoding hurts commitment tracking in complex reasoning.
Models must track which attribute belongs to which position across multiple clues.

This task measures format tax specifically on logic puzzles — where
JSON-constrained decoding often forces premature commitment before
all clues are integrated.
"""

from __future__ import annotations

import json
from typing import Any

_PROBLEMS: list[dict[str, Any]] = [
    {
        "prompt": (
            "Five people live in a row of houses numbered 1 to 5 (left to right). "
            "Each person has a unique nationality, drink, pet, color house, and hobby.\n\n"
            "Clues:\n"
            "1. The Brit lives in the red house.\n"
            "2. The Swede has a dog.\n"
            "3. The Dane drinks tea.\n"
            "4. The green house is immediately to the left of the white house.\n"
            "5. The green house owner drinks coffee.\n"
            "6. The person who smokes Pall Mall has birds.\n"
            "7. The owner of the yellow house smokes Dunhill.\n"
            "8. The man in the center house drinks milk.\n"
            "9. The Norwegian lives in the first house.\n"
            "10. The person who smokes Blends lives next to the one who has cats.\n"
            "11. The person who has horses lives next to the Dunhill smoker.\n"
            "12. The German smokes Prince.\n"
            "13. The Norwegian lives next to the blue house.\n"
            "14. The person who smokes Blends has a neighbor who drinks water.\n\n"
            "Who has the fish? "
            "(Note: in this simplified version, one person has fish as their pet.)\n"
            "Output the house number (1-5) of the fish owner."
        ),
        "ground_truth": {"fish_owner_house": 4, "nationality": "German"},
        "schema": {
            "type": "object",
            "properties": {
                "fish_owner_house": {"type": "integer", "minimum": 1, "maximum": 5},
                "reasoning": {"type": "string"},
                "nationality": {"type": "string"},
            },
            "required": ["fish_owner_house", "reasoning"],
        },
        "puzzle_type": "constraint_satisfaction",
        "n_constraints": 14,
    },
    {
        "prompt": (
            "Three people (Alice, Bob, Carol) each own one animal (cat, dog, fish) "
            "and one vehicle (car, bike, bus).\n\n"
            "Clues:\n"
            "1. The person with the cat does not have a car.\n"
            "2. Bob does not have a fish.\n"
            "3. The person with the bike has a dog.\n"
            "4. Alice does not have a bike.\n\n"
            "Who has the cat?"
        ),
        "ground_truth": {"person": "Carol", "animal": "cat", "vehicle": "bus"},
        "schema": {
            "type": "object",
            "properties": {
                "person": {"type": "string"},
                "animal": {"type": "string"},
                "vehicle": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["person", "reasoning"],
        },
        "puzzle_type": "constraint_satisfaction",
        "n_constraints": 4,
    },
    {
        "prompt": (
            "Four friends (Anna, Ben, Clara, David) sit in a row of seats 1-4.\n\n"
            "Clues:\n"
            "1. Anna sits next to Ben.\n"
            "2. Clara is not in seat 1 or 4.\n"
            "3. David is in seat 1 or 4.\n"
            "4. Ben is to the right of Anna.\n\n"
            "What seat is Clara in? (Output seat number 1-4)"
        ),
        "ground_truth": {
            "clara_seat": 2,
            "arrangement": "Anna-Ben-Clara-David or David-Clara-Anna-Ben",
        },
        "schema": {
            "type": "object",
            "properties": {
                "clara_seat": {"type": "integer", "minimum": 1, "maximum": 4},
                "reasoning": {"type": "string"},
            },
            "required": ["clara_seat", "reasoning"],
        },
        "puzzle_type": "seating",
        "n_constraints": 4,
    },
    {
        "prompt": (
            "A farmer needs to cross a river with a fox, a chicken, and a bag of grain. "
            "The boat holds the farmer and one item. The fox eats the chicken if left alone. "
            "The chicken eats the grain if left alone.\n\n"
            "What is the minimum number of river crossings needed to get everything across?"
        ),
        "ground_truth": {"min_crossings": 7},
        "schema": {
            "type": "object",
            "properties": {
                "min_crossings": {"type": "integer"},
                "reasoning": {"type": "string"},
            },
            "required": ["min_crossings", "reasoning"],
        },
        "puzzle_type": "river_crossing",
        "n_constraints": 2,
    },
    {
        "prompt": (
            "In a tournament, 5 teams (A, B, C, D, E) each play every other team once. "
            "Results: A beat B, A beat C, B beat C, B beat D, C beat D, C beat E, D beat E, "
            "A beat D, B beat E, A beat E.\n\n"
            "Rank the teams from 1st to 5th by number of wins."
        ),
        "ground_truth": {
            "ranking": ["A", "B", "C", "D", "E"],
            "wins": {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0},
        },
        "schema": {
            "type": "object",
            "properties": {
                "ranking": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 5,
                    "maxItems": 5,
                },
                "reasoning": {"type": "string"},
            },
            "required": ["ranking", "reasoning"],
        },
        "puzzle_type": "ranking",
        "n_constraints": 10,
    },
    {
        "prompt": (
            "Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. "
            "All three labels are WRONG. You may draw one fruit from one box. "
            "From which box should you draw to determine all three contents?"
        ),
        "ground_truth": {
            "box": "Mixed",
            "reasoning": "Drawing from the mislabeled Mixed box reveals all",
        },
        "schema": {
            "type": "object",
            "properties": {
                "box": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["box", "reasoning"],
        },
        "puzzle_type": "logic",
        "n_constraints": 1,
    },
    {
        "prompt": (
            "You have 9 balls, one of which is slightly heavier. "
            "You have a balance scale. What is the minimum number of weighings "
            "needed to guarantee finding the heavy ball?"
        ),
        "ground_truth": {"min_weighings": 2},
        "schema": {
            "type": "object",
            "properties": {
                "min_weighings": {"type": "integer"},
                "reasoning": {"type": "string"},
            },
            "required": ["min_weighings", "reasoning"],
        },
        "puzzle_type": "optimization",
        "n_constraints": 1,
    },
    {
        "prompt": (
            "Five colored blocks (red, blue, green, yellow, white) are stacked.\n\n"
            "Clues:\n"
            "1. Red is directly above blue.\n"
            "2. Green is somewhere above red.\n"
            "3. Yellow is at the bottom.\n"
            "4. White is directly above yellow.\n\n"
            "What is the order from bottom to top?"
        ),
        "ground_truth": {"order": ["yellow", "white", "blue", "red", "green"]},
        "schema": {
            "type": "object",
            "properties": {
                "order": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 5,
                    "maxItems": 5,
                },
                "reasoning": {"type": "string"},
            },
            "required": ["order", "reasoning"],
        },
        "puzzle_type": "ordering",
        "n_constraints": 4,
    },
    {
        "prompt": (
            "A, B, C, D are four suspects. Exactly one is guilty.\n"
            "A says: 'I am innocent.'\n"
            "B says: 'A is guilty.'\n"
            "C says: 'B is innocent.'\n"
            "D says: 'C is innocent.'\n"
            "Only the guilty person lies. Who is guilty?"
        ),
        "ground_truth": {"guilty": "B"},
        "schema": {
            "type": "object",
            "properties": {
                "guilty": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "reasoning": {"type": "string"},
            },
            "required": ["guilty", "reasoning"],
        },
        "puzzle_type": "truth_teller",
        "n_constraints": 4,
    },
    {
        "prompt": (
            "Six houses in a street are colored differently: "
            "red, blue, green, yellow, orange, purple.\n\n"
            "Clues:\n"
            "1. The red house is not next to the blue house.\n"
            "2. Green is to the right of yellow.\n"
            "3. Orange is between red and purple.\n"
            "4. Blue is at position 1 or 6.\n"
            "5. Purple is at an even position.\n"
            "6. Red is at position 3.\n\n"
            "What color is at position 5?"
        ),
        "ground_truth": {"position_5": "orange"},
        "schema": {
            "type": "object",
            "properties": {
                "position_5": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["position_5", "reasoning"],
        },
        "puzzle_type": "constraint_satisfaction",
        "n_constraints": 6,
    },
]


def get_problems(quick: bool = False) -> list[dict[str, Any]]:
    """Return ZebraLogic benchmark problems.

    Args:
        quick: If True, return a small subset for CI/smoke tests.

    Returns:
        List of problem dicts with keys: 'prompt', 'ground_truth', 'schema',
        'puzzle_type', 'n_constraints'.
    """
    return _PROBLEMS[:2] if quick else _PROBLEMS


def score_response(predicted: str, ground_truth: Any) -> float:
    """Score a model response against the ground truth.

    Tries JSON parsing first, then returns 0.0 on failure.
    Partial credit: each correct field worth an equal share.

    Args:
        predicted: Raw string output from the model.
        ground_truth: Dict of expected field values from get_problems().

    Returns:
        Float in [0.0, 1.0]. 1.0 = all fields match. Partial credit given.
    """
    if not isinstance(ground_truth, dict):
        return 0.0

    try:
        parsed = json.loads(predicted)
    except (json.JSONDecodeError, ValueError):
        return 0.0

    if not isinstance(parsed, dict):
        return 0.0

    # Score each field that exists in ground truth (excluding reasoning).
    scoreable_fields = {k: v for k, v in ground_truth.items() if k != "reasoning"}
    if not scoreable_fields:
        return 0.0

    correct = 0
    for field, expected in scoreable_fields.items():
        predicted_val = parsed.get(field)
        if isinstance(expected, list) and isinstance(predicted_val, list):
            # Lists: check if they match exactly (order matters for rankings).
            if [str(x).lower() for x in predicted_val] == [str(x).lower() for x in expected]:
                correct += 1
        elif isinstance(expected, int | float):
            # Numbers: allow for string representation.
            if predicted_val is not None:
                try:
                    if int(predicted_val) == int(expected):
                        correct += 1
                except (TypeError, ValueError):
                    pass
        elif isinstance(expected, str):
            if str(predicted_val).strip().lower() == expected.strip().lower():
                correct += 1
        elif isinstance(expected, dict):
            # Nested dict: give partial credit for structure match.
            if isinstance(predicted_val, dict):
                correct += 0.5

    return correct / len(scoreable_fields)
