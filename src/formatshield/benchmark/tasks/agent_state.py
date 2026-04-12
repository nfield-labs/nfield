"""
AgentStateTask — Agent state tracking benchmark task for FormatShield.

This task contains 12 hardcoded narrative descriptions of an agent performing
a sequence of actions.  Models must read the narrative and extract the agent's
current state into a structured :class:`AgentState` response.

Tracking which steps are complete, which are pending, and what blockers
currently exist requires careful reading of multi-step narratives, making
this task HIGH complexity with an expected benefit from TTF routing.

Complexity: HIGH
Expected TTF benefit: True
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """Structured schema representing the current state of an autonomous agent."""

    current_goal: str
    """The agent's active high-level objective."""

    completed_steps: list[str]
    """Steps the agent has already successfully completed."""

    pending_steps: list[str]
    """Steps that still need to be done to achieve the current goal."""

    blockers: list[str]
    """Issues or dependencies currently preventing progress."""

    confidence: float
    """Agent's estimated confidence it can complete the goal, in [0.0, 1.0]."""


# ---------------------------------------------------------------------------
# 12 hardcoded agent narrative problems
# ---------------------------------------------------------------------------

_PROBLEMS: list[dict[str, Any]] = [
    {
        "narrative": (
            "The research agent was tasked with preparing a competitive analysis report on "
            "the electric vehicle market.  It has already completed a web search for the top "
            "five EV manufacturers and downloaded their latest annual reports.  It has also "
            "extracted revenue and market share data from those reports.  Currently it is "
            "trying to access a premium industry database, but the API key it was given has "
            "expired and no replacement has been provided.  Once database access is restored, "
            "it still needs to synthesise the data into a narrative and format the final report."
        ),
        "current_goal": "Prepare a competitive analysis report on the electric vehicle market",
        "completed_steps_min": 2,
    },
    {
        "narrative": (
            "A deployment agent was instructed to roll out a new microservice to production. "
            "It has pulled the latest Docker image from the container registry, run the "
            "integration test suite (all 312 tests passed), and successfully deployed the "
            "service to the staging environment.  Promotion to production is blocked because "
            "the required security scan has not yet been approved by the infosec team.  "
            "After approval, the agent still needs to update the load balancer configuration "
            "and notify the on-call engineer."
        ),
        "current_goal": "Roll out the new microservice to production",
        "completed_steps_min": 2,
    },
    {
        "narrative": (
            "The data-pipeline agent was asked to migrate historical sales records from a "
            "legacy PostgreSQL database to the new data warehouse.  It has finished "
            "schema mapping, created the target tables, and migrated records for the years "
            "2015 through 2020.  Migration of 2021–2023 data is in progress but has stalled "
            "because several rows contain malformed date values that fail the warehouse's "
            "validation checks.  The agent is waiting for a data-cleaning script from the "
            "engineering team before it can continue.  Post-migration validation and indexing "
            "remain outstanding."
        ),
        "current_goal": "Migrate historical sales records to the new data warehouse",
        "completed_steps_min": 2,
    },
    {
        "narrative": (
            "A scheduling agent was given the goal of booking a week-long team offsite. "
            "It has already polled all twelve team members for their availability, "
            "identified three possible date windows where everyone is free, and obtained "
            "quotes from four venue providers.  It has just sent a shortlist of venues to "
            "the team manager for approval.  Once the manager responds, the agent needs to "
            "confirm the venue booking, arrange group travel, and book accommodation for "
            "remote attendees.  No blockers exist at the moment."
        ),
        "current_goal": "Book a week-long team offsite",
        "completed_steps_min": 3,
    },
    {
        "narrative": (
            "The code-review agent was assigned to audit the authentication module for "
            "security vulnerabilities.  It has scanned the codebase with a static analysis "
            "tool and reviewed the output — finding three high-severity issues.  It has "
            "drafted a remediation report detailing each vulnerability and proposed fixes.  "
            "The report is ready but the agent cannot submit it because the project's issue "
            "tracker is undergoing scheduled maintenance and the API is returning 503 errors. "
            "After submission, the agent still needs to verify the fixes once the developers "
            "address the issues."
        ),
        "current_goal": "Audit the authentication module for security vulnerabilities",
        "completed_steps_min": 2,
    },
    {
        "narrative": (
            "An e-commerce fulfilment agent was instructed to process all outstanding "
            "refund requests from the previous 48 hours.  It has identified 47 eligible "
            "refund requests, validated each one against the return policy, and approved "
            "39 of them.  Refunds for 8 orders have been flagged for manual review because "
            "the customers' stated reason does not match the product category's return "
            "window.  The agent has notified the customer support team about the flagged "
            "cases and is waiting for their decisions before closing those tickets.  "
            "It still needs to send confirmation emails to all approved refund recipients."
        ),
        "current_goal": "Process all outstanding refund requests from the previous 48 hours",
        "completed_steps_min": 2,
    },
    {
        "narrative": (
            "A document-translation agent was tasked with translating a 200-page technical "
            "manual from German to English.  It has translated chapters 1 through 8, "
            "applied terminology consistency checks, and formatted the output to match the "
            "original layout.  It is currently working on chapter 9 but has encountered a "
            "section containing highly specialised metallurgy terminology that its glossary "
            "does not cover.  It has requested a domain expert to review that section. "
            "Chapters 10 through 15 and the final proofreading pass are still pending."
        ),
        "current_goal": "Translate a 200-page technical manual from German to English",
        "completed_steps_min": 2,
    },
    {
        "narrative": (
            "The monitoring agent was given the objective of diagnosing a production latency "
            "spike reported at 14:32 UTC.  It has gathered metrics from the APM dashboard, "
            "correlated the spike with a deployment that occurred at 14:28 UTC, and confirmed "
            "that the new version introduced an inefficient database query on the user-profile "
            "endpoint.  It has already triggered an automatic rollback of the deployment.  "
            "It is now waiting to confirm that latency metrics return to baseline before "
            "writing a post-incident report and alerting the engineering lead."
        ),
        "current_goal": "Diagnose and resolve the production latency spike reported at 14:32 UTC",
        "completed_steps_min": 3,
    },
    {
        "narrative": (
            "A content-moderation agent was assigned to review flagged posts from the past "
            "24 hours.  It has reviewed 230 of the 310 flagged posts, removing 88 that "
            "violated community guidelines and clearing 142 as false positives.  Progress "
            "has slowed because 40 posts require a second human review under the platform's "
            "edge-case policy, and the human review queue is backed up by three hours.  "
            "The remaining 80 unflagged posts have not been started yet."
        ),
        "current_goal": "Review all flagged posts from the past 24 hours",
        "completed_steps_min": 1,
    },
    {
        "narrative": (
            "A financial reconciliation agent was instructed to reconcile last month's "
            "expense reports against the company's bank statements.  It has ingested all "
            "432 expense report line items and all 518 bank transactions, matched 410 of "
            "the line items to corresponding transactions, and flagged 22 unmatched items. "
            "It has also categorised each matched transaction by department.  It is now "
            "attempting to resolve the unmatched items but is blocked because it lacks "
            "write access to the accounting system needed to post manual journal entries.  "
            "It still needs to generate the final reconciliation summary report."
        ),
        "current_goal": "Reconcile last month's expense reports against bank statements",
        "completed_steps_min": 3,
    },
    {
        "narrative": (
            "The onboarding agent was tasked with setting up access and accounts for a new "
            "engineer joining the team.  It has created the employee's Google Workspace "
            "account, added them to the correct GitHub organisation with the appropriate "
            "team permissions, provisioned their Jira and Confluence access, and sent a "
            "welcome email with first-day instructions.  It is waiting for IT to ship the "
            "hardware before it can enrol the laptop in mobile device management.  "
            "After MDM enrolment, a final access audit is still needed."
        ),
        "current_goal": "Set up access and accounts for a new engineer joining the team",
        "completed_steps_min": 4,
    },
    {
        "narrative": (
            "A marketing automation agent was given the goal of launching an email campaign "
            "for the upcoming product release.  It has drafted three email variants for A/B "
            "testing, generated a segmented audience list of 45,000 subscribers, and "
            "configured the send schedule in the email platform.  It submitted the campaign "
            "for compliance review this morning.  The legal team has placed a hold on the "
            "campaign pending approval of updated privacy disclosures required by new "
            "regulations.  Once approved, the agent is ready to launch and then monitor "
            "open and click-through rates over the first 72 hours."
        ),
        "current_goal": "Launch an email campaign for the upcoming product release",
        "completed_steps_min": 3,
    },
]

# Quick-mode uses the first 5 problems only
_QUICK_SLICE = 5


class AgentStateTask:
    """
    Agent state tracking benchmark task.

    Contains 12 hardcoded narratives describing an autonomous agent performing
    a sequence of actions toward a high-level goal.  Models must extract the
    agent's current state — its goal, completed steps, pending steps, blockers,
    and confidence — into a structured :class:`AgentState` response.

    Scoring checks structural completeness: the ``current_goal`` field must be
    non-empty and ``completed_steps`` must contain at least one entry.

    Attributes
    ----------
    name:
        Stable task identifier used in benchmark result records.
    expected_ttf_benefit:
        ``True`` because tracking multi-step narratives and distinguishing
        completed from pending actions requires careful sequential reasoning.
    schema:
        The Pydantic model class that defines the expected output shape.
    complexity:
        Qualitative complexity label consumed by the harness for reporting.
    """

    name: str = "agent_state"
    expected_ttf_benefit: bool = True
    schema = AgentState
    complexity: str = "HIGH"

    def get_problems(self, quick: bool = False) -> list[dict[str, Any]]:
        """
        Return the list of benchmark problems.

        Parameters
        ----------
        quick:
            When ``True`` returns only the first 5 problems, enabling fast
            smoke-test runs without hitting rate limits or wasting API budget.

        Returns
        -------
        list[dict]
            Each element has keys:

            ``"narrative"`` : str
                The agent action narrative passed to the model.
            ``"current_goal"`` : str
                Ground-truth current goal string.
            ``"completed_steps_min"`` : int
                Minimum number of completed steps expected in a correct
                response (informational; scoring checks >= 1).
        """
        problems = _PROBLEMS[:_QUICK_SLICE] if quick else _PROBLEMS
        return [
            {
                "narrative": p["narrative"],
                "current_goal": p["current_goal"],
                "completed_steps_min": p["completed_steps_min"],
            }
            for p in problems
        ]

    def score_response(self, predicted: dict[str, Any], ground_truth: dict[str, Any]) -> float:
        """
        Score a model response based on structural completeness.

        A response is considered correct when ``current_goal`` is a non-empty
        string AND ``completed_steps`` contains at least one non-empty entry.
        This verifies that the model extracted the core state elements rather
        than returning an empty or malformed structure.

        Parameters
        ----------
        predicted:
            A dict representation of an :class:`AgentState` instance produced
            by the model.  Expected keys: ``current_goal``, ``completed_steps``,
            ``pending_steps``, ``blockers``, ``confidence``.
        ground_truth:
            The annotated ground-truth dict for the narrative (used for
            logging context; primary scoring is structural).

        Returns
        -------
        float
            ``1.0`` if ``current_goal`` is non-empty AND ``completed_steps``
            has at least one non-empty entry, otherwise ``0.0``.
        """
        if not isinstance(predicted, dict):
            logger.debug("score_response: predicted is not a dict, got %r", type(predicted))
            return 0.0

        goal = predicted.get("current_goal") or ""
        completed = predicted.get("completed_steps") or []

        if not isinstance(completed, list):
            completed = []

        goal_ok = bool(str(goal).strip())
        completed_ok = len([s for s in completed if str(s).strip()]) >= 1

        if goal_ok and completed_ok:
            return 1.0

        logger.debug(
            "score_response: missing required fields — goal_ok=%r completed_ok=%r",
            goal_ok,
            completed_ok,
        )
        return 0.0

    def build_prompt(self, narrative: str) -> str:
        """
        Construct the full prompt string sent to the model.

        Parameters
        ----------
        narrative:
            The agent action narrative describing what the agent has done and
            what remains to be completed.

        Returns
        -------
        str
            A formatted prompt instructing the model to extract agent state
            as structured JSON.
        """
        return (
            "You are an agent state tracker.  Read the following narrative describing what "
            "an autonomous agent has done and what still needs to happen.  Extract the "
            "agent's current state and return it as structured JSON.\n\n"
            "Return a JSON object with exactly these five fields:\n"
            "  - current_goal: the agent's active high-level objective (string)\n"
            "  - completed_steps: list of steps the agent has already finished\n"
            "  - pending_steps: list of steps that still need to be done\n"
            "  - blockers: list of issues currently preventing progress (empty list if none)\n"
            "  - confidence: estimated probability the agent can complete the goal, "
            "as a float in [0.0, 1.0]\n\n"
            "Return only the JSON object.  Do not include text outside the JSON.\n\n"
            f"Agent narrative:\n{narrative}"
        )
