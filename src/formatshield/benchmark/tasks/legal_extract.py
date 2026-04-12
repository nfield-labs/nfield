"""
LegalExtractTask — Legal entity extraction benchmark task for FormatShield.

This task contains 15 hardcoded contract and legal clause snippets.  Models
must extract key legal entities — parties, effective date, obligations, and
termination conditions — into a structured :class:`LegalEntities` response.

Because legal text is dense, ambiguous, and requires understanding clause
structure, this task is classified HIGH complexity and is expected to benefit
from TTF routing.

Complexity: HIGH
Expected TTF benefit: True
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LegalEntities(BaseModel):
    """Structured schema for legal entity extraction from contract clauses."""

    parties: list[str]
    """Names of all contracting parties identified in the clause."""

    effective_date: str
    """The date on which the contract or clause becomes effective."""

    obligations: list[str]
    """Specific duties or obligations imposed on the parties by the clause."""

    termination_conditions: list[str]
    """Conditions or events under which the contract may be terminated."""


# ---------------------------------------------------------------------------
# 15 hardcoded contract / legal clause problems
# ---------------------------------------------------------------------------

_PROBLEMS: list[dict[str, Any]] = [
    {
        "text": (
            "This Software License Agreement ('Agreement') is entered into as of January 1, 2024 "
            "('Effective Date') by and between Nexus Technologies Inc. ('Licensor') and Orion "
            "Solutions LLC ('Licensee').  Licensee shall pay an annual license fee of $50,000 "
            "within 30 days of the Effective Date.  Licensor shall provide technical support "
            "during normal business hours.  Either party may terminate this Agreement upon "
            "60 days' written notice, or immediately upon a material breach that remains uncured "
            "for 30 days after written notice."
        ),
        "parties": ["Nexus Technologies Inc.", "Orion Solutions LLC"],
        "effective_date": "January 1, 2024",
        "obligations": [
            "Licensee shall pay annual license fee of $50,000 within 30 days",
            "Licensor shall provide technical support during normal business hours",
        ],
        "termination_conditions": [
            "60 days written notice by either party",
            "immediately upon material breach uncured for 30 days",
        ],
    },
    {
        "text": (
            "This Non-Disclosure Agreement is made effective as of March 15, 2023, between "
            "Vertex Capital Partners ('Disclosing Party') and Alpine Research Group ('Receiving "
            "Party').  The Receiving Party agrees to hold all confidential information in strict "
            "confidence and not to disclose it to any third party without prior written consent. "
            "The Receiving Party shall use the confidential information solely for evaluating a "
            "potential business relationship.  This Agreement terminates automatically after two "
            "years, or immediately if the Receiving Party breaches any confidentiality obligation."
        ),
        "parties": ["Vertex Capital Partners", "Alpine Research Group"],
        "effective_date": "March 15, 2023",
        "obligations": [
            "Receiving Party shall hold confidential information in strict confidence",
            "Receiving Party shall not disclose to third parties without written consent",
            "Receiving Party shall use information solely for evaluating potential"
            " business relationship",
        ],
        "termination_conditions": [
            "automatically after two years",
            "immediately upon breach of confidentiality obligation",
        ],
    },
    {
        "text": (
            "This Services Agreement ('Agreement') is dated February 28, 2024, and is between "
            "Brightline Consulting Group ('Consultant') and Meadowbrook Financial Services "
            "('Client').  Consultant shall deliver a written project report within 90 days of "
            "commencement.  Client shall remit payment of $120,000 in three equal instalments "
            "on the first day of each project month.  The Client may terminate this Agreement "
            "for convenience upon 30 days' written notice; the Consultant may terminate if "
            "any instalment payment is not received within 15 days of its due date."
        ),
        "parties": ["Brightline Consulting Group", "Meadowbrook Financial Services"],
        "effective_date": "February 28, 2024",
        "obligations": [
            "Consultant shall deliver written project report within 90 days",
            "Client shall remit payment of $120,000 in three equal instalments",
        ],
        "termination_conditions": [
            "Client may terminate for convenience upon 30 days written notice",
            "Consultant may terminate if instalment payment not received"
            " within 15 days of due date",
        ],
    },
    {
        "text": (
            "Effective July 1, 2022, Ironclad Manufacturing Corp. ('Supplier') and Cascade "
            "Retail Holdings ('Buyer') enter into this Supply Agreement.  Supplier shall "
            "deliver a minimum of 500 units per month in accordance with the delivery schedule "
            "attached hereto as Exhibit A.  Buyer shall issue purchase orders at least 45 days "
            "in advance and pay invoices within 30 days of receipt.  This Agreement may be "
            "terminated by either party upon 90 days' written notice or immediately by Buyer "
            "if Supplier fails to meet quality standards set forth in Exhibit B for three "
            "consecutive months."
        ),
        "parties": ["Ironclad Manufacturing Corp.", "Cascade Retail Holdings"],
        "effective_date": "July 1, 2022",
        "obligations": [
            "Supplier shall deliver minimum 500 units per month per Exhibit A",
            "Buyer shall issue purchase orders at least 45 days in advance",
            "Buyer shall pay invoices within 30 days of receipt",
        ],
        "termination_conditions": [
            "90 days written notice by either party",
            "immediately by Buyer if Supplier fails quality standards for three consecutive months",
        ],
    },
    {
        "text": (
            "This Employment Agreement is entered into on September 1, 2023 between Summit "
            "Pharmaceuticals Ltd. ('Employer') and Dr. Eleanor Voss ('Employee').  Employee "
            "shall devote her full professional time to the duties of Chief Scientific Officer "
            "and shall not engage in any competing employment without prior written approval. "
            "Employer shall pay Employee an annual base salary of $280,000 plus a performance "
            "bonus of up to 30% of base salary.  Either party may terminate this Agreement "
            "with 90 days' written notice; Employer may terminate immediately for cause, "
            "including gross misconduct or material breach of fiduciary duty."
        ),
        "parties": ["Summit Pharmaceuticals Ltd.", "Dr. Eleanor Voss"],
        "effective_date": "September 1, 2023",
        "obligations": [
            "Employee shall devote full professional time to Chief Scientific Officer duties",
            "Employee shall not engage in competing employment without written approval",
            "Employer shall pay annual base salary of $280,000 plus performance bonus",
        ],
        "termination_conditions": [
            "90 days written notice by either party",
            "immediately for cause including gross misconduct or material breach of fiduciary duty",
        ],
    },
    {
        "text": (
            "This Lease Agreement is effective October 1, 2023 between Harrington Property "
            "Group ('Landlord') and BlueSky Startups Inc. ('Tenant').  Tenant shall pay "
            "monthly rent of $8,500 on the first of each month without deduction or set-off. "
            "Tenant shall maintain the premises in good condition and obtain Landlord's prior "
            "written consent for any alterations.  This Lease terminates on September 30, 2026; "
            "Landlord may terminate early if Tenant fails to pay rent for two consecutive months "
            "or causes material damage to the property."
        ),
        "parties": ["Harrington Property Group", "BlueSky Startups Inc."],
        "effective_date": "October 1, 2023",
        "obligations": [
            "Tenant shall pay monthly rent of $8,500 on the first of each month",
            "Tenant shall maintain premises in good condition",
            "Tenant shall obtain Landlord written consent for alterations",
        ],
        "termination_conditions": [
            "terminates on September 30, 2026",
            "Landlord may terminate if Tenant fails to pay rent for two consecutive months",
            "Landlord may terminate if Tenant causes material damage",
        ],
    },
    {
        "text": (
            "This Joint Venture Agreement ('Agreement') is dated May 1, 2024 and is made "
            "between Titan Energy Corp. ('Party A') and Solaris Renewables BV ('Party B'). "
            "Each party shall contribute $5,000,000 in capital to the joint venture entity "
            "within 60 days of the Effective Date.  Party A shall manage day-to-day operations; "
            "Party B shall provide proprietary solar technology under a separate licence. "
            "The Agreement may be terminated by mutual written consent of both parties, or by "
            "either party if the other becomes insolvent or files for bankruptcy protection."
        ),
        "parties": ["Titan Energy Corp.", "Solaris Renewables BV"],
        "effective_date": "May 1, 2024",
        "obligations": [
            "Each party shall contribute $5,000,000 in capital within 60 days",
            "Party A shall manage day-to-day operations",
            "Party B shall provide proprietary solar technology under separate licence",
        ],
        "termination_conditions": [
            "mutual written consent of both parties",
            "either party may terminate if the other becomes insolvent or files for bankruptcy",
        ],
    },
    {
        "text": (
            "Effective as of August 15, 2022, this Distribution Agreement is entered into by "
            "Arcadia Foods International ('Manufacturer') and Pacific Rim Distributors Co. "
            "('Distributor').  Manufacturer grants Distributor the exclusive right to distribute "
            "its products in the territory of Australia and New Zealand.  Distributor shall "
            "achieve minimum annual sales targets of AUD 2,000,000 and shall not distribute "
            "competing products without written consent.  Either party may terminate upon "
            "120 days' notice; Manufacturer may terminate immediately if Distributor fails "
            "to meet annual sales targets for two consecutive years."
        ),
        "parties": ["Arcadia Foods International", "Pacific Rim Distributors Co."],
        "effective_date": "August 15, 2022",
        "obligations": [
            "Manufacturer grants exclusive distribution rights in Australia and New Zealand",
            "Distributor shall achieve minimum annual sales of AUD 2,000,000",
            "Distributor shall not distribute competing products without written consent",
        ],
        "termination_conditions": [
            "120 days notice by either party",
            "immediately by Manufacturer if Distributor fails annual sales targets"
            " two consecutive years",
        ],
    },
    {
        "text": (
            "This Research Collaboration Agreement is effective as of November 1, 2023 between "
            "the University of Westfield ('University') and BioCore Therapeutics Inc. "
            "('Company').  University shall conduct research as described in Annex I and "
            "provide quarterly progress reports.  Company shall fund the research with "
            "$750,000 per year payable in quarterly instalments.  The Agreement shall "
            "continue for three years unless terminated earlier by mutual consent, or by "
            "University if Company fails to make any payment within 45 days of its due date."
        ),
        "parties": ["University of Westfield", "BioCore Therapeutics Inc."],
        "effective_date": "November 1, 2023",
        "obligations": [
            "University shall conduct research as described in Annex I",
            "University shall provide quarterly progress reports",
            "Company shall fund research with $750,000 per year in quarterly instalments",
        ],
        "termination_conditions": [
            "after three years unless terminated earlier",
            "mutual consent",
            "University may terminate if Company fails payment within 45 days of due date",
        ],
    },
    {
        "text": (
            "This Technology Transfer Agreement ('Agreement') is dated April 10, 2024 between "
            "Quantum Dynamics GmbH ('Transferor') and Vega Systems Inc. ('Transferee'). "
            "Transferor shall deliver all technical documentation and source code within "
            "30 days of the Effective Date.  Transferee shall pay a lump-sum transfer fee "
            "of EUR 2,500,000 upon signing and a royalty of 3% on net sales using the "
            "transferred technology.  The Agreement terminates if Transferee fails to "
            "commercialise the technology within five years, or upon material breach by "
            "either party that is not remedied within 60 days of written notice."
        ),
        "parties": ["Quantum Dynamics GmbH", "Vega Systems Inc."],
        "effective_date": "April 10, 2024",
        "obligations": [
            "Transferor shall deliver technical documentation and source code within 30 days",
            "Transferee shall pay lump-sum of EUR 2,500,000 upon signing",
            "Transferee shall pay 3% royalty on net sales using transferred technology",
        ],
        "termination_conditions": [
            "if Transferee fails to commercialise technology within five years",
            "material breach by either party not remedied within 60 days of written notice",
        ],
    },
    {
        "text": (
            "This Franchise Agreement is made as of June 1, 2023 between FastBite Brands Corp. "
            "('Franchisor') and Harper Family Restaurants LLC ('Franchisee').  Franchisee shall "
            "operate the franchise location in strict accordance with the Operations Manual and "
            "pay a weekly royalty of 6% of gross revenues.  Franchisor shall provide initial "
            "training for up to ten employees and ongoing marketing support.  Franchisor may "
            "terminate this Agreement immediately if Franchisee violates food safety regulations "
            "or fails to pay royalties for 14 consecutive days."
        ),
        "parties": ["FastBite Brands Corp.", "Harper Family Restaurants LLC"],
        "effective_date": "June 1, 2023",
        "obligations": [
            "Franchisee shall operate in strict accordance with Operations Manual",
            "Franchisee shall pay weekly royalty of 6% of gross revenues",
            "Franchisor shall provide initial training for up to ten employees",
            "Franchisor shall provide ongoing marketing support",
        ],
        "termination_conditions": [
            "immediately if Franchisee violates food safety regulations",
            "immediately if Franchisee fails to pay royalties for 14 consecutive days",
        ],
    },
    {
        "text": (
            "This Asset Purchase Agreement is entered into on December 1, 2023 between "
            "Sterling Media Holdings ('Seller') and Harbour Point Investments ('Buyer'). "
            "Seller agrees to transfer ownership of the identified assets free and clear "
            "of all liens and encumbrances by the closing date.  Buyer shall pay the "
            "purchase price of $4,200,000 at closing by wire transfer.  The Agreement "
            "may be terminated by either party if closing does not occur by February 28, 2024, "
            "or if any representation or warranty of the other party is materially inaccurate."
        ),
        "parties": ["Sterling Media Holdings", "Harbour Point Investments"],
        "effective_date": "December 1, 2023",
        "obligations": [
            "Seller shall transfer assets free and clear of all liens by closing date",
            "Buyer shall pay $4,200,000 at closing by wire transfer",
        ],
        "termination_conditions": [
            "if closing does not occur by February 28, 2024",
            "if any representation or warranty is materially inaccurate",
        ],
    },
    {
        "text": (
            "This Marketing Services Agreement is effective from January 15, 2024 between "
            "Prism Digital Agency ('Agency') and Coral Cosmetics Group ('Brand').  Agency "
            "shall develop and execute digital marketing campaigns across all agreed channels "
            "and deliver monthly performance reports by the 5th of each following month. "
            "Brand shall provide all required creative assets within 10 business days of "
            "each campaign brief and pay Agency a monthly retainer of $25,000.  Either party "
            "may terminate this Agreement with 30 days' written notice."
        ),
        "parties": ["Prism Digital Agency", "Coral Cosmetics Group"],
        "effective_date": "January 15, 2024",
        "obligations": [
            "Agency shall develop and execute digital marketing campaigns",
            "Agency shall deliver monthly performance reports by the 5th of each following month",
            "Brand shall provide creative assets within 10 business days of campaign brief",
            "Brand shall pay monthly retainer of $25,000",
        ],
        "termination_conditions": [
            "30 days written notice by either party",
        ],
    },
    {
        "text": (
            "This Shareholders' Agreement is dated March 1, 2024 between Pinnacle Ventures "
            "('Investor') and GreenLeaf Technologies ('Company') and its founders, "
            "Maya Patel and Riku Tanaka (collectively 'Founders').  The Investor shall "
            "subscribe for 20% equity in the Company at a pre-money valuation of $10,000,000. "
            "Founders shall not transfer or encumber their shares for a period of 36 months "
            "without Investor's prior written consent.  The Agreement terminates upon an "
            "initial public offering of the Company's shares, a trade sale, or unanimous "
            "written consent of all shareholders."
        ),
        "parties": ["Pinnacle Ventures", "GreenLeaf Technologies", "Maya Patel", "Riku Tanaka"],
        "effective_date": "March 1, 2024",
        "obligations": [
            "Investor shall subscribe for 20% equity at pre-money valuation of $10,000,000",
            "Founders shall not transfer or encumber shares for 36 months without written consent",
        ],
        "termination_conditions": [
            "upon initial public offering of the Company",
            "upon trade sale",
            "unanimous written consent of all shareholders",
        ],
    },
    {
        "text": (
            "This Indemnification and Hold Harmless Agreement is effective as of October 10, "
            "2023 between Atlas Construction LLC ('Contractor') and Riverbend Development "
            "Corp. ('Owner').  Contractor shall indemnify, defend, and hold Owner harmless "
            "from any claims, liabilities, or costs arising from Contractor's work on the "
            "project site.  Contractor shall maintain general liability insurance of not less "
            "than $5,000,000 per occurrence throughout the project.  This Agreement "
            "terminates upon final acceptance of the completed project by Owner, or upon "
            "written agreement of both parties to terminate earlier."
        ),
        "parties": ["Atlas Construction LLC", "Riverbend Development Corp."],
        "effective_date": "October 10, 2023",
        "obligations": [
            "Contractor shall indemnify and hold Owner harmless from project-related claims",
            "Contractor shall maintain general liability insurance of at least"
            " $5,000,000 per occurrence",
        ],
        "termination_conditions": [
            "upon final acceptance of completed project by Owner",
            "written agreement of both parties to terminate earlier",
        ],
    },
]

# Quick-mode uses the first 5 problems only
_QUICK_SLICE = 5


class LegalExtractTask:
    """
    Legal entity extraction benchmark task.

    Contains 15 hardcoded contract and legal clause snippets pre-annotated
    with four entity categories: parties, effective date, obligations, and
    termination conditions.

    The model must extract these into a structured :class:`LegalEntities`
    response.  Scoring checks structural completeness: both the parties list
    and the obligations list must be non-empty to receive a passing score.

    Attributes
    ----------
    name:
        Stable task identifier used in benchmark result records.
    expected_ttf_benefit:
        ``True`` because dense legal language requires careful reasoning to
        identify clause-level obligations and termination triggers.
    schema:
        The Pydantic model class that defines the expected output shape.
    complexity:
        Qualitative complexity label consumed by the harness for reporting.
    """

    name: str = "legal_extract"
    expected_ttf_benefit: bool = True
    schema = LegalEntities
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

            ``"text"`` : str
                The contract clause text passed to the model as input.
            ``"parties"`` : list[str]
                Ground-truth list of contracting party names.
            ``"effective_date"`` : str
                Ground-truth effective date string.
            ``"obligations"`` : list[str]
                Ground-truth list of obligation strings.
            ``"termination_conditions"`` : list[str]
                Ground-truth list of termination condition strings.
        """
        problems = _PROBLEMS[:_QUICK_SLICE] if quick else _PROBLEMS
        return [
            {
                "text": p["text"],
                "parties": p["parties"],
                "effective_date": p["effective_date"],
                "obligations": p["obligations"],
                "termination_conditions": p["termination_conditions"],
            }
            for p in problems
        ]

    def score_response(self, predicted: dict[str, Any], ground_truth: dict[str, Any]) -> float:
        """
        Score a model response based on structural completeness.

        A response is considered correct when both the ``parties`` field and
        the ``obligations`` field are non-empty lists.  This checks that the
        model identified the core structural elements of the contract rather
        than returning an empty or malformed response.

        Parameters
        ----------
        predicted:
            A dict representation of a :class:`LegalEntities` instance
            produced by the model.  Expected keys: ``parties``,
            ``effective_date``, ``obligations``, ``termination_conditions``.
        ground_truth:
            The annotated entity dict for the legal clause (unused in scoring
            beyond logging, as scoring is structural).

        Returns
        -------
        float
            ``1.0`` if ``parties`` is non-empty AND ``obligations`` is
            non-empty, otherwise ``0.0``.
        """
        if not isinstance(predicted, dict):
            logger.debug("score_response: predicted is not a dict, got %r", type(predicted))
            return 0.0

        parties = predicted.get("parties") or []
        obligations = predicted.get("obligations") or []

        if not isinstance(parties, list):
            parties = []
        if not isinstance(obligations, list):
            obligations = []

        parties_ok = len([p for p in parties if str(p).strip()]) > 0
        obligations_ok = len([o for o in obligations if str(o).strip()]) > 0

        if parties_ok and obligations_ok:
            return 1.0

        logger.debug(
            "score_response: missing required fields — parties=%r obligations=%r",
            bool(parties_ok),
            bool(obligations_ok),
        )
        return 0.0

    def build_prompt(self, text: str) -> str:
        """
        Construct the full prompt string sent to the model.

        Parameters
        ----------
        text:
            The contract clause or legal text snippet.

        Returns
        -------
        str
            A formatted prompt instructing the model to extract legal entities
            as structured JSON.
        """
        return (
            "You are a legal document analysis assistant.  Extract the key legal entities "
            "from the following contract clause and return them as structured JSON.\n\n"
            "Extract exactly these four fields:\n"
            "  - parties: list of all contracting party names mentioned\n"
            "  - effective_date: the date on which the contract or clause takes effect (string)\n"
            "  - obligations: list of specific duties or obligations imposed on the parties\n"
            "  - termination_conditions: list of conditions or events that allow termination\n\n"
            "Return only the JSON object.  Use empty lists or an empty string where a field "
            "has no applicable content.  Do not include explanation text outside the JSON.\n\n"
            f"Contract clause:\n{text}"
        )
