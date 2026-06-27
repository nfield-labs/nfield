"""Generator for the realistic 10-K financial extraction schema fixture.

Builds ``financial_10k_realistic.json`` from real US-GAAP XBRL concept names and
SEC-DEI cover-page tags, fanned out across multiple fiscal years to reach ~1000
leaf fields. Run: ``uv run python tests/fixtures/schemas/_generate_financial_10k.py``
"""

from __future__ import annotations

import json
import pathlib

# 1. Real SEC-DEI cover-page concepts (company profile): (type, constraints, desc)
PROFILE: dict[str, tuple[str, dict, str]] = {
    "entity_registrant_name": ("string", {"minLength": 1}, "dei:EntityRegistrantName"),
    "entity_central_index_key": (
        "string",
        {"pattern": r"^\d{10}$"},
        "dei:EntityCentralIndexKey (CIK)",
    ),
    "trading_symbol": ("string", {}, "dei:TradingSymbol"),
    "security_exchange_name": (
        "enum",
        {"enum": ["NYSE", "NASDAQ", "NYSEAMER", "CBOE", "OTC"]},
        "dei:SecurityExchangeName",
    ),
    "entity_tax_identification_number": (
        "string",
        {"pattern": r"^\d{2}-\d{7}$"},
        "IRS Employer ID / EIN",
    ),
    "entity_incorporation_state_country_code": (
        "string",
        {"maxLength": 2},
        "State/country of incorporation",
    ),
    "entity_address_state_or_province": ("string", {}, "HQ state/province"),
    "entity_address_city_or_town": ("string", {}, "HQ city"),
    "entity_address_postal_zip_code": ("string", {}, "HQ postal code"),
    "city_area_code": ("string", {}, "Phone area code"),
    "local_phone_number": ("string", {}, "Phone number"),
    "current_fiscal_year_end_date": (
        "string",
        {"pattern": r"^--\d{2}-\d{2}$"},
        "Fiscal year end --MM-DD",
    ),
    "fiscal_year_end_month": (
        "enum",
        {
            "enum": [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        },
        "Fiscal year end month",
    ),
    "document_fiscal_year_focus": ("integer", {"minimum": 1990}, "dei:DocumentFiscalYearFocus"),
    "document_fiscal_period_focus": (
        "enum",
        {"enum": ["FY", "Q1", "Q2", "Q3"]},
        "dei:DocumentFiscalPeriodFocus",
    ),
    "document_type": (
        "enum",
        {"enum": ["10-K", "10-Q", "20-F", "40-F", "8-K"]},
        "dei:DocumentType",
    ),
    "document_period_end_date": ("string", {"format": "date"}, "dei:DocumentPeriodEndDate"),
    "amendment_flag": ("boolean", {}, "dei:AmendmentFlag"),
    "standard_industrial_classification_code": (
        "integer",
        {"minimum": 100, "maximum": 9999},
        "SIC code",
    ),
    "entity_filer_category": (
        "enum",
        {
            "enum": [
                "Large Accelerated Filer",
                "Accelerated Filer",
                "Non-accelerated Filer",
                "Smaller Reporting Company",
            ]
        },
        "dei:EntityFilerCategory",
    ),
    "entity_emerging_growth_company": ("boolean", {}, "dei:EntityEmergingGrowthCompany"),
    "entity_shell_company": ("boolean", {}, "dei:EntityShellCompany"),
    "entity_common_stock_shares_outstanding": (
        "integer",
        {"minimum": 0},
        "dei:EntityCommonStockSharesOutstanding",
    ),
    "entity_public_float": ("number", {"minimum": 0}, "dei:EntityPublicFloat"),
    "number_of_employees": ("integer", {"minimum": 0}, "Full-time employees (Item 1)"),
    "auditor_firm_name": ("string", {}, "dei:AuditorName"),
    "auditor_firm_id": ("string", {}, "dei:AuditorFirmId (PCAOB ID)"),
    "auditor_location": ("string", {}, "dei:AuditorLocation"),
}

# minzero flag: True -> minimum 0; False -> signed (can be loss); "rate" -> [0,1]
INCOME: dict[str, tuple[str, object, str]] = {
    "revenue_from_contract_with_customer": (
        "number",
        True,
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    ),
    "revenues": ("number", True, "Revenues"),
    "cost_of_revenue": ("number", True, "CostOfRevenue"),
    "cost_of_goods_and_services_sold": ("number", True, "CostOfGoodsAndServicesSold"),
    "gross_profit": ("number", False, "GrossProfit"),
    "research_and_development_expense": ("number", True, "ResearchAndDevelopmentExpense"),
    "selling_general_and_administrative_expense": (
        "number",
        True,
        "SellingGeneralAndAdministrativeExpense",
    ),
    "selling_and_marketing_expense": ("number", True, "SellingAndMarketingExpense"),
    "general_and_administrative_expense": ("number", True, "GeneralAndAdministrativeExpense"),
    "restructuring_charges": ("number", True, "RestructuringCharges"),
    "amortization_of_intangible_assets": ("number", True, "AmortizationOfIntangibleAssets"),
    "operating_expenses": ("number", True, "OperatingExpenses"),
    "costs_and_expenses": ("number", True, "CostsAndExpenses"),
    "operating_income_loss": ("number", False, "OperatingIncomeLoss"),
    "interest_expense": ("number", True, "InterestExpense"),
    "interest_income_expense_net": ("number", False, "InterestIncomeExpenseNet"),
    "investment_income_interest": ("number", True, "InvestmentIncomeInterest"),
    "other_nonoperating_income_expense": ("number", False, "OtherNonoperatingIncomeExpense"),
    "nonoperating_income_expense": ("number", False, "NonoperatingIncomeExpense"),
    "income_before_income_taxes": (
        "number",
        False,
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
    ),
    "income_tax_expense_benefit": ("number", False, "IncomeTaxExpenseBenefit"),
    "effective_income_tax_rate": ("number", "rate", "EffectiveIncomeTaxRateContinuingOperations"),
    "income_loss_from_continuing_operations": (
        "number",
        False,
        "IncomeLossFromContinuingOperations",
    ),
    "income_loss_from_discontinued_operations_net_of_tax": (
        "number",
        False,
        "IncomeLossFromDiscontinuedOperationsNetOfTax",
    ),
    "net_income_loss": ("number", False, "NetIncomeLoss"),
    "net_income_available_to_common_stockholders_basic": (
        "number",
        False,
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ),
    "net_income_attributable_to_noncontrolling_interest": (
        "number",
        False,
        "NetIncomeLossAttributableToNoncontrollingInterest",
    ),
    "profit_loss": ("number", False, "ProfitLoss"),
    "earnings_per_share_basic": ("number", False, "EarningsPerShareBasic"),
    "earnings_per_share_diluted": ("number", False, "EarningsPerShareDiluted"),
    "weighted_average_shares_outstanding_basic": (
        "integer",
        True,
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ),
    "weighted_average_diluted_shares_outstanding": (
        "integer",
        True,
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ),
    "common_stock_dividends_per_share_declared": (
        "number",
        True,
        "CommonStockDividendsPerShareDeclared",
    ),
    "comprehensive_income_net_of_tax": ("number", False, "ComprehensiveIncomeNetOfTax"),
    "other_comprehensive_income_loss_net_of_tax": (
        "number",
        False,
        "OtherComprehensiveIncomeLossNetOfTax",
    ),
    "interest_and_debt_expense": ("number", True, "InterestAndDebtExpense"),
    "impairment_of_long_lived_assets": ("number", True, "ImpairmentOfLongLivedAssetsHeldForUse"),
    "goodwill_impairment_loss": ("number", True, "GoodwillImpairmentLoss"),
    "revenue_remaining_performance_obligation": (
        "number",
        True,
        "RevenueRemainingPerformanceObligation",
    ),
    "depreciation_amortization_accretion_net": (
        "number",
        True,
        "DepreciationAmortizationAndAccretionNet",
    ),
}

# balance sheet: value is minzero flag, or "int" for share counts
BALANCE: dict[str, object] = {
    "cash_and_cash_equivalents_at_carrying_value": True,
    "short_term_investments": True,
    "marketable_securities_current": True,
    "accounts_receivable_net_current": True,
    "allowance_for_doubtful_accounts_current": True,
    "inventory_net": True,
    "prepaid_expense_and_other_assets_current": True,
    "assets_held_for_sale_current": True,
    "assets_current": True,
    "property_plant_and_equipment_net": True,
    "property_plant_and_equipment_gross": True,
    "accumulated_depreciation": True,
    "operating_lease_right_of_use_asset": True,
    "long_term_investments": True,
    "equity_method_investments": True,
    "goodwill": True,
    "intangible_assets_net_excluding_goodwill": True,
    "finite_lived_intangible_assets_net": True,
    "deferred_tax_assets_net_noncurrent": True,
    "other_assets_noncurrent": True,
    "assets_noncurrent": True,
    "assets": True,
    "accounts_payable_current": True,
    "accrued_liabilities_current": True,
    "employee_related_liabilities_current": True,
    "contract_with_customer_liability_current": True,
    "deferred_revenue_current": True,
    "short_term_borrowings": True,
    "long_term_debt_current": True,
    "operating_lease_liability_current": True,
    "income_taxes_payable_current": True,
    "liabilities_current": True,
    "long_term_debt_noncurrent": True,
    "long_term_debt": True,
    "operating_lease_liability_noncurrent": True,
    "deferred_tax_liabilities_noncurrent": True,
    "deferred_revenue_noncurrent": True,
    "other_liabilities_noncurrent": True,
    "liabilities_noncurrent": True,
    "liabilities": True,
    "preferred_stock_value": True,
    "common_stock_value": True,
    "common_stock_shares_authorized": "int",
    "common_stock_shares_issued": "int",
    "common_stock_shares_outstanding": "int",
    "common_stock_par_value_per_share": True,
    "additional_paid_in_capital": True,
    "treasury_stock_value": True,
    "retained_earnings_accumulated_deficit": False,
    "accumulated_other_comprehensive_income_loss": False,
    "stockholders_equity": False,
    "minority_interest": False,
    "total_equity_including_noncontrolling": False,
    "liabilities_and_stockholders_equity": True,
}

CASHFLOW: dict[str, object] = {
    "depreciation_depletion_and_amortization": True,
    "depreciation": True,
    "share_based_compensation": True,
    "deferred_income_tax_expense_benefit": False,
    "provision_for_doubtful_accounts": False,
    "gain_loss_on_sale_of_ppe": False,
    "increase_decrease_in_accounts_receivable": False,
    "increase_decrease_in_inventories": False,
    "increase_decrease_in_accounts_payable": False,
    "increase_decrease_in_accrued_liabilities": False,
    "increase_decrease_in_deferred_revenue": False,
    "increase_decrease_in_other_operating_capital": False,
    "net_cash_provided_by_operating_activities": False,
    "payments_to_acquire_property_plant_and_equipment": True,
    "payments_to_acquire_businesses_net_of_cash": True,
    "payments_to_acquire_investments": True,
    "proceeds_from_sale_maturity_of_investments": True,
    "payments_to_acquire_intangible_assets": True,
    "proceeds_from_sale_of_ppe": True,
    "net_cash_provided_by_investing_activities": False,
    "proceeds_from_issuance_of_long_term_debt": True,
    "repayments_of_long_term_debt": True,
    "proceeds_from_issuance_of_common_stock": True,
    "payments_for_repurchase_of_common_stock": True,
    "payments_of_dividends": True,
    "payments_of_dividends_common_stock": True,
    "proceeds_from_payments_for_other_financing": False,
    "net_cash_provided_by_financing_activities": False,
    "effect_of_exchange_rate_on_cash": False,
    "cash_period_increase_decrease": False,
    "cash_beginning_of_period": True,
    "cash_end_of_period": True,
    "income_taxes_paid_net": True,
    "interest_paid_net": True,
    "capital_expenditures": True,
}

RATIOS: dict[str, object] = {
    "gross_margin": "rate",
    "operating_margin": "rate",
    "net_profit_margin": "rate",
    "ebitda": False,
    "ebitda_margin": "rate",
    "current_ratio": True,
    "quick_ratio": True,
    "debt_to_equity_ratio": False,
    "debt_to_assets_ratio": "rate",
    "interest_coverage_ratio": False,
    "return_on_equity": False,
    "return_on_assets": False,
    "return_on_invested_capital": False,
    "asset_turnover_ratio": True,
    "inventory_turnover_ratio": True,
    "free_cash_flow": False,
}

FISCAL_YEARS = list(range(2019, 2026))  # 7 fiscal years


def amount(minzero: object, desc: str) -> dict:
    node: dict = {"type": "number", "description": desc}
    if minzero is True:
        node["minimum"] = 0
    elif minzero == "rate":
        node["minimum"] = 0
        node["maximum"] = 1
    return node


def build() -> dict:
    props: dict = {}

    profile_props = {
        k: {"type": "string" if t == "enum" else t, "description": d, **c}
        for k, (t, c, d) in PROFILE.items()
    }
    props["company_profile"] = {
        "type": "object",
        "description": "SEC cover-page entity information (DEI)",
        "properties": profile_props,
    }

    for yr in FISCAL_YEARS:
        inc = {
            k: (
                amount(mz, d)
                if t == "number"
                else {"type": t, "description": d, **({"minimum": 0} if mz is True else {})}
            )
            for k, (t, mz, d) in INCOME.items()
        }
        bal = {
            k: (
                {"type": "integer", "minimum": 0, "description": f"us-gaap balance item {k}"}
                if v == "int"
                else amount(v, f"us-gaap balance item {k}")
            )
            for k, v in BALANCE.items()
        }
        cf = {k: amount(v, f"us-gaap cash-flow item {k}") for k, v in CASHFLOW.items()}
        rt = {k: amount(v, f"financial ratio {k}") for k, v in RATIOS.items()}
        props[f"fiscal_year_{yr}"] = {
            "type": "object",
            "description": f"Financial statements for fiscal year {yr}",
            "properties": {
                "income_statement": {"type": "object", "properties": inc},
                "balance_sheet": {"type": "object", "properties": bal},
                "cash_flow": {"type": "object", "properties": cf},
                "ratios": {"type": "object", "properties": rt},
            },
        }

    props["reportable_segments"] = {
        "type": "array",
        "description": "ASC 280 operating segments",
        "items": {
            "type": "object",
            "properties": {
                "segment_name": {"type": "string", "description": "Reportable segment name"},
                "revenue": amount(True, "Segment revenue"),
                "operating_income_loss": amount(False, "Segment operating income/loss"),
                "depreciation_and_amortization": amount(True, "Segment D&A"),
                "total_assets": amount(True, "Segment assets"),
                "capital_expenditure": amount(True, "Segment CapEx"),
            },
        },
    }
    props["geographic_revenue"] = {
        "type": "array",
        "description": "Revenue disaggregated by geography",
        "items": {
            "type": "object",
            "properties": {
                "region_name": {
                    "type": "string",
                    "enum": [
                        "United States",
                        "Americas",
                        "EMEA",
                        "Europe",
                        "Greater China",
                        "Japan",
                        "Asia Pacific",
                        "Rest of World",
                    ],
                    "description": "Geographic region",
                },
                "country_code": {"type": "string", "description": "ISO-3166 alpha-2"},
                "revenue": amount(True, "Revenue for the geography"),
                "long_lived_assets": amount(True, "Long-lived assets by geography"),
            },
        },
    }
    props["product_revenue"] = {
        "type": "array",
        "description": "Revenue by product/service line",
        "items": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Product or service line"},
                "revenue": amount(True, "Product line revenue"),
            },
        },
    }

    props["governance"] = {
        "type": "object",
        "description": "10-K Item 10 governance",
        "properties": {
            "chief_executive_officer_name": {"type": "string", "description": "CEO full name"},
            "chief_financial_officer_name": {"type": "string", "description": "CFO full name"},
            "chairperson_name": {"type": "string", "description": "Board chair name"},
            "board_size": {"type": "integer", "minimum": 1, "description": "Number of directors"},
            "number_of_independent_directors": {
                "type": "integer",
                "minimum": 0,
                "description": "Independent directors",
            },
            "ceo_and_chair_roles_separated": {"type": "boolean", "description": "CEO/Chair split"},
            "audit_committee_size": {
                "type": "integer",
                "minimum": 0,
                "description": "Audit committee members",
            },
            "auditor_opinion": {
                "type": "string",
                "enum": ["unqualified", "qualified", "adverse", "disclaimer"],
                "description": "Auditor opinion",
            },
            "going_concern_doubt_flag": {"type": "boolean", "description": "Going-concern doubt"},
            "icfr_material_weakness_flag": {
                "type": "boolean",
                "description": "ICFR material weakness",
            },
            "dual_class_share_structure": {
                "type": "boolean",
                "description": "Multiple share classes",
            },
            "board_of_directors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Director name"},
                        "role": {
                            "type": "string",
                            "enum": ["chair", "lead_independent", "member", "vice_chair"],
                            "description": "Board role",
                        },
                        "is_independent": {
                            "type": "boolean",
                            "description": "Independence status",
                        },
                        "year_first_elected": {
                            "type": "integer",
                            "minimum": 1950,
                            "description": "Year joined board",
                        },
                    },
                },
            },
            "executive_officers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Officer name"},
                        "title": {"type": "string", "description": "Officer title"},
                        "total_compensation": amount(True, "Total compensation"),
                        "age": {
                            "type": "integer",
                            "minimum": 18,
                            "maximum": 100,
                            "description": "Officer age",
                        },
                    },
                },
            },
        },
    }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Annual Report (Form 10-K) Financial Extraction Schema",
        "description": "Real US-GAAP / SEC-DEI concepts across multiple fiscal years.",
        "type": "object",
        "properties": props,
    }


if __name__ == "__main__":
    schema = build()
    out = pathlib.Path(__file__).parent / "financial_10k_realistic.json"
    out.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    from nfield.schema._flatten import flatten_schema

    fields = flatten_schema(schema)
    print(f"wrote {out.name} ({out.stat().st_size} bytes)")
    print(f"leaf fields = {len(fields)}")
