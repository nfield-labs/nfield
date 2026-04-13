"""
SQLExtractionTask — SQL generation from natural language benchmark task for FormatShield.

This task contains 15 hardcoded natural language questions paired with
SQL queries and the relevant tables. Models must extract the SQL query
and the table names involved, given a natural language description.

Because SQL generation requires understanding schema structure and mapping
natural language concepts to precise query syntax, this task is classified
HIGH complexity and is expected to benefit from TTF routing.

Complexity: HIGH
Expected TTF benefit: True
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SQLExtraction(BaseModel):
    """Structured schema for SQL extraction from natural language."""

    query: str
    """The SQL query that answers the natural language question."""

    tables: list[str]
    """The table names referenced in the query."""

    explanation: str
    """Brief explanation of what the query does."""


# ---------------------------------------------------------------------------
# 15 hardcoded NL-to-SQL problems with ground-truth queries and tables
# ---------------------------------------------------------------------------

_PROBLEMS: list[dict[str, Any]] = [
    {
        "question": (
            "Find all employees in the Sales department who earn more than $75,000 per year. "
            "Return their names and salaries, sorted by salary descending. "
            "Tables: employees(id, name, department_id, salary), departments(id, name)"
        ),
        "ground_truth": {
            "query": (
                "SELECT e.name, e.salary FROM employees e "
                "JOIN departments d ON e.department_id = d.id "
                "WHERE d.name = 'Sales' AND e.salary > 75000 "
                "ORDER BY e.salary DESC"
            ),
            "tables": ["employees", "departments"],
        },
    },
    {
        "question": (
            "Count how many orders were placed each month in 2024. "
            "Return month number and order count. "
            "Tables: orders(id, customer_id, order_date, total_amount)"
        ),
        "ground_truth": {
            "query": (
                "SELECT EXTRACT(MONTH FROM order_date) AS month, COUNT(*) AS order_count "
                "FROM orders "
                "WHERE EXTRACT(YEAR FROM order_date) = 2024 "
                "GROUP BY EXTRACT(MONTH FROM order_date) "
                "ORDER BY month"
            ),
            "tables": ["orders"],
        },
    },
    {
        "question": (
            "Find customers who have never placed an order. Return their names and emails. "
            "Tables: customers(id, name, email), orders(id, customer_id, order_date)"
        ),
        "ground_truth": {
            "query": (
                "SELECT c.name, c.email FROM customers c "
                "LEFT JOIN orders o ON c.id = o.customer_id "
                "WHERE o.id IS NULL"
            ),
            "tables": ["customers", "orders"],
        },
    },
    {
        "question": (
            "Get the top 5 products by total revenue. "
            "Tables: order_items(id, order_id, product_id, quantity, unit_price), "
            "products(id, name, category)"
        ),
        "ground_truth": {
            "query": (
                "SELECT p.name, SUM(oi.quantity * oi.unit_price) AS total_revenue "
                "FROM order_items oi "
                "JOIN products p ON oi.product_id = p.id "
                "GROUP BY p.id, p.name "
                "ORDER BY total_revenue DESC "
                "LIMIT 5"
            ),
            "tables": ["order_items", "products"],
        },
    },
    {
        "question": (
            "Find all students who scored above average on the final exam. "
            "Return student name and score. "
            "Tables: students(id, name, class_id), exam_scores(id, student_id, exam_type, score)"
        ),
        "ground_truth": {
            "query": (
                "SELECT s.name, es.score FROM students s "
                "JOIN exam_scores es ON s.id = es.student_id "
                "WHERE es.exam_type = 'final' "
                "AND es.score > (SELECT AVG(score) FROM exam_scores WHERE exam_type = 'final')"
            ),
            "tables": ["students", "exam_scores"],
        },
    },
    {
        "question": (
            "Find the second highest salary across all employees. "
            "Tables: employees(id, name, salary, department_id)"
        ),
        "ground_truth": {
            "query": (
                "SELECT MAX(salary) AS second_highest_salary "
                "FROM employees "
                "WHERE salary < (SELECT MAX(salary) FROM employees)"
            ),
            "tables": ["employees"],
        },
    },
    {
        "question": (
            "List all products that have never been ordered. "
            "Tables: products(id, name, price), order_items(id, order_id, product_id, quantity)"
        ),
        "ground_truth": {
            "query": (
                "SELECT p.name, p.price FROM products p "
                "LEFT JOIN order_items oi ON p.id = oi.product_id "
                "WHERE oi.id IS NULL"
            ),
            "tables": ["products", "order_items"],
        },
    },
    {
        "question": (
            "Calculate the average order value per customer, only for customers with more than "
            "3 orders. Return customer name and average order value. "
            "Tables: customers(id, name), orders(id, customer_id, total_amount)"
        ),
        "ground_truth": {
            "query": (
                "SELECT c.name, AVG(o.total_amount) AS avg_order_value "
                "FROM customers c "
                "JOIN orders o ON c.id = o.customer_id "
                "GROUP BY c.id, c.name "
                "HAVING COUNT(o.id) > 3"
            ),
            "tables": ["customers", "orders"],
        },
    },
    {
        "question": (
            "Find all managers who manage more than 5 employees. "
            "Tables: employees(id, name, manager_id, department_id)"
        ),
        "ground_truth": {
            "query": (
                "SELECT m.name, COUNT(e.id) AS employee_count "
                "FROM employees m "
                "JOIN employees e ON e.manager_id = m.id "
                "GROUP BY m.id, m.name "
                "HAVING COUNT(e.id) > 5"
            ),
            "tables": ["employees"],
        },
    },
    {
        "question": (
            "Get the running total of sales by date for the last 30 days. "
            "Tables: sales(id, sale_date, amount)"
        ),
        "ground_truth": {
            "query": (
                "SELECT sale_date, amount, "
                "SUM(amount) OVER (ORDER BY sale_date) AS running_total "
                "FROM sales "
                "WHERE sale_date >= CURRENT_DATE - INTERVAL '30 days' "
                "ORDER BY sale_date"
            ),
            "tables": ["sales"],
        },
    },
    {
        "question": (
            "Find all duplicate email addresses in the users table. "
            "Tables: users(id, name, email, created_at)"
        ),
        "ground_truth": {
            "query": (
                "SELECT email, COUNT(*) AS occurrences "
                "FROM users "
                "GROUP BY email "
                "HAVING COUNT(*) > 1"
            ),
            "tables": ["users"],
        },
    },
    {
        "question": (
            "Find the department with the highest average salary. "
            "Tables: employees(id, name, salary, department_id), departments(id, name)"
        ),
        "ground_truth": {
            "query": (
                "SELECT d.name, AVG(e.salary) AS avg_salary "
                "FROM departments d "
                "JOIN employees e ON d.id = e.department_id "
                "GROUP BY d.id, d.name "
                "ORDER BY avg_salary DESC "
                "LIMIT 1"
            ),
            "tables": ["employees", "departments"],
        },
    },
    {
        "question": (
            "Retrieve the 3 most recent login timestamps for each user. "
            "Tables: user_sessions(id, user_id, login_at, logout_at)"
        ),
        "ground_truth": {
            "query": (
                "SELECT user_id, login_at "
                "FROM ("
                "  SELECT user_id, login_at, "
                "  ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_at DESC) AS rn "
                "  FROM user_sessions"
                ") ranked "
                "WHERE rn <= 3"
            ),
            "tables": ["user_sessions"],
        },
    },
    {
        "question": (
            "Find all pairs of products that were ordered together in the same order. "
            "Tables: order_items(id, order_id, product_id), products(id, name)"
        ),
        "ground_truth": {
            "query": (
                "SELECT DISTINCT p1.name AS product_1, p2.name AS product_2 "
                "FROM order_items oi1 "
                "JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id "  # noqa: E501
                "JOIN products p1 ON oi1.product_id = p1.id "
                "JOIN products p2 ON oi2.product_id = p2.id"
            ),
            "tables": ["order_items", "products"],
        },
    },
    {
        "question": (
            "Calculate the month-over-month percentage change in revenue. "
            "Tables: sales(id, sale_date, amount)"
        ),
        "ground_truth": {
            "query": (
                "WITH monthly AS ("
                "  SELECT DATE_TRUNC('month', sale_date) AS month, SUM(amount) AS revenue "
                "  FROM sales GROUP BY DATE_TRUNC('month', sale_date)"
                ") "
                "SELECT month, revenue, "
                "ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month)) / "
                "NULLIF(LAG(revenue) OVER (ORDER BY month), 0), 2) AS pct_change "
                "FROM monthly ORDER BY month"
            ),
            "tables": ["sales"],
        },
    },
]


def _score_sql_extraction(predicted: str, ground_truth: dict[str, Any]) -> float:
    """Score SQL extraction response against ground truth.

    Uses a lenient scoring approach: full credit if both the table names
    are correct AND the SQL contains the key clauses (table names as tokens).
    Partial credit for table names only.

    Returns:
        Float in [0.0, 1.0].
    """
    try:
        parsed = json.loads(predicted)
    except (json.JSONDecodeError, ValueError):
        return 0.0

    if not isinstance(parsed, dict):
        return 0.0

    gt_tables = {t.lower() for t in ground_truth.get("tables", [])}
    pred_tables = {t.lower() for t in parsed.get("tables", [])}

    if not gt_tables:
        return 0.0

    table_score = len(gt_tables & pred_tables) / len(gt_tables)

    pred_query = parsed.get("query", "").lower()
    query_mentions_tables = all(t in pred_query for t in gt_tables)

    if table_score == 1.0 and query_mentions_tables:
        return 1.0
    return table_score * 0.5


class SQLExtractionTask:
    """SQL extraction from natural language benchmark task.

    Measures FormatShield's routing accuracy on NL-to-SQL generation,
    a HIGH complexity task expected to benefit from TTF routing.
    """

    name = "sql_extraction"
    complexity = "HIGH"
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
            schema = SQLExtraction.model_json_schema()
            problems.append(
                {
                    "prompt": (
                        f"Extract the SQL query and referenced table names from this question:\n\n"
                        f"{p['question']}"
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
            ground_truth: Ground truth dict with 'query' and 'tables' keys.

        Returns:
            Float in [0.0, 1.0] where 1.0 = perfect match.
        """
        return _score_sql_extraction(predicted, ground_truth)
