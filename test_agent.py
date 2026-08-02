"""
test_agent.py
-------------
Basic sanity tests for the CSV Q&A agent. These run against the
deterministic fallback engine (no API key needed), so they work the
same way for any reviewer with just `pip install pandas`.

Run with:
    python test_agent.py
or, if pytest is installed:
    pytest test_agent.py
"""

import pandas as pd

import fallback_engine


def load_test_df():
    return pd.read_csv("data/sales_data.csv")


def test_top_products_by_revenue():
    df = load_test_df()
    result, _ = fallback_engine.answer("Top 3 products by revenue", df)
    assert result is not None
    top_product = result.index[0]
    assert top_product == "Widget A", f"expected Widget A on top, got {top_product}"
    print("PASS: test_top_products_by_revenue")


def test_highest_revenue_region_q4():
    df = load_test_df()
    result, _ = fallback_engine.answer(
        "Which region had the highest total revenue in Q4 2025?", df
    )
    assert result is not None
    assert result["region"] == "West", f"expected West, got {result['region']}"
    print("PASS: test_highest_revenue_region_q4")


def test_total_revenue_2025():
    df = load_test_df()
    result, _ = fallback_engine.answer("What is the total revenue in 2025?", df)
    assert result is not None
    assert result["total_revenue"] == df["revenue"].sum(), "total revenue mismatch"
    print("PASS: test_total_revenue_2025")


def test_unmatched_question_returns_none():
    df = load_test_df()
    result, explanation = fallback_engine.answer("asdkjaskjd nonsense question", df)
    assert result is None
    assert "No matching pattern" in explanation
    print("PASS: test_unmatched_question_returns_none")


def test_csv_loads_with_expected_columns():
    df = load_test_df()
    expected_cols = {"region", "quarter", "year", "product", "units_sold", "revenue"}
    assert expected_cols.issubset(set(df.columns)), "sales_data.csv is missing expected columns"
    assert len(df) > 0, "sales_data.csv loaded empty"
    print("PASS: test_csv_loads_with_expected_columns")


if __name__ == "__main__":
    test_top_products_by_revenue()
    test_highest_revenue_region_q4()
    test_total_revenue_2025()
    test_unmatched_question_returns_none()
    test_csv_loads_with_expected_columns()
    print("\nAll tests passed.")
