
import re
import pandas as pd


def answer(question: str, df: pd.DataFrame):
    q = question.lower()

    # "top N products/regions by revenue/units"
    m = re.search(r"top\s+(\d+)\s+(product|region)s?\s+by\s+(revenue|units)", q)
    if m:
        n, group_col_raw, metric_raw = m.groups()
        group_col = "product" if group_col_raw == "product" else "region"
        metric = "revenue" if metric_raw == "revenue" else "units_sold"
        s = df.groupby(group_col)[metric].sum().sort_values(ascending=False).head(int(n))
        return s, f"Grouped by {group_col}, summed {metric}, took top {n}."

    # "which region/product had highest/lowest total revenue/units in <period>"
    m = re.search(r"(highest|lowest)\s+total\s+(revenue|units)(?:\s+in\s+(q[1-4])\s*(\d{4})?)?", q)
    if m and ("region" in q or "product" in q):
        direction, metric_raw, quarter, year = m.groups()
        metric = "revenue" if metric_raw == "revenue" else "units_sold"
        group_col = "region" if "region" in q else "product"
        sub = df.copy()
        if quarter:
            sub = sub[sub["quarter"].str.lower() == quarter]
        if year:
            sub = sub[sub["year"].astype(str) == year]
        s = sub.groupby(group_col)[metric].sum().sort_values(ascending=(direction == "lowest"))
        top = s.index[0]
        val = s.iloc[0]
        label = f"{quarter.upper() if quarter else ''} {year or ''}".strip()
        return {group_col: top, metric: float(val)}, (
            f"Grouped by {group_col}" + (f" for {label}" if label else "") +
            f", summed {metric}, took {direction}."
        )

    # "which region grew fastest from <period A> to <period B>"
    m = re.search(r"grew\s+fastest.*from\s+(q[1-4])\s*(\d{4})?\s+to\s+(q[1-4])\s*(\d{4})?", q)
    if m:
        q1, y1, q2, y2 = m.groups()
        y1 = y1 or y2
        y2 = y2 or y1
        base = df[(df["quarter"].str.lower() == q1) & (df["year"].astype(str) == y1)]
        end = df[(df["quarter"].str.lower() == q2) & (df["year"].astype(str) == y2)]
        base_rev = base.groupby("region")["revenue"].sum()
        end_rev = end.groupby("region")["revenue"].sum()
        growth = ((end_rev - base_rev) / base_rev * 100).sort_values(ascending=False)
        top_region = growth.index[0]
        return {"region": top_region, "growth_pct": round(float(growth.iloc[0]), 2)}, (
            f"Compared total revenue per region in {q1.upper()} vs {q2.upper()}, "
            f"computed % growth, took the max."
        )

    # "growth rate for <region> between <period A> and <period B>"
    m = re.search(r"growth (?:rate )?for (\w+).*between\s+(q[1-4])\s*(\d{4})?\s+and\s+(q[1-4])\s*(\d{4})?", q)
    if m:
        region, q1, y1, q2, y2 = m.groups()
        y1 = y1 or y2
        y2 = y2 or y1
        region = region.capitalize()
        base = df[(df["region"] == region) & (df["quarter"].str.lower() == q1) & (df["year"].astype(str) == y1)]["revenue"].sum()
        end = df[(df["region"] == region) & (df["quarter"].str.lower() == q2) & (df["year"].astype(str) == y2)]["revenue"].sum()
        pct = (end - base) / base * 100 if base else None
        return {"region": region, "growth_pct": round(pct, 2) if pct is not None else None}, (
            f"Summed {region} revenue in {q1.upper()} and {q2.upper()}, computed % change."
        )

    # "which quarter had the lowest/highest total revenue overall"
    m = re.search(r"(highest|lowest).*quarter.*(revenue|units)", q) or re.search(r"quarter.*(highest|lowest).*(revenue|units)", q)
    if m and "quarter" in q:
        groups = re.findall(r"highest|lowest", q)
        direction = groups[0] if groups else "lowest"
        metric = "revenue" if "revenue" in q else "units_sold"
        s = df.groupby(["year", "quarter"])[metric].sum().sort_values(ascending=(direction == "lowest"))
        top = s.index[0]
        return {"year": top[0], "quarter": top[1], metric: float(s.iloc[0])}, (
            f"Grouped by year+quarter, summed {metric}, took {direction}."
        )

    # "total revenue in <year>"
    m = re.search(r"total revenue in\s+(\d{4})", q)
    if m:
        year = m.group(1)
        total = df[df["year"].astype(str) == year]["revenue"].sum()
        return {"year": year, "total_revenue": float(total)}, f"Filtered to year {year}, summed revenue."

    # "most/least consistent (variable) revenue across quarters"
    if "consistent" in q or "variable" in q or "variability" in q:
        std_by_region = df.groupby(["region", "quarter"])["revenue"].sum().groupby("region").std().sort_values()
        if "least" in q or "most consistent" in q:
            winner = std_by_region.index[0]
        else:
            winner = std_by_region.index[-1]
        return {"region": winner, "revenue_std_across_quarters": round(float(std_by_region[winner]), 2)}, (
            "Computed standard deviation of quarterly revenue per region; "
            "lower std = more consistent."
        )

    # "average revenue per region per quarter"
    if "average revenue" in q and "region" in q:
        s = df.groupby(["region", "quarter"])["revenue"].mean()
        return s, "Grouped by region and quarter, took mean revenue."

    return None, "No matching pattern in fallback engine for this question."
