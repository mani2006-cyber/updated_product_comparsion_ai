"""
data_quality/validator.py
==========================
Runs the rules in contradiction_rules.py over a dataframe of products and
produces a per-row quality report. Use this BEFORE any row is used to build
training pairs (exact-match or relationship-dataset generation) -- see
Phase 0 in the migration plan.

Usage:
    from data_quality.validator import DataQualityValidator

    validator = DataQualityValidator()
    report_df = validator.validate_dataframe(products_df)
    clean_df = validator.filter_clean(products_df, report_df, max_severity="medium")
"""

import json
import os
from typing import List, Optional

import pandas as pd

from data_quality.contradiction_rules import BRAND_RULES, TEXT_ONLY_RULES, Issue

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


class DataQualityValidator:
    def __init__(self, id_col: str = "product_id", title_col: str = "title",
                 description_col: str = "description", brand_col: Optional[str] = "brand"):
        self.id_col = id_col
        self.title_col = title_col
        self.description_col = description_col
        self.brand_col = brand_col

    def _validate_row(self, row: pd.Series) -> List[Issue]:
        title = str(row.get(self.title_col, "") or "")
        description = str(row.get(self.description_col, "") or "")
        combined_text = f"{title} {description}"

        issues: List[Issue] = []
        for rule in TEXT_ONLY_RULES:
            result = rule(combined_text)
            if result is not None:
                issues.append(result)

        if self.brand_col:
            declared_brand = row.get(self.brand_col)
            for rule in BRAND_RULES:
                result = rule(combined_text, declared_brand)
                if result is not None:
                    issues.append(result)

        return issues

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns one report row per input row:
        [id_col, issue_count, max_severity, issue_codes, issues_json]
        """
        records = []
        for _, row in df.iterrows():
            issues = self._validate_row(row)
            max_severity = max((i.severity for i in issues), key=lambda s: SEVERITY_ORDER[s], default="none")
            records.append({
                self.id_col: row.get(self.id_col),
                "issue_count": len(issues),
                "max_severity": max_severity,
                "issue_codes": [i.code for i in issues],
                "issues_json": json.dumps([i.__dict__ for i in issues]),
            })
        return pd.DataFrame.from_records(records)

    def filter_clean(self, df: pd.DataFrame, report_df: pd.DataFrame,
                      max_severity: str = "medium") -> pd.DataFrame:
        """Drops rows whose max_severity exceeds the given threshold.
        max_severity="medium" keeps low/medium/none, drops "high".
        """
        threshold = SEVERITY_ORDER.get(max_severity, 1)
        keep_ids = report_df[
            report_df["max_severity"].map(lambda s: SEVERITY_ORDER.get(s, -1)) <= threshold
        ][self.id_col]
        return df[df[self.id_col].isin(keep_ids)].reset_index(drop=True)

    def save_report(self, report_df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        report_df.to_csv(path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data-quality validation over a products CSV.")
    parser.add_argument("--input", required=True, help="Path to products CSV")
    parser.add_argument("--id-col", default="product_id")
    parser.add_argument("--title-col", default="title")
    parser.add_argument("--description-col", default="description")
    parser.add_argument("--brand-col", default="brand")
    parser.add_argument("--out", default="data_quality/reports/quality_report.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    validator = DataQualityValidator(
        id_col=args.id_col, title_col=args.title_col,
        description_col=args.description_col, brand_col=args.brand_col,
    )
    report = validator.validate_dataframe(df)
    validator.save_report(report, args.out)

    n_flagged = (report["issue_count"] > 0).sum()
    print(f"Validated {len(df)} rows -> {n_flagged} flagged with at least one issue.")
    print(report["max_severity"].value_counts())