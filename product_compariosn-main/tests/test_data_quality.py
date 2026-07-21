import pandas as pd

from data_quality.validator import DataQualityValidator

SAMPLE_PRODUCTS = pd.DataFrame([
    {
        # From the spec: contradictory CPU + RAM copied from another listing
        "product_id": "P003433",
        "title": "Non-Touch Display Modern 14 1TB SSD Silver - MSI",
        "brand": "MSI",
        "description": (
            "Modern 14 by MSI: Core i7 RTX 3050, Non-Touch Display, 1TB SSD, "
            "8GB RAM. Also available with Ryzen 5 5600H, 32GB RAM."
        ),
    },
    {
        # Clean listing -- should raise no issues
        "product_id": "P001569",
        "title": "Full Frame Sensor Fujifilm X-T5 (with 18-55mm Lens) Black",
        "brand": "Fujifilm",
        "description": "X-T5 by Fujifilm: Full-frame sensor, with 18-55mm Lens, 24.2MP resolution.",
    },
    {
        # Legitimate multi-SKU listing -- should NOT trigger storage_conflict
        # because "available in" signals real variants, not a contradiction.
        "product_id": "P009001",
        "title": "Galaxy Tab S9",
        "brand": "Samsung",
        "description": "Available in 128GB, 256GB, or 512GB storage options.",
    },
    {
        # Brand mismatch
        "product_id": "P009002",
        "title": "Modern 14 Laptop",
        "brand": "MSI",
        "description": "A sleek ultrabook by Acer with all-day battery life.",
    },
])


def test_cpu_family_conflict_detected():
    validator = DataQualityValidator()
    report = validator.validate_dataframe(SAMPLE_PRODUCTS)
    row = report[report["product_id"] == "P003433"].iloc[0]
    assert "cpu_family_conflict" in row["issue_codes"]
    assert "ram_conflict" in row["issue_codes"]
    assert row["max_severity"] == "high"


def test_clean_listing_has_no_issues():
    validator = DataQualityValidator()
    report = validator.validate_dataframe(SAMPLE_PRODUCTS)
    row = report[report["product_id"] == "P001569"].iloc[0]
    assert row["issue_count"] == 0
    assert row["max_severity"] == "none"


def test_variant_language_avoids_false_positive():
    validator = DataQualityValidator()
    report = validator.validate_dataframe(SAMPLE_PRODUCTS)
    row = report[report["product_id"] == "P009001"].iloc[0]
    assert "storage_conflict" not in row["issue_codes"]


def test_brand_mismatch_detected():
    validator = DataQualityValidator()
    report = validator.validate_dataframe(SAMPLE_PRODUCTS)
    row = report[report["product_id"] == "P009002"].iloc[0]
    assert "brand_mismatch" in row["issue_codes"]


def test_filter_clean_drops_high_severity_rows():
    validator = DataQualityValidator()
    report = validator.validate_dataframe(SAMPLE_PRODUCTS)
    clean = validator.filter_clean(SAMPLE_PRODUCTS, report, max_severity="medium")
    assert "P003433" not in clean["product_id"].values
    assert "P001569" in clean["product_id"].values