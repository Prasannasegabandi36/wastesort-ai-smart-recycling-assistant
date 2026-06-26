from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT_DIR / "data" / "waste_rules.csv"

DEFAULT_RULE = {
    "class_name": "unknown",
    "display_name": "Unknown Waste Item",
    "category": "Needs Manual Review",
    "bin_type": "Check Local Waste Rules",
    "action": "The image is unclear. Please retake the photo with one waste item in good lighting.",
    "tip": "Correct sorting depends on local recycling rules. When unsure, keep wet and dry waste separate.",
    "journey": "Retake image;Classify clearly;Check bin guide;Dispose safely",
    "points": 0,
    "warning": "Low confidence result. Please verify before disposal.",
    "color_tag": "Review",
}


def load_rules(path: Path | str = RULES_PATH) -> pd.DataFrame:
    """Load the waste disposal rule table."""
    path = Path(path)

    if not path.exists():
        return pd.DataFrame([DEFAULT_RULE])

    rules = pd.read_csv(path, encoding="utf-8-sig")

    required_columns = {
        "class_name",
        "display_name",
        "category",
        "bin_type",
        "action",
        "tip",
        "journey",
        "points",
        "warning",
        "color_tag",
    }

    missing_columns = required_columns - set(rules.columns)
    if missing_columns:
        raise ValueError(f"waste_rules.csv is missing columns: {sorted(missing_columns)}")

    rules["class_name"] = rules["class_name"].astype(str).str.lower().str.strip()
    rules["points"] = pd.to_numeric(rules["points"], errors="coerce").fillna(0).astype(int)

    return rules


def get_rule(class_name: str, rules: pd.DataFrame | None = None) -> dict:
    """Return recycling rule details for a predicted class."""
    if rules is None:
        rules = load_rules()

    class_name = str(class_name).lower().strip()
    matched = rules[rules["class_name"] == class_name]

    if matched.empty:
        return DEFAULT_RULE.copy()

    row = matched.iloc[0].to_dict()
    row["points"] = int(row.get("points", 0))
    return row


def get_unique_categories(rules: pd.DataFrame | None = None) -> list[str]:
    if rules is None:
        rules = load_rules()

    return sorted(rules["category"].dropna().unique().tolist())
