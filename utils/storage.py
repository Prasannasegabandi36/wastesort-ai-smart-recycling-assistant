from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT_DIR / "data" / "scan_history.csv"
HISTORY_COLUMNS = ["timestamp", "item", "category", "bin_type", "points", "confidence", "status"]


def ensure_history_file(path: Path | str = HISTORY_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(path, index=False)
    return path


def load_history(path: Path | str = HISTORY_PATH) -> pd.DataFrame:
    path = ensure_history_file(path)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=HISTORY_COLUMNS)

    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[HISTORY_COLUMNS]

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0).astype(int)
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    return df


def save_scan(item: str, category: str, bin_type: str, points: int, confidence: float, status: str, path: Path | str = HISTORY_PATH) -> None:
    path = ensure_history_file(path)
    df = load_history(path)
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "item": item,
        "category": category,
        "bin_type": bin_type,
        "points": int(points),
        "confidence": round(float(confidence), 4),
        "status": status,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(path, index=False)


def clear_history(path: Path | str = HISTORY_PATH) -> None:
    path = ensure_history_file(path)
    pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(path, index=False)
