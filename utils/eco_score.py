from __future__ import annotations


def get_badge(points: int) -> tuple[str, str]:
    """Return badge title and message based on total green points."""
    points = int(points or 0)
    if points >= 500:
        return "🌍 Planet Protector", "Excellent impact. You are leading sustainable waste habits."
    if points >= 300:
        return "🏆 WasteSort Champion", "Strong progress. Your sorting habits are becoming powerful."
    if points >= 150:
        return "🌿 Green Warrior", "Great work. You are building consistent eco-friendly habits."
    if points >= 50:
        return "♻️ Eco Learner", "Nice start. Keep scanning and sorting correctly."
    return "🌱 Recycling Beginner", "Start scanning waste items to grow your green score."


def confidence_status(confidence: float) -> str:
    """Convert confidence into a friendly label."""
    confidence = float(confidence or 0)
    if confidence >= 0.85:
        return "High Confidence"
    if confidence >= 0.65:
        return "Medium Confidence"
    return "Needs Review"


def category_icon(category: str) -> str:
    category = str(category).lower()
    if "hazard" in category:
        return "⚠️"
    if "organic" in category or "wet" in category:
        return "🍃"
    if "recycl" in category:
        return "♻️"
    if "reuse" in category or "textile" in category:
        return "👕"
    if "trash" in category or "residual" in category:
        return "🗑️"
    return "🔎"


def disposal_priority(category: str) -> str:
    category = str(category).lower()
    if "hazard" in category:
        return "Critical: separate immediately"
    if "organic" in category or "wet" in category:
        return "Compost-friendly"
    if "recycl" in category:
        return "High recycling value"
    if "reuse" in category:
        return "Reuse or donate first"
    return "Reduce and verify"
