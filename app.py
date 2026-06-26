from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from utils.eco_score import category_icon, confidence_status, disposal_priority, get_badge
from utils.prediction import predict_waste
from utils.recycling_rules import get_rule, load_rules
from utils.storage import clear_history, load_history, save_scan

ROOT_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="WasteSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Green creative UI styling
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(141, 255, 170, 0.25), transparent 35%),
            radial-gradient(circle at top right, rgba(0, 150, 96, 0.16), transparent 30%),
            linear-gradient(135deg, #f7fff7 0%, #eefcf1 38%, #e8f8ec 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #083b2f 0%, #0b5a40 48%, #0d7c55 100%);
        border-right: 1px solid rgba(255,255,255,0.14);
    }

    section[data-testid="stSidebar"] * {
        color: #f4fff7 !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero-card {
        padding: 2.4rem;
        border-radius: 34px;
        background:
            linear-gradient(135deg, rgba(6, 95, 70, 0.96), rgba(22, 163, 74, 0.92)),
            url('');
        box-shadow: 0 26px 70px rgba(7, 89, 48, 0.24);
        color: white;
        position: relative;
        overflow: hidden;
    }

    .hero-card:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        right: -60px;
        top: -70px;
        background: rgba(255,255,255,0.14);
    }

    .hero-title {
        font-size: 3.4rem;
        line-height: 1;
        font-weight: 800;
        margin-bottom: 0.6rem;
        letter-spacing: -0.05em;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: rgba(255,255,255,0.90);
        max-width: 760px;
        line-height: 1.7;
    }

    .mini-pill {
        display: inline-block;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.17);
        border: 1px solid rgba(255,255,255,0.25);
        margin-right: 0.45rem;
        margin-top: 0.75rem;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .glass-card {
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(34, 197, 94, 0.18);
        border-radius: 26px;
        padding: 1.25rem;
        box-shadow: 0 16px 45px rgba(20, 83, 45, 0.10);
        backdrop-filter: blur(10px);
        height: 100%;
    }

    .metric-card {
        border-radius: 24px;
        padding: 1.2rem;
        background: linear-gradient(145deg, #ffffff, #effdf3);
        border: 1px solid rgba(22, 163, 74, 0.18);
        box-shadow: 0 14px 35px rgba(21, 128, 61, 0.10);
    }

    .metric-number {
        font-size: 2.1rem;
        font-weight: 800;
        color: #08603f;
        margin-bottom: 0.25rem;
    }

    .metric-label {
        color: #3f604c;
        font-weight: 700;
        font-size: 0.92rem;
    }

    .result-card {
        padding: 1.6rem;
        border-radius: 30px;
        background: linear-gradient(135deg, #ffffff 0%, #f1fff4 100%);
        border: 1px solid rgba(22, 163, 74, 0.22);
        box-shadow: 0 24px 60px rgba(21, 128, 61, 0.16);
    }

    .warning-card {
        padding: 1rem 1.2rem;
        border-radius: 20px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        font-weight: 700;
    }

    .success-card {
        padding: 1rem 1.2rem;
        border-radius: 20px;
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        color: #166534;
        font-weight: 700;
    }

    .timeline-step {
        display: flex;
        gap: 0.9rem;
        align-items: flex-start;
        margin: 0.7rem 0;
        padding: 0.9rem;
        border-radius: 18px;
        background: rgba(240, 253, 244, 0.9);
        border: 1px solid rgba(34,197,94,0.16);
    }

    .timeline-dot {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #16a34a, #065f46);
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        font-weight: 800;
        flex-shrink: 0;
    }

    .badge {
        display: inline-block;
        background: #dcfce7;
        border: 1px solid #86efac;
        color: #14532d;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 800;
        margin-right: 0.3rem;
        margin-bottom: 0.35rem;
    }

    .big-badge {
        padding: 1.4rem;
        border-radius: 28px;
        color: white;
        background: linear-gradient(135deg, #047857, #22c55e);
        box-shadow: 0 18px 40px rgba(4, 120, 87, 0.25);
        text-align: center;
    }

    .big-badge h2 {
        margin-bottom: 0.3rem;
        color: white;
    }

    .pipeline-box {
        padding: 1rem;
        border-radius: 18px;
        background: #ffffff;
        border: 1px solid rgba(22,163,74,0.16);
        text-align: center;
        box-shadow: 0 8px 24px rgba(21,128,61,0.08);
        font-weight: 800;
        color: #065f46;
    }

    .footer-note {
        color: #52705d;
        font-size: 0.9rem;
        text-align: center;
        padding-top: 2rem;
    }

    div[data-testid="stMetricValue"] {
        color: #087344;
        font-weight: 800;
    }

    .stButton > button {
        border-radius: 999px;
        border: 0;
        padding: 0.65rem 1.15rem;
        background: linear-gradient(135deg, #16a34a, #047857);
        color: white;
        font-weight: 800;
        box-shadow: 0 10px 24px rgba(22, 163, 74, 0.22);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #15803d, #065f46);
        color: white;
        border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helper functions
# -----------------------------
def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">WasteSort AI ♻️</div>
            <div class="hero-subtitle">
                A creative AI-powered smart recycling assistant that scans waste, recommends the correct bin,
                gives safety guidance, and tracks green impact through a live eco dashboard.
            </div>
            <span class="mini-pill">Scan → Sort → Suggest → Save → Sustain</span>
            <span class="mini-pill">Green Points</span>
            <span class="mini-pill">Smart Bin Guide</span>
            <span class="mini-pill">Eco Dashboard</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, icon: str = "🌿") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{icon} {value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline() -> None:
    steps = ["📷 Scan", "🧠 Sort", "🗑️ Suggest", "🌿 Save", "📊 Sustain"]
    cols = st.columns(len(steps))
    for col, step in zip(cols, steps):
        with col:
            st.markdown(f"<div class='pipeline-box'>{step}</div>", unsafe_allow_html=True)


def render_rule_card(rule: dict, confidence: float) -> None:
    status = confidence_status(confidence)
    icon = category_icon(rule["category"])
    priority = disposal_priority(rule["category"])

    st.markdown(
        f"""
        <div class="result-card">
            <span class="badge">{status}</span>
            <span class="badge">{rule['color_tag']}</span>
            <span class="badge">+{rule['points']} Green Points</span>
            <h2>{icon} {rule['display_name']}</h2>
            <h4 style="color:#065f46; margin-top:-0.4rem;">{rule['category']}</h4>
            <p><b>Recommended Bin:</b> {rule['bin_type']}</p>
            <p><b>Action:</b> {rule['action']}</p>
            <p><b>Eco Tip:</b> {rule['tip']}</p>
            <p><b>Disposal Priority:</b> {priority}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if str(rule.get("warning", "")).strip():
        st.markdown(
            f"<div class='warning-card'>⚠️ Safety Note: {rule['warning']}</div>",
            unsafe_allow_html=True,
        )


def render_journey(rule: dict) -> None:
    journey = str(rule.get("journey", "")).split(";")
    st.markdown("### 🌍 Waste Journey Map")
    for i, step in enumerate(journey, start=1):
        st.markdown(
            f"""
            <div class="timeline-step">
                <div class="timeline-dot">{i}</div>
                <div><b>{step.strip()}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_dashboard_summary(history: pd.DataFrame) -> dict:
    if history.empty:
        return {
            "total_scans": 0,
            "total_points": 0,
            "recyclable": 0,
            "hazardous": 0,
            "most_common": "No scans yet",
            "avg_confidence": 0,
        }

    total_scans = len(history)
    total_points = int(history["points"].sum())
    recyclable = int(history[history["category"].str.contains("Recyclable", case=False, na=False)].shape[0])
    hazardous = int(history[history["category"].str.contains("Hazardous", case=False, na=False)].shape[0])
    most_common = history["item"].mode().iloc[0] if not history["item"].mode().empty else "No scans yet"
    avg_confidence = round(float(history["confidence"].mean()) * 100, 1)
    return {
        "total_scans": total_scans,
        "total_points": total_points,
        "recyclable": recyclable,
        "hazardous": hazardous,
        "most_common": most_common,
        "avg_confidence": avg_confidence,
    }


def create_sample_dashboard_message() -> None:
    st.info("Start scanning waste items to activate the live dashboard. Your scan history will appear here automatically.")


# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.markdown("# ♻️ WasteSort AI")
st.sidebar.markdown("Smart recycling assistant")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📷 AI Waste Scanner",
        "📊 Eco Dashboard",
        "🗑️ Smart Bin Guide",
        "🌍 Waste Journey",
        "🏫 Campus Insights",
        "ℹ️ About Project",
    ],
)

history = load_history()
rules = load_rules()
summary = get_dashboard_summary(history)

st.sidebar.markdown("---")
st.sidebar.metric("Total Scans", summary["total_scans"])
st.sidebar.metric("Green Points", summary["total_points"])
st.sidebar.markdown("---")
st.sidebar.caption("Mode: real model if model/waste_model.h5 exists, otherwise demo mode.")


# -----------------------------
# Pages
# -----------------------------
if page == "🏠 Home":
    render_hero()
    st.write("")

    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Total Items Scanned", str(summary["total_scans"]), "📦")
    with cols[1]:
        render_metric_card("Green Points Earned", str(summary["total_points"]), "🌿")
    with cols[2]:
        render_metric_card("Recyclable Items", str(summary["recyclable"]), "♻️")
    with cols[3]:
        render_metric_card("Hazardous Alerts", str(summary["hazardous"]), "⚠️")

    st.write("")
    st.markdown("## Creative Project Pipeline")
    render_pipeline()

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <h3>🧠 AI Sorting</h3>
                <p>Upload or capture a waste image and get an AI-based category prediction with confidence score.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>🗑️ Smart Bin Advice</h3>
                <p>The app recommends the correct bin, disposal action, safety warning, and recycling tip.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="glass-card">
                <h3>📊 Eco Dashboard</h3>
                <p>Every scan updates green points, waste distribution, category trends, and sustainability badges.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("## Why WasteSort AI is different")
    st.markdown(
        """
        - It is not only a garbage classifier; it behaves like a **smart recycling assistant**.
        - It includes **green points**, **badges**, **safety alerts**, **bin guidance**, and **waste journey maps**.
        - It can be extended for **college hostels, campuses, apartments, and smart cities**.
        """
    )

elif page == "📷 AI Waste Scanner":
    st.markdown("# 📷 AI Waste Scanner")
    st.markdown("Upload a waste item image or capture using camera. Keep one item in the frame for better prediction.")

    tab1, tab2 = st.tabs(["Upload Image", "Camera Scan"])
    uploaded_file = None

    with tab1:
        uploaded_file = st.file_uploader("Upload waste image", type=["jpg", "jpeg", "png", "webp"])

    with tab2:
        camera_file = st.camera_input("Capture waste item")
        if camera_file is not None:
            uploaded_file = camera_file

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        col_img, col_result = st.columns([1, 1.25])
        with col_img:
            st.image(image, caption="Selected waste image", use_container_width=True)

        with col_result:
            with st.spinner("AI is analyzing your waste item..."):
                result = predict_waste(image, filename=getattr(uploaded_file, "name", None))
                rule = get_rule(result["class_name"], rules)
                confidence = float(result["confidence"])
                status = confidence_status(confidence)

            if result["mode"] == "demo_mode":
                st.info("Demo mode is active because no trained model file was found in model/waste_model.h5. The app UI and pipeline are fully working; add a trained model for real predictions.")
            else:
                st.success("Trained AI model prediction completed.")

            st.progress(min(max(confidence, 0), 1), text=f"Confidence: {confidence * 100:.1f}%")
            render_rule_card(rule, confidence)

            save_col1, save_col2 = st.columns(2)
            with save_col1:
                if st.button("Save Scan to Dashboard"):
                    save_scan(
                        item=rule["display_name"],
                        category=rule["category"],
                        bin_type=rule["bin_type"],
                        points=int(rule["points"]),
                        confidence=confidence,
                        status=status,
                    )
                    st.success("Scan saved. Open Eco Dashboard to see updated analytics.")
            with save_col2:
                st.download_button(
                    "Download Result Text",
                    data=(
                        f"WasteSort AI Result\n"
                        f"Detected Item: {rule['display_name']}\n"
                        f"Category: {rule['category']}\n"
                        f"Bin: {rule['bin_type']}\n"
                        f"Action: {rule['action']}\n"
                        f"Tip: {rule['tip']}\n"
                        f"Confidence: {confidence * 100:.1f}%\n"
                        f"Green Points: +{rule['points']}\n"
                    ),
                    file_name="wastesort_result.txt",
                )

        st.write("")
        render_journey(rule)

        st.markdown("### Top AI Scores")
        scores_df = pd.DataFrame(
            {"Waste Type": list(result["top_scores"].keys()), "Score": list(result["top_scores"].values())}
        )
        fig_scores = px.bar(scores_df, x="Score", y="Waste Type", orientation="h", text="Score")
        fig_scores.update_layout(height=330, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_scores, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="success-card">
                🌿 Upload a waste image to begin. Example items: plastic bottle, paper, cardboard, battery, food waste, metal can, glass bottle, clothes, shoes.
            </div>
            """,
            unsafe_allow_html=True,
        )

elif page == "📊 Eco Dashboard":
    st.markdown("# 📊 Green Impact Dashboard")
    st.markdown("A live sustainability dashboard built from every scan saved by the user.")

    history = load_history()
    summary = get_dashboard_summary(history)
    badge_title, badge_message = get_badge(summary["total_points"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Scans", summary["total_scans"])
    with c2:
        st.metric("Green Points", summary["total_points"])
    with c3:
        st.metric("Recyclable Items", summary["recyclable"])
    with c4:
        st.metric("Avg Confidence", f"{summary['avg_confidence']}%")

    st.write("")
    st.markdown(
        f"""
        <div class="big-badge">
            <h2>{badge_title}</h2>
            <p>{badge_message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if history.empty:
        create_sample_dashboard_message()
    else:
        history["date"] = pd.to_datetime(history["timestamp"]).dt.date

        col_a, col_b = st.columns(2)
        with col_a:
            category_counts = history["category"].value_counts().reset_index()
            category_counts.columns = ["Category", "Count"]
            fig_pie = px.pie(
                category_counts,
                values="Count",
                names="Category",
                hole=0.55,
                title="Waste Category Distribution",
            )
            fig_pie.update_layout(height=430, margin=dict(l=20, r=20, t=55, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            daily = history.groupby("date", as_index=False).agg(scans=("item", "count"), points=("points", "sum"))
            fig_daily = px.bar(daily, x="date", y="scans", title="Daily Scan Activity", text="scans")
            fig_daily.update_layout(height=430, margin=dict(l=20, r=20, t=55, b=20))
            st.plotly_chart(fig_daily, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            points_daily = history.groupby("date", as_index=False)["points"].sum()
            points_daily["cumulative_points"] = points_daily["points"].cumsum()
            fig_line = px.line(points_daily, x="date", y="cumulative_points", markers=True, title="Green Points Growth")
            fig_line.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
            st.plotly_chart(fig_line, use_container_width=True)

        with col_d:
            item_counts = history["item"].value_counts().reset_index().head(10)
            item_counts.columns = ["Waste Item", "Count"]
            fig_items = px.bar(item_counts, x="Count", y="Waste Item", orientation="h", title="Most Scanned Waste Items", text="Count")
            fig_items.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
            st.plotly_chart(fig_items, use_container_width=True)

        st.markdown("## AI-Generated Campus Suggestions")
        most_category = history["category"].mode().iloc[0] if not history["category"].mode().empty else "Unknown"
        most_item = history["item"].mode().iloc[0] if not history["item"].mode().empty else "Unknown"
        suggestions = []
        if "Recyclable" in most_category:
            suggestions.append("Place more blue/dry waste bins near high-traffic areas.")
            suggestions.append("Run a recycling awareness drive for plastic, paper, glass, and metal waste.")
        if "Organic" in most_category or "Wet" in most_category:
            suggestions.append("Add compost bins near canteens and food zones.")
        if "Hazardous" in most_category:
            suggestions.append("Create a separate e-waste and battery collection point with warning labels.")
        if not suggestions:
            suggestions.append("Encourage users to scan more waste items for stronger insights.")

        st.markdown(
            f"""
            <div class="glass-card">
                <h3>🌱 Insight Summary</h3>
                <p><b>Most common item:</b> {most_item}</p>
                <p><b>Most common category:</b> {most_category}</p>
                <p><b>Recommended action:</b> {suggestions[0]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View Scan History Table"):
            st.dataframe(history.sort_values("timestamp", ascending=False), use_container_width=True)

        col_clear, col_download = st.columns(2)
        with col_download:
            st.download_button(
                "Download Scan History CSV",
                data=history.to_csv(index=False),
                file_name="wastesort_scan_history.csv",
                mime="text/csv",
            )
        with col_clear:
            if st.button("Clear Scan History"):
                clear_history()
                st.success("Scan history cleared. Refresh or revisit the dashboard.")

elif page == "🗑️ Smart Bin Guide":
    st.markdown("# 🗑️ Smart Bin Guide")
    st.markdown("Learn where each waste item should go and what action to take before disposal.")

    display_rules = rules[["display_name", "category", "bin_type", "action", "tip", "points"]].copy()
    display_rules.columns = ["Waste Item", "Category", "Recommended Bin", "Action", "Eco Tip", "Points"]
    st.dataframe(display_rules, use_container_width=True, hide_index=True)

    st.markdown("## Bin Color Logic")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='glass-card'><h3>🟦 Blue Bin</h3><p>Dry recyclable waste like paper, plastic, glass, metal, and cardboard.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='glass-card'><h3>🟩 Green Bin</h3><p>Wet or organic waste like food scraps, leaves, and compostable waste.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='glass-card'><h3>🟥 Red Bin</h3><p>Hazardous or e-waste like batteries, electronics, and chemical waste.</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='glass-card'><h3>⬛ Black Bin</h3><p>Residual waste that cannot be safely recycled or composted.</p></div>", unsafe_allow_html=True)

elif page == "🌍 Waste Journey":
    st.markdown("# 🌍 Waste Journey Map")
    st.markdown("This feature shows what happens after the user correctly disposes of an item.")

    selected = st.selectbox("Choose a waste item", rules["display_name"].tolist())
    selected_rule = rules[rules["display_name"] == selected].iloc[0].to_dict()

    render_rule_card(selected_rule, confidence=0.95)
    render_journey(selected_rule)

elif page == "🏫 Campus Insights":
    st.markdown("# 🏫 Campus Waste Intelligence")
    st.markdown("A future-ready view for colleges, hostels, apartments, and smart campuses.")

    zone = st.selectbox("Select campus zone", ["Hostel Area", "Canteen", "Library", "Classroom Block", "Sports Ground", "Main Gate"])
    st.markdown(
        f"""
        <div class="glass-card">
            <h3>📍 Zone Selected: {zone}</h3>
            <p>This module can be connected with QR codes near dustbins. Users scan the QR, upload waste image, and the dashboard tracks zone-level waste patterns.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = load_history()
    if history.empty:
        st.info("No real scan data yet. Once users save scans, this page can show zone-level insights.")
    else:
        category_counts = history["category"].value_counts().reset_index()
        category_counts.columns = ["Category", "Count"]
        st.plotly_chart(px.bar(category_counts, x="Category", y="Count", title=f"Waste Pattern for {zone}"), use_container_width=True)

        st.markdown("## Suggested Campus Actions")
        st.markdown(
            """
            - Put smart QR labels on dustbins so students can scan before throwing waste.
            - Add separate e-waste collection boxes for batteries and electronics.
            - Use the dashboard weekly to identify where extra bins are needed.
            - Run hostel-wise green points competition to encourage participation.
            """
        )

    st.markdown("## Future Upgrade Pipeline")
    st.code(
        """
QR Code on Bin
   ↓
Student scans waste item
   ↓
WasteSort AI predicts category
   ↓
Zone-level scan is saved
   ↓
Admin dashboard tracks waste patterns
   ↓
Campus team improves bin placement and recycling awareness
        """.strip()
    )

elif page == "ℹ️ About Project":
    st.markdown("# ℹ️ About WasteSort AI")
    st.markdown(
        """
        **WasteSort AI** is an AI-powered smart waste sorting and recycling guidance system.
        It is designed as a portfolio-ready computer vision project with a professional product feel.

        ### Current Features
        - Waste image upload and camera scan
        - AI prediction pipeline
        - Smart bin recommendation
        - Recycling and safety tips
        - Eco points and badge system
        - Green impact dashboard
        - Waste journey map
        - Campus insights concept

        ### Important Note
        The app runs in **demo mode** until you train and place your model at:

        `model/waste_model.h5`

        After adding the trained model and `class_names.json`, the same app automatically switches to real AI model mode.
        """
    )

    st.markdown("## Complete Technical Pipeline")
    st.code(
        """
Dataset Collection
   ↓
Image Preprocessing
   ↓
Transfer Learning Model Training
   ↓
Model Evaluation
   ↓
Save waste_model.h5 and class_names.json
   ↓
Streamlit App Prediction
   ↓
Recycling Rule Engine
   ↓
Eco Score Calculator
   ↓
CSV Scan History Storage
   ↓
Interactive Green Dashboard
        """.strip()
    )

st.markdown("<div class='footer-note'>Built with ♻️ for smarter recycling, cleaner campuses, and greener cities.</div>", unsafe_allow_html=True)
