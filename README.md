# ♻️ WasteSort AI – Smart Waste Sorting & Recycling Assistant

WasteSort AI is a creative AI-powered waste sorting web app that helps users scan waste items, identify the waste category, recommend the correct bin, give recycling/safety guidance, and track green impact through an interactive dashboard.

This project is designed for a strong portfolio because it is not just a normal image classifier. It works like a real smart recycling assistant for colleges, hostels, apartments, and smart city use cases.

---

## 🌍 Tagline

**Scan. Sort. Recycle. Save the Planet.**

---

## ✨ Key Features

- AI waste image scanner
- Upload image or camera capture
- Waste category prediction
- Confidence score
- Smart bin recommendation
- Recycling and safety tips
- Green points system
- Eco badge system
- Live scan history
- Interactive green impact dashboard
- Waste journey map
- Smart bin guide
- Campus waste intelligence concept
- Works in demo mode before adding trained model

---

## 🧠 Project Pipeline

```text
Waste Image / Camera Scan
        ↓
Image Preprocessing
        ↓
AI Waste Classification Model
        ↓
Prediction + Confidence Score
        ↓
Smart Recycling Rule Engine
        ↓
Correct Bin Recommendation
        ↓
Safety Warning + Recycling Tip
        ↓
Eco Points Calculation
        ↓
Scan History Storage
        ↓
Interactive Green Dashboard
        ↓
Campus-Level Waste Insights
```

Creative product pipeline:

```text
Scan → Sort → Suggest → Save → Sustain
```

---

## 🖼️ Website Pages

### 1. Home
- Creative green landing page
- Project pipeline
- Sustainability cards
- Live metrics

### 2. AI Waste Scanner
- Upload waste image
- Capture using camera
- AI prediction result
- Confidence score
- Bin recommendation
- Recycling action
- Safety warning
- Waste journey map

### 3. Eco Dashboard
- Total scans
- Total green points
- Recyclable count
- Hazardous alerts
- Waste category pie chart
- Daily scan activity
- Green points growth
- Most scanned waste items
- AI-generated campus suggestions

### 4. Smart Bin Guide
- Waste item table
- Correct bin type
- Disposal action
- Eco tips

### 5. Waste Journey
- Shows what happens after correct disposal
- Example: plastic bottle → dry bin → recycling center → flakes → new product

### 6. Campus Insights
- Future-ready concept for colleges and hostels
- QR code bin idea
- Zone-level waste analytics

---

## 📁 Folder Structure

```text
wastesort-ai-smart-recycling-assistant/
│
├── app.py
├── train_model.py
├── requirements.txt
├── requirements-training.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── waste_rules.csv
│   └── scan_history.csv
│
├── model/
│   └── .gitkeep
│
├── sample_images/
│   └── README.md
│
└── utils/
    ├── prediction.py
    ├── recycling_rules.py
    ├── eco_score.py
    └── storage.py
```

---

## 🚀 How to Run Locally

### Step 1: Create virtual environment

```bash
python -m venv venv
```

### Step 2: Activate virtual environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### Step 3: Install requirements

```bash
pip install -r requirements.txt
```

### Step 4: Run Streamlit app

```bash
streamlit run app.py
```

---

## 🧪 Demo Mode vs Real Model Mode

The app has two modes.

### Demo Mode

If `model/waste_model.h5` is not present, the website still runs using demo prediction logic. This is useful for showing UI, dashboard, and project pipeline before training.

### Real AI Model Mode

After training, place these files inside the `model/` folder:

```text
model/waste_model.h5
model/class_names.json
```

Then the app automatically switches to trained model mode.

---

## 🏋️ How to Train the Model

First download a garbage classification image dataset and arrange it like this:

```text
dataset/
├── battery/
├── biological/
├── brown-glass/
├── cardboard/
├── clothes/
├── green-glass/
├── metal/
├── paper/
├── plastic/
├── shoes/
├── trash/
└── white-glass/
```

Install training requirements:

```bash
pip install -r requirements-training.txt
```

Train using MobileNetV2:

```bash
python train_model.py --dataset_dir path/to/dataset --epochs 12 --model_name mobilenet
```

Train using EfficientNetB0:

```bash
python train_model.py --dataset_dir path/to/dataset --epochs 12 --model_name efficientnet
```

The script saves:

```text
model/waste_model.h5
model/class_names.json
```

---

## 🌿 Green Points Logic

| Waste Type | Points |
|---|---:|
| Battery / E-waste | 15 |
| Metal | 12 |
| Clothes / Reusable items | 10–12 |
| Plastic | 10 |
| Glass | 10 |
| Paper | 8 |
| Cardboard | 8 |
| Organic Waste | 8 |
| General Trash | 2 |

---

## 🏆 Badge System

| Points | Badge |
|---:|---|
| 0–49 | Recycling Beginner |
| 50–149 | Eco Learner |
| 150–299 | Green Warrior |
| 300–499 | WasteSort Champion |
| 500+ | Planet Protector |

---

## 🏫 Real-World Use Case

WasteSort AI can be used in:

- College campuses
- Hostels
- Apartments
- Schools
- Offices
- Public parks
- Smart city waste awareness campaigns

Example campus idea:

```text
QR Code on Dustbin
        ↓
Student scans waste item
        ↓
WasteSort AI recommends bin
        ↓
Scan data is saved
        ↓
Admin dashboard shows waste pattern
        ↓
Campus team improves bin placement
```

---

## 🔮 Future Enhancements

- Voice assistant in English, Hindi, and Telugu
- YOLO-based multiple waste object detection
- QR-code smart bin mapping
- User login and leaderboard
- Hostel-wise green points competition
- Municipality dashboard
- E-waste collection center locator
- Weekly PDF sustainability report

---

## 🧰 Tech Stack

- Python
- Streamlit
- Pandas
- Plotly
- Pillow
- TensorFlow / Keras for model training
- CSV storage for scan history

---

## 💚 Project Summary

WasteSort AI follows a complete **Scan → Sort → Suggest → Save → Sustain** pipeline. Users scan waste items, AI classifies them, the app recommends correct disposal methods, green points are calculated, and the dashboard visualizes the user's environmental impact.
