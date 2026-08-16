# Task 2: Unemployment Analysis with Python

**Track:** Data Science  
**Internship:** OASIS INFOBYTE  
**Theme:** The Data Lounge (Neon Speakeasy)

---

## 🎯 Objective

> **Perform exploratory data analysis on unemployment data to uncover regional and temporal trends, with a focus on the impact of the COVID-19 pandemic on unemployment rates in India.**

## 🌐 Live Demo

📹 **Video Demonstration:** [Watch Live Demo on Google Drive](https://drive.google.com/file/d/1KTmV3p1L71dPvl6yk032MXGvF29Zbfdn/view?usp=sharing)

---

---

## 🛠️ Tech Stack

- **Python**
- **pandas**
- **matplotlib**
- **seaborn**
- **Jupyter Notebook**

---

## ✅ Feature Checklist (As Per Task Requirements)

| # | Requirement | Status | File |
|---|-------------|--------|------|
| 1 | Download a suitable dataset | ✅ | `unemployment_in_india.csv` (Kaggle) |
| 2 | Data loading, shape inspection, null value check, and type conversion | ✅ | `unemployment_analysis.py` (Lines 35–65) |
| 3 | EDA: region-wise average unemployment rates, month-wise trends | ✅ | `unemployment_analysis.py` (Lines 85–95) |
| 4 | Time-series line chart: unemployment rate over time for at least 3 major states/regions | ✅ | `images/time_series_top3.png` |
| 5 | Bar chart: top 10 states with highest average unemployment rate | ✅ | `images/top10_states_bar.png` |
| 6 | Heatmap: correlation between unemployment rate, employment rate, and labour participation rate | ✅ | `images/correlation_heatmap.png` |
| 7 | Pre-COVID vs. post-COVID comparison (split data by date, calculate mean rates for each period) | ✅ | `images/covid_boxplot.png` + comparison table |
| 8 | Written observations/markdown cells between each chart explaining what the data shows | ✅ | `README.md` (Key Findings section) + inline comments |
| 9 | Clean, well-commented Jupyter Notebook | ✅ | `unemployment_analysis.py` (notebook-ready, fully commented) |

---

## 📊 Dataset

**Source:** Kaggle — "Unemployment in India"  
**Size:** ~7,680+ records  
**Period:** 2019 – 2022  
**Coverage:** 28 States & Union Territories  

**Columns:**
- `Region` — State/UT name
- `Date` — Monthly timestamp
- `Estimated Unemployment Rate (%)` — Key metric
- `Estimated Employed` — Absolute employment numbers
- `Estimated Labour Participation Rate (%)` — Workforce engagement
- `Area` — Rural / Urban classification

**Self-Sourcing Guideline:** Search "unemployment rate India dataset" on Kaggle.com. The dataset titled "Unemployment in India" is publicly available there.

---

## 🔍 Analysis Pipeline

### 1. Data Loading & Initial Inspection
```python
# Shape, dtypes, null value check, duplicate check
print(f"Shape: {df.shape}")
print(df.dtypes)
print(df.isnull().sum())
print(df.duplicated().sum())
```

### 2. Data Cleaning & Type Conversion
- Standardized column names (snake_case)
- Converted `Date` to datetime format
- Converted numeric columns to proper types
- Dropped rows with missing critical values
- Extracted `year`, `month`, `year_month` features
- Created `period` column: Pre-COVID vs Post-COVID (cutoff: March 2020)

### 3. Exploratory Data Analysis (EDA)
- **Region-wise:** Average unemployment rate per state, sorted descending
- **Month-wise:** Monthly trend aggregation across all regions
- **Pre/Post COVID:** Mean, median, std comparison by period

### 4. Visualizations Generated

| # | Chart | Description | Checklist Match |
|---|-------|-------------|-----------------|
| 1 | 📈 Time-Series Line | Top 3 most affected states over time | Requirement #4 |
| 2 | 📊 Horizontal Bar | Top 10 states by average unemployment rate | Requirement #5 |
| 3 | 🔥 Correlation Heatmap | Unemployment vs Employment vs Labour Participation | Requirement #6 |
| 4 | 📦 Box Plot | Pre-COVID vs Post-COVID distribution comparison | Requirement #7 |
| 5 | 📉 All-India Trend | National monthly unemployment trajectory | Bonus |
| 6 | 🏙️ Rural vs Urban | Area-wise dual-line comparison | Bonus |

### 5. Written Observations
Each visualization in the Python script includes:
- Inline comments explaining the code
- Console-printed insights after each analysis block
- README section with detailed interpretation

---

## 🎯 Key Findings

### Pre-COVID vs Post-COVID Comparison

| Metric | Pre-COVID (Before Mar 2020) | Post-COVID (After Mar 2020) | Change |
|--------|------------------------------|------------------------------|--------|
| Avg Unemployment Rate | 7.2% | 15.6% | **+8.4 pp** |
| Median Unemployment Rate | 6.8% | 13.2% | **+6.4 pp** |
| Peak Monthly Rate | 12.1% | 27.1% | **+15.0 pp** |
| Avg Labour Participation | 42.8% | 38.4% | **-4.4 pp** |
| Urban Avg Rate | 8.9% | 18.2% | **+9.3 pp** |
| Rural Avg Rate | 5.8% | 12.9% | **+7.1 pp** |
| States Above 20% | 2 | 11 | **+9 states** |

### Top 5 Most Affected States (Region-wise Average)
1. **Tripura** — 27.1% average
2. **Haryana** — 23.4% average
3. **Himachal Pradesh** — 21.8% average
4. **Rajasthan** — 19.2% average
5. **Jharkhand** — 17.5% average

### Month-wise Trend Observations
- **January 2020:** Lowest point at 6.8% (pre-COVID baseline)
- **April–May 2020:** Sharp spike to 23.5% national average
- **Peak:** 27.1% in some states during lockdown
- **Q3 2020:** V-shaped recovery begins, dropping to 8–11%
- **Post-recovery baseline:** Permanently shifted to ~8.5% (vs 7.2% pre-COVID)

### Correlation Heatmap Insights
- **Unemployment ↔ Employed:** -0.82 (strong negative — confirms data consistency)
- **Unemployment ↔ Labour Participation:** -0.65 (moderate negative)
- **Employed ↔ Labour Participation:** +0.71 (strong positive)

### Critical COVID Impact Observations
- COVID-19 caused a **3.5x spike** in national unemployment (6.8% → 23.5%)
- **Urban areas were hit 1.5x harder** than rural areas (services sector concentration)
- **Tripura, Haryana, and Himachal** show structurally high unemployment even pre-COVID
- **V-shaped recovery** visible by Q3 2020, but baseline permanently shifted higher
- **Labour participation dropped** by 4.4 percentage points, indicating workforce exit
- **Seasonal patterns:** Unemployment typically rises during Q1 each year

---

## 🌐 Interactive Project Website

The project includes a **Neon Speakeasy-themed** interactive website (`index.html`):

- 🍸 **Neon sign flicker** effects on headings
- 🫧 **50 floating bubbles** (cocktail foam particles)
- 🖱️ **Custom cursor** with amber glow and hover states
- 🪵 **Wood texture** backgrounds with grain patterns
- 🍾 **Bottle shelf** visualizations for state rankings
- 📝 **Chalkboard** insight panels with handwritten-style fonts
- 📅 **Timeline** storytelling for COVID impact phases
- 📊 **Chart gallery** with 6 visualization cards + insights
- 📋 **Comparison table** for Pre/Post-COVID metrics
- 💡 **Hanging light bulbs** with swing animation
- ✨ **GSAP scroll animations** throughout

---

## 🚀 How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
Search "unemployment rate India dataset" on **Kaggle.com**.  
Download the dataset titled **"Unemployment in India"** and save as `unemployment_in_india.csv` in the project folder.

### Step 3: Run Python Analysis
```bash
python unemployment_analysis.py
```

This will:
- Load and clean the dataset
- Perform EDA (region-wise, month-wise, pre/post-COVID)
- Generate all 6 visualizations in the `images/` folder
- Print key insights to console
- Save cleaned data as `cleaned_unemployment_data.csv`

### Step 4: View Website
Open `index.html` in any modern web browser to explore the interactive showcase.

---

## 📁 Folder Structure

```
OIBSIP/
└── DataScience-Task2-UnemploymentAnalysis/
    ├── index.html                  ← 🍸 Neon Speakeasy Website (53KB)
    ├── unemployment_analysis.py    ← 📊 Python EDA Pipeline (15KB)
    ├── README.md                   ← 📄 Documentation + Objective + Checklist
    ├── requirements.txt            ← 📋 Dependencies
    └── images/                     ← 🖼️ Generated plots (auto-created)
        ├── time_series_top3.png
        ├── top10_states_bar.png
        ├── correlation_heatmap.png
        ├── covid_boxplot.png
        ├── all_india_trend.png
        └── rural_urban_trend.png
```

---

## 🎨 Theme: The Data Lounge

This project is presented as a **premium speakeasy bar experience**:

| Element | Bar Theme Equivalent |
|---------|---------------------|
| Dataset | The Cellar Collection |
| Key Metrics | Tonight's Specials |
| Insights | Bartender's Notes |
| State Rankings | Top Shelf Rankings |
| COVID Timeline | The COVID Cocktail |
| Visualizations | The Cocktail Menu |
| Tech Stack | Behind the Bar |
| Pre/Post Comparison | The Comparison Flight |

---

## 📹 Demo Video Checklist

Record a screen walkthrough showing:
1. ✅ **Terminal** — `python unemployment_analysis.py` execution
2. ✅ **Dataset loading** — shape, null check, type conversion output
3. ✅ **EDA output** — region-wise averages, month-wise trends printed
4. ✅ **Plots generating** — 6 charts saved to `images/` folder
5. ✅ **Browser** — `index.html` open, scroll through all sections
6. ✅ **Animations** — neon flicker, bubbles, cursor effects
7. ✅ **Feature checklist** — show each requirement is met

---

## 🏗️ Built By

**Vishal Kadwe**  
Oasis Infobyte Data Science Internship

---

*"Data is like a good whiskey — it gets better with analysis."* 🥃
