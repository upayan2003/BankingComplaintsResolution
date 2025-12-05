import streamlit as st
import pandas as pd
import plotly.express as px
import time
import random
import os

# Import modules created in previous steps
from utils import load_global_analytics, predict_complaint_class, load_geo_analytics
from llm_agent import generate_ai_response

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ZeroLedger - Banking Complaints Resolution App",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CONFIGURATION CHECK ---
if "gcp" in st.secrets and "GROQ_API_KEY" in st.secrets["gcp"]:
    os.environ["GROQ_API_KEY"] = st.secrets["gcp"]["GROQ_API_KEY"]

elif os.path.exists("GroqAPI_Key.txt"):
    with open("GroqAPI_Key.txt", "r") as f:
        os.environ["GROQ_API_KEY"] = f.read().strip()

# --- CUSTOM CSS FOR UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .big-font {
        font-size: 72px !important;
        font-weight: 700;
        color: #1f77b4;
    }
    .resolution-card {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 20px;
        font-family: sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# --- UI LAYOUT ---

# --- SECTION: HEADER ---
col1, col2 = st.columns([2, 1])
df_global = load_global_analytics()

with col1:
    st.markdown(
    f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: -15px;
    ">
        <img src="logo.png" 
             style="width:70px; height:auto; border-radius:8px;">
        <p class="big-font" style="margin:0;">ZeroLedger</p>
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown("## Intelligent Banking Complaint Resolution System")
    st.markdown("##### Powered by Big Data Analytics & Deep Learning")

with col2:
    # Compute KPIs
    total_complaints = 4582954
    avg_timely_rate = df_global["timely_response_rate"].mean()
    avg_dispute_rate = df_global["dispute_rate"].mean()

    timely_pct = f"{avg_timely_rate * 100:.2f}%"
    dispute_pct = f"{avg_dispute_rate * 100:.2f}%"

    total_timely_resolved = int(total_complaints * avg_timely_rate)

    kpi1, kpi2 = st.columns(2)
    kpi1.metric(
        "Total Complaints",
        f"{total_complaints:,}",
        None
    )

    kpi2.metric(
        "Avg Timely Response Rate",
        timely_pct,
        None
    )

st.divider()

# --- SECTION: GENERAL ANALYTICS ---
st.subheader("📊 Global Complaint Landscape")
st.markdown("##### Real-time analysis of Banking Complaints in the US")
viz_col1, viz_col2 = st.columns([2, 1])

# --------------------------
# LEFT: Bar Chart
# --------------------------
with viz_col1:
    top11 = df_global.sort_values("total_complaints", ascending=False).head(11)

    fig_bar = px.bar(
        top11,
        x='sub-issues',
        y='total_complaints',
        color='total_complaints',
        title="Top Complaint Sub-Issues",
        color_continuous_scale='Blues'
    )
    fig_bar.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Sub-Issue",
        yaxis_title="Total Complaints"
    )
    st.plotly_chart(fig_bar, width='stretch')

# --------------------------
# RIGHT: Donut Chart
# --------------------------
with viz_col2:
    # Based on timely response rate and dispute rate
    timely = df_global['timely_response_rate'].mean()
    not_timely = 1 - timely

    fig_pie = px.pie(
        values=[timely, not_timely],
        names=['Timely Response', 'Not Timely'],
        title="Timely Response Rate (Overall)",
        hole=0.4,
    )
    
    fig_pie.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_pie, width='stretch')

st.divider()

# --- SECTION: GEOGRAPHIC MAP ---
st.subheader("🗺️ Geographic Complaint Intensity")
st.markdown("##### Interactive map showing complaint volume by state. Hover for details.")

# 1. Load Data
df_geo = load_geo_analytics()
print(df_geo.head())

# 2. Create Map
fig_map = px.choropleth(
    df_geo,
    locations='State',
    locationmode="USA-states",
    color='Total Complaints',
    scope="usa",
    hover_name='Hover_State',
    hover_data=['Most Common Issue', 'Company Most Commonly Complained About', 'No of Complaints Closed Timely'],
    color_continuous_scale='Blues',
    labels={'Complaints': 'Total Complaints'}
)

# 3. Layout Adjustments
fig_map.update_layout(
    geo=dict(bgcolor='rgba(0,0,0.5,0.05)'),
    margin={"r":0,"t":0,"l":0,"b":0},  # Tight margins
    height=400
)

st.plotly_chart(fig_map, width='stretch')

st.divider()

# --- SECTION: AI AGENT ---
st.markdown("## 🤖 AI Resolution Assistant")
st.info("Describe your banking issue below. Our Deep Learning model will classify it, and the Agent will generate a solution.")

# Input area
user_complaint = st.text_area("Complaint Details", height=300, placeholder="Example: I noticed a charge of $500 on my credit card that I did not authorize...")

if st.button("Analyze & Resolve", type="primary"):
    if not user_complaint:
        st.warning("Please enter a complaint description.")
    else:
        # 1. Processing UI
        with st.status("Processing Complaint...", expanded=True) as status:
            st.write("🔍 Tokenizing text...")
            time.sleep(0.5)
            st.write("🧠 Running Classification Model...")
            
            # 2. Call Classification Model
            label = predict_complaint_class(user_complaint)

            labels = {
                "LABEL_0": 'Information belongs to someone else',
                "LABEL_1": 'Reporting company used your report improperly',
                "LABEL_2": 'Their investigation did not fix an error on your report',
                "LABEL_3": 'Account information incorrect',
                "LABEL_4": 'Account status incorrect',
                "LABEL_5": "Credit inquiries on your report that you don't recognize",
                "LABEL_6": 'Investigation took more than 30 days',
                "LABEL_7": 'Debt is not yours',
                "LABEL_8": 'Was not notified of investigation status or results',
                "LABEL_9": 'Personal information incorrect',
                "LABEL_10": "Other"
            }

            predicted_category = labels.get(label, "Unknown Category")
            st.write(f"✅ Classified as: **{predicted_category}** ({label})")
            
            # 3. Call GenAI
            st.write("🤖 Retrieving Policy Documents & Generating Solution...")
            
            # Passing both complaint and category to the agent
            ai_response = generate_ai_response(user_complaint, predicted_category)
            time.sleep(0.5)
            
            status.update(label="Analysis Complete", state="complete", expanded=False)

        # 4. Display Results
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.subheader("📂 Classification")
            st.success(f"**Issue Identified**: {predicted_category}")

            # Show the GenAI Solution
            st.markdown(f"""
            <div class="resolution-card">
            <i><b><h4>AI-Generated Resolution:</h4></b></i>
            <br>
            {ai_response}
            </div>
            """, unsafe_allow_html=True)

        # --- SECTION: SPECIALIZED ANALYTICS (Contextual) ---
        with res_col2:
            st.subheader("📈 Category Analytics")
            st.info(f"Historical trends for: **{predicted_category}**")
            
            # Mock trend data for this specific category
            dates = pd.date_range(start='2023-01-01', periods=12, freq='ME')
            trend_data = pd.DataFrame({
                'Date': dates,
                'Complaints': [random.randint(100, 500) for _ in range(12)]
            })
            
            # Specific Line Chart
            fig_trend = px.area(
                trend_data, x='Date', y='Complaints', 
                title=f"Volume Trend (Last 12 Months)",
                line_shape='spline'
            )
            fig_trend.update_traces(line_color='#1f77b4', fillcolor='rgba(31, 119, 180, 0.3)')
            fig_trend.update_layout(height=300)
            st.plotly_chart(fig_trend, width='stretch')
            
            # Specific Metric
            st.metric(
                label=f"Avg. Compensation", 
                value="$342.50", 
                delta="+$12.00 vs Global Avg"

            )





