import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os
st.set_page_config(page_title="Retail vs Wholesale Strategy Dashboard", layout="wide", page_icon="🛍️")

# =========================================================================
# COLOR PALETTE & GLOBAL STYLE
# =========================================================================
PRIMARY = "#4C3B8C"      # deep purple
ACCENT = "#E8833A"       # warm orange
RETAIL_COLOR = "#4C72B0" # blue
WHOLESALE_COLOR = "#C44E52"  # red
BG_CARD = "#F7F5FB"
TEXT_DARK = "#2B2140"

PLOTLY_TEMPLATE = "simple_white"
COLOR_SEQ = ["#4C3B8C", "#E8833A", "#4C72B0", "#C44E52", "#64B5CD", "#8172B2"]

st.markdown(f"""
<style>
    .main {{ background-color: #FFFFFF; }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

    .hero {{
        background: linear-gradient(135deg, {PRIMARY} 0%, #6B4FA0 60%, {ACCENT} 130%);
        padding: 2.2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        color: white;
    }}
    .hero h1 {{
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        color: white;
    }}
    .hero p {{
        font-size: 1.05rem;
        opacity: 0.92;
        margin: 0;
    }}

    div[data-testid="stMetric"] {{
        background-color: {BG_CARD};
        border: 1px solid #E4DFF2;
        border-left: 5px solid {PRIMARY};
        border-radius: 10px;
        padding: 1rem 1.1rem 0.6rem 1.1rem;
    }}
    div[data-testid="stMetricLabel"] {{
        font-weight: 600;
        color: {TEXT_DARK};
    }}
    div[data-testid="stMetricValue"] {{
        color: {PRIMARY};
        font-weight: 800;
    }}

    .section-title {{
        font-size: 1.35rem;
        font-weight: 700;
        color: {TEXT_DARK};
        border-bottom: 3px solid {ACCENT};
        display: inline-block;
        padding-bottom: 4px;
        margin-top: 1.6rem;
        margin-bottom: 1rem;
    }}

    .insight-box {{
        background-color: {BG_CARD};
        border-radius: 10px;
        border-left: 5px solid {ACCENT};
        padding: 1rem 1.3rem;
        margin: 0.8rem 0 1.4rem 0;
        font-size: 0.97rem;
        line-height: 1.55;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {BG_CARD};
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        color: {TEXT_DARK};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY} !important;
        color: white !important;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
    <h1>From Market Basket Insights to Strategic Positioning</h1>
    <p>Online Retail Dataset — Retail vs Wholesale Segment Strategy Dashboard</p>
</div>
""", unsafe_allow_html=True)

# =========================================================================
# Load all outputs
# =========================================================================
rules_retail = pd.read_csv("association_rules_retail.csv")
rules_wholesale = pd.read_csv("association_rules_wholesale.csv")
freq_retail = pd.read_csv("frequent_itemsets_retail.csv")
freq_wholesale = pd.read_csv("frequent_itemsets_wholesale.csv")

mmm_roi = pd.read_csv("mmm_roi_results.csv")
mmm_comparison = pd.read_csv("mmm_model_comparison.csv")
mmm_dataset = pd.read_csv("mmm_dataset.csv")

perceptual_coords = pd.read_csv("perceptual_map_coords.csv")
perceptual_means = pd.read_csv("perceptual_brand_means.csv")

tab1, tab2, tab3, tab4 = st.tabs([
    "🧺  Market Basket Analysis", "📈  Marketing Mix Modelling",
    "🗺️  Perceptual Mapping", "📋  Executive Summary"
])

# =========================================================================
# TAB 1: Market Basket Analysis
# =========================================================================
with tab1:
    st.markdown('<div class="section-title">Segment Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retail Rules", len(rules_retail))
    c2.metric("Retail Top Lift", round(rules_retail['lift'].max(), 2))
    c3.metric("Wholesale Rules", len(rules_wholesale))
    c4.metric("Wholesale Top Lift", round(rules_wholesale['lift'].max(), 2))

    st.markdown("""
    <div class="insight-box">
    <b>Key insight:</b> Wholesale generates <b>4.3× more association rules</b> than Retail despite
    19× fewer transactions — wholesale baskets are far more repetitive and predictable,
    while Retail baskets are diverse with a few very strong "matching-set" patterns.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Top Association Rules</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(f"**Retail** &nbsp;<span style='color:{RETAIL_COLOR}'>●</span>", unsafe_allow_html=True)
        st.dataframe(rules_retail.head(8)[['antecedent', 'consequent', 'confidence', 'lift']],
                     width='stretch', hide_index=True)
    with rc2:
        st.markdown(f"**Wholesale** &nbsp;<span style='color:{WHOLESALE_COLOR}'>●</span>", unsafe_allow_html=True)
        st.dataframe(rules_wholesale.head(8)[['antecedent', 'consequent', 'confidence', 'lift']],
                     width='stretch', hide_index=True)

    st.markdown('<div class="section-title">Item Frequency</div>', unsafe_allow_html=True)
    freq_col1, freq_col2 = st.columns(2)
    with freq_col1:
        top15_retail = freq_retail[freq_retail['size'] == 1].sort_values('support', ascending=False).head(12)
        fig_r = px.bar(top15_retail, x='support', y='itemset', orientation='h',
                        title="Top Items — Retail", template=PLOTLY_TEMPLATE,
                        color_discrete_sequence=[RETAIL_COLOR])
        fig_r.update_layout(yaxis_title=None, xaxis_title="Support", height=420,
                             title_font_size=15, margin=dict(l=10, r=10, t=40, b=10))
        fig_r.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_r, width='stretch')
    with freq_col2:
        top15_wholesale = freq_wholesale[freq_wholesale['size'] == 1].sort_values('support', ascending=False).head(12)
        fig_w = px.bar(top15_wholesale, x='support', y='itemset', orientation='h',
                        title="Top Items — Wholesale", template=PLOTLY_TEMPLATE,
                        color_discrete_sequence=[WHOLESALE_COLOR])
        fig_w.update_layout(yaxis_title=None, xaxis_title="Support", height=420,
                             title_font_size=15, margin=dict(l=10, r=10, t=40, b=10))
        fig_w.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_w, width='stretch')

    st.markdown('<div class="section-title">Rules Generated by Level</div>', unsafe_allow_html=True)
    level_summary = pd.DataFrame({
        "Segment": ["Retail", "Retail", "Retail", "Wholesale", "Wholesale", "Wholesale"],
        "Level": ["1-item", "2-item", "3-item", "1-item", "2-item", "3-item"],
        "Count": [240, 52, 1, 183, 137, 23],
    })
    fig_lvl = px.bar(level_summary, x="Level", y="Count", color="Segment", barmode="group",
                      template=PLOTLY_TEMPLATE, color_discrete_map={"Retail": RETAIL_COLOR, "Wholesale": WHOLESALE_COLOR})
    fig_lvl.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_lvl, width='stretch')

# =========================================================================
# TAB 2: Marketing Mix Modelling
# =========================================================================
with tab2:
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    best_r2 = mmm_comparison['R2'].max()
    best_rmse = mmm_comparison.loc[mmm_comparison['R2'].idxmax(), 'RMSE']
    best_channel = mmm_roi.loc[mmm_roi['roi'].idxmax(), 'channel']
    worst_channel = mmm_roi.loc[mmm_roi['roi'].idxmin(), 'channel']

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model R²", round(best_r2, 3))
    c2.metric("RMSE", f"{best_rmse:,.0f}")
    c3.metric("Best ROI Channel", best_channel.replace('_spend', '').replace('_', ' ').title())
    c4.metric("Weakest Channel", worst_channel.replace('_spend', '').replace('_', ' ').title())

    st.markdown("""
    <div class="insight-box">
    <b>Key insight:</b> Digital/Social spend delivers the strongest ROI, more than double
    TV/Print's return. Ridge regularisation converged to plain Linear Regression
    (alpha ≈ 0.01) — confirming multicollinearity was not an issue in this model.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">ROI by Channel</div>', unsafe_allow_html=True)
    roi_sorted = mmm_roi.sort_values('roi')
    roi_sorted['channel_clean'] = roi_sorted['channel'].str.replace('_spend', '').str.replace('_', ' ').str.title()
    fig_roi = px.bar(roi_sorted, x='roi', y='channel_clean', orientation='h',
                      color='roi', color_continuous_scale=[WHOLESALE_COLOR, "#F2E9D8", "#2E7D32"],
                      template=PLOTLY_TEMPLATE, text='roi')
    fig_roi.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_roi.update_layout(yaxis_title=None, xaxis_title="ROI (₤ sales per ₤ spent)",
                           height=350, coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_roi, width='stretch')

    m1, m2 = st.columns([1, 1])
    with m1:
        st.markdown("**Model Comparison**")
        st.dataframe(mmm_comparison, width='stretch', hide_index=True)
    with m2:
        st.markdown("**Sales Trend**")
        day_col = mmm_dataset.columns[0]
        mmm_dataset_sorted = mmm_dataset.sort_values(day_col)
        fig_trend = px.area(mmm_dataset_sorted, x=day_col, y='sales_for_model',
                             template=PLOTLY_TEMPLATE, color_discrete_sequence=[PRIMARY])
        fig_trend.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                 yaxis_title="Sales", xaxis_title=None)
        st.plotly_chart(fig_trend, width='stretch')

# =========================================================================
# TAB 3: Perceptual Mapping
# =========================================================================
with tab3:
    st.markdown('<div class="section-title">Brand Positioning Map</div>', unsafe_allow_html=True)

    fig_map = go.Figure()
    for _, row in perceptual_coords.iterrows():
        color = PRIMARY if row['brand'] == 'This Company' else ACCENT
        size = 34 if row['brand'] == 'This Company' else 24
        fig_map.add_trace(go.Scatter(
            x=[row['PC1']], y=[row['PC2']], mode='markers+text',
            text=[row['brand']], textposition='top center',
            marker=dict(size=size, color=color, line=dict(width=2, color='white')),
            name=row['brand'], showlegend=False
        ))
    fig_map.add_hline(y=0, line_dash="dot", line_color="lightgrey")
    fig_map.add_vline(x=0, line_dash="dot", line_color="lightgrey")
    fig_map.update_layout(template=PLOTLY_TEMPLATE, height=480,
                           xaxis_title="PC1 — Design/Premium ↔ Value/Bulk",
                           yaxis_title="PC2 — Catalogue Breadth & Wholesale Capability",
                           margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_map, width='stretch')

    st.markdown("""
    <div class="insight-box">
    <b>Key insight:</b> This Company is closest to Trade Gift Wholesalers but is the only
    brand combining bulk/wholesale capability with genuine design credibility and catalogue
    breadth — a dual-audience positioning no competitor currently occupies.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Brand Attribute Scores</div>', unsafe_allow_html=True)
    attr_cols = [c for c in perceptual_means.columns if c != 'brand']
    fig_radar = go.Figure()
    for _, row in perceptual_means.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[c] for c in attr_cols] + [row[attr_cols[0]]],
            theta=attr_cols + [attr_cols[0]],
            fill='toself', name=row['brand'], opacity=0.55
        ))
    fig_radar.update_layout(template=PLOTLY_TEMPLATE, height=460,
                             polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                             margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_radar, width='stretch')

# =========================================================================
# TAB 4: Executive Summary
# =========================================================================
with tab4:
    st.markdown('<div class="section-title">Key Findings</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f"""
        <div class="insight-box" style="border-left-color:{RETAIL_COLOR}">
        <b>🧺 Market Basket</b><br><br>
        Retail and Wholesale show fundamentally different basket logic — Retail favours
        matching-set decor bundles (lift up to 20.4); Wholesale favours stable, repeatable
        bulk combinations.
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown(f"""
        <div class="insight-box" style="border-left-color:{ACCENT}">
        <b>📈 Marketing Mix</b><br><br>
        Digital/Social delivers the strongest ROI (0.35), more than double TV/Print (0.13).
        Recommend shifting ~₤120K from TV/Print to Digital/Social.
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown(f"""
        <div class="insight-box" style="border-left-color:{PRIMARY}">
        <b>🗺️ Positioning</b><br><br>
        This Company uniquely combines wholesale capability with design credibility and
        catalogue breadth — a white-space dual-audience position.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Strategic Recommendations</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Segment-specific bundling** — decor-completion sets for Retail, stable bulk combinations for Wholesale
    - **Reallocate marketing budget** toward Digital/Social, away from TV/Print
    - **Dual-audience brand positioning** — serve both Retail and Wholesale without compromise
    - **Two-tier pricing** — formalise the informal quantity-based segmentation into explicit retail vs bulk pricing
    """)

st.markdown("---")
st.caption("CIA-4 Final Project — Marketing Analytics | From Market Basket Insights to Strategic Positioning")
