import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules import eda
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# ---------------- PDF Export Function ----------------
def generate_pdf_report(df, missing, outliers, correlations, health_score, breakdown):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("📊 Automated EDA Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}", styles['Normal']))
    story.append(Paragraph(f"Data Health Score: {health_score}%", styles['Normal']))
    story.append(Paragraph(str(breakdown), styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("❌ Missing Values", styles['Heading2']))
    if missing.empty:
        story.append(Paragraph("No missing values ✅", styles['Normal']))
    else:
        story.append(Paragraph(missing.to_html(), styles['Normal']))

    story.append(Spacer(1, 12))
    story.append(Paragraph("⚠ Outliers", styles['Heading2']))
    story.append(Paragraph(str(outliers), styles['Normal']))

    story.append(Spacer(1, 12))
    story.append(Paragraph("🔥 Top Correlations", styles['Heading2']))
    if not correlations.empty:
        story.append(Paragraph(correlations.to_html(index=False), styles['Normal']))
    else:
        story.append(Paragraph("No correlations available", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="🚀 Smart EDA Dashboard", layout="wide")
st.title("🚀 Smart EDA Dashboard")
st.markdown("An **interactive** and automated EDA tool with data quality scoring 📊")

uploaded = st.file_uploader("📂 Upload CSV/Excel", type=["csv", "xlsx"])

if uploaded:
    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

    st.sidebar.header("📌 Dataset Info")
    st.sidebar.write(f"Rows: {df.shape[0]}")
    st.sidebar.write(f"Columns: {df.shape[1]}")

    tabs = st.tabs(["🔍 Overview", "📈 Numeric", "📊 Categorical", "🔥 Correlations", "⚠ Outliers", "🩺 Health Report"])

    # Overview
    with tabs[0]:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.subheader("Missing Values")
        st.dataframe(eda.missing_summary(df))

    # Numeric
    with tabs[1]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.histogram(df, x=col, nbins=20, title=f"Histogram: {col}"), use_container_width=True)
            with col2:
                st.plotly_chart(px.box(df, y=col, title=f"Boxplot: {col}"), use_container_width=True)

        if len(numeric_cols) >= 2:
            st.subheader("Scatter Plot (Select Variables)")
            x_var = st.selectbox("X-axis", numeric_cols)
            y_var = st.selectbox("Y-axis", numeric_cols, index=1)
            st.plotly_chart(px.scatter(df, x=x_var, y=y_var, title=f"{x_var} vs {y_var}"), use_container_width=True)

    # Categorical
    with tabs[2]:
        cat_cols = df.select_dtypes(exclude=[np.number, "datetime64[ns]"]).columns.tolist()
        for col in cat_cols:
            col_df = df[col].value_counts().reset_index()
            col_df.columns = [col, "count"]
            st.plotly_chart(px.bar(col_df, x=col, y="count", title=f"Bar Chart: {col}"), use_container_width=True)
            st.plotly_chart(px.pie(col_df, names=col, values="count", title=f"Pie Chart: {col}"), use_container_width=True)

    # Correlations
    with tabs[3]:
        if numeric_cols:
            corr = df[numeric_cols].corr()
            st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale="RdBu_r"), use_container_width=True)
            st.subheader("Top Correlations")
            st.dataframe(eda.top_correlations(df, numeric_cols))
        else:
            st.info("No numeric columns found")

    # Outliers
    with tabs[4]:
        st.json(eda.iqr_outlier_summary(df, numeric_cols))

    # Health Report
    with tabs[5]:
        st.subheader("🩺 Data Health Score")
        score, breakdown = eda.data_health_score(df)
        st.metric("Overall Data Quality", f"{score}%")
        st.progress(score / 100)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Data Health"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "green" if score > 70 else "orange" if score > 40 else "red"},
                   'steps': [
                       {'range': [0, 40], 'color': "red"},
                       {'range': [40, 70], 'color': "orange"},
                       {'range': [70, 100], 'color': "lightgreen"}] }
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.write("🔎 Breakdown:", breakdown)

    # PDF Export
    pdf_buffer = generate_pdf_report(df,
                                     eda.missing_summary(df),
                                     eda.iqr_outlier_summary(df, numeric_cols),
                                     eda.top_correlations(df, numeric_cols),
                                     score, breakdown)
    st.download_button("📥 Download EDA Report (PDF)", data=pdf_buffer, file_name="eda_report.pdf", mime="application/pdf")
