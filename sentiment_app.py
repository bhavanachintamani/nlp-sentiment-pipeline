import streamlit as st
from transformers import pipeline as hf_pipeline
import plotly.graph_objects as go
import pandas as pd
import re

st.set_page_config(
    page_title="NLP Sentiment Analyser",
    page_icon="💬",
    layout="wide"
)

st.markdown("""
<style>
.main-title{font-size:2rem;font-weight:700;color:#185FA5;}
.subtitle{font-size:1rem;color:#5F5E5A;margin-bottom:2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">💬 NLP Sentiment Analyser</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyse product and movie review sentiment using DistilBERT transformer | Built for Canadian retail and media sectors</p>',
            unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )

with st.spinner("Loading DistilBERT model..."):
    nlp = load_model()

tab1, tab2, tab3 = st.tabs(["Single Review", "Batch Analysis", "About"])

with tab1:
    st.subheader("Analyse a Single Review")
    review = st.text_area("Paste your review here:", height=150,
        placeholder="Type or paste a product or movie review here...")

    if st.button("Analyse Sentiment", type="primary"):
        if review.strip():
            result    = nlp(review[:512])[0]
            label     = result["label"]
            score     = result["score"]
            sentiment = "POSITIVE" if label == "POSITIVE" else "NEGATIVE"
            emoji     = "😊" if sentiment == "POSITIVE" else "😞"
            color     = "#27500A" if sentiment == "POSITIVE" else "#A32D2D"
            bg        = "#EAF3DE" if sentiment == "POSITIVE" else "#FCEBEB"

            st.markdown(f"""
            <div style="background:{bg};border-radius:10px;padding:16px;text-align:center;margin-bottom:16px">
                <h2 style="color:{color};margin:0">{emoji} {sentiment}</h2>
                <p style="color:{color};margin:4px 0 0 0">Confidence: {score*100:.1f}%</p>
            </div>""", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment", f"{emoji} {sentiment}")
            col2.metric("Confidence", f"{score*100:.1f}%")
            col3.metric("Word Count", f"{len(review.split())} words")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score*100,
                domain={"x":[0,1],"y":[0,1]},
                title={"text":f"Confidence Score"},
                gauge={
                    "axis":{"range":[0,100]},
                    "bar":{"color":"#378ADD"},
                    "steps":[
                        {"range":[0,50],"color":"#FCEBEB"},
                        {"range":[50,100],"color":"#EAF3DE"}
                    ],
                    "threshold":{
                        "line":{"color":"red","width":4},
                        "thickness":0.75,
                        "value":50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            if sentiment == "POSITIVE":
                st.success("This review expresses a positive experience. Good for business reputation.")
            else:
                st.error("This review expresses a negative experience. Needs attention from customer service.")
        else:
            st.warning("Please enter a review first.")

with tab2:
    st.subheader("Batch Analysis — Multiple Reviews at Once")
    st.caption("Enter one review per line")
    batch_text = st.text_area("Paste multiple reviews:", height=200,
        placeholder="The product was great!\nTerrible experience.\nAverage quality.")

    if st.button("Analyse All Reviews", type="primary"):
        reviews = [r.strip() for r in batch_text.split("\n") if r.strip()]
        if reviews:
            results = []
            progress = st.progress(0)
            for i, r in enumerate(reviews):
                res = nlp(r[:512])[0]
                results.append({
                    "Review": r[:80] + "..." if len(r) > 80 else r,
                    "Sentiment": "POSITIVE 😊" if res["label"]=="POSITIVE" else "NEGATIVE 😞",
                    "Confidence": f"{res['score']*100:.1f}%"
                })
                progress.progress((i+1)/len(reviews))

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            pos = sum(1 for r in results if "POSITIVE" in r["Sentiment"])
            neg = len(results) - pos

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Reviews", len(results))
            col2.metric("Positive", pos)
            col3.metric("Negative", neg)
            col4.metric("Positive Rate", f"{pos/len(results)*100:.1f}%")

            import plotly.express as px
            fig = px.pie(values=[pos, neg],
                        names=["Positive", "Negative"],
                        color_discrete_sequence=["#378ADD","#D85A30"],
                        title="Sentiment Breakdown")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter at least one review.")

with tab3:
    st.subheader("About This Project")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Model:** DistilBERT (fine-tuned on SST-2)
        
        **Dataset:** IMDB Movie Reviews (50,000 samples)
        
        **Task:** Binary Sentiment Classification
        
        **Baseline AUC:** ~0.93 (TF-IDF + Logistic Regression)
        
        **DistilBERT AUC:** ~0.97
        """)
    with col2:
        st.markdown("""
        **Tech Stack:**
        - Python
        - HuggingFace Transformers
        - DistilBERT
        - Scikit-learn
        - Streamlit + Plotly
        - MLflow
        """)

st.markdown("---")
st.caption("Built by Bhavana Aswin | B.Tech CS (AI) | DistilBERT + HuggingFace + Streamlit")
