import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️")
st.title("🕵️ Fraudulent Job Posting Detector")
st.caption("Paste a job post. The ML model estimates how likely it is to be fake.")

@st.cache_resource
def load():
    return joblib.load("model.joblib")

m = load()
title = st.text_input("Job title")
company = st.text_area("Company profile", height=80)
desc = st.text_area("Job description", height=150)
req = st.text_area("Requirements", height=80)
c1, c2, c3 = st.columns(3)
logo = c1.checkbox("Has company logo", True)
ques = c2.checkbox("Has screening questions", True)
sal = c3.checkbox("Salary mentioned", True)

if st.button("Check posting", type="primary"):
    text = f"{title} {company} {desc} {req}"
    text += "" if logo else " nologo"
    text += "" if ques else " noquestions"
    text += "" if sal else " nosalary"
    X = m["vec"].transform([text])
    p = m["clf"].predict_proba(X)[0, 1]
    if p >= 0.5:
        st.error(f"⚠️ Likely FAKE — {p:.0%} fraud probability")
    else:
        st.success(f"✅ Likely genuine — {p:.0%} fraud probability")
    st.progress(float(p))

    # explain: words pushing towards "fake"
    names = np.array(m["vec"].get_feature_names_out())
    contrib = X.toarray()[0] * m["clf"].coef_[0]
    top = np.argsort(contrib)[::-1][:8]
    words = [names[i] for i in top if contrib[i] > 0]
    if words:
        st.write("**Words that raised suspicion:**", ", ".join(words))
