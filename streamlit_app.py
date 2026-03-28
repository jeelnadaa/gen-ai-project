import json
import os
from pathlib import Path

import streamlit as st
from app import process_pdf

st.set_page_config(page_title="Legal Clause Simplifier", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Satoshi:wght@400;500;600;700&display=swap');
    html, body, [class^='css'] {
        font-family: 'Satoshi', 'Inter', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Legal Clause Simplifier (Streamlit)")
st.write("Upload a PDF and generate clause simplification + importance + evaluation JSON.")

with st.sidebar:
    st.header("Settings")

    if "show_info" not in st.session_state:
        st.session_state.show_info = ""

    def info_button(key: str, title: str, tooltip: str):
        left_col, right_col = st.columns([0.6, 9.4])
        with left_col:
            clicked = st.button("i", key=f"{key}_button")
            if clicked:
                if st.session_state.show_info == key:
                    st.session_state.show_info = ""
                else:
                    st.session_state.show_info = key
        with right_col:
            st.markdown(f"**{title}**")

        if st.session_state.show_info == key:
            st.success(tooltip)

    info_button(
        "groq_model_info",
        "Groq Model",
        "Select a model for Groq API calls. llama-3.3-70b-versatile = best quality; "
        "llama-3.1-8b-instant = faster/cheaper; gemma2-9b-it = balanced.",
    )
    groq_model = st.selectbox(
        "Groq model",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ],
        index=0,
    )

    info_button(
        "Max Clauses",
        "Max Clauses",
        "How many clauses to analyze. Set 0 to analyze all extracted clauses.",
    )
    max_clauses = st.number_input("Max clauses (0 = all)", min_value=0, value=0, step=1)

    info_button(
        "Min Clause Length",
        "Min Clause Length",
        "Ignore clauses shorter than this character length; helps reduce noise.",
    )
    min_clause_length = st.number_input("Min clause length", min_value=10, value=30, step=1)

    info_button(
        "Reference Summary",
        "Reference Summary",
        "Optional gold summary text used only for evaluation in ROUGE/metrics.",
    )
    reference_summary = st.text_area("Reference summary (optional)")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    tmp_dir = Path("./.tmp_streamlit_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_dir / uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Process PDF"):
        if not os.environ.get("GROQ_API_KEY"):
            st.error("Please set GROQ_API_KEY in environment variables before running")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(processed: int, total: int):
                progress = int(processed / total * 100)
                progress_bar.progress(progress)
                status_text.info(f"Processing clause {processed}/{total} ({progress}%)")

            with st.spinner("Processing PDF (this may take a while)..."):
                try:
                    result = process_pdf(
                        str(pdf_path),
                        groq_model=groq_model.strip() or None,
                        max_clauses=int(max_clauses) if max_clauses > 0 else None,
                        reference_summary=reference_summary.strip() or None,
                        ground_truth_labels=None,
                        min_clause_length=int(min_clause_length),
                        progress_callback=progress_callback,
                    )

                    progress_bar.progress(100)
                    status_text.success("All clauses processed")
                    st.success("Processing complete")

                    st.subheader("Summary")
                    st.write(result.get("summary", ""))

                    st.subheader("Evaluation")
                    st.json(result.get("evaluation", {}))

                    st.subheader(f"Clauses ({len(result.get('clauses', []))})")
                    clauses = result.get("clauses", [])
                    if clauses:
                        st.dataframe(
                            [
                                {
                                    "#": i + 1,
                                    "Original": c.get("original", ""),
                                    "Simplified": c.get("simplified", ""),
                                    "Importance": c.get("importance", ""),
                                    "Semantic similarity": c.get("semantic_similarity", ""),
                                }
                                for i, c in enumerate(clauses)
                            ]
                        )

                    st.subheader("JSON Output")
                    st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")

                    json_path = tmp_dir / f"{pdf_path.stem}.results.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(result, indent=2, ensure_ascii=False),
                        file_name=f"{pdf_path.stem}.results.json",
                        mime="application/json",
                    )
                except Exception as exc:
                    st.error(f"Error processing PDF: {exc}")
