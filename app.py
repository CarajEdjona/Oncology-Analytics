
import streamlit as st
import pandas as pd
import pickle

with open('cancer_model.pkl', 'rb') as f:
    trained_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Cancer Diagnostics & Genomics Workspace", layout="wide")
st.title("Cancer Diagnostics & Genomics Workspace")
st.write("An independent project using data science and genetics to analyze breast cancer cell features and scan for DNA mutations.")

tab1, tab2 = st.tabs(["Cell Shape Predictor (Phenotype)", "BRCA1 DNA Scanner (Genotype)"])

with tab1:
    st.header("Clinical Tumor Shape Predictor")
    st.write("Using machine learning trained on real patient data to analyze cell measurements and predict cancer risk.")
    st.subheader("Input Cell Measurements:")

    if "concavity_val" not in st.session_state: st.session_state.concavity_val = 0.30
    if "compactness_val" not in st.session_state: st.session_state.compactness_val = 0.20
    if "radius_val" not in st.session_state: st.session_state.radius_val = 14.0

    def update_c_slide(): st.session_state.concavity_val = st.session_state.c_slide
    def update_c_num(): st.session_state.concavity_val = st.session_state.c_num
    def update_comp_slide(): st.session_state.compactness_val = st.session_state.comp_slide
    def update_comp_num(): st.session_state.compactness_val = st.session_state.comp_num
    def update_r_slide(): st.session_state.radius_val = st.session_state.r_slide
    def update_r_num(): st.session_state.radius_val = st.session_state.r_num

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Surface Concavity**")
        st.slider("Slide to adjust:", 0.0, 1.3, key="c_slide", on_change=update_c_slide, value=st.session_state.concavity_val, help="How big or small the cell is.")
        concavity = st.number_input("Or type exact value:", 0.0, 1.3, key="c_num", on_change=update_c_num, value=st.session_state.concavity_val, step=0.01)
    with col2:
        st.markdown("**2. Cell Compactness**")
        st.slider("Slide to adjust:", 0.0, 1.1, key="comp_slide", on_change=update_comp_slide, value=st.session_state.compactness_val, help="How tightly packed together the cells look.")
        compactness = st.number_input("Or type exact value:", 0.0, 1.1, key="comp_num", on_change=update_comp_num, value=st.session_state.compactness_val, step=0.01)
    with col3:
        st.markdown("**3. Mean Radius (Cell Size)**")
        st.slider("Slide to adjust:", 5.0, 30.0, key="r_slide", on_change=update_r_slide, value=st.session_state.radius_val, help="How many hollow dents there are on the outside edge of the cells.")
        radius_val = st.number_input("Or type exact value:", 5.0, 30.0, key="r_num", on_change=update_r_num, value=st.session_state.radius_val, step=0.1)

    st.markdown("---")
    if st.button("Run Cancer Prediction"):
        user_features = pd.DataFrame([[st.session_state.concavity_val, st.session_state.compactness_val, st.session_state.radius_val]],
                                     columns=['worst concavity', 'worst compactness', 'mean radius'])
        user_features_scaled = scaler.transform(user_features)
        prediction = trained_model.predict(user_features_scaled)

        if prediction == 0:
            st.error("Result: High Probability of Malignancy (Cancerous Cell Shapes)")
        else:
            st.success("Result: High Probability of Benign (Safe/Healthy Cell Shapes)")


with tab2:
    st.header("BRCA1 Genetic Mutation Scanner")
    st.write("Scanning raw DNA sequences for known genetic mutations that increase breast cancer risk.")

    st.info("BRCA stands for BReast CAncer gene. When it is working normally, it repairs damaged DNA to stop tumors from forming. If it mutates, the body loses that protection, which increases breast cancer risks." )

    st.subheader("BRCA1 Genetic Reference Guide:")
    variant_data = {
        'BRCA1 Location': ['BRCA1 Exon 2 (185delAG Variant)', 'BRCA1 Exon 2 (Wild-Type Control)', 'BRCA1 Exon 11 (C61G Variant)', 'BRCA1 Exon 20 (5382insC Variant)'],
        'DNA Sequence Marker': ['AGAGAA', 'AGGGAA', 'TTTATC', 'AAGAAA'],
        'Mutation Type': ['Frameshift Deletion', 'Normal Sequence', 'Missense Mutation', 'Frameshift Insertion'],
        'What it does to the cell': ['Premature Stop Codon (leaves it unfinished)', 'Healthy DNA', 'Deforms the protein so it cannot stick to damaged DNA', 'Protein folds incorrectly, making it useless']
    }
    variant_df = pd.DataFrame(variant_data)
    st.dataframe(variant_df, use_container_width=True)

    st.markdown("---")
    st.subheader("DNA Sequence Scanner:")
    st.write("Input a DNA Sequence from the table above to check for dangerous genetic mutations:")

    dna_input = st.text_input("Enter DNA sequence marker (A, T, C, G):", value="AGAGAA")

    if st.button("Run DNA Mutation Scan"):
        clean_dna = dna_input.upper().strip().replace(" ", "")
        rna = clean_dna.replace("T", "U")
        st.write(f"**Step 1: Generated Transcription mRNA Sequence:** `{rna}`")

        if "AGAGAA" in clean_dna:
            st.error("Mutation Detected: Confirmed BRCA1 frameshift deletion.")
            st.markdown("**Biological Impact:** This mutation destroys the cell's ability to repair severe DNA damage. Because the cell cannot fix itself, it begins to divide out of control. This rapid growth creates the oversized and dented cell shapes flagged on **Tab 1**. ")
        elif "AGGGAA" in clean_dna:
            st.success("Healthy Sequence Verified: Confirmed normal BRCA1 gene.")
            st.markdown("**Biological Impact:** This gene is working normally to repair DNA damage. Cell growth remains under control, so tissue cells stay small and smooth, as predicted on **Tab 1**.")
        elif "TTTATC" in clean_dna:
            st.error("Mutation Detected: Confirmed BRCA1 missense mutation.")
            st.markdown("**Biological Impact:** This mutation changes the protein's shape, which prevents it from holding onto and repairing broken DNA. Over time, this allows cells to crowd together, leading to the cell shapes detected on **Tab 1**.")
        elif "AAGAAA" in clean_dna:
            st.error("Mutation Detected: Confirmed BRCA1 frameshift insertion.")
            st.markdown("**Biological Impact:** This mutation deforms the end of the protein chain, making it useless at cell checkpoints. This causes the cells to multiply rapidly and bloat, causing the shapes detected on **Tab 1**.")
        else:
            st.info("This DNA Sequence does not match any of the high-risk breast cancer mutations in the database above.")
