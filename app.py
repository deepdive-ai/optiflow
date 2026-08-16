import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="OptiFlow", layout="wide")

VOICE_NOTES_FILE = Path("voice_notes.json")

# Cheap to swap (e.g. to a Sonnet/Haiku model) without touching the call site below.
ANTHROPIC_MODEL = "claude-opus-5"

CLINICAL_SUMMARY_SYSTEM_PROMPT = """You are an AI clinical assistant supporting an optometrist. You are not an autonomous clinician and you do not provide diagnoses.

Given structured examination data for one patient visit, write a concise clinical summary for the optometrist. You may:
- Summarize the key examination findings
- Note values or patterns that may be outside typical ranges
- Highlight areas the optometrist may want to review
- Note where further evaluation may be appropriate
- Reference the assessment category and referral status already recorded in the data

You must NOT:
- State or imply a definitive diagnosis (e.g. never say "the patient has glaucoma")
- Claim that you diagnosed anything
- Present your output as medical advice

When flagging a finding, use cautious language such as "potential finding to review," "consider further evaluation," "potential referral consideration," or "this finding may warrant additional assessment."

If no previous visit is available for comparison, state that plainly rather than guessing at a trend.

Keep the summary concise (roughly 100-200 words), scannable, and useful to a busy optometrist."""


def build_patient_summary_prompt(patient: pd.Series) -> str:
    referral_specialty = patient["Referral_Specialty"] if pd.notna(patient["Referral_Specialty"]) else "N/A"
    return (
        f"Patient ID: {patient['Patient_ID']}\n"
        f"Age: {patient['Age']}\n"
        f"Gender: {patient['Gender']}\n"
        f"Visit Date: {patient['Visit_Date'].date()}\n"
        f"Visual Acuity OD/OS: {patient['Visual_Acuity_OD']} / {patient['Visual_Acuity_OS']}\n"
        f"IOP OD/OS (mmHg): {patient['IOP_OD_mmHg']} / {patient['IOP_OS_mmHg']}\n"
        f"Spherical Equivalent OD/OS (D): {patient['Spherical_Equivalent_OD_D']} / {patient['Spherical_Equivalent_OS_D']}\n"
        f"C/D Ratio OD/OS: {patient['CD_Ratio_OD']} / {patient['CD_Ratio_OS']}\n"
        f"Assessment Category: {patient['Assessment_Category']}\n"
        f"Referral Required: {patient['Referral_Required']}\n"
        f"Referral Specialty: {referral_specialty}\n\n"
        "Previous visits on file: None — this is the only recorded visit for this patient "
        "in the current dataset."
    )


def generate_ai_summary(patient: pd.Series) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=CLINICAL_SUMMARY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_patient_summary_prompt(patient)}],
    )
    return next(block.text for block in response.content if block.type == "text")


@st.cache_resource
def load_whisper_model():
    from faster_whisper import WhisperModel

    return WhisperModel("small", device="cpu", compute_type="int8")


def transcribe_audio(audio_file) -> str:
    model = load_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp.flush()
        segments, _ = model.transcribe(tmp.name)
        return " ".join(segment.text.strip() for segment in segments)


def load_voice_notes() -> dict:
    if not VOICE_NOTES_FILE.exists():
        return {}
    return json.loads(VOICE_NOTES_FILE.read_text())


def save_voice_note(patient_id: str, text: str) -> None:
    notes = load_voice_notes()
    notes.setdefault(patient_id, []).append(
        {"timestamp": datetime.now().isoformat(timespec="seconds"), "text": text}
    )
    VOICE_NOTES_FILE.write_text(json.dumps(notes, indent=2))


@st.cache_data
def load_examinations() -> pd.DataFrame:
    df = pd.read_csv("patients_examinations.csv")
    df["Visit_Date"] = pd.to_datetime(df["Visit_Date"])
    return df


@st.cache_data
def load_ophthalmologists() -> pd.DataFrame:
    return pd.read_csv("ophthalmologists.csv")


def render_filters(examinations_df: pd.DataFrame) -> pd.DataFrame:
    """Sidebar filter widgets for the exam data. Keys keep selections in sync
    when the user switches between pages that both call this function."""
    st.sidebar.header("Filters")

    patient_id_query = st.sidebar.text_input("Patient ID contains", key="filter_patient_id")

    selected_assessments = st.sidebar.multiselect(
        "Assessment Category",
        sorted(examinations_df["Assessment_Category"].unique()),
        key="filter_assessment",
    )

    referral_status = st.sidebar.selectbox(
        "Referral Required", ["All", "Yes", "No"], key="filter_referral_status"
    )

    selected_specialties = st.sidebar.multiselect(
        "Referral Specialty",
        sorted(examinations_df["Referral_Specialty"].dropna().unique()),
        key="filter_specialty",
    )

    age_bounds = (int(examinations_df["Age"].min()), int(examinations_df["Age"].max()))
    selected_age_range = st.sidebar.slider(
        "Age Range", *age_bounds, value=age_bounds, key="filter_age"
    )

    date_bounds = (
        examinations_df["Visit_Date"].min().date(),
        examinations_df["Visit_Date"].max().date(),
    )
    selected_date_range = st.sidebar.date_input(
        "Visit Date Range",
        value=date_bounds,
        min_value=date_bounds[0],
        max_value=date_bounds[1],
        key="filter_date",
    )

    iop_od_bounds = (
        int(examinations_df["IOP_OD_mmHg"].min()),
        int(examinations_df["IOP_OD_mmHg"].max()),
    )
    selected_iop_od_range = st.sidebar.slider(
        "IOP OD Range (mmHg)", *iop_od_bounds, value=iop_od_bounds, key="filter_iop_od"
    )

    iop_os_bounds = (
        int(examinations_df["IOP_OS_mmHg"].min()),
        int(examinations_df["IOP_OS_mmHg"].max()),
    )
    selected_iop_os_range = st.sidebar.slider(
        "IOP OS Range (mmHg)", *iop_os_bounds, value=iop_os_bounds, key="filter_iop_os"
    )

    filtered = examinations_df.copy()

    if patient_id_query:
        filtered = filtered[
            filtered["Patient_ID"].str.contains(patient_id_query, case=False, na=False)
        ]

    if selected_assessments:
        filtered = filtered[filtered["Assessment_Category"].isin(selected_assessments)]

    if referral_status != "All":
        filtered = filtered[filtered["Referral_Required"] == referral_status]

    if selected_specialties:
        filtered = filtered[filtered["Referral_Specialty"].isin(selected_specialties)]

    filtered = filtered[filtered["Age"].between(*selected_age_range)]

    if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        filtered = filtered[filtered["Visit_Date"].dt.date.between(start_date, end_date)]

    filtered = filtered[filtered["IOP_OD_mmHg"].between(*selected_iop_od_range)]
    filtered = filtered[filtered["IOP_OS_mmHg"].between(*selected_iop_os_range)]

    return filtered


def dashboard_page() -> None:
    filtered = st.session_state["filtered_examinations"]
    st.title("Dashboard")

    total_patients = filtered["Patient_ID"].nunique()
    total_examinations = len(filtered)
    referrals_required = (filtered["Referral_Required"] == "Yes").sum()
    referral_rate = referrals_required / total_examinations if total_examinations else 0

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Patients", total_patients)
    kpi_cols[1].metric("Total Examinations", total_examinations)
    kpi_cols[2].metric("Referrals Required", referrals_required)
    kpi_cols[3].metric("Referral Rate", f"{referral_rate:.0%}")

    if filtered.empty:
        st.warning("No patients match the current filter selection. Adjust the filters in the sidebar.")
        return

    referral_by_specialty = (
        filtered.loc[filtered["Referral_Required"] == "Yes", "Referral_Specialty"]
        .value_counts()
        .sort_values(ascending=False)
    )
    assessment_category_counts = filtered["Assessment_Category"].value_counts().sort_values(
        ascending=False
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Assessment Categories")
        st.bar_chart(assessment_category_counts)
    with chart_cols[1]:
        st.subheader("Referrals by Specialty")
        if referral_by_specialty.empty:
            st.info("No referrals in the current filter selection.")
        else:
            referral_pie = px.pie(
                referral_by_specialty.reset_index(name="Count"),
                names="Referral_Specialty",
                values="Count",
            )
            referral_pie.update_traces(textinfo="label+percent")
            st.plotly_chart(referral_pie, use_container_width=True)


def patient_examinations_page() -> None:
    examinations = load_examinations()
    filtered = st.session_state["filtered_examinations"]
    st.title("Patient Examinations")
    st.caption(f"Showing {len(filtered)} of {len(examinations)} examinations.")
    st.dataframe(
        filtered,
        width="stretch",
        column_config={"Visit_Date": st.column_config.DateColumn("Visit_Date", format="YYYY-MM-DD")},
    )


def referrals_page() -> None:
    ophthalmologists = load_ophthalmologists()
    st.title("Ophthalmology Referral Directory")
    st.caption("Synthetic/demo referral contacts only.")
    st.dataframe(ophthalmologists, width="stretch")


def ai_summary_page() -> None:
    st.title("AI Clinical Summary")
    st.caption(
        "AI-assisted summary of the selected patient's examination. Synthetic data only. "
        "The optometrist retains final clinical judgment — this is not a diagnosis."
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY is not set in the environment. Set it before generating a summary.")
        return

    examinations = load_examinations()
    patient_id = st.selectbox(
        "Patient", sorted(examinations["Patient_ID"].unique()), key="ai_summary_patient"
    )
    patient = examinations[examinations["Patient_ID"] == patient_id].iloc[0]

    with st.container(border=True):
        st.subheader(f"Current Examination — {patient_id}")
        cols = st.columns(4)
        cols[0].metric("Visual Acuity OD/OS", f"{patient['Visual_Acuity_OD']} / {patient['Visual_Acuity_OS']}")
        cols[1].metric("IOP OD/OS", f"{patient['IOP_OD_mmHg']} / {patient['IOP_OS_mmHg']}")
        cols[2].metric("C/D Ratio OD/OS", f"{patient['CD_Ratio_OD']} / {patient['CD_Ratio_OS']}")
        cols[3].metric("Assessment", patient["Assessment_Category"])

    if st.button("Generate AI Clinical Summary"):
        with st.spinner("Generating summary..."):
            try:
                st.session_state["ai_summary_text"] = generate_ai_summary(patient)
                st.session_state["ai_summary_patient_id"] = patient_id
            except Exception as exc:
                st.error(f"Could not generate summary: {exc}")

    if (
        st.session_state.get("ai_summary_text")
        and st.session_state.get("ai_summary_patient_id") == patient_id
    ):
        st.info("**AI-generated summary — optometrist review required.**")
        st.write(st.session_state["ai_summary_text"])


def submit_case_page() -> None:
    st.title("Submit a Case for Second Opinion")
    st.warning(
        "**Concept demo only — not a real upload.** Do not upload real patient photos "
        "or medical reports. Nothing on this page is stored, encrypted, or sent to any "
        "AI service; it exists to preview the intended UX flow only."
    )
    st.caption(
        "Previews how a patient might submit prior reports or eye photos for a second "
        "opinion. Use placeholder/dummy files only — e.g. a random test image or PDF."
    )

    uploaded_files = st.file_uploader(
        "Upload prior reports or eye photos (dummy files only)",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.subheader("Preview")
        for file in uploaded_files:
            if file.type.startswith("image/"):
                st.image(file, caption=file.name, width=300)
            else:
                st.info(f"📄 {file.name} ({file.size / 1024:.1f} KB)")

    consent = st.checkbox(
        "I consent to this case, including any uploaded files, being reviewed by a "
        "licensed ophthalmologist, and understand any AI assistance is for "
        "summarization only and subject to the reviewing doctor's judgment. "
        "(Placeholder consent text for this demo — not a real agreement.)"
    )

    if st.button("Submit Case", disabled=not (uploaded_files and consent)):
        st.session_state["demo_case_id"] = f"DEMO-{uuid.uuid4().hex[:8].upper()}"

    if st.session_state.get("demo_case_id"):
        st.success(
            f"Case submitted. Reference ID: {st.session_state['demo_case_id']} "
            "— demo only, not real, nothing stored."
        )
        st.subheader("AI Summary")
        st.info(
            "🔒 An AI-generated summary would appear here once connected to a real "
            "summarization service. Not implemented in this demo — no file content "
            "has been sent anywhere."
        )


def voice_notes_page() -> None:
    st.title("Dictate Exam Note")
    st.caption(
        "Prototype: doctor records a note about a patient exam. Transcription runs "
        "locally on this machine via faster-whisper — audio and text never leave "
        "this device, no third-party service involved. Synthetic patients only. "
        "Saved notes are stored in a plain local file (voice_notes.json) — fine for "
        "this prototype, not the encrypted/access-controlled storage real patient "
        "data would need."
    )

    examinations = load_examinations()
    patient_id = st.selectbox("Patient", sorted(examinations["Patient_ID"].unique()))

    audio = st.audio_input("Record note")

    if audio is not None and st.button("Transcribe"):
        with st.spinner("Transcribing locally..."):
            st.session_state["voice_note_transcript"] = transcribe_audio(audio)

    if st.session_state.get("voice_note_transcript"):
        st.subheader(f"Transcript for {patient_id} (editable)")
        edited_text = st.text_area(
            "Note",
            value=st.session_state["voice_note_transcript"],
            height=200,
            key="voice_note_edit",
        )
        if st.button("Save Note"):
            save_voice_note(patient_id, edited_text)
            st.session_state["voice_note_transcript"] = None
            st.success(f"Note saved for {patient_id}.")
            st.rerun()

    st.divider()
    st.subheader(f"Saved notes for {patient_id}")
    patient_notes = load_voice_notes().get(patient_id, [])
    if not patient_notes:
        st.caption("No saved notes yet for this patient.")
    else:
        for note in reversed(patient_notes):
            with st.container(border=True):
                st.caption(note["timestamp"])
                st.write(note["text"])


pages = st.navigation(
    {
        "OptiFlow": [
            st.Page(dashboard_page, title="Dashboard", icon="📊", url_path="dashboard", default=True),
            st.Page(
                patient_examinations_page,
                title="Patient Examinations",
                icon="📋",
                url_path="patient-examinations",
            ),
            st.Page(referrals_page, title="Ophthalmology Referrals", icon="🏥", url_path="referrals"),
            st.Page(ai_summary_page, title="AI Clinical Summary", icon="🤖", url_path="ai-summary"),
            st.Page(submit_case_page, title="Submit a Case", icon="📤", url_path="submit-case"),
            st.Page(voice_notes_page, title="Dictate Note", icon="🎙️", url_path="voice-notes"),
        ]
    }
)

st.sidebar.caption(
    "AI-assisted clinical workflow assistant for optometrists — "
    "all data shown here is synthetic/demo data only."
)

# Filter widgets live here, at a fixed point in the script that runs on every
# rerun regardless of which page is selected. Streamlit's page router resets
# widget state that isn't instantiated at a consistent script position across
# page switches, so rendering them inside each page function (and relying on
# `key=` alone) silently loses the selection when navigating. Rendering them
# once here and handing the result down via session_state avoids that.
if pages.title in ("Dashboard", "Patient Examinations"):
    st.session_state["filtered_examinations"] = render_filters(load_examinations())

pages.run()
