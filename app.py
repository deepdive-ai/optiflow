from __future__ import annotations

import base64
import json
import os
import tempfile
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
ANTHROPIC_MODEL = "claude-haiku-4-5"

CLINICAL_SUMMARY_SYSTEM_PROMPT = """You are an AI clinical assistant supporting an optometrist. You are not an autonomous clinician and you do not provide diagnoses.

Given structured examination data for one patient visit, write a concise clinical summary for the optometrist. You may:
- Summarize the key examination findings
- Note values or patterns that may be outside typical ranges
- Highlight areas the optometrist may want to review
- Note where further evaluation may be appropriate
- Reference the assessment category and referral status already recorded in the data
- If data from a previous visit is provided, note which values have changed and by how much, using cautious language (e.g. "IOP has increased since the last visit, which may warrant closer monitoring")

You must NOT:
- State or imply a definitive diagnosis (e.g. never say "the patient has glaucoma")
- Claim that you diagnosed anything
- Present your output as medical advice
- Claim that a change between visits confirms disease progression — describe it as a change worth reviewing, not a confirmed trend

When flagging a finding, use cautious language such as "potential finding to review," "consider further evaluation," "potential referral consideration," or "this finding may warrant additional assessment."

If no previous visit is available for comparison, state that plainly rather than guessing at a trend.

Keep the summary concise (roughly 100-200 words), scannable, and useful to a busy optometrist."""


def _format_visit_block(visit: pd.Series, heading: str | None = None) -> str:
    lines = [f"{heading}\n"] if heading else []
    lines += [
        f"Visit Date: {visit['Visit_Date'].date()}\n",
        f"Visual Acuity OD/OS: {visit['Visual_Acuity_OD']} / {visit['Visual_Acuity_OS']}\n",
        f"IOP OD/OS (mmHg): {visit['IOP_OD_mmHg']} / {visit['IOP_OS_mmHg']}\n",
        f"Spherical Equivalent OD/OS (D): {visit['Spherical_Equivalent_OD_D']} / {visit['Spherical_Equivalent_OS_D']}\n",
        f"C/D Ratio OD/OS: {visit['CD_Ratio_OD']} / {visit['CD_Ratio_OS']}\n",
        f"Assessment Category: {visit['Assessment_Category']}\n",
    ]
    return "".join(lines)


def build_patient_summary_prompt(current: pd.Series, previous: pd.Series | None) -> str:
    referral_specialty = current["Referral_Specialty"] if pd.notna(current["Referral_Specialty"]) else "N/A"
    current_block = (
        f"Patient ID: {current['Patient_ID']}\n"
        f"Age: {current['Age']}\n"
        f"Gender: {current['Gender']}\n"
        + _format_visit_block(current)
        + f"Referral Required: {current['Referral_Required']}\n"
        f"Referral Specialty: {referral_specialty}\n"
    )

    if previous is None:
        return (
            current_block
            + "\nPrevious visits on file: None — this is the only recorded visit for this patient "
            "in the current dataset."
        )

    previous_block = _format_visit_block(previous, heading="Previous Visit")
    return (
        current_block
        + "\n"
        + previous_block
        + "\nA previous visit is available above for comparison — note any notable changes "
        "between the previous and current visit per your instructions."
    )


def generate_ai_summary(current: pd.Series, previous: pd.Series | None) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=CLINICAL_SUMMARY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_patient_summary_prompt(current, previous)}],
    )
    return next(block.text for block in response.content if block.type == "text")


CASE_SUMMARY_SYSTEM_PROMPT = """You are an AI clinical assistant supporting an optometrist. You are not an autonomous clinician and you do not provide diagnoses.

The optometrist has uploaded a prior report or eye photo for a patient they are seeing. Write a concise summary of what is visible or stated in the uploaded material, to help the optometrist review it quickly. You may:
- Describe key details you can identify (reported findings, measurements, visible image characteristics)
- Note anything that looks unclear, illegible, or outside typical ranges, using cautious language such as "potential finding to review" or "may warrant closer inspection"
- Suggest what the optometrist may want to look at more closely in the original file

You must NOT:
- State or imply a definitive diagnosis
- Claim you diagnosed anything from the image or report
- Present your output as medical advice

If the uploaded content is illegible, not a clinical document or eye photo, or otherwise unclear, say so plainly rather than guessing.

Keep the summary concise (roughly 100-200 words)."""


def build_case_content_blocks(files) -> list[dict]:
    blocks = []
    for file in files:
        data = base64.standard_b64encode(file.getvalue()).decode("utf-8")
        block_type = "document" if file.type == "application/pdf" else "image"
        blocks.append({"type": block_type, "source": {"type": "base64", "media_type": file.type, "data": data}})
    blocks.append({
        "type": "text",
        "text": "Summarize the uploaded prior report/photo(s) above for the optometrist reviewing this patient.",
    })
    return blocks


def generate_case_summary(files) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=CASE_SUMMARY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_case_content_blocks(files)}],
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


VA_SCALE = ["6/6", "6/9", "6/12", "6/18", "6/24"]


def va_line_diff(before: str, after: str) -> int:
    """Positive = worse (moved down the Snellen scale), negative = improved."""
    return VA_SCALE.index(after) - VA_SCALE.index(before)


def patient_history_page() -> None:
    examinations = load_examinations()
    st.title("Patient History")
    st.caption(
        "Compare a patient's examinations across visits. Only patients with more "
        "than one recorded visit have a history to show — most patients in this "
        "synthetic dataset still have just one."
    )

    patient_id = st.selectbox(
        "Patient", sorted(examinations["Patient_ID"].unique()), key="history_patient"
    )
    patient_visits = examinations[examinations["Patient_ID"] == patient_id].sort_values("Visit_Date")

    if len(patient_visits) < 2:
        st.info(f"Only one recorded visit on file for {patient_id} — no history to compare yet.")
        st.dataframe(
            patient_visits,
            width="stretch",
            column_config={"Visit_Date": st.column_config.DateColumn("Visit_Date", format="YYYY-MM-DD")},
        )
        return

    st.caption(f"{len(patient_visits)} visits on file for {patient_id}.")

    visit1, visit2 = patient_visits.iloc[-2], patient_visits.iloc[-1]

    visit_cols = st.columns(2)
    for col, visit, label in zip(visit_cols, (visit1, visit2), ("Earlier Visit", "Most Recent Visit")):
        with col, st.container(border=True):
            st.subheader(f"{label} — {visit['Visit_Date'].date()}")
            metric_cols = st.columns(2)
            metric_cols[0].metric("Visual Acuity OD/OS", f"{visit['Visual_Acuity_OD']} / {visit['Visual_Acuity_OS']}")
            metric_cols[1].metric("IOP OD/OS", f"{visit['IOP_OD_mmHg']} / {visit['IOP_OS_mmHg']}")
            metric_cols = st.columns(2)
            metric_cols[0].metric("C/D Ratio OD/OS", f"{visit['CD_Ratio_OD']} / {visit['CD_Ratio_OS']}")
            metric_cols[1].metric("Assessment", visit["Assessment_Category"])

    st.subheader("Change Since Earlier Visit")
    st.caption("Red/upward deltas flag values where an increase is the finding worth reviewing (IOP, C/D ratio, worse visual acuity). Spherical equivalent shift has no inherent 'better' direction, so it's shown without color.")

    change_row1 = st.columns(3)
    change_row1[0].metric(
        "IOP OD (mmHg)", visit2["IOP_OD_mmHg"],
        delta=round(float(visit2["IOP_OD_mmHg"] - visit1["IOP_OD_mmHg"]), 2), delta_color="inverse",
    )
    change_row1[1].metric(
        "IOP OS (mmHg)", visit2["IOP_OS_mmHg"],
        delta=round(float(visit2["IOP_OS_mmHg"] - visit1["IOP_OS_mmHg"]), 2), delta_color="inverse",
    )
    change_row1[2].metric(
        "Visual Acuity OD", visit2["Visual_Acuity_OD"],
        delta=va_line_diff(visit1["Visual_Acuity_OD"], visit2["Visual_Acuity_OD"]), delta_color="inverse",
    )

    change_row2 = st.columns(3)
    change_row2[0].metric(
        "C/D Ratio OD", visit2["CD_Ratio_OD"],
        delta=round(float(visit2["CD_Ratio_OD"] - visit1["CD_Ratio_OD"]), 2), delta_color="inverse",
    )
    change_row2[1].metric(
        "C/D Ratio OS", visit2["CD_Ratio_OS"],
        delta=round(float(visit2["CD_Ratio_OS"] - visit1["CD_Ratio_OS"]), 2), delta_color="inverse",
    )
    change_row2[2].metric(
        "Visual Acuity OS", visit2["Visual_Acuity_OS"],
        delta=va_line_diff(visit1["Visual_Acuity_OS"], visit2["Visual_Acuity_OS"]), delta_color="inverse",
    )

    change_row3 = st.columns(2)
    change_row3[0].metric(
        "Spherical Equivalent OD (D)", visit2["Spherical_Equivalent_OD_D"],
        delta=round(float(visit2["Spherical_Equivalent_OD_D"] - visit1["Spherical_Equivalent_OD_D"]), 2),
        delta_color="off",
    )
    change_row3[1].metric(
        "Spherical Equivalent OS (D)", visit2["Spherical_Equivalent_OS_D"],
        delta=round(float(visit2["Spherical_Equivalent_OS_D"] - visit1["Spherical_Equivalent_OS_D"]), 2),
        delta_color="off",
    )

    st.subheader("IOP Trend")
    iop_trend = px.line(
        patient_visits,
        x="Visit_Date",
        y=["IOP_OD_mmHg", "IOP_OS_mmHg"],
        markers=True,
        labels={"value": "IOP (mmHg)", "Visit_Date": "Visit Date", "variable": "Eye"},
    )
    st.plotly_chart(iop_trend, use_container_width=True)

    with st.expander("All recorded visits"):
        st.dataframe(
            patient_visits,
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
    patient_visits = examinations[examinations["Patient_ID"] == patient_id].sort_values("Visit_Date")
    patient = patient_visits.iloc[-1]
    previous_visit = patient_visits.iloc[-2] if len(patient_visits) > 1 else None

    st.caption(
        f"{len(patient_visits)} visit(s) on file for {patient_id}."
        + (f" Comparing against the previous visit on {previous_visit['Visit_Date'].date()}."
           if previous_visit is not None else "")
    )

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
                st.session_state["ai_summary_text"] = generate_ai_summary(patient, previous_visit)
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
    st.title("Upload Prior Report")
    st.caption(
        "For the optometrist: upload a patient's prior report or eye photo to get an "
        "AI-assisted summary before the exam. Synthetic/demo data only."
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY is not set in the environment. Set it before generating a summary.")
        return

    st.warning(
        "**Do not upload real patient photos or medical reports.** Uploaded file "
        "content is sent to Anthropic's API to generate the summary below — use "
        "placeholder/dummy files only, e.g. a random test image or PDF. Nothing is "
        "persisted to disk by this app."
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

    acknowledged = st.checkbox(
        "I confirm these are placeholder/dummy files only — not real patient data — "
        "and understand they will be sent to Anthropic's API for summarization."
    )

    if st.button("Generate AI Summary", disabled=not (uploaded_files and acknowledged)):
        with st.spinner("Generating summary..."):
            try:
                st.session_state["case_summary_text"] = generate_case_summary(uploaded_files)
            except Exception as exc:
                st.error(f"Could not generate summary: {exc}")

    if st.session_state.get("case_summary_text"):
        st.info("**AI-generated summary — optometrist review required.**")
        st.write(st.session_state["case_summary_text"])


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
            st.Page(patient_history_page, title="Patient History", icon="📈", url_path="patient-history"),
            st.Page(referrals_page, title="Ophthalmology Referrals", icon="🏥", url_path="referrals"),
            st.Page(ai_summary_page, title="AI Clinical Summary", icon="🤖", url_path="ai-summary"),
            st.Page(submit_case_page, title="Upload Prior Report", icon="📤", url_path="submit-case"),
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
