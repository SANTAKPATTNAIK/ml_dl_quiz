import uuid
import streamlit as st
from openai import OpenAI, AuthenticationError
from langgraph.types import Command
from ingest import extract_and_generate_bank
from quiz_graph import build_quiz_graph

st.set_page_config(
    page_title="ML & DL Assessment Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.uploaded_names = []
    st.session_state.user_api_key = ""
    st.session_state.key_verified = False

def verify_openai_key(key: str) -> tuple[bool, str]:
    if not key or not key.startswith("sk-"):
        return False, "API key must start with 'sk-'."
    try:
        client = OpenAI(api_key=key)
        client.models.list()
        return True, "API key verified!"
    except AuthenticationError:
        return False, "Invalid OpenAI API key. Please check and try again."
    except Exception as e:
        return False, f"Connection error: {e}"

def get_latest_interrupt(graph, cfg):
    state = graph.get_state(cfg)
    if state.tasks and state.tasks[0].interrupts:
        return state.tasks[0].interrupts[0].value
    return None

# --- Sidebar: API Key Management & Session Controls ---
with st.sidebar:
    st.header("🧠 Assessment Studio")
    st.caption("LangGraph • BYOK (Bring Your Own Key)")
    st.divider()

    st.subheader("🔑 API Configuration")
    sidebar_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.user_api_key,
        placeholder="sk-proj-...",
        help="Your key is stored only in your current browser session and never saved on the server."
    )

    if sidebar_key != st.session_state.user_api_key:
        st.session_state.user_api_key = sidebar_key.strip()
        st.session_state.key_verified = False
        st.session_state.pop("graph", None)

    if st.session_state.user_api_key:
        if st.session_state.key_verified:
            st.success("Key Connected & Verified", icon="✅")
        else:
            if st.button("Verify Key", width="stretch"):
                ok, msg = verify_openai_key(st.session_state.user_api_key)
                if ok:
                    st.session_state.key_verified = True
                    st.rerun()
                else:
                    st.error(msg)
    else:
        st.info("Paste your OpenAI API key above to unlock PDF ingestion & grading.")

    if st.session_state.uploaded_names:
        st.divider()
        st.subheader("📚 Active Study Material")
        for name in st.session_state.uploaded_names:
            st.markdown(f"- `{name}`")

    st.divider()
    col_reset, col_clear = st.columns(2)
    with col_reset:
        if st.button("🔄 New Quiz", width="stretch"):
            saved_key = st.session_state.user_api_key
            verified = st.session_state.key_verified
            st.session_state.clear()
            st.session_state.user_api_key = saved_key
            st.session_state.key_verified = verified
            st.rerun()
    with col_clear:
        if st.button("🗑️ Clear Key", width="stretch"):
            st.session_state.clear()
            st.rerun()

# Initialize user-scoped LangGraph once key is provided
api_key = st.session_state.user_api_key
if api_key and "graph" not in st.session_state:
    st.session_state.graph = build_quiz_graph(api_key=api_key)

config = {"configurable": {"thread_id": st.session_state.thread_id}}
payload = st.session_state.interrupt_payload
graph_state = st.session_state.graph.get_state(config).values if "graph" in st.session_state else {}

# ==========================================
# STAGE 1: ONBOARDING & PDF UPLOAD
# ==========================================
if payload is None and not graph_state:
    st.title("ML & Deep Learning Question Bank")
    st.markdown(
        "Upload ML/DL textbooks or research papers in PDF format, choose **MCQ** or **Written** mode, "
        "and get graded by a stateful LangGraph examiner."
    )

    # Prominent Step 1 card if API key isn't entered yet
    if not api_key:
        with st.container(border=True):
            st.subheader("Step 1: Connect Your OpenAI API Key")
            st.caption("Your key stays strictly in your session memory and is discarded when you close the tab.")
            col_k, col_b = st.columns([4, 1], vertical_alignment="bottom")
            with col_k:
                main_key_input = st.text_input(
                    "Enter OpenAI API Key",
                    type="password",
                    placeholder="sk-proj-...",
                    label_visibility="collapsed"
                )
            with col_b:
                if st.button("Connect Key", type="primary", width="stretch"):
                    ok, msg = verify_openai_key(main_key_input.strip())
                    if ok:
                        st.session_state.user_api_key = main_key_input.strip()
                        st.session_state.key_verified = True
                        st.rerun()
                    else:
                        st.error(msg)

    with st.container(border=True):
        st.subheader("Step 2: Upload Reference PDFs" if not api_key else "Upload Reference Material")
        uploaded_files = st.file_uploader(
            "Drop one or more PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            disabled=not bool(api_key),
            label_visibility="collapsed"
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            max_chunks = st.number_input(
                "Max sections per PDF",
                min_value=2,
                max_value=25,
                value=6,
                disabled=not bool(api_key)
            )

        if st.button(
            "Build Question Bank →",
            type="primary",
            disabled=not (uploaded_files and api_key)
        ):
            try:
                all_questions = []
                progress_bar = st.progress(0, text="Initializing PDF parser...")
                for idx, pdf in enumerate(uploaded_files):
                    progress_bar.progress(
                        int((idx / len(uploaded_files)) * 100),
                        text=f"Extracting concepts from {pdf.name}..."
                    )
                    all_questions.extend(
                        extract_and_generate_bank(pdf.read(), api_key=api_key, max_chunks=max_chunks)
                    )
                progress_bar.progress(100, text="Finalizing question bank...")

                if all_questions:
                    st.session_state.uploaded_names = [f.name for f in uploaded_files]
                    st.session_state.graph.invoke({"question_bank": all_questions, "results": []}, config)
                    st.session_state.interrupt_payload = get_latest_interrupt(st.session_state.graph, config)
                    st.rerun()
                else:
                    st.error("Not enough readable text found in the uploaded PDFs.")
            except AuthenticationError:
                st.error("Authentication failed. Please check your OpenAI API key in the sidebar.")
            except Exception as e:
                st.error(f"Error generating question bank: {e}")

# ==========================================
# STAGE 2: OUTSET TEST FORMAT SELECTION
# ==========================================
elif payload and payload.get("type") == "config":
    st.title("Configure Your Assessment")
    st.success(f"Generated **{payload['total_available']}** dual-format questions from your PDFs.", icon="📚")

    with st.container(border=True):
        st.subheader("How would you like to answer?")
        mode_choice = st.segmented_control(
            "Select assessment format:",
            options=["Multiple Choice (MCQ)", "Written / Descriptive"],
            default="Multiple Choice (MCQ)"
        )

        if mode_choice == "Multiple Choice (MCQ)":
            st.info("4 options per question with instant answer verification and concept explanations.")
            quiz_mode = "mcq"
        else:
            st.info("Write free-form technical responses evaluated by an LLM-as-a-Judge against a rubric.")
            quiz_mode = "descriptive"

        num_q = st.slider(
            "Number of questions",
            min_value=1,
            max_value=payload["total_available"],
            value=min(5, payload["total_available"])
        )

        if st.button("Start Exam →", type="primary"):
            st.session_state.graph.invoke(
                Command(resume={"quiz_mode": quiz_mode, "num_questions": num_q}),
                config
            )
            st.session_state.interrupt_payload = get_latest_interrupt(st.session_state.graph, config)
            st.rerun()

# ==========================================
# STAGE 3: ACTIVE INTERACTIVE QUIZ LOOP
# ==========================================
elif payload and payload.get("type") == "question":
    q_num = payload["number"]
    total = payload["total"]

    col_prog, col_badge = st.columns([4, 1])
    with col_prog:
        st.progress((q_num - 1) / total, text=f"Question {q_num} of {total}")
    with col_badge:
        st.markdown(f"**Difficulty:** `{payload['difficulty']}`")

    with st.container(border=True):
        st.caption(f"TOPIC: {payload['topic'].upper()}")
        st.markdown(f"### {payload['question']}")
        st.divider()

        if payload["mode"] == "mcq":
            choice = st.radio(
                "Select the best answer:",
                options=payload["options"],
                index=None,
                key=f"mcq_{q_num}"
            )
            if st.button("Submit Answer", type="primary", disabled=choice is None):
                selected_letter = choice.split(".")[0].strip()
                st.session_state.graph.invoke(Command(resume=selected_letter), config)
                st.session_state.interrupt_payload = get_latest_interrupt(st.session_state.graph, config)
                st.rerun()
        else:
            written_ans = st.text_area(
                "Your Technical Response:",
                placeholder="Write a clear explanation covering key intuition, math, or trade-offs...",
                height=200,
                key=f"desc_{q_num}"
            )
            if st.button("Submit Response for Grading", type="primary", disabled=not written_ans.strip()):
                with st.spinner("Evaluating response against grading rubric..."):
                    st.session_state.graph.invoke(Command(resume=written_ans), config)
                    st.session_state.interrupt_payload = get_latest_interrupt(st.session_state.graph, config)
                st.rerun()

    prior_results = graph_state.get("results", [])
    if prior_results:
        last = prior_results[-1]
        with st.expander(f"Last Question Result ({last['topic']}): {last['score']}/10", expanded=True):
            st.write(last["feedback"])

# ==========================================
# STAGE 4: FINAL SCORECARD & DIAGNOSTICS
# ==========================================
elif graph_state.get("results"):
    results = graph_state["results"]
    total_score = sum(r["score"] for r in results)
    max_score = len(results) * 10
    pct = round((total_score / max_score) * 100)

    st.title("Assessment Report")

    m1, m2, m3 = st.columns(3)
    m1.metric("Final Score", f"{total_score} / {max_score}")
    m2.metric("Accuracy", f"{pct}%")
    m3.metric("Questions Completed", len(results))

    st.divider()
    st.subheader("Detailed Question-by-Question Review")

    for idx, r in enumerate(results, 1):
        status_icon = "🟢" if r["score"] >= 8 else ("🟡" if r["score"] >= 5 else "🔴")
        with st.expander(f"{status_icon} Q{idx}: {r['topic']} — Score: {r['score']}/10", expanded=(r["score"] < 8)):
            st.markdown(f"**Question:** {r['question']}")
            st.markdown(f"**Your Answer:** {r['user_answer']}")
            st.info(f"**Examiner Feedback:** {r['feedback']}")
            if r["missing"]:
                st.warning("**Key Concepts Missed:** " + " • ".join(r["missing"]))
            st.markdown(f"**Reference Answer:** {r['correct_answer']}")