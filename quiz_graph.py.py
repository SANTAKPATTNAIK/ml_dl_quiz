from typing import TypedDict, List, Literal, Annotated
import operator
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

class DescriptiveGrade(BaseModel):
    score: int = Field(description="Score from 0 to 10")
    feedback: str = Field(description="Detailed evaluation of technical accuracy")
    missing_points: List[str] = Field(description="Rubric points missed by the candidate")

class QuizState(TypedDict):
    question_bank: List[dict]
    quiz_mode: Literal["mcq", "descriptive"]
    num_questions: int
    current_index: int
    user_answer: str
    results: Annotated[List[dict], operator.add]

def build_quiz_graph(api_key: str):
    # Scope the LLM client to this user's session API key
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    descriptive_grader = llm.with_structured_output(DescriptiveGrade)

    def configure_quiz(state: QuizState):
        prefs = interrupt({
            "type": "config",
            "message": "Choose your assessment format before starting.",
            "total_available": len(state["question_bank"])
        })
        return {
            "quiz_mode": prefs["quiz_mode"],
            "num_questions": min(prefs["num_questions"], len(state["question_bank"])),
            "current_index": 0
        }

    def ask_question(state: QuizState):
        q = state["question_bank"][state["current_index"]]
        mode = state["quiz_mode"]

        if mode == "mcq":
            payload = {
                "type": "question",
                "mode": "mcq",
                "number": state["current_index"] + 1,
                "total": state["num_questions"],
                "topic": q["topic"],
                "difficulty": q["difficulty"],
                "question": q["mcq_question"],
                "options": q["mcq_options"]
            }
        else:
            payload = {
                "type": "question",
                "mode": "descriptive",
                "number": state["current_index"] + 1,
                "total": state["num_questions"],
                "topic": q["topic"],
                "difficulty": q["difficulty"],
                "question": q["descriptive_question"]
            }

        answer = interrupt(payload)
        return {"user_answer": answer}

    def evaluate_answer(state: QuizState):
        q = state["question_bank"][state["current_index"]]
        mode = state["quiz_mode"]

        if mode == "mcq":
            selected = state["user_answer"].strip().upper()[:1]
            is_correct = selected == q["mcq_correct_option"]
            record = {
                "topic": q["topic"],
                "question": q["mcq_question"],
                "user_answer": selected,
                "correct_answer": q["mcq_correct_option"],
                "score": 10 if is_correct else 0,
                "feedback": q["mcq_explanation"],
                "missing": [] if is_correct else [f"Correct option was {q['mcq_correct_option']}"]
            }
        else:
            prompt = f"""
            Evaluate the candidate's ML/DL response against the reference answer and rubric.
            Question: {q['descriptive_question']}
            Reference Answer: {q['reference_answer']}
            Grading Rubric: {q['grading_rubric']}
            Candidate Response: {state['user_answer']}
            """
            grade: DescriptiveGrade = descriptive_grader.invoke(prompt)
            record = {
                "topic": q["topic"],
                "question": q["descriptive_question"],
                "user_answer": state["user_answer"],
                "correct_answer": q["reference_answer"],
                "score": grade.score,
                "feedback": grade.feedback,
                "missing": grade.missing_points
            }

        return {
            "results": [record],
            "current_index": state["current_index"] + 1
        }

    def should_continue(state: QuizState):
        if state["current_index"] < state["num_questions"]:
            return "ask_question"
        return END

    builder = StateGraph(QuizState)
    builder.add_node("configure_quiz", configure_quiz)
    builder.add_node("ask_question", ask_question)
    builder.add_node("evaluate_answer", evaluate_answer)

    builder.add_edge(START, "configure_quiz")
    builder.add_edge("configure_quiz", "ask_question")
    builder.add_edge("ask_question", "evaluate_answer")
    builder.add_conditional_edges("evaluate_answer", should_continue)

    return builder.compile(checkpointer=MemorySaver())