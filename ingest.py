import fitz  # PyMuPDF
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class QuestionItem(BaseModel):
    topic: str = Field(description="Specific ML/DL concept, e.g., Batch Normalization, Adam Optimizer")
    difficulty: Literal["Easy", "Medium", "Hard"]
    mcq_question: str = Field(description="Multiple-choice question stem")
    mcq_options: List[str] = Field(description="Exactly 4 options prefixed with A., B., C., D.")
    mcq_correct_option: Literal["A", "B", "C", "D"]
    mcq_explanation: str = Field(description="Why the correct option is right and others are wrong")
    descriptive_question: str = Field(description="Open-ended conceptual or derivation question")
    reference_answer: str = Field(description="Comprehensive model answer")
    grading_rubric: List[str] = Field(description="3-4 key technical points required for full credit")

class QuestionBankBatch(BaseModel):
    questions: List[QuestionItem]

def extract_and_generate_bank(pdf_bytes: bytes, api_key: str, max_chunks: int = 8) -> List[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = "\n".join(page.get_text("text") for page in doc)
    chunks = [full_text[i:i+3000] for i in range(0, len(full_text), 3000)][:max_chunks]

    # Use the user-supplied API key directly (thread-safe across Streamlit sessions)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)
    generator = llm.with_structured_output(QuestionBankBatch)

    question_bank = []
    for chunk in chunks:
        if len(chunk.strip()) < 200:
            continue
        prompt = f"""
        Extract 2 high-value Machine Learning / Deep Learning interview or exam questions from the textbook excerpt below.
        For each concept, create BOTH a rigorous MCQ and a deep Descriptive question with a grading rubric.
        
        Excerpt:
        {chunk}
        """
        batch = generator.invoke(prompt)
        for q in batch.questions:
            question_bank.append(q.model_dump())
            
    return question_bank