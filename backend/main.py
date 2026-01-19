from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import traceback

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client with OpenRouter
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("ERROR: OPENROUTER_API_KEY environment variable not set!")
else:
    print("✅ OPENROUTER_API_KEY found")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

class CodeRequest(BaseModel):
    code: str
    language: str

@app.get("/")
async def root():
    return {"message": "Code Quality Assistant API is running"}

@app.post("/analyze")
async def analyze_code(request: CodeRequest):
    try:
        print(f"\n=== Analyzing {request.language} code ===")
        print(f"Code length: {len(request.code)} characters")
        
        # Agent 1: Code Analyzer
        print("🔍 Starting CodeAnalyzer agent...")
        analyzer_response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free",  # ✅ UPDATED MODEL
            messages=[
                {
                    "role": "system",
                    "content": f"You are a code quality analyzer. Analyze the following {request.language} code and provide: 1) Quality score (0-100), 2) Issues found, 3) Best practices recommendations. Format your response clearly."
                },
                {
                    "role": "user",
                    "content": request.code
                }
            ],
        )
        code_analysis = analyzer_response.choices[0].message.content
        print(f"✅ CodeAnalyzer complete: {len(code_analysis)} chars")

        # Agent 2: Test Generator
        print("🧪 Starting TestGenerator agent...")
        test_response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free",  # ✅ UPDATED MODEL
            messages=[
                {
                    "role": "system",
                    "content": f"You are a test case generator. Generate comprehensive unit tests for the following {request.language} code. Include edge cases and expected outputs."
                },
                {
                    "role": "user",
                    "content": request.code
                }
            ],
        )
        test_cases = test_response.choices[0].message.content
        print(f"✅ TestGenerator complete: {len(test_cases)} chars")

        # Agent 3: Documentation Writer
        print("📝 Starting DocumentationWriter agent...")
        doc_response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free",  # ✅ UPDATED MODEL
            messages=[
                {
                    "role": "system",
                    "content": f"You are a technical documentation writer. Write clear, comprehensive documentation for the following {request.language} code. Include purpose, parameters, return values, and usage examples."
                },
                {
                    "role": "user",
                    "content": request.code
                }
            ],
        )
        documentation = doc_response.choices[0].message.content
        print(f"✅ DocumentationWriter complete: {len(documentation)} chars")
        
        print("🎉 All agents completed successfully!\n")

        return {
            "code_analysis": code_analysis,
            "test_cases": test_cases,
            "documentation": documentation
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"\n❌ ERROR in /analyze endpoint:")
        print(f"Error message: {error_msg}")
        print(f"Full traceback:\n{error_trace}\n")
        
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {error_msg}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
