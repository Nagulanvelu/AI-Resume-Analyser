import io
import re
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="AI Resume Analyser", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Skill definitions ───────────────────────────────────────────────────────

SKILL_CATEGORIES = {
    "Programming Languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
        "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl", "dart"
    ],
    "Frontend": [
        "react", "vue", "angular", "svelte", "html", "css", "sass", "tailwind",
        "bootstrap", "next.js", "nuxt", "gatsby", "webpack", "vite", "redux",
        "jquery", "flutter", "react native"
    ],
    "Backend": [
        "fastapi", "django", "flask", "express", "node.js", "spring", "laravel",
        "rails", "asp.net", "nestjs", "fastify", "graphql", "rest api", "grpc"
    ],
    "Databases": [
        "postgresql", "mysql", "mongodb", "sqlite", "redis", "elasticsearch",
        "cassandra", "dynamodb", "firebase", "supabase", "sql", "nosql", "oracle"
    ],
    "DevOps & Cloud": [
        "docker", "kubernetes", "aws", "azure", "gcp", "ci/cd", "jenkins",
        "github actions", "terraform", "ansible", "linux", "nginx", "git"
    ],
    "AI / ML": [
        "machine learning", "deep learning", "tensorflow", "pytorch", "keras",
        "scikit-learn", "nlp", "computer vision", "langchain", "openai",
        "pandas", "numpy", "data science", "rag", "llm"
    ],
    "Soft Skills": [
        "leadership", "communication", "teamwork", "problem solving", "agile",
        "scrum", "project management", "mentoring", "collaboration", "presentation"
    ]
}

SENIORITY_SIGNALS = {
    "senior": ["senior", "lead", "principal", "staff", "architect", "head of", "director", "vp", "chief"],
    "mid": ["mid", "intermediate", "engineer ii", "developer ii", "3+ years", "4+ years", "5+ years"],
    "junior": ["junior", "entry", "associate", "intern", "graduate", "fresher", "1+ year", "0-2 years"]
}

SECTION_PATTERNS = {
    "experience": r"(experience|work history|employment|career|professional background)",
    "education": r"(education|academic|qualification|degree|university|college|school)",
    "skills": r"(skills|technologies|competencies|expertise|proficiencies|tech stack)",
    "projects": r"(projects|portfolio|work samples|contributions)",
    "certifications": r"(certification|certificate|license|credential)",
    "summary": r"(summary|objective|profile|about me|introduction)"
}

# ─── Text extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(content: bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        raise HTTPException(400, f"PDF parsing error: {e}")

def extract_text_from_docx(content: bytes) -> str:
    try:
        import docx
        doc = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        raise HTTPException(400, f"DOCX parsing error: {e}")

def extract_text(content: bytes, filename: str) -> str:
    fname = filename.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf(content)
    elif fname.endswith((".docx", ".doc")):
        return extract_text_from_docx(content)
    else:
        return content.decode("utf-8", errors="ignore")

# ─── Analysis engine ──────────────────────────────────────────────────────────

def detect_skills(text: str) -> dict:
    text_lower = text.lower()
    found = {}
    for category, skills in SKILL_CATEGORIES.items():
        matched = [s for s in skills if re.search(r'\b' + re.escape(s) + r'\b', text_lower)]
        if matched:
            found[category] = matched
    return found

def detect_seniority(text: str) -> str:
    text_lower = text.lower()
    for level, signals in SENIORITY_SIGNALS.items():
        for signal in signals:
            if signal in text_lower:
                return level
    # Estimate from years of experience mentions
    years = re.findall(r'(\d+)\+?\s*year', text_lower)
    if years:
        max_years = max(int(y) for y in years)
        if max_years >= 7: return "senior"
        if max_years >= 3: return "mid"
    return "junior"

def detect_sections(text: str) -> List[str]:
    found = []
    text_lower = text.lower()
    for section, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, text_lower):
            found.append(section)
    return found

def extract_contact_info(text: str) -> dict:
    info = {}
    # Email
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        info["email"] = email_match.group()
    # Phone
    phone_match = re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
    if phone_match:
        info["phone"] = phone_match.group().strip()
    # LinkedIn
    linkedin_match = re.search(r'linkedin\.com/in/[\w\-]+', text, re.IGNORECASE)
    if linkedin_match:
        info["linkedin"] = linkedin_match.group()
    # GitHub
    github_match = re.search(r'github\.com/[\w\-]+', text, re.IGNORECASE)
    if github_match:
        info["github"] = github_match.group()
    return info

def calculate_overall_score(skills: dict, sections: List[str], word_count: int, jd_match: Optional[dict]) -> int:
    score = 0
    # Skills breadth (max 35)
    total_skills = sum(len(v) for v in skills.values())
    score += min(35, total_skills * 2)
    # Sections completeness (max 30)
    key_sections = ["experience", "education", "skills", "summary"]
    present = sum(1 for s in key_sections if s in sections)
    score += (present / len(key_sections)) * 30
    # Resume length/detail (max 15)
    if word_count > 400: score += 15
    elif word_count > 200: score += 8
    else: score += 3
    # JD match bonus (max 20)
    if jd_match:
        score += (jd_match["match_percentage"] / 100) * 20
    else:
        score += 10  # neutral bonus when no JD
    return min(100, max(10, int(score)))

def match_with_jd(resume_skills: dict, jd_text: str) -> dict:
    jd_lower = jd_text.lower()
    all_resume_skills = [s for skills in resume_skills.values() for s in skills]
    
    jd_skills = []
    for skills in SKILL_CATEGORIES.values():
        for skill in skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', jd_lower):
                jd_skills.append(skill)
    
    matched = [s for s in jd_skills if s in all_resume_skills]
    missing = [s for s in jd_skills if s not in all_resume_skills]
    
    pct = round(len(matched) / max(len(jd_skills), 1) * 100, 1)
    
    return {
        "jd_skills_required": jd_skills,
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percentage": pct
    }

def generate_recommendations(skills: dict, sections: List[str], seniority: str, jd_match: Optional[dict]) -> List[str]:
    recs = []
    
    if "summary" not in sections:
        recs.append("📝 Add a professional summary/objective at the top to quickly convey your value proposition.")
    if "projects" not in sections:
        recs.append("🛠️ Include a Projects section with links to demonstrate practical experience.")
    if "certifications" not in sections:
        recs.append("🏆 Add relevant certifications (AWS, Google Cloud, etc.) to strengthen credibility.")
    
    if "AI / ML" not in skills and seniority in ["mid", "senior"]:
        recs.append("🤖 Consider adding AI/ML skills or tools to stay current with industry trends.")
    if "DevOps & Cloud" not in skills:
        recs.append("☁️ Cloud/DevOps skills (Docker, AWS, CI/CD) are highly valued — consider upskilling.")
    
    if jd_match and jd_match["missing_skills"]:
        top_missing = jd_match["missing_skills"][:4]
        recs.append(f"🎯 Close skill gaps for this role: {', '.join(s.title() for s in top_missing)}")
    
    total_skills = sum(len(v) for v in skills.values())
    if total_skills < 5:
        recs.append("📈 Your skills section is sparse — list all relevant technologies you've used.")
    
    if not recs:
        recs.append("✅ Great resume! Focus on quantifying achievements with metrics (%, $, time saved).")
        recs.append("🔗 Ensure all project links and GitHub are up-to-date and accessible.")
    
    return recs[:6]

def generate_strengths(skills: dict, sections: List[str], contact: dict) -> List[str]:
    strengths = []
    if len(skills) >= 4:
        strengths.append(f"Strong multi-domain skill set across {len(skills)} technical areas")
    if "AI / ML" in skills:
        strengths.append("AI/ML expertise — highly in-demand skillset")
    if "DevOps & Cloud" in skills:
        strengths.append("Cloud & DevOps proficiency (scalable, modern stack)")
    if len(sections) >= 4:
        strengths.append("Well-structured resume with all key sections present")
    if "github" in contact:
        strengths.append("GitHub profile linked — demonstrates active open source presence")
    if "linkedin" in contact:
        strengths.append("LinkedIn profile included — professional network visibility")
    total = sum(len(v) for v in skills.values())
    if total >= 10:
        strengths.append(f"Broad technical arsenal with {total}+ specific technologies listed")
    return strengths[:5] or ["Resume submitted — provide a Job Description for targeted strengths analysis"]

# ─── API Routes ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

@app.post("/analyse")
async def analyse_resume(
    file: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    job_description: Optional[str] = Form(None)
):
    # Extract text
    if file and file.filename:
        content = await file.read()
        text = extract_text(content, file.filename)
    elif resume_text and resume_text.strip():
        text = resume_text.strip()
    else:
        raise HTTPException(400, "Please upload a file or paste your resume text.")

    if len(text.strip()) < 50:
        raise HTTPException(400, "Resume text is too short. Please provide more content.")

    words = text.split()
    word_count = len(words)

    # Run analysis
    skills = detect_skills(text)
    sections = detect_sections(text)
    seniority = detect_seniority(text)
    contact = extract_contact_info(text)
    
    jd_match = None
    if job_description and job_description.strip():
        jd_match = match_with_jd(skills, job_description)

    overall_score = calculate_overall_score(skills, sections, word_count, jd_match)
    recommendations = generate_recommendations(skills, sections, seniority, jd_match)
    strengths = generate_strengths(skills, sections, contact)

    # Score label
    if overall_score >= 80: grade = "Excellent"; grade_color = "#10B981"
    elif overall_score >= 65: grade = "Good"; grade_color = "#6366F1"
    elif overall_score >= 45: grade = "Average"; grade_color = "#F59E0B"
    else: grade = "Needs Work"; grade_color = "#EF4444"

    return JSONResponse({
        "overall_score": overall_score,
        "grade": grade,
        "grade_color": grade_color,
        "seniority_level": seniority,
        "word_count": word_count,
        "skills_found": skills,
        "total_skills": sum(len(v) for v in skills.values()),
        "sections_detected": sections,
        "contact_info": contact,
        "jd_match": jd_match,
        "strengths": strengths,
        "recommendations": recommendations
    })

# Mount static files for frontend
import os
os.makedirs("frontend", exist_ok=True)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
