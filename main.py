from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from duckduckgo_search import DDGS
import docx
import io
import json
import os

app = FastAPI(title="Aishka API")
security = HTTPBasic()

vectorizer = TfidfVectorizer()
topics_text = []
topics_vectors = None
DB_FILE = "database.json"

@app.on_event("startup")
async def load_database():
    global topics_text, topics_vectors
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            topics_text = json.load(f)
        if topics_text:
            topics_vectors = vectorizer.fit_transform(topics_text)

def check_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "admin" or credentials.password != "1234":
        raise HTTPException(status_code=401, detail="Помилка доступу", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

@app.get("/")
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return {"error": "Файл index.html не знайдено"}

@app.post("/ask")
def ask_question(data: dict):
    try:
        question_text = data.get("question", "").strip()
        if not question_text:
            return {"answer": "Порожній запит."}
        
        # 1. Бронебійний пошук (якщо слова із запиту точно є в тексті конспекту)
        q_lower = question_text.lower()
        for block in topics_text:
            if q_lower in block.lower():
                return {"answer": block}

        # 2. Розумний пошук (якщо слова трохи змінені)
        if topics_vectors is not None and len(topics_text) > 0:
            query_vec = vectorizer.transform([question_text])
            similarities = cosine_similarity(query_vec, topics_vectors)[0]
            
            best_match_idx = similarities.argmax()
            score = similarities[best_match_idx]
            
            # Знизили поріг з 0.1 до 0.02
            if score > 0.02:
                return {"answer": topics_text[best_match_idx]}

        # 3. Інтернет
        try:
            results = DDGS().text(question_text, max_results=1)
            if results:
                web_answer = results[0]['body']
                return {"answer": f"🌐 <b>Знайдено в інтернеті:</b>\n<br>{web_answer}"}
        except Exception:
            pass
            
        return {"answer": "Я не знайшла відповіді ні в конспекті, ні в інтернеті."}
        
    except Exception as e:
        return {"answer": f"Помилка ШІ: {str(e)}"}

@app.post("/ask")
def ask_question(data: dict):
    try:
        question_text = data.get("question", "")
        
        if topics_vectors is not None and len(topics_text) > 0:
            query_vec = vectorizer.transform([question_text])
            similarities = cosine_similarity(query_vec, topics_vectors)[0]
            
            best_match_idx = similarities.argmax()
            score = similarities[best_match_idx]
            
            if score > 0.1:
                return {"answer": topics_text[best_match_idx]}

        results = DDGS().text(question_text, max_results=1)
        if results:
            web_answer = results[0]['body']
            return {"answer": f"🌐 <b>Знайдено в інтернеті:</b>\n{web_answer}"}
            
        return {"answer": "Я не знайшла відповіді ні в конспекті, ні в інтернеті."}
        
    except Exception as e:
        return {"answer": f"Помилка: {str(e)}"}
