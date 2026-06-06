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

@app.post("/upload")
async def upload_notes(file: UploadFile = File(...), admin: str = Depends(check_admin)):
    global topics_text, topics_vectors
    content = await file.read()
    
    if file.filename.endswith('.docx'):
        doc = docx.Document(io.BytesIO(content))
        raw_blocks = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    else:
        text = content.decode("utf-8")
        raw_blocks = [block.strip() for block in text.split('\n') if block.strip()]
    
    if not raw_blocks:
        return {"message": "Файл порожній."}

    blocks = []
    current_block = ""
    
    for text_part in raw_blocks:
        words = text_part.split()
        # Якщо рядок короткий (менше 12 слів), робимо його новим заголовком
        if len(words) < 12:
            if current_block:
                blocks.append(current_block.strip())
            current_block = f"<b>📌 {text_part}</b>\n\n"
        else:
            # Всі наступні абзаци приклеюємо до поточного заголовку
            if not current_block:
                current_block = text_part + "\n"
            else:
                current_block += text_part + "\n"
            
    if current_block:
        blocks.append(current_block.strip())

    # Очищаємо стару базу при новому завантаженні, щоб не було дублів старих обрізаних текстів
    topics_text = blocks
    
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(topics_text, f, ensure_ascii=False, indent=4)
        
    if len(topics_text) > 0:
        topics_vectors = vectorizer.fit_transform(topics_text)
    
    return {"message": f"Успішно! Збережено {len(blocks)} повноцінних тем."}

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
