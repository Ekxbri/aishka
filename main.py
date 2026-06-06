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
    current_title = ""
    current_body = []
    
    for text_part in raw_blocks:
        words = text_part.split()
        # Якщо рядок короткий (це заголовок або підзаголовок)
        if len(words) < 15:
            if current_body:
                # Зберігаємо попередню тему (бо знайшли новий заголовок після тексту)
                title_str = f"<b>📌 {current_title}</b>\n\n" if current_title else ""
                blocks.append(title_str + "\n\n".join(current_body))
                
                # Починаємо нову тему
                current_title = text_part
                current_body = []
            else:
                # Якщо тексту ще не було, склеюємо заголовки разом (наприклад "1. Тема..." і "1.1 Підтема...")
                if current_title:
                    current_title += " " + text_part
                else:
                    current_title = text_part
        else:
            # Це основний довгий текст
            current_body.append(text_part)
            
    # Зберігаємо останню тему в кінці файлу
    if current_title or current_body:
        title_str = f"<b>📌 {current_title}</b>\n\n" if current_title else ""
        if current_body:
            blocks.append(title_str + "\n\n".join(current_body))

    topics_text = blocks
    
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(topics_text, f, ensure_ascii=False, indent=4)
        
    if len(topics_text) > 0:
        topics_vectors = vectorizer.fit_transform(topics_text)
    
    return {"message": f"Успішно! Збережено {len(blocks)} правильно склеєних тем."}

@app.post("/ask")
def ask_question(data: dict):
    try:
        question_text = data.get("question", "").strip()
        if not question_text:
            return {"answer": "Порожній запит."}
        
        q_lower = question_text.lower()
        
        # 1. Пріоритет: точний пошук лише у ЗАГОЛОВКАХ конспекту
        for block in topics_text:
            first_line = block.split('\n')[0].lower()
            if q_lower in first_line:
                return {"answer": block}

        # 2. Розумний пошук по всьому тексту
        if topics_vectors is not None and len(topics_text) > 0:
            query_vec = vectorizer.transform([question_text])
            similarities = cosine_similarity(query_vec, topics_vectors)[0]
            
            best_match_idx = similarities.argmax()
            score = similarities[best_match_idx]
            
            if score > 0.02:
                return {"answer": topics_text[best_match_idx]}

        # 3. Інтернет
        try:
            results = DDGS().text(question_text, max_results=1)
            if results:
                web_answer = results[0]['body']
                return {"answer": f"🌐 <b>Знайдено в інтернеті:</b><br><br>{web_answer}"}
        except Exception:
            pass
            
        return {"answer": "Я не знайшла відповіді ні в конспекті, ні в інтернеті."}
        
    except Exception as e:
        return {"answer": f"Помилка ШІ: {str(e)}"}
