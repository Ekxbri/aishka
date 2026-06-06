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

print("Завантаження оптимізованого алгоритму...")
vectorizer = TfidfVectorizer()

# Пам'ять
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
            print(f"Відновлено {len(topics_text)} тем. Векторизую...")
            topics_vectors = vectorizer.fit_transform(topics_text)
            print("ШІ готовий до роботи!")

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
        raw_blocks = [block.strip() for block in text.split('\n\n') if block.strip()]
    
    if not raw_blocks:
        return {"message": "Файл порожній."}

    blocks = []
    current_topic = ""
    for text_part in raw_blocks:
        if len(text_part.split()) < 15:
            current_topic += text_part + "\n\n"
        else:
            current_topic += text_part
            blocks.append(current_topic)
            current_topic = ""
            
    if current_topic:
        if blocks: blocks[-1] += "\n\n" + current_topic
        else: blocks.append(current_topic)

    topics_text.extend(blocks)
    
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(topics_text, f, ensure_ascii=False, indent=4)
        
    if len(topics_text) > 0:
        topics_vectors = vectorizer.fit_transform(topics_text)
    
    return {"message": f"Успішно! Додано {len(blocks)} нових тем. Всього в базі: {len(topics_text)}."}

@app.post("/ask")
def ask_question(data: dict):
    try:
        question_text = data.get("question", "")
        
        # 1. Шукаємо в конспекті (через математичну подібність)
        if topics_vectors is not None and len(topics_text) > 0:
            query_vec = vectorizer.transform([question_text])
            similarities = cosine_similarity(query_vec, topics_vectors)[0]
            
            best_match_idx = similarities.argmax()
            score = similarities[best_match_idx]
            
            # Поріг збігу (10% ключових слів)
            if score > 0.1:
                return {"answer": topics_text[best_match_idx]}

        # 2. Якщо немає в конспекті - йдемо в інтернет
        results = DDGS().text(question_text, max_results=1)
        if results:
            web_answer = results[0]['body']
            return {"answer": f"🌐 Знайдено в інтернеті: {web_answer}"}
            
        return {"answer": "Я не знайшла відповіді ні в конспекті, ні в інтернеті."}
        
    except Exception as e:
        print(f"Критична помилка: {e}")
        return {"answer": f"Внутрішня помилка ШІ: {str(e)}"}
