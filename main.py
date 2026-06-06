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

last_question = ""

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
        # Нове правило: ігноруємо рядки з двокрапкою та списки
        is_list_item = text_part.lstrip().startswith(('-', '•', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
        
        if len(words) < 15 and not text_part.endswith(':') and not is_list_item:
            if current_body:
                title_str = f"<b>📌 {current_title}</b>\n\n" if current_title else ""
                blocks.append(title_str + "\n\n".join(current_body))
                current_title = text_part
                current_body = []
            else:
                if current_title:
                    current_title += " " + text_part
                else:
                    current_title = text_part
        else:
            current_body.append(text_part)
            
    if current_title or current_body:
        title_str = f"<b>📌 {current_title}</b>\n\n" if current_title else ""
        if current_body:
            blocks.append(title_str + "\n\n".join(current_body))

    topics_text = blocks
    
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(topics_text, f, ensure_ascii=False, indent=4)
        
    if len(topics_text) > 0:
        topics_vectors = vectorizer.fit_transform(topics_text)
    
    return {"message": f"Успішно! Збережено {len(blocks)} тем. Списки обробляються коректно."}

@app.post("/ask")
def ask_question(data: dict):
    global last_question
    try:
        question_text = data.get("question", "").strip()
        if not question_text:
            return {"answer": "Порожній запит."}
        
        q_lower = question_text.lower()
        is_short = "коротк" in q_lower or "стисл" in q_lower
        is_long = "детальн" in q_lower or "розгорнут" in q_lower or "все про" in q_lower
        
        search_query = question_text
        if len(question_text.split()) <= 3 and last_question:
            search_query = f"{last_question} {question_text}"
        
        if not is_short and not is_long and len(question_text.split()) > 2:
            last_question = question_text
        
        for block in topics_text:
            first_line = block.split('\n')[0].lower()
            if q_lower in first_line:
                return {"answer": block}

        if topics_vectors is not None and len(topics_text) > 0:
            query_vec = vectorizer.transform([search_query])
            similarities = cosine_similarity(query_vec, topics_vectors)[0]
            
            best_match_idx = similarities.argmax()
            score = similarities[best_match_idx]
            
            if score > 0.02:
                full_topic = topics_text[best_match_idx]
                parts = full_topic.split('\n\n')
                title = parts[0]
                paragraphs = [p.strip() for p in parts[1:] if p.strip()]
                
                if not paragraphs:
                    return {"answer": full_topic}
                
                if is_short:
                    first_sentence = paragraphs[0].split('.')[0] + "."
                    return {"answer": f"{title}\n\n{first_sentence}"}
                elif is_long:
                    return {"answer": full_topic}
                else:
                    preview = "\n\n".join(paragraphs[:2])
                    if len(paragraphs) > 2:
                        preview += "\n\n<i>...у цьому розділі є ще інформація. Додай слово 'детально' до запиту, щоб побачити весь текст.</i>"
                    return {"answer": f"{title}\n\n{preview}"}

        try:
            results = DDGS().text(search_query, max_results=1)
            if results:
                web_answer = results[0]['body']
                return {"answer": f"🌐 <b>Знайдено в інтернеті:</b><br><br>{web_answer}"}
        except Exception:
            pass
            
        return {"answer": "Я не знайшла відповіді ні в конспекті, ні в інтернеті."}
        
    except Exception as e:
        return {"answer": f"Помилка ШІ: {str(e)}"}
