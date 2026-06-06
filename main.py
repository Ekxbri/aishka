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
    current_topic = ""
    
    for text_part in raw_blocks:
        words = text_part.split()
        # Якщо рядок короткий (менше 15 слів) — це заголовок нової теми
        if len(words) < 15:
            if current_topic:
                blocks.append(current_topic.strip())
            current_topic = f"<b>📌 {text_part}</b>\n\n"
        else:
            # Усі наступні абзаци приклеюємо ДО ЦІЄЇ Ж ТЕМИ, а не створюємо нові блоки
            if not current_topic:
                current_topic = "<b>📌 Загальна інформація</b>\n\n"
            current_topic += text_part + "\n\n"
            
    if current_topic:
        blocks.append(current_topic.strip())

    # Перезаписуємо базу правильними великими темами
    topics_text = blocks
    
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(topics_text, f, ensure_ascii=False, indent=4)
        
    if len(topics_text) > 0:
        topics_vectors = vectorizer.fit_transform(topics_text)
    
    return {"message": f"Успішно! Збережено {len(blocks)} згрупованих тем. Пошук тепер буде точним."}

@app.post("/ask")
def ask_question(data: dict):
    try:
        question_text = data.get("question", "").strip()
        if not question_text:
            return {"answer": "Порожній запит."}
        
        q_lower = question_text.lower()
        is_short = "коротк" in q_lower or "стисл" in q_lower
        is_long = "детальн" in q_lower or "розгорнут" in q_lower or "все про" in q_lower
        
        if topics_vectors is not None and len(topics_text) > 0:
            query_vec = vectorizer.transform([question_text])
            similarities = cosine_similarity(query_vec, topics_vectors)[0]
            
            best_match_idx = similarities.argmax()
            score = similarities[best_match_idx]
            
            if score > 0.02:
                full_topic = topics_text[best_match_idx]
                
                # Розбиваємо знайдену тему на заголовок та окремі абзаци
                parts = full_topic.split('\n\n')
                title = parts[0]
                paragraphs = [p.strip() for p in parts[1:] if p.strip()]
                
                if not paragraphs:
                    return {"answer": full_topic}
                
                # 1. Режим: КОРОТКО
                if is_short:
                    first_sentence = paragraphs[0].split('.')[0] + "."
                    return {"answer": f"{title}\n\n{first_sentence}"}
                
                # 2. Режим: ДЕТАЛЬНО
                elif is_long:
                    return {"answer": full_topic}
                
                # 3. Режим: СТАНДАРТНО (Видаємо лише перші 2 абзаци, щоб не перевантажувати)
                else:
                    preview = "\n\n".join(paragraphs[:2])
                    if len(paragraphs) > 2:
                        preview += "\n\n<i>...у цьому розділі є ще інформація. Додай слово 'детально' до запиту, щоб побачити весь текст.</i>"
                    return {"answer": f"{title}\n\n{preview}"}

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

@app.post("/ask")
def ask_question(data: dict):
    try:
        question_text = data.get("question", "").strip()
        if not question_text:
            return {"answer": "Порожній запит."}
        
        # Перевіряємо, яку відповідь хоче користувач
        q_lower = question_text.lower()
        is_short = "коротк" in q_lower or "стисл" in q_lower
        is_long = "детальн" in q_lower or "розгорнут" in q_lower or "все про" in q_lower
        
        if topics_vectors is not None and len(topics_text) > 0:
            query_vec = vectorizer.transform([question_text])
            similarities = cosine_similarity(query_vec, topics_vectors)[0]
            
            best_match_idx = similarities.argmax()
            score = similarities[best_match_idx]
            
            if score > 0.05:
                base_chunk = topics_text[best_match_idx]
                
                # 1. Якщо просять КОРОТКО (видаємо лише перше речення)
                if is_short:
                    parts = base_chunk.split('\n\n', 1)
                    if len(parts) > 1:
                        first_sentence = parts[1].split('.')[0] + "."
                        return {"answer": f"{parts[0]}\n\n{first_sentence}"}
                    return {"answer": base_chunk}
                    
                # 2. Якщо просять ДЕТАЛЬНО (додаємо ще 2 абзаци з цієї ж теми)
                elif is_long:
                    answer = base_chunk
                    parts = base_chunk.split('\n\n', 1)
                    title = parts[0] if len(parts) > 1 else ""
                    
                    for i in range(best_match_idx + 1, min(len(topics_text), best_match_idx + 3)):
                        next_chunk = topics_text[i]
                        if title and next_chunk.startswith(title):
                            next_body = next_chunk.split('\n\n', 1)[1]
                            answer += f"\n\n{next_body}"
                    return {"answer": answer}
                    
                # 3. СТАНДАРТНО (один абзац)
                return {"answer": base_chunk}

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
