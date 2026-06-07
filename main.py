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
import re

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
    current_title = "Загальна інформація"
    current_body = []
    
    for text_part in raw_blocks:
        text_part = text_part.strip()
        if not text_part:
            continue
            
        words = text_part.split()
        ends_with_punct = text_part.endswith(('.', ',', ';', ':'))
        is_list = text_part.startswith(('-', '•', '*', '–', '+')) or re.match(r"^\d+[\)\\]", text_part)
        
        is_header = False
        if len(words) < 12 and not is_list:
            if not text_part[0].islower():
                if not ends_with_punct:
                    is_header = True
                elif text_part.isupper():
                    is_header = True
                elif len(words) <= 5:
                    is_header = True
                    
        if is_header:
            if current_body:
                blocks.append(f"<b>📌 {current_title}</b>\n\n" + "\n\n".join(current_body))
                current_body = []
            current_title = text_part
        else:
            current_body.append(text_part)
            if sum(len(p.split()) for p in current_body) > 250:
                blocks.append(f"<b>📌 {current_title}</b>\n\n" + "\n\n".join(current_body))
                current_body = []

    if current_body or (not blocks and current_title):
        blocks.append(f"<b>📌 {current_title}</b>\n\n" + "\n\n".join(current_body))

    topics_text = blocks
    
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(topics_text, f, ensure_ascii=False, indent=4)
        
    if len(topics_text) > 0:
        topics_vectors = vectorizer.fit_transform(topics_text)
    
    return {"message": f"Успішно! Збережено {len(blocks)} тем. Працює дворівневий пошук."}

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
        
        search_query = re.sub(r'(?i)(коротко|стисло|детально|розгорнуто|все про)', '', question_text).strip()
        if not search_query:
            search_query = question_text
            
        if len(search_query.split()) <= 3 and last_question:
            search_query = f"{last_question} {search_query}"
        
        if not is_short and not is_long and len(question_text.split()) > 2:
            last_question = question_text

        best_match_idx = -1
        best_score = 0
        
        if topics_vectors is not None and len(topics_text) > 0:
            query_vec = vectorizer.transform([search_query])
            similarities = cosine_similarity(query_vec, topics_vectors)[0]
            
            for i, block in enumerate(topics_text):
                if search_query.lower() in block.lower():
                    similarities[i] += 0.5 
                    
            best_match_idx = similarities.argmax()
            best_score = similarities[best_match_idx]

        if best_score > 0.05 and best_match_idx != -1:
            full_topic = topics_text[best_match_idx]
            parts = full_topic.split('\n\n')
            title = parts[0]
            paragraphs = [p.strip() for p in parts[1:] if p.strip()]
            
            if not paragraphs:
                return {"answer": full_topic}
            
            # === ДРУГИЙ ЕТАП: ШУКАЄМО ТОЧНИЙ АБЗАЦ ===
            best_p_idx = 0
            if len(paragraphs) > 1:
                p_vecs = vectorizer.transform(paragraphs)
                q_vec = vectorizer.transform([search_query])
                p_sims = cosine_similarity(q_vec, p_vecs)[0]
                
                # Бонус за точний збіг слова в абзаці
                for i, p in enumerate(paragraphs):
                    if search_query.lower() in p.lower():
                        p_sims[i] += 2.0 
                        
                best_p_idx = p_sims.argmax()

            # Формуємо відповідь ПОЧИНАЮЧИ зі знайденого абзацу
            if is_short:
                target_p = paragraphs[best_p_idx]
                first_sentence = target_p.split('.')[0] + "." if '.' in target_p else target_p
                return {"answer": f"{title}\n\n{first_sentence}"}
            elif is_long:
                # Видаємо потрібний абзац + 2 наступні (а не весь розділ з початку)
                end_idx = min(len(paragraphs), best_p_idx + 3)
                detailed_text = "\n\n".join(paragraphs[best_p_idx:end_idx])
                if best_p_idx > 0:
                    detailed_text = f"<i>(Фрагмент із середини розділу)</i>\n\n{detailed_text}"
                return {"answer": f"{title}\n\n{detailed_text}"}
            else:
                # Стандарт: 1 знайдений абзац + 1 наступний
                end_idx = min(len(paragraphs), best_p_idx + 2)
                preview = "\n\n".join(paragraphs[best_p_idx:end_idx])
                if len(paragraphs) > end_idx:
                    preview += "\n\n<i>...додай слово 'детально', щоб читати далі.</i>"
                if best_p_idx > 0:
                    preview = f"<i>(Фрагмент із середини розділу)</i>\n\n{preview}"
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
