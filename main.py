from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from duckduckgo_search import DDGS
import docx
import io
import os
import re
import pymongo

app = FastAPI(title="Aishka Pro")

TEACHER_PASSWORD = "teacher" # Пароль для завантаження конспектів

# --- ПІДКЛЮЧЕННЯ ДО ХМАРНОЇ БАЗИ MONGODB ---
# Встав свій пароль замість <db_password> (без дужок < >)
MONGO_URI = "mongodb+srv://rgbdf969_db_user:<1234qwer>@cluster0.wrhsfpw.mongodb.net/?appName=Cluster0"

client = pymongo.MongoClient(MONGO_URI)
db = client["aishka_database"]
collection = db["topics"]

db_data = []
corpus_texts = []
corpus_meta = []
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
topics_vectors = None

def rebuild_vectors():
    global corpus_texts, corpus_meta, topics_vectors
    corpus_texts = []
    corpus_meta = []
    
    for item_idx, item in enumerate(db_data):
        subject = item.get("subject", "Загальне")
        title = item.get("title", "Без назви")
        
        corpus_texts.append(f"ПРЕДМЕТ: {subject}. ТЕМА: {title}")
        corpus_meta.append({"type": "title", "item_idx": item_idx})
        
        for p_idx, para in enumerate(item.get("paragraphs", [])):
            corpus_texts.append(f"{subject}. {title}. {para}")
            corpus_meta.append({"type": "paragraph", "item_idx": item_idx, "p_idx": p_idx})
            
    if corpus_texts:
        topics_vectors = vectorizer.fit_transform(corpus_texts)

@app.on_event("startup")
async def load_database():
    global db_data
    # При запуску сервера вантажимо все з MongoDB
    db_data = list(collection.find({}, {"_id": 0}))
    if db_data:
        rebuild_vectors()
    print(f"Завантажено {len(db_data)} тем з хмари MongoDB.")

@app.get("/")
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return {"error": "Файл index.html не знайдено"}

@app.post("/upload")
async def upload_notes(
    file: UploadFile = File(...), 
    subject: str = Form(...), 
    password: str = Form(...)
):
    global db_data
    if password != TEACHER_PASSWORD:
        return {"message": "❌ Невірний пароль вчителя!"}
        
    content = await file.read()
    if file.filename.endswith('.docx'):
        doc = docx.Document(io.BytesIO(content))
        raw_blocks = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    else:
        text = content.decode("utf-8")
        raw_blocks = [block.strip() for block in text.split('\n') if block.strip()]
    
    if not raw_blocks:
        return {"message": "Файл порожній."}

    current_title = "Загальна інформація"
    current_body = []
    blocks_to_add = []
    
    for text_part in raw_blocks:
        text_part = text_part.strip()
        if not text_part: continue
            
        words = text_part.split()
        ends_with_punct = text_part.endswith(('.', ',', ';', ':'))
        is_list = text_part.startswith(('-', '•', '*', '–', '+')) or re.match(r"^\d+[\)\\]", text_part)
        
        is_header = False
        if len(words) < 12 and not is_list:
            if not text_part[0].islower():
                if not ends_with_punct or text_part.isupper() or len(words) <= 5:
                    is_header = True
                    
        if is_header:
            if current_body:
                blocks_to_add.append({"subject": subject.strip(), "title": current_title, "paragraphs": current_body})
                current_body = []
            current_title = text_part
        else:
            current_body.append(text_part)

    if current_body or current_title != "Загальна інформація":
        blocks_to_add.append({"subject": subject.strip(), "title": current_title, "paragraphs": current_body})

    # Відправляємо в MongoDB (уникаючи дублікатів)
    added_count = 0
    for new_topic in blocks_to_add:
        exists = collection.find_one({"subject": new_topic["subject"], "title": new_topic["title"]})
        if not exists:
            collection.insert_one(new_topic)
            new_topic.pop("_id", None) # Видаляємо службовий ID від Монго перед додаванням в оперативку
            db_data.append(new_topic)
            added_count += 1

    rebuild_vectors()
    return {"message": f"✅ Успішно! В хмарну базу додано {added_count} нових тем."}

@app.post("/ask")
def ask_question(data: dict):
    try:
        q = data.get("question", "").strip()
        if not q: return {"answer": "Порожній запит."}
        
        q_lower = q.lower()
        
        summary_match = re.search(r'(головне|основне|суть|коротко|тези)\s*(по|про|в)?\s*тем[іі]\s*(.+)', q_lower)
        if summary_match and topics_vectors is not None:
            target_topic = summary_match.group(3).strip()
            
            query_vec = vectorizer.transform([target_topic])
            sims = cosine_similarity(query_vec, topics_vectors)[0]
            
            best_score = 0
            best_item = None
            
            for idx, score in enumerate(sims):
                meta = corpus_meta[idx]
                if meta["type"] == "title" and score > best_score:
                    best_score = score
                    best_item = db_data[meta["item_idx"]]
                    
            if best_score > 0.1 and best_item:
                subj = best_item["subject"]
                title = best_item["title"]
                paras = best_item["paragraphs"]
                
                summary = []
                for p in paras:
                    if not p.startswith(('-', '•')):
                        first_sentence = p.split('.')[0] + "."
                        summary.append(first_sentence)
                    else:
                        summary.append(p)
                
                res_text = f"📚 <b>Предмет:</b> {subj}\n📌 <b>Тема:</b> {title}\n\n<b>Головні тези:</b>\n\n"
                res_text += "\n\n".join(summary[:5])
                if len(summary) > 5:
                    res_text += "\n\n<i>...тема містить більше інформації, запитай детальніше, якщо потрібно.</i>"
                return {"answer": res_text}

        stop_words = r'(?i)\b(коротко|стисло|детально|розгорнуто|все про|що|таке|це|які|є|як|чому|навіщо|предмет|тема)\b'
        search_query = re.sub(stop_words, '', q).strip()
        if not search_query: search_query = q
        
        if topics_vectors is not None and len(corpus_texts) > 0:
            query_vec = vectorizer.transform([search_query])
            sims = cosine_similarity(query_vec, topics_vectors)[0]
            
            for i, text in enumerate(corpus_texts):
                if search_query.lower() in text.lower():
                    sims[i] += 2.0
                    
            best_idx = sims.argmax()
            best_score = sims[best_idx]
            
            if best_score > 0.05:
                meta = corpus_meta[best_idx]
                item = db_data[meta["item_idx"]]
                subj = item["subject"]
                title = item["title"]
                
                header = f"📚 <b>{subj}</b>\n📌 <b>{title}</b>\n\n"
                
                if meta["type"] == "title":
                    preview = "\n\n".join(item["paragraphs"][:2])
                    return {"answer": header + preview + ("\n\n<i>...запитай детальніше, щоб читати далі.</i>" if len(item["paragraphs"]) > 2 else "")}
                else:
                    p_idx = meta["p_idx"]
                    ans = item["paragraphs"][p_idx]
                    if p_idx + 1 < len(item["paragraphs"]):
                        ans += "\n\n" + item["paragraphs"][p_idx + 1]
                    return {"answer": header + ans}

        try:
            results = DDGS().text(search_query, max_results=1)
            if results:
                web_answer = results[0]['body']
                return {"answer": f"🌐 <b>Знайдено в інтернеті:</b><br><br>{web_answer}"}
        except Exception:
            pass
            
        return {"answer": "На жаль, я не знайшла цього у базі конспектів та інтернеті."}
        
    except Exception as e:
        return {"answer": f"Помилка ШІ: {str(e)}"}
