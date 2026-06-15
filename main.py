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
import certifi

app = FastAPI(title="Aishka Pro")
TEACHER_PASSWORD = "teacher"
MONGO_URI = "mongodb+srv://rgbdf969_db_user:1234qwer@cluster0.wrhsfpw.mongodb.net/?appName=Cluster0"

client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["aishka_database"]
collection = db["topics"]

db_data = []
corpus_texts = []
corpus_meta = []
# Збільшено діапазон для кращого розпізнавання фраз із різних предметів
vectorizer = TfidfVectorizer(ngram_range=(1, 3)) 
topics_vectors = None

def rebuild_vectors():
    global corpus_texts, corpus_meta, topics_vectors
    corpus_texts = []
    corpus_meta = []
    
    for item_idx, item in enumerate(db_data):
        subject = item.get("subject", "Загальне")
        title = item.get("title", "Без назви")
        
        # Посилюємо вагу назви та предмета для точного пошуку
        corpus_texts.append(f"ПРЕДМЕТ {subject} ТЕМА {title} {title} {title}")
        corpus_meta.append({"type": "title", "item_idx": item_idx})
        
        for p_idx, para in enumerate(item.get("paragraphs", [])):
            corpus_texts.append(f"{subject} {title} {para}")
            corpus_meta.append({"type": "paragraph", "item_idx": item_idx, "p_idx": p_idx})
            
    if corpus_texts:
        topics_vectors = vectorizer.fit_transform(corpus_texts)

@app.on_event("startup")
async def load_database():
    global db_data
    db_data = list(collection.find({}, {"_id": 0}))
    if db_data: rebuild_vectors()

@app.get("/")
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return {"error": "Файл index.html не знайдено"}

@app.post("/upload")
async def upload_notes(file: UploadFile = File(...), subject: str = Form(...), password: str = Form(...)):
    global db_data
    if password != TEACHER_PASSWORD: return {"message": "❌ Невірний пароль!"}
        
    content = await file.read()
    if file.filename.endswith('.docx'):
        doc = docx.Document(io.BytesIO(content))
        raw_blocks = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    else:
        text = content.decode("utf-8")
        raw_blocks = [block.strip() for block in text.split('\n') if block.strip()]
    
    if not raw_blocks: return {"message": "Файл порожній."}

    current_title = "Загальна інформація"
    current_body = []
    blocks_to_add = []
    
    for text_part in raw_blocks:
        words = text_part.split()
        ends_with_punct = text_part.endswith(('.', ',', ';', ':'))
        is_list = text_part.startswith(('-', '•', '*', '–', '+')) or re.match(r"^\d+[\)\\]", text_part)
        
        is_header = False
        if len(words) < 12 and not is_list and not text_part[0].islower():
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

    added_count = 0
    for new_topic in blocks_to_add:
        exists = collection.find_one({"subject": new_topic["subject"], "title": new_topic["title"]})
        if not exists:
            collection.insert_one(new_topic)
            new_topic.pop("_id", None)
            db_data.append(new_topic)
            added_count += 1

    rebuild_vectors()
    return {"message": f"✅ Додано {added_count} нових тем до предмета '{subject}'."}

@app.post("/ask")
def ask_question(data: dict):
    try:
        q = data.get("question", "").strip()
        if not q: return {"answer": "Порожній запит."}
        
        q_lower = q.lower()
        is_detailed = bool(re.search(r'(детальніше|детально|повністю|весь текст|розгорнуто)', q_lower))
        
        # 1. Спец-режим: "головне по темі"
        summary_match = re.search(r'(головне|основне|суть|коротко|тези)\s*(по|про|в)?\s*тем[іі]\s*(.+)', q_lower)
        if summary_match and topics_vectors is not None:
            target_topic = summary_match.group(3).strip()
            query_vec = vectorizer.transform([target_topic])
            sims = cosine_similarity(query_vec, topics_vectors)[0]
            
            best_idx = sims.argmax()
            if sims[best_idx] > 0.05:
                meta = corpus_meta[best_idx]
                best_item = db_data[meta["item_idx"]]
                summary = [p for p in best_item["paragraphs"] if p.startswith(('-', '•'))] + \
                          [p.split('.')[0] + "." for p in best_item["paragraphs"] if not p.startswith(('-', '•'))]
                
                res = f"📚 <b>{best_item['subject']}</b>\n📌 <b>{best_item['title']}</b>\n\n<b>Головні тези:</b>\n\n" + "\n\n".join(summary[:5])
                if len(summary) > 5: res += "\n\n<i>...напиши «[назва теми] детальніше», щоб побачити все.</i>"
                return {"answer": res}

        # 2. Розумний пошук з розпізнаванням команди "детальніше"
        stop_words = r'(?i)\b(коротко|стисло|детально|детальніше|повністю|розгорнуто|все про|що|таке|це|які|є|як|чому|навіщо|предмет|тема)\b'
        search_query = re.sub(stop_words, '', q).strip()
        if not search_query: search_query = q
        
        if topics_vectors is not None and len(corpus_texts) > 0:
            query_vec = vectorizer.transform([search_query])
            sims = cosine_similarity(query_vec, topics_vectors)[0]
            
            for i, text in enumerate(corpus_texts):
                if search_query.lower() in text.lower(): sims[i] += 2.0
                    
            best_idx = sims.argmax()
            if sims[best_idx] > 0.05:
                meta = corpus_meta[best_idx]
                item = db_data[meta["item_idx"]]
                header = f"📚 <b>{item['subject']}</b>\n📌 <b>{item['title']}</b>\n\n"
                
                if meta["type"] == "title" or is_detailed:
                    if is_detailed:
                        ans = "\n\n".join(item["paragraphs"])
                    else:
                        ans = "\n\n".join(item["paragraphs"][:2]) + ("\n\n<i>...додай «детальніше» до запиту, щоб прочитати всю тему.</i>" if len(item["paragraphs"]) > 2 else "")
                    return {"answer": header + ans}
                else:
                    p_idx = meta["p_idx"]
                    ans = item["paragraphs"][p_idx]
                    if p_idx + 1 < len(item["paragraphs"]): ans += "\n\n" + item["paragraphs"][p_idx + 1]
                    return {"answer": header + ans}

        try:
            results = DDGS().text(search_query, max_results=1)
            if results: return {"answer": f"🌐 <b>Знайдено в інтернеті:</b><br><br>{results[0]['body']}"}
        except: pass
            
        return {"answer": "На жаль, я не знайшла цього у базі конспектів та інтернеті."}
    except Exception as e:
        return {"answer": f"Помилка ШІ: {str(e)}"}
