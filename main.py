from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS
import docx
import io

app = FastAPI(title="Aishka API")
security = HTTPBasic()

# Завантажуємо розумну мовну модель (розуміє зміст, а не просто букви)
print("Завантаження моделі... (потрібно почекати)")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Пам'ять нашої Aishka
topics_text = []
topics_embeddings = None

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
    global topics_text, topics_embeddings
    content = await file.read()
    
    # Читаємо файл (Word або TXT)
    if file.filename.endswith('.docx'):
        doc = docx.Document(io.BytesIO(content))
        raw_blocks = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    else:
        text = content.decode("utf-8")
        raw_blocks = [block.strip() for block in text.split('\n\n') if block.strip()]
    
    if not raw_blocks:
        return {"message": "Файл порожній."}

    # РОЗУМНЕ ОБ'ЄДНАННЯ (клеїмо заголовки до тексту)
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

    topics_text = blocks
    # Перетворюємо весь конспект на вектори змісту
    topics_embeddings = embedder.encode(topics_text, convert_to_tensor=True)
    
    return {"message": f"Успішно! Я проаналізувала {len(blocks)} тем і готова відповідати."}

@app.post("/ask")
async def ask_question(data: dict):
    question_text = data.get("question", "")
    print(f"\n--- НОВИЙ ЗАПИТ ВІД КОРИСТУВАЧА: '{question_text}' ---")
    
    # 1. Перевіряємо конспект
    if topics_embeddings is not None and len(topics_text) > 0:
        print(f"База конспекту завантажена. Кількість тем: {len(topics_text)}")
        
        query_embedding = embedder.encode(question_text, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, topics_embeddings, top_k=1)
        
        best_match_idx = hits[0][0]['corpus_id']
        score = hits[0][0]['score']
        
        print(f"Найкращий збіг: Тема №{best_match_idx} з точністю {score:.2f}")
        
        if score > 0.4:
            answer = topics_text[best_match_idx]
            print(f"Віддаю відповідь з конспекту: {answer[:50]}...")
            return {"answer": answer}
        else:
            print("Точність нижче 0.4. В конспекті цього немає.")
    else:
        print("ПОМИЛКА: Конспект не завантажено або він порожній!")

    # 2. Шукаємо в інтернеті
    print("Пробую знайти в DuckDuckGo...")
    try:
        results = DDGS().text(question_text, max_results=1)
        if results:
            web_answer = results[0]['body']
            print("Успішно знайдено в інтернеті.")
            return {"answer": f"🌐 Знайдено в інтернеті: {web_answer}"}
        else:
            print("DuckDuckGo нічого не знайшов.")
    except Exception as e:
        print(f"Помилка під час пошуку в інтернеті: {e}")
        return {"answer": "Помилка пошуку в інтернеті."}
        
    return {"answer": "Я не знайшла відповіді ні в конспекті, ні в інтернеті."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)