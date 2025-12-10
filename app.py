from datetime import date
import os
import re

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal



load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")          # örn: appOrTVQJzXgO4oNg
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")    # örn: "Leads"

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE_NAME:
    raise RuntimeError("Airtable env variables are not configured")



def is_valid_email(email: str) -> bool:
    if not email:
        return False
    return EMAIL_REGEX.match(email.strip()) is not None




# -----------------------------
# OpenAI client
# -----------------------------
client = OpenAI()

# -----------------------------
# Your vectorstore ID
# -----------------------------
VECTOR_STORE_ID = "vs_68e13ee74bc88191becbd2061ca7de01"   # <-- kendi gerçek ID'ni koy

# -----------------------------
# Basit IP bazlı rate limiting (günlük)
# -----------------------------
WEBSEARCH_DAILY_LIMIT = 3   # keloid dışı / websearch soruları
TOTAL_DAILY_LIMIT = 15      # toplam cevap limiti

# ip_stats[ip] = {"total": int, "websearch": int, "date": date}
ip_stats: dict[str, dict] = {}


def _get_ip(request: Request) -> str:
    """
    Gerçek projede X-Forwarded-For başlığını da okumak isteyebilirsin.
    Şimdilik doğrudan client.host kullanıyoruz.
    """
    return request.client.host or "unknown"


def _get_daily_counters(ip: str) -> dict:
    """
    Her IP için günlük sayaç tutar.
    Gün değişince sayaç otomatik sıfırlanır.
    """
    today = date.today()
    stats = ip_stats.get(ip)
    if not stats or stats.get("date") != today:
        stats = {"total": 0, "websearch": 0, "date": today}
        ip_stats[ip] = stats
    return stats


# ============================================================
# ====================== ROUTER ==============================
# ============================================================

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Choose whether to use vectorstore or web search."
    )

class StageRoute(BaseModel):
    stage: Literal["info", "nurture", "close"] = Field(
        ...,
        description=(
            "Conversation stage based on the user's question. "
            "'info' = only information seeking, "
            "'nurture' = evaluating options / has concerns, "
            "'close' = asking about price, appointment, or concrete action."
        )
    )



# Router LLM
llm_router = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_router = llm_router.with_structured_output(RouteQuery)

system_router = """
You are an expert router.

If the user question is about:
- keloids
- keloid treatments
- cryotherapy
- laser therapy
- corticosteroid injections
- recurrence rates
- medical literature
- wound healing
- clinical guidelines

ALWAYS choose: vectorstore.

For all unrelated questions choose: websearch.
"""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_router),
    ("human", "{question}")
])

question_router = route_prompt | structured_llm_router

def fast_route(question: str) -> Literal["vectorstore", "websearch"]:
    """
    Basit keyword kontrolüyle sorunun keloid ile ilgili olup olmadığını tahmin eder.
    Bariz keloid / skar sorularında direkt vectorstore döner, diğerlerinde websearch.
    """
    q = (question or "").lower()

    keloid_keywords = [
        "keloid",
        "keloit",
        "keloidcare",
        "skar",
        "scar",
        "yara izi",
        "yara izi tedavisi",
        "keloid tedavisi",
        "hypertrophic",
        "hipertrofik",
    ]

    if any(kw in q for kw in keloid_keywords):
        return "vectorstore"
    return "websearch"


# ============================================================
# ============== CONVERSATION STAGE ROUTER ===================
# ============================================================

stage_system_router = """
You are a triage assistant that classifies the user's question
into one of three conversation stages for a medical clinic chatbot.

Stages:
- "info": The user is only seeking general information about keloids,
  causes, treatments, risks, healing, etc. No mention of price, cost,
  appointment, booking, or specific offers.
- "nurture": The user is comparing options, expressing fears or hopes,
  asking about side effects, recurrence, success rates, or suitability.
  They are not yet explicitly asking for prices or appointments,
  but they seem to be evaluating whether this clinic / treatment is right for them.
- "close": The user is asking about price, cost, campaign, discount,
  package details, location, exact dates, appointment, booking, or similar
  decision-oriented topics.

Rules:
- If the question explicitly mentions price, cost, fee, campaign,
  discount, installment, appointment, booking, available dates, or schedule,
  choose "close".
- Else if the question expresses fear, concerns, comparison, or is
  clearly evaluating whether to do treatment (e.g. "should I do it",
  "is it worth it", "is it better than X", "what happens if I don't"),
  choose "nurture".
- Else if it is mostly general medical information about keloids,
  definitions, causes, or treatments, choose "info".
- If in doubt between "info" and "nurture", prefer "nurture".
"""

stage_route_prompt = ChatPromptTemplate.from_messages([
    ("system", stage_system_router),
    ("human", "{question}")
])

# Stage router LLM (aynı modeli kullanabiliriz)
stage_structured_llm = llm_router.with_structured_output(StageRoute)
conversation_stage_router = stage_route_prompt | stage_structured_llm


def detect_stage(user_question: str) -> str:
    """Kullanıcının sorusuna göre info / nurture / close aşamasını belirler."""
    route = conversation_stage_router.invoke({"question": user_question})
    print("STAGE DECISION:", route.stage)
    return route.stage

def build_limit_message(lang_code: str, limit_type: str) -> str:
    """
    limit_type: "total" veya "websearch"
    """
    if lang_code == "tr":
        if limit_type == "total":
            return (
                "Güvenlik nedeniyle, bugün bu asistandan en fazla 15 yanıt alabiliyoruz. "
                "Bugünkü sınır doldu. Yeni soruların için lütfen yarın tekrar yazabilir "
                "veya doğrudan kliniğimizle telefon ya da WhatsApp üzerinden iletişime geçebilirsin."
            )
        else:  # websearch
            return (
                "Klinik dışı, keloid ile ilgisi olmayan sorularda güvenlik nedeniyle günde en fazla "
                "3 yanıt verebiliyoruz. Bu sınırı doldurdun. "
                "Keloid ve tedavileriyle ilgili sorularını ise dilediğin kadar sorabilirsin."
            )
    else:
        if limit_type == "total":
            return (
                "For security reasons you can receive up to 15 answers from this assistant per day. "
                "You’ve reached today’s limit. For new questions, please try again tomorrow or contact "
                "our clinic directly by phone or WhatsApp."
            )
        else:
            return (
                "For non-keloid, general questions we can only provide up to 3 answers per day. "
                "You’ve reached this limit. You can still ask us as many questions as you like "
                "about keloid and scar treatments."
            )

# ============================================================
# ================== GRADE DOCUMENTS (RELEVANCE) =============
# ============================================================

class GradeDocuments(BaseModel):
    """Binary relevance score for a single document."""
    binary_score: str = Field(
        description="Reply 'yes' if the document is relevant to the question, otherwise 'no'."
    )

# LLM for grading
llm_grader = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm_grader.with_structured_output(GradeDocuments)

grade_docs_system = """
You are a grader checking if a single document is relevant to a user question.

If the document helps answer the question, reply 'yes'.
If it is irrelevant or only weakly related, reply 'no'.

Answer strictly with 'yes' or 'no'.
"""

grade_docs_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grade_docs_system),
        (
            "human",
            "Question:\n{question}\n\nDocument:\n{document}\n\nIs this document relevant? (yes or no)"
        ),
    ]
)

grade_documents_chain: RunnableSequence = grade_docs_prompt | structured_llm_grader


def filter_relevant_chunks(question: str, chunks: list[str]) -> list[str]:
    """LLM ile her chunk için 'relevant mı?' filtresi uygular."""
    relevant = []

    for idx, ch in enumerate(chunks, start=1):
        try:
            result = grade_documents_chain.invoke(
                {"question": question, "document": ch}
            )
            decision = (result.binary_score or "").strip().lower()
            print(f"[GRADE_DOCS] Chunk {idx}: {decision}")

            if decision.startswith("y"):  # yes
                relevant.append(ch)
        except Exception as e:
            print(f"[GRADE_DOCS] Error grading chunk {idx}: {e}")
            # Hata olursa chunk'ı atlayabiliriz

    print(f"[GRADE_DOCS] Kept {len(relevant)} / {len(chunks)} chunks")
    return relevant


# ============================================================
# ================== HALLUCINATION CHECKER ===================
# ============================================================

def check_hallucination(documents: str, answer: str) -> bool:
    """Cevabın verilen dokümanlarla uyumlu olup olmadığını kontrol eder."""

    prompt = f"""
You are a hallucination checker.
Your task is to determine whether the assistant's answer is fully grounded in the provided context.

If the answer is supported → reply only: YES
If the answer is not supported → reply only: NO

CONTEXT:
{documents}

ANSWER:
{answer}
    """

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=32,  # <<< BURAYI DEĞİŞTİRDİK
    )

    result = resp.output_text.strip().upper()
    return result == "YES"

# ============================================================
# ================== LANGUAGE DETECTION ======================
# ============================================================

LANG_NAME_MAP = {
    "tr": "Turkish",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "bg": "Bulgarian",
}


def detect_language(text: str) -> tuple[str, str]:
    """
    Kullanıcının soru dilini tespit eder.
    ISO 639-1 kodu (tr, en, es...) ve İngilizce adını döner.
    """
    prompt = f"""
Detect the primary language of the following user text.
Respond ONLY with the two-letter ISO 639-1 language code
(e.g. "tr", "en", "es", "fr", "de") and nothing else.

Text:
{text}
"""

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=16,   # <<< BURAYI 16 YAP
    )

    code = resp.output_text.strip().lower()
    if code not in LANG_NAME_MAP:
        code = "tr"  # default
    lang_name = LANG_NAME_MAP[code]
    return code, lang_name

WELCOME_MESSAGES = {
    "en": (
        "Hi There! Welcome to The Keloidcare clinic.\n"
        "What should I call you?"
    ),
    "tr": (
        "Merhaba! Keloidcare Kliniğine hoş geldin.\n"
        "Sana nasıl hitap edelim?"
    ),
    "fr": (
        "Salut ! Bienvenue à la clinique Keloidcare.\n"
        "Comment veux-tu qu’on t’appelle ?"
    ),
    "de": (
        "Hi! Willkommen in der Keloidcare Klinik.\n"
        "Wie dürfen wir dich nennen?"
    ),
    "el": (
        "Γεια σου! Καλώς ήρθες στην κλινική Keloidcare.\n"
        "Πώς να σε φωνάζουμε;"
    ),
    "bg": (
        "Здрасти! Добре дошъл/дошла в клиниката Keloidcare.\n"
        "Как да те наричаме?"
    ),
    "es": (
        "¡Hola! Bienvenido a la clínica Keloidcare.\n"
        "¿Cómo quieres que te llamemos?"
    ),
}

# IP'den ülke kodu geldiğinde hangi dili kullanacağımız
COUNTRY_LANG_MAP = {
    "FR": "fr",
    "DE": "de",
    "GR": "el",   # Yunanistan
    "BG": "bg",
    "TR": "tr",
}

def get_country_code_from_ip(ip: str) -> str | None:
    """
    IP'den ülke kodu almak için basit bir servis kullanıyoruz.
    Prod ortamında istersen farklı bir provider'a geçebilirsin.
    """
    try:
        if not ip or ip in ("127.0.0.1", "::1", "localhost"):
            return None

        resp = requests.get(f"https://ipapi.co/{ip}/json/", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            code = data.get("country_code")
            if isinstance(code, str):
                return code.upper()
    except Exception as e:
        print("IP geolocation error:", e)
    return None


def get_preferred_lang_from_request(request: Request) -> str:
    """
    1) Önce IP'den ülke kodunu bul.
       - FR -> Fransızca
       - DE -> Almanca
       - GR -> Yunanca
       - BG -> Bulgarca
       - TR -> Türkçe
       - Diğer tüm ülkeler -> İngilizce
    2) IP'den ülke alınamazsa (localhost vs.)
       Accept-Language'e göre tahmin et, yine yoksa İngilizce.
    """
    ip = _get_ip(request)
    country_code = get_country_code_from_ip(ip)

    if country_code:
        if country_code in COUNTRY_LANG_MAP:
            return COUNTRY_LANG_MAP[country_code]
        # tanıdığımız ama map'te olmayan ülke -> İngilizce
        return "en"

    # fallback: Accept-Language
    header = request.headers.get("accept-language", "")
    if header:
        first = header.split(",")[0].strip()
        if "-" in first:
            first = first.split("-")[0]
        code = first.lower()
        if code in LANG_NAME_MAP:
            return code

    return "en"



def get_welcome_message(lang_code: str) -> str:
    return WELCOME_MESSAGES.get(lang_code, WELCOME_MESSAGES["en"])


def build_intro_messages(lang_code: str, name: str) -> list[str]:
    """
    İlk karşılama sonrası isim ve e-posta soruları.
    Dil bulunamazsa İngilizce döner.
    """
    templates = {
        "en": (
            "I am Nicole! Nice to meet you, {name}!",
            "May I know your email {name}? so I can get back to you if needed."
        ),
        "tr": (
            "Ben Nicole! Tanıştığımıza memnun oldum, {name}!",
            "{name}, e-posta adresini alabilir miyim? Gerekirse sana dönebilmem için."
        ),
        "fr": (
            "Je suis Nicole ! Ravi de te rencontrer, {name} !",
            "Puis-je avoir ton email {name} ? Ainsi je pourrai te recontacter si besoin."
        ),
        "de": (
            "Ich bin Nicole! Schön, dich kennenzulernen, {name}!",
            "Darf ich deine E-Mail {name} haben, damit ich mich bei Bedarf melden kann?"
        ),
        "el": (
            "Είμαι η Nicole! Χαίρομαι που σε γνωρίζω, {name}!",
            "Μπορώ να έχω το email σου {name}; ώστε να μπορώ να σε ενημερώσω αν χρειαστεί."
        ),
        "bg": (
            "Аз съм Никол! Приятно ми е да се запознаем, {name}!",
            "Мога ли да взема имейла ти, {name}, за да се свържа при нужда?"
        ),
    }

    msg1, msg2 = templates.get(lang_code, templates["en"])
    return [msg1.format(name=name), msg2.format(name=name)]


def build_invalid_email_message(lang_code: str, name: str) -> str:
    """
    Hatalı e-posta mesajını diline göre döner, bulunamazsa İngilizce.
    """
    templates = {
        "en": (
            "No problem {name}, if you do not want to share your email address! "
            "I am here to find what you need. What are you looking for?"
        ),
        "tr": (
            "Hiç sorun değil {name}, e-posta adresini paylaşmak istemezsen! "
            "İhtiyacın olanı bulmana yardım etmek için buradayım. Ne arıyorsun?"
        ),
        "fr": (
            "Pas de souci {name} si tu ne veux pas partager ton email ! "
            "Je suis là pour t’aider. Que recherches-tu ?"
        ),
        "de": (
            "Kein Problem {name}, falls du deine E-Mail nicht teilen möchtest! "
            "Ich bin hier, um zu finden, was du brauchst. Wonach suchst du?"
        ),
        "el": (
            "Κανένα πρόβλημα {name} αν δεν θέλεις να μοιραστείς το email σου! "
            "Είμαι εδώ για να βρω αυτό που χρειάζεσαι. Τι ψάχνεις;"
        ),
        "bg": (
            "Няма проблем {name}, ако не искаш да споделиш имейла си! "
            "Тук съм, за да намеря от какво имаш нужда. Какво търсиш?"
        ),
    }
    return templates.get(lang_code, templates["en"]).format(name=name)


def build_email_thanks_message(lang_code: str, name: str) -> str:
    """
    Geçerli e-posta sonrası teşekkür mesajı, bulunamazsa İngilizce.
    """
    templates = {
        "en": (
            "Thank you {name}! I am here to find what you need. What are you looking for?"
        ),
        "tr": (
            "Teşekkürler {name}! İhtiyacın olanı bulmana yardım etmek için buradayım. Ne arıyorsun?"
        ),
        "fr": (
            "Merci {name} ! Je suis là pour t’aider à trouver ce dont tu as besoin. Que recherches-tu ?"
        ),
        "de": (
            "Danke {name}! Ich bin hier, um zu finden, was du brauchst. Wonach suchst du?"
        ),
        "el": (
            "Ευχαριστούμε {name}! Είμαι εδώ για να βρω αυτό που χρειάζεσαι. Τι ψάχνεις;"
        ),
        "bg": (
            "Благодаря ти, {name}! Тук съм, за да намерим от какво имаш нужда. Какво търсиш?"
        ),
    }
    return templates.get(lang_code, templates["en"]).format(name=name)



# ================== LEAD MODELLERİ ve AIRTABLE ENTEGRASYONU ==================

class ChatMessage(BaseModel):
    role: str    # "user" veya "assistant"
    content: str

class NameRequest(BaseModel):
    name: str


class EmailRequest(BaseModel):
    name: str
    email: str

class LeadPayload(BaseModel):
    name: str | None = None
    phone: str | None = None
    email: str | None = None
    message: str | None = None               # Son mesaj / not (opsiyonel)
    conversation: list[ChatMessage] | None = None  # Tüm sohbet (opsiyonel)


def summarize_conversation(conversation: list[ChatMessage]) -> str | None:
    """Sohbeti doktor için 5–7 madde halinde özetler."""
    if not conversation:
        return None

    text = "\n".join([f"{m.role}: {m.content}" for m in conversation])

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=f"""
You are a medical assistant at a keloid clinic.

Summarize the following conversation between a patient and the KeloidCare Clinic AI
in 5–7 concise bullet points in English.

Focus on:
- main keloid complaints (location, duration, symptoms)
- previous treatments and responses
- patient concerns and expectations
- any mentioned comorbidities or medications

CONVERSATION:
{text}
""",
        max_output_tokens=250,
    )
    return resp.output_text.strip()

def send_lead_to_airtable(payload: LeadPayload):
    """Hasta iletişim bilgilerini + sohbet özetini Airtable'a kaydeder."""

    summary = summarize_conversation(payload.conversation) if payload.conversation else None

    fields: dict[str, str] = {}

    if payload.name:
        fields["Name"] = payload.name
    if payload.phone:
        fields["Phone"] = payload.phone
    if payload.email:
        fields["Email"] = payload.email
    if payload.message:
        fields["PatientMessage"] = payload.message
    if summary:
        fields["ConversationSummary"] = summary

    if not fields:
        raise RuntimeError("No lead fields to send to Airtable")

    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "records": [
            {"fields": fields}
        ]
    }

    r = requests.post(url, headers=headers, json=data)

    # DEBUG
    print("\n--- AIRTABLE DEBUG LOG ---")
    print("URL:", url)
    print("Headers:", headers)
    print("Payload:", data)
    print("Response Code:", r.status_code)
    print("Response Body:", r.text)
    print("--- END DEBUG LOG ---\n")

    if r.status_code not in (200, 201, 202):
        raise RuntimeError(f"Airtable error: {r.status_code} {r.text}")

    return r.json()





# ============================================================
# ====================== RAG System ==========================
# ============================================================

def rag_answer(query: str, lang_info: tuple[str, str] | None = None) -> tuple[str, str]:
    """
    RAG pipeline — translate → retrieve → GRADE_DOCUMENTS → answer.
    Returns (answer, source), where source ∈ {"rag", "websearch", "empty"}.

    LLM, keloid kliniği adına konuşur ve cevabı her zaman
    kullanıcının soru dilinde verir.
    """

    # 0) Kullanıcının dilini tespit et (hazır varsa kullan)
    if lang_info is None:
        lang_code, lang_name = detect_language(query)
        lang_info = (lang_code, lang_name)
    else:
        lang_code, lang_name = lang_info

    # 1) Soru İngilizceye çevrilir (sadece arama için kullanıyoruz)
    translation_resp = client.responses.create(
        model="gpt-4o-mini",
        input=f"Translate this query to English (only English output): {query}",
        max_output_tokens=64,
    )
    translated_query = translation_resp.output_text.strip()

    # 2) Vector store'dan chunk'lar çekilir
    search = client.responses.create(
        model="gpt-4o-mini",
        input=translated_query,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID],
            # grade_documents kullanılacağı için 6–8 gibi biraz yüksek tutmak mantıklı
            "max_num_results": 4
        }],
        include=["file_search_call.results"]
    )

    chunks: list[str] = []
    for item in search.output:
        if item.type == "file_search_call":
            for r in item.results:
                chunks.append(r.text)

    print(f"\nRetrieved chunks (raw): {len(chunks)}")

    # 2.5) Hiç chunk yoksa → internal dokümanlardan cevap yok
    if len(chunks) == 0:
        no_answer_prompt = f"""
You are a medical assistant working for a specialized keloid clinic.

The user asked (in {lang_name}):
{query}

There is no relevant information about this in the clinic's internal documents.
You must answer in {lang_name} only.

Politely explain that our clinic does not have enough document-based data
for this specific question and that the user should consult our doctors
directly for a personalized evaluation.

Speak as "we" / "our clinic", not as an AI model.
"""
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=no_answer_prompt,
            max_output_tokens=200,
        )
        return resp.output_text.strip(), "empty"

    # 3) GRADE_DOCUMENTS → sadece ilgili chunk'lar kalsın
    # Burada istersen translated_query yerine query de kullanabilirsin.
    graded_chunks = filter_relevant_chunks(translated_query, chunks)

    print(f"[GRADE_DOCS] After grading: {len(graded_chunks)} relevant chunks")

    # Eğer grading sonrası hiç chunk kalmadıysa, dokümanlardan cevap veremiyoruz
    if len(graded_chunks) == 0:
        no_answer_prompt = f"""
You are a medical assistant working for a specialized keloid clinic.

The user asked (in {lang_name}):
{query}

We found documents, but none of them are clearly relevant to this question.
You must answer in {lang_name} only.

Politely explain that our clinic does not have enough document-based data
for this specific question and that the user should consult our doctors
directly for a personalized evaluation.

Speak as "we" / "our clinic", not as an AI model.
"""
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=no_answer_prompt,
            max_output_tokens=200,
        )
        return resp.output_text.strip(), "empty"

    # 4) Context artık SADECE ilgili chunk'lardan oluşuyor
    context = "\n\n".join(graded_chunks)

    # 5) Bu context'e dayanarak (kullanıcı dili ile) cevap üret
    final = client.responses.create(
        model="gpt-4o",
        input=f"""
You are a medical assistant for a specialized keloid clinic.

The user's language is: {lang_name} (code: {lang_code}).
User question (keep its original wording and answer in this language):
{query}

You speak on behalf of the clinic using "we" / "our clinic", not as an abstract AI.

Use ONLY the context below to answer the user's question.
If the context does not contain the answer, say that we don't know from
our documents and recommend consulting our doctors directly.

You MUST answer ONLY in {lang_name}, even if the context is in a different language.
Translate any necessary information from the context into {lang_name}.

CONTEXT:
{context}
""",
        max_output_tokens=350
    )

    answer = final.output_text

    # 6) HALLUCINATION CHECK (artık graded context'e göre)
    is_grounded = check_hallucination(context, answer)

    if not is_grounded:
        print("⚠️ HALLUCINATION DETECTED → FALLBACK WEBSEARCH")
        ws_answer = websearch_answer(query, lang_info=lang_info)
        return ws_answer, "websearch"

    # 7) Her şey yolundaysa → cevabı döndür
    # (CTA akışı satış katmanında eklenecek)
    return answer, "rag"




# ============================================================
# =============== SALES STYLE POST-PROCESSOR =================
# ============================================================

SALES_PLAYBOOK = {
    "info": {
        "tone": (
            "Calm, educational, empowering. Recognize that learning about keloids early is wise."
        ),
        "intro": (
            "Open with gratitude for reaching out and reinforce that sharing knowledge freely is part of our care."
        ),
        "closing": (
            "Invite them to request complimentary resources (PDF, başarı hikayeleri, kısa video) "
            "and remind that final decisions follow a doctor evaluation."
        ),
        "tactics": [
            "Reciprocity: mention that we happily share analyses, başarı hikayeleri or guidance without obligation.",
            "Authority: highlight that our medical team and technology (lazer, soğutma, bakım protokolleri) are curated for keloids without adding new clinical claims.",
            "Consistency: assure that the same information is reflected across WhatsApp, PDF'ler ve klinik görüşmeleri.",
        ],
        "cues": [
            "Offer fast follow-up options (ör. WhatsApp, sesli mesaj) even if they cannot respond immediately.",
            "Encourage them to keep asking questions and remind that sabır + düzen takibi kritik.",
        ],
    },
    "nurture": {
        "tone": (
            "Warm, story-driven, empathetic. Normalize their kararsızlık and create emotional closeness."
        ),
        "intro": (
            "Reference that many visitors feel the same and we listen closely before öneri yapmak."
        ),
        "closing": (
            "Invite them to share birkaç detay so we can craft a kişiye özel plan and schedule a free evaluation call if they wish."
        ),
        "tactics": [
            "Liking: use their name if available, mirror their vibe, show gerçek bir insan ilgisi.",
            "Yes Ladder: build küçük onaylar (süreç uzun, kaşıntı olabilir, çözüm görmek ister misiniz?) before proposing evaluation.",
            "Storytelling & Social Proof: refer to anonymized başarı hikayeleri or how birçok hastaya eşlik ettiğimizi belirt (no numbers).",
            "Repetition & Patience: reassure that it's normal to revisit sorular and we gladly explain again.",
        ],
        "cues": [
            "Mention that we can send kısa video / ses notu if they prefer to feel the team's presence.",
            "Hint that we log their notes so nothing kaybolur and we follow up nazikçe (ör. 3 gün / 7 gün ritmi).",
        ],
    },
    "close": {
        "tone": (
            "Confident, action-oriented but gentle. Reduce friction around booking or fiyat konuşmasının yöntemi."
        ),
        "intro": (
            "Affirm that taking action now prevents keloidların sertleşip yayılmasını and that we already reserve time for them."
        ),
        "closing": (
            "Use assumption language (hangi gün uygun olur) + scarcity/deadline cues (kontenjan dolmadan) "
            "and remind that full plan & ücret only netleşir after evaluation. Offer ücretsiz analiz or memnuniyet garantili ilk görüşme opsiyonu."
        ),
        "tactics": [
            "Fear/Responsibility: gently outline that gecikmek daha agresif tedavilere neden olabilir, so erken planlama değerlidir.",
            "Freebie: highlight ücretsiz cilt analizi veya ilk görüşmede memnuniyet garantisi.",
            "Assumption Technique: speak as if they already chose us and only scheduling details remain.",
            "Deadline & Scarcity: mention limited kampanya süreleri veya doktor kontenjanlarının hızla dolduğu (no exact numbers/dates).",
        ],
        "cues": [
            "Offer to lock a provisional slot and promise nazik hatırlatmalar (WhatsApp ping, kısa arama).",
            "Reassure that we document seçimlerini in CRM so fiyat / kampanya tutarlılığı bozulmaz.",
        ],
    },
}

COMMON_SALES_GUARDRAILS = """
- Keep intro + closing together under roughly four sentences; be concise.
- Mirror the user's language style and emoji usage (if they add 😊 you may add one benzer emoji).
- Never invent medical details, prices, rakamlar or guarantees; reference doctor evaluation for personalization.
- Highlight that we are reachable via WhatsApp, sesli mesaj or video if they prefer warmer iletişim.
- Mention that we log follow-up notes (CRM) so nobody feels forgotten and bilgilerin tutarlılığı korunur.
"""


def _build_stage_instruction(stage: str) -> str:
    data = SALES_PLAYBOOK.get(stage, SALES_PLAYBOOK["info"])
    tactics_block = "\n".join(f"- {item}" for item in data.get("tactics", []))
    cues_block = "\n".join(f"- {item}" for item in data.get("cues", []))
    return f"""
Tone focus: {data['tone']}
Intro focus: {data['intro']}
Closing/CTA focus: {data['closing']}

Preferred persuasion cues (weave in naturally, maks 1 cümle):
{tactics_block}

Conversation micro-cues:
{cues_block}
"""

def build_flow4_cta(lang_code: str) -> str:
    """
    Kullanıcının durumuna özel randevu / tedavi seçenekleri için
    iletişim izni isteyen CTA.
    """
    if lang_code == "tr":
        return (
            "İstersen senin durumuna özel randevu ve tedavi seçeneklerini de paylaşabiliriz. "
            "Bunun için bir-iki iletişim bilgine ihtiyacımız olacak. Uygun mu?"
        )
    elif lang_code == "fr":
        return (
            "Si tu veux, nous pouvons aussi te proposer des options de rendez-vous et de traitement adaptées à ta situation. "
            "Pour cela, nous aurons besoin de quelques informations de contact. Est-ce que ça te convient ?"
        )
    else:
        return (
            "If you like, we can also share appointment and treatment options tailored to your situation. "
            "For that we’ll need one or two contact details from you. Is that okay?"
        )

def apply_sales_style(
    user_question: str,
    base_answer: str,
    stage: str,
    lang_info: tuple[str, str] | None = None,
    user_name: str | None = None,
) -> str:

    """
    Tıbbi olarak doğrulanmış cevabı (base_answer) HİÇ DEĞİŞTİRMEZ.
    Sadece, aşamaya (info / nurture / close) göre:
    - kısa bir giriş (intro)
    - kısa bir kapanış (closing / CTA) ekler.

    'close' aşamasında, kapanışın sonuna Flow 4'te tarif ettiğin
    özel CTA cümlesini ekliyoruz.
    """
    if lang_info is None:
        lang_code, lang_name = detect_language(user_question)
        lang_info = (lang_code, lang_name)
    else:
        lang_code, lang_name = lang_info

    stage_instructions = _build_stage_instruction(stage)

    safe_name = (user_name or "").strip()
    if safe_name:
        name_instruction = (
            f'The user\'s name is "{safe_name}". '
            f'Use this name at most once, either in the intro or in the closing, '
            f'in a natural way of addressing them. Do NOT repeat the name in every sentence.'
        )
    else:
        name_instruction = "You do not know the user's name. Do not invent one."

    prompt = f"""
You are a patient advisor and sales-oriented representative for a specialized keloid clinic.

The user's language is: {lang_name} (code: {lang_code}).
You MUST write both paragraphs ONLY in {lang_name}.
If the user's question is in Turkish, write in Turkish.
If it is in English, write in English.

{name_instruction}

You will NOT write the medical explanation. That part is already prepared.
Your job is ONLY to write:
- a SHORT intro paragraph
- and a SHORT closing / CTA paragraph.

You MUST:
- Speak on behalf of the clinic as "we" / "our clinic", not as an AI model.
- NOT describe specific medical procedures, drugs, doses, or success rates.
- Refer to the medical explanation only in general terms, like
  "aşağıdaki bilgiler", "aşağıda paylaşılan tedavi seçenekleri" etc.
- NOT add any numerical claims (%, number of patients, years, etc.).
- NOT give any guarantee of results.

Conversation stage: {stage}

Stage-specific guidelines:
{stage_instructions}

Universal guardrails:
{COMMON_SALES_GUARDRAILS}

OUTPUT FORMAT (important):
[INTRO]
<your intro paragraph in {lang_name}>

[CLOSING]
<your closing / CTA paragraph in {lang_name}>

Do not mention the tags [INTRO] or [CLOSING] to the user; they are just for parsing.
"""


    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=300,
    )

    raw = resp.output_text.strip()
    intro = ""
    closing = ""

    if "[INTRO]" in raw and "[CLOSING]" in raw:
        try:
            after_intro = raw.split("[INTRO]", 1)[1]
            intro_part, closing_part = after_intro.split("[CLOSING]", 1)
            intro = intro_part.strip()
            closing = closing_part.strip()
        except Exception:
            intro = raw.strip()
            closing = ""
    else:
        intro = raw

    # Flow 4 CTA: sadece "close" aşamasında, kapanışın sonuna ekle
    if stage == "close":
        flow_cta = build_flow4_cta(lang_code)
        if closing:
            closing = f"{closing}\n\n{flow_cta}"
        else:
            closing = flow_cta

    parts = []
    if intro:
        parts.append(intro)
    if base_answer:
        parts.append(base_answer)
    if closing:
        parts.append(closing)

    final_answer = "\n\n".join(parts)
    return final_answer


# ============================================================
# ====================== WEBSEARCH ============================
# ============================================================

def websearch_answer(query: str, lang_info: tuple[str, str] | None = None) -> str:
    """
    If RAG fails or router decides, use websearch.
    AMA:
    - Başka kliniklerin fiyatını/verdiği rakamları ASLA söyleme.
    - Hiçbir şekilde net fiyat, aralık, USD/Euro/TL rakamı verme.
    - Başka klinik / hastane / marka adı verme.
    """
    if lang_info is None:
        lang_code, lang_name = detect_language(query)
        lang_info = (lang_code, lang_name)
    else:
        lang_code, lang_name = lang_info

    prompt = f"""
You are an assistant speaking on behalf of our specialized keloid clinic.

The user's language is: {lang_name} (code: {lang_code}).
You MUST answer ONLY in {lang_name}.

IMPORTANT RESTRICTIONS:
- Do NOT mention any specific prices, cost ranges, currencies, or numeric estimates.
- Do NOT mention or promote other clinics, brand names, hospitals, or websites.
- Even if web search results contain prices or other clinics, you MUST ignore them.
- You can explain which factors affect the cost (lesion size, location, number of sessions, etc.),
  but you must NOT give numbers.
- ALWAYS say that exact pricing in our clinic is determined only after an in-person or online
  evaluation by our doctors.

Use web search ONLY to improve the quality of general medical information (e.g. treatment options),
but NEVER to give concrete costs or name other providers.

User question:
{query}
"""

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        tools=[{"type": "web_search"}],
        max_output_tokens=350,
    )
    return resp.output_text


# ============================================================
# ====================== SMART ANSWER =========================
# ============================================================

def smart_answer(
    user_question: str,
    precomputed_route=None,
    lang_info: tuple[str, str] | None = None,
    user_name: str | None = None,
) -> tuple[str, dict]:

    """
    Cevap + metadata döner.
    meta = {
      "datasource": "vectorstore" | "websearch",
      "source": "rag" | "websearch" | "empty",
      "stage": "info" | "nurture" | "close" | None
    }
    """
    if lang_info is None:
        lang_info = detect_language(user_question)

    # 🔹 BURASI DEĞİŞTİ
    if precomputed_route is None:
        # Önce hızlı kural tabanlı router ile tahmin et
        fast_ds = fast_route(user_question)

        if fast_ds == "vectorstore":
            # Bariz keloid / skar sorularında LLM router'a gerek yok
            route = RouteQuery(datasource="vectorstore")
        else:
            # Diğer sorularda LLM router devreye girsin
            route = question_router.invoke({"question": user_question})
    else:
        # Dışarıdan hazır route geldiyse onu kullan
        route = precomputed_route

    datasource = route.datasource
    print("ROUTER DECISION:", datasource)

    meta = {"datasource": datasource, "source": None, "stage": None}

    # 2) Eğer keloid / medikal içerik → vectorstore + RAG
    if datasource == "vectorstore":
        base_answer, source = rag_answer(user_question, lang_info=lang_info)
        meta["source"] = source

        # RAG hiç cevap bulamadıysa → direkt websearch
        if source == "empty" or (base_answer is None):
            print("FALLBACK → WEBSEARCH (no RAG answer)")
            answer = websearch_answer(user_question, lang_info=lang_info)
            meta["source"] = "websearch"
            return answer, meta

        # Eğer cevap zaten websearch'ten geldiyse (hallucination fallback)
        if source == "websearch":
            stage = detect_stage(user_question)
            meta["stage"] = stage
            styled = apply_sales_style(
                user_question, base_answer, stage, lang_info=lang_info, user_name=user_name
            )
            return styled, meta

        # Kaynak gerçekten RAG ise → satış stil filtresi uygula
        stage = detect_stage(user_question)
        meta["stage"] = stage
        styled_answer = apply_sales_style(
            user_question, base_answer, stage, lang_info=lang_info, user_name=user_name
        )
        return styled_answer, meta

    # 3) Keloid dışı sorularda → direkt websearch (yine soru diliyle, klinik adına)
    else:
        answer = websearch_answer(user_question, lang_info=lang_info)
        meta["source"] = "websearch"
        return answer, meta

# ============================================================
# =========================== TEST ===========================
# ============================================================

if __name__ == "__main__":
    print("\n--- TEST 1 (Keloid Question → Vectorstore) ---")
    print(smart_answer("Keloidlerim kizardi, cok korkuyorum ne yapmaliyim"))


# ============================================================
# ========================== FASTAPI =========================
# ============================================================

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/welcome")
async def welcome(request: Request):
    """
    Chat açıldığında front-end burayı çağırıp
    uygun dilde karşılama metnini alabilir.
    """
    lang_code = get_preferred_lang_from_request(request)
    return {"lang": lang_code, "message": get_welcome_message(lang_code)}

@app.post("/intro/name")
async def intro_name(payload: NameRequest, request: Request):
    """
    Kullanıcı adını yazdıktan sonraki adım.
    2 mesaj döner (IP / Accept-Language ile tespit edilen dilde):
      1) I am Nicole! Nice to meet you, <name>!
      2) May I know your email <name>? so I can get back to you if needed.
    """
    name = (payload.name or "").strip() or "there"
    lang_code = get_preferred_lang_from_request(request)
    messages = build_intro_messages(lang_code, name)

    return {
        "name": name,
        "messages": messages,
    }


@app.post("/intro/email")
async def intro_email(payload: EmailRequest, request: Request):
    """
    Kullanıcı email girdikten sonraki adım.
    - Email formatı doğruysa -> Airtable'a (name + email) kaydedilir,
      sonra CTA mesajı döner.
    - Email formatı yanlışsa -> Airtable'a hiçbir şey gönderilmez,
      şu mesaj döner:
      "No problem name, if you do not want to share your email address! ..."
    """
    name = (payload.name or "").strip() or "there"
    lang_code = get_preferred_lang_from_request(request)
    email = (payload.email or "").strip()

    # 4. madde: yanlış format veya kullanıcı vermek istemiyor
    if not is_valid_email(email):
        msg = build_invalid_email_message(lang_code, name)
        return {
            "name": name,
            "email": None,
            "valid": False,
            "messages": [msg],
        }

    # 3. madde: email formatı doğru -> Airtable'a yolla
    try:
        lead_payload = LeadPayload(
            name=name,
            email=email,
            message=None,
            conversation=None,
        )
        send_lead_to_airtable(lead_payload)
    except Exception as e:
        # Airtable hatasını logla ama kullanıcıya hata gösterme
        print("Error sending lead to Airtable:", e)

    msg = build_email_thanks_message(lang_code, name)

    return {
        "name": name,
        "email": email,
        "valid": True,
        "messages": [msg],
    }


@app.post("/ask")
async def ask_api(payload: dict, request: Request):
    question = (payload.get("question") or "").strip()
    if not question:
        return {"answer": ""}

    # 👇 front-end'ten (widget'tan) gelen isim
    user_name = (payload.get("name") or "").strip() or None

    # IP bazlı sayaçları al / güncelle
    ip = _get_ip(request)
    stats = _get_daily_counters(ip)

    # Dil tespiti (limit mesajları için)
    lang_seed = question or "Merhaba"
    lang_code, lang_name = detect_language(lang_seed)
    lang_info = (lang_code, lang_name)

    # 🔹 Önce hızlı router ile kaba bir tahmin yap
    fast_ds = fast_route(question)

    # 1) Toplam günlük limit kontrolü (15 cevap)
    if stats["total"] >= TOTAL_DAILY_LIMIT:
        msg = build_limit_message(lang_code, "total")
        return {"answer": msg, "limit_reached": True}

    # 2) Eğer soru bariz keloid ile alakasızsa ve fast_route = websearch ise,
    #    websearch limitini kontrol et
    if fast_ds == "websearch" and stats["websearch"] >= WEBSEARCH_DAILY_LIMIT:
        msg = build_limit_message(lang_code, "websearch")
        return {"answer": msg, "limit_reached": True}

    # 3) Normal akış: cevabı üret
    #    Burada precomputed_route GÖNDERMİYORUZ ki smart_answer kendi router'ını kullansın
    answer, meta = smart_answer(
        question,
        precomputed_route=None,
        lang_info=lang_info,
        user_name=user_name,
    )

    # Gerçek datasource'u meta'dan al
    datasource = meta.get("datasource", fast_ds)

    # 4) Sayaçları cevaptan sonra artır
    stats["total"] += 1
    if datasource == "websearch":
        stats["websearch"] += 1

    return {
        "answer": answer,
        "meta": {
            "datasource": datasource,
            "source": meta.get("source"),
            "stage": meta.get("stage"),
            "limits": {
                "total_used": stats["total"],
                "total_limit": TOTAL_DAILY_LIMIT,
                "websearch_used": stats["websearch"],
                "websearch_limit": WEBSEARCH_DAILY_LIMIT,
            },
        },
    }



@app.post("/lead")
async def create_lead_endpoint(lead: LeadPayload):
    """
    İletişim bilgilerini + (varsa) sohbet geçmişini alır,
    Airtable'a kaydeder.
    """
    try:
        airtable_resp = send_lead_to_airtable(lead)
        return {"status": "ok", "airtable": airtable_resp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
