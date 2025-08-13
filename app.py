import json
import numpy as np
import gradio as gr
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

class ChatbotEngine:
    def __init__(self):
        # Load data
        with open("intent_dataset.json", "r", encoding="utf-8") as f:
             self.faq_data = json.load(f)

        with open("clarification_question_model/clarifying_questions_main.json", "r", encoding="utf-8") as f:
             self.clarifying_questions = json.load(f)


        self.intent_labels = sorted(set(entry["intent"] for entry in self.faq_data))
        self.FALLBACK = "😕 Sorry, I didn't get that. Can you rephrase?"

        # Load models
        self.intent_model = TFAutoModelForSequenceClassification.from_pretrained("saved_bert_model")
        self.intent_tokenizer = AutoTokenizer.from_pretrained("saved_bert_model")
        self.retriever_sbert = SentenceTransformer("fine_tuned_models/sbert_finetuned_small")
        self.clarifier_sbert = SentenceTransformer("clarification_question_model/clarification_sbert_model")

        # Prepare clarification embeddings
        self.clarification_corpus = []
        for intent, questions in self.clarifying_questions.items():
            self.clarification_corpus += questions
        self.clarification_embeddings = self.clarifier_sbert.encode(self.clarification_corpus, convert_to_tensor=True)

        # Prepare FAQ embeddings
        self.intent_to_qas = {}
        self.intent_to_embeddings = {}
        for entry in self.faq_data:
            intent = entry["intent"]
            q, a = entry["question"], entry["answer"]
            emb = self.retriever_sbert.encode(q)
            if intent not in self.intent_to_qas:
                self.intent_to_qas[intent] = []
                self.intent_to_embeddings[intent] = []
            self.intent_to_qas[intent].append({"q": q, "a": a})
            self.intent_to_embeddings[intent].append(emb)

    def predict_intent(self, query):
        inputs = self.intent_tokenizer(query, return_tensors="tf", padding=True, truncation=True, max_length=128)
        output = self.intent_model(**inputs)
        logits = output.logits.numpy()
        idx = np.argmax(logits, axis=1)[0]
        confidence = tf.nn.softmax(logits)[0][idx].numpy()
        return self.intent_labels[idx], confidence

    def retrieve_answer(self, query, intent, threshold=0.6):
        if intent not in self.intent_to_qas:
            return None, 0.0
        qa = self.intent_to_qas[intent]
        embeddings = self.intent_to_embeddings[intent]
        query_emb = self.retriever_sbert.encode(query).reshape(1, -1)
        scores = cosine_similarity(query_emb, np.stack(embeddings))[0]
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        return (qa[best_idx]["a"], best_score) if best_score >= threshold else (None, best_score)

    def get_clarification(self, query):
        query_emb = self.clarifier_sbert.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, self.clarification_embeddings, top_k=1)
        return self.clarification_corpus[hits[0][0]["corpus_id"]]

    def respond(self, query, intent_thresh=0.5, sim_thresh=0.6, fallback_thresh=0.4):
        intent, intent_conf = self.predict_intent(query)

        if intent_conf < fallback_thresh:
            return self.FALLBACK

        if intent_conf < intent_thresh or intent not in self.intent_to_qas:
            return self.get_clarification(query)

        answer, sim_score = self.retrieve_answer(query, intent, threshold=sim_thresh)
        return answer if answer else self.get_clarification(query)

# Initialize chatbot engine
bot = ChatbotEngine()

def chat_interface(message, history):
    try:
        response = bot.respond(message)
    except Exception as e:
        response = f"⚠️ Internal error: {str(e)}"
    return response



gr.ChatInterface(
    fn=chat_interface,
    title="🎓 EduBot - Student Help Assistant",
    description="Ask me about exams, deadlines, modules, or login issues!",
    theme="default",
    textbox=gr.Textbox(placeholder="Type your question here... 👇"),
    chatbot=gr.Chatbot()
).launch()
