#EDUbot: University FAQ Chatbot
EduBot is an intelligent chatbot designed to assist students with frequently asked university questions. It used a hybrid Nlp pipeline model involving:

- **BERT**  intent classifer
- **SBERT** for answer retrival 
- **Clarification Engine** for fallback handling

## features
- Conversational AI bot
-Handles multiple intent related question( exam, deadlines, accomodation etc.)
-Clarifes ambiguous questions
-respond with relevant answer or fallback prompt
-Fully intractive via gardio web interface

## Technologies used
- `Transformers` (Hugging Face)
- `TensorFlow` for BERT-based intent classification
- `Sentence-Transformers` (SBERT) for semantic search
- `Gradio` for web-based chat UI
- `scikit-learn` for cosine similarity

## Project structure
├── app.py # Main Gradio interface
├── chatbot_pipeline.py # Core logic and response flow
├── intent_dataset.json # FAQ question-answer pairs
├── clarification_question_model/
│ ├── clarification_sbert_model # Fine-tuned SBERT for clarification
│ └── clarifying_questions_main.json
├── fine_tuned_models/
│ └── sbert_finetuned_small # Fine-tuned SBERT for answer retrieval
├── saved_bert_model/ # Fine-tuned BERT intent classifier
├── Requirements.txt # Required dependencies
└── README.md # Project overview
