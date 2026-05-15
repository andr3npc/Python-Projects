// !pip install SpeechRecognition
// !pip install pydub
// !pip install spacy
// !python3 -m spacy download en_core_web_sm

# Import required libraries
// import pandas as pd

// import nltk
// nltk.download('vader_lexicon')
// from nltk.sentiment.vader import SentimentIntensityAnalyzer

// import speech_recognition as sr
// from pydub import AudioSegment

// import spacy

# Start coding here

# ============================================================
# Task 1: Convert audio to text, get frame rate and channels
# ============================================================
audio = AudioSegment.from_wav("sample_customer_call.wav")
frame_rate = audio.frame_rate
number_channels = audio.channels

recognizer = sr.Recognizer()
with sr.AudioFile("sample_customer_call.wav") as source:
    audio_data = recognizer.record(source)
    transcribed_text = recognizer.recognize_google(audio_data)

print(f"Frame rate: {frame_rate}")
print(f"Number of channels: {number_channels}")
print(f"Transcribed text: {transcribed_text}")

# ============================================================
# Task 2: Sentiment analysis - count true positives
# ============================================================
df = pd.read_csv("customer_call_transcriptions.csv")

sia = SentimentIntensityAnalyzer()
df['compound'] = df['text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
df['predicted'] = df['compound'].apply(
    lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
)

true_positive = int(((df['predicted'] == 'positive') & (df['sentiment_label'] == 'positive')).sum())
print(f"True positives: {true_positive}")

# ============================================================
# Task 3: Most frequent named entity
# ============================================================
nlp = spacy.load("en_core_web_sm")

all_entities = []
for text_val in df['text']:
    doc = nlp(str(text_val))
    for ent in doc.ents:
        all_entities.append(ent.text)

from collections import Counter
entity_counts = Counter(all_entities)
most_freq_ent = entity_counts.most_common(1)[0][0]
print(f"Most frequent entity: {most_freq_ent}")

# ============================================================
# Task 4: Most similar call to "wrong package delivery"
# ============================================================
query = "wrong package delivery"
query_doc = nlp(query)

best_similarity = -1
most_similar_text = ""

for text_val in df['text']:
    doc = nlp(str(text_val))
    similarity = query_doc.similarity(doc)
    if similarity > best_similarity:
        best_similarity = similarity
        most_similar_text = str(text_val)

print(f"Most similar text: {most_similar_text}")
