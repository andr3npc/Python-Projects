import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# --- Step 1: Speech-to-Text using Whisper-1 ---
audio_file_path = "data/sample.wav"

with open(audio_file_path, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

transcription_text = transcription.text
print("Transcription:\n", transcription_text)

# --- Step 2: Translation using GPT-4o-mini ---
target_language = "Spanish"

translation_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": f"You are a professional translator. Translate the following text into {target_language}. Provide only the translated text without any explanations."
        },
        {
            "role": "user",
            "content": transcription_text
        }
    ]
)

translated_text = translation_response.choices[0].message.content
print("\nTranslated Text:\n", translated_text)

# --- Step 3: Grammar Feedback using GPT-4o-mini ---
grammar_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a grammar expert. Analyze the following text for grammatical errors. Provide the corrected version of the text along with explanations of any errors found."
        },
        {
            "role": "user",
            "content": translated_text
        }
    ]
)

grammar_feedback = grammar_response.choices[0].message.content
print("\nGrammar Feedback:\n", grammar_feedback)

# --- Step 4: Pronunciation Feedback using GPT-4o-mini ---
target_text = "The stale smell of old beer lingers."

pronunciation_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a pronunciation coach. Compare the user's spoken transcription with the target text. Identify differences and provide specific suggestions for pronunciation improvement, including which sounds or words need more practice."
        },
        {
            "role": "user",
            "content": f"Original transcription (what the user said): {transcription_text}\n\nTarget text (what they should have said): {target_text}\n\nPlease compare these and provide pronunciation feedback with suggestions for improvement."
        }
    ]
)

pronunciation_feedback = pronunciation_response.choices[0].message.content
print("\nPronunciation Feedback:\n", pronunciation_feedback)
