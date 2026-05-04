import pandas as pd
import json
from openai import OpenAI

# Load the data
df = pd.read_csv("data/transcriptions.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())

# Initialize the OpenAI client
client = OpenAI()

# ---------- Step 1: Extract structured info from each transcription ----------

def extract_medical_info(transcription, medical_specialty):
    """Extract age, recommended treatment, and ICD-10 code from a transcription."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical coding expert. Extract structured information "
                    "from the following medical transcription. Return ONLY a valid JSON "
                    "object with these exact keys:\n"
                    '- "age": the patient\'s age as an integer (use 0 if not mentioned)\n'
                    '- "medical_specialty": the medical specialty as a string\n'
                    '- "recommended_treatment": a concise description of the recommended treatment\n'
                    '- "icd_10_code": the most appropriate ICD-10 code for the primary diagnosis\n'
                    "Return ONLY the JSON object, no markdown, no explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Medical specialty: {medical_specialty}\n\n"
                    f"Transcription:\n{transcription}"
                ),
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# Process each row
results = []
for idx, row in df.iterrows():
    print(f"Processing row {idx + 1}/{len(df)}...")
    try:
        raw = extract_medical_info(row["transcription"], row["medical_specialty"])
        # Clean potential markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()

        parsed = json.loads(raw)
        results.append(
            {
                "age": parsed.get("age", 0),
                "medical_specialty": parsed.get(
                    "medical_specialty", row["medical_specialty"]
                ),
                "recommended_treatment": parsed.get("recommended_treatment", "N/A"),
                "icd_10_code": parsed.get("icd_10_code", "N/A"),
            }
        )
    except Exception as e:
        print(f"  Error on row {idx}: {e}")
        results.append(
            {
                "age": 0,
                "medical_specialty": row["medical_specialty"],
                "recommended_treatment": "N/A",
                "icd_10_code": "N/A",
            }
        )

# ---------- Step 2: Build the structured DataFrame ----------
df_structured = pd.DataFrame(results)

print("\n--- df_structured ---")
print(df_structured.head())
print(f"\nShape: {df_structured.shape}")
print(f"Columns: {list(df_structured.columns)}")
