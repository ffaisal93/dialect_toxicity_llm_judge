import os
import json
import pickle
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from tqdm import tqdm

PROJECT_ID = "genuine-haiku-438514-f4"

# Set up the environment variable for Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'key/stt_key.json'

# Function to transcribe audio using an existing recognizer
def transcribe_reuse_recognizer(audio_file: str, recognizer_id: str) -> cloud_speech.RecognizeResponse:
    """Transcribe an audio file using an existing recognizer.
    Args:
        audio_file (str): Path to the local audio file to be transcribed.
            Example: "resources/audio.wav"
        recognizer_id (str): The ID of the existing recognizer to be used for transcription.
    Returns:
        cloud_speech.RecognizeResponse: The response containing the transcription results.
    """
    # Instantiates a client
    client = SpeechClient()

    # Reads a file as bytes
    with open(audio_file, "rb") as f:
        audio_content = f.read()

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/global/recognizers/{recognizer_id}",
        content=audio_content,
    )

    # Transcribes the audio into text
    response = client.recognize(request=request)

    return response

# Iterate through recordings and collect transcriptions
dhaka_transcriptions = []
recordings_dir = 'data/ben_stt/recording_dhaka'
recognizer_id = 'bn-rec'

for i in tqdm(range(1, 381), desc="Processing recordings"):
    audio_file = os.path.join(recordings_dir, f"Recording {i}.wav")
    try:
        response = transcribe_reuse_recognizer(audio_file, recognizer_id)
        transcription = response.results[0].alternatives[0].transcript if response.results else ''
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        transcription = ''
    dhaka_transcriptions.append(transcription)

# Load the standard Bengali sentences from the pickle file
with open('data/nllb_toxigen_test/ben_Beng.pkl', 'rb') as pkl_file:
    standard_sentences = pickle.load(pkl_file)

# Create the final dictionary
bengali_dict = {
    'dhaka': dhaka_transcriptions,
    'standard': standard_sentences
}

# Save the dictionary as a JSON file
output_path = 'data/processed_data/bengali.json'
with open(output_path, 'w') as json_file:
    json.dump(bengali_dict, json_file, ensure_ascii=False, indent=4)

print(f"Bengali data saved to {output_path}")