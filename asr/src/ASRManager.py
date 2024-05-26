import jsonlines
import torchaudio
from datasets import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from pathlib import Path
import torch
import librosa
import IPython.display as ipd
import jiwer
import noisereduce as nr
import soundfile as sf
import os

class ASRManager:
    def __init__(self):
        # initialize the model here
        model_name = 'results/best_model'
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
    

    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription
        # Load the saved best model

        # Convert raw audio bytes to a NumPy array
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        audio_input, sample_rate = librosa.load(audio_path, sr=16000)
        # print(f"Loaded audio: {audio_input.shape}, Sampling rate: {sample_rate}")
        input_values = self.processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
        # print(f"Input values: {input_values.shape}")

        with torch.no_grad():
            logits = self.model(input_values).logits
            print(f"Logits: {logits.shape}")
            print("Logits tensor:", logits)

        predicted_ids = torch.argmax(logits, dim=-1)
        print(f"Predicted IDs: {predicted_ids}")
        transcription = self.processor.batch_decode(predicted_ids)[0]

        # print("Transcription:", transcription)

        return transcription
