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
import torchaudio.transforms as T

model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

def reduce_noise(audio_path):
    audio, rate = librosa.load(audio_path, sr=None)

    # Perform noise reduction
    noisy_part = audio[0:int(rate*0.5)]  # Identify the noisy part
    reduced_noise_audio = nr.reduce_noise(y=audio, sr=rate, y_noise=noisy_part)

    sf.write(audio_path, reduced_noise_audio, rate)

def extract_features(waveform, sample_rate, n_mfcc=13):
    # Compute standard Spectrogram
    spectrogram_transform = T.Spectrogram()
    spectrogram = spectrogram_transform(waveform)

    # Compute Mel Spectrogram
    mel_spectrogram_transform = T.MelSpectrogram(sample_rate=sample_rate)
    mel_spectrogram = mel_spectrogram_transform(waveform)

    # Compute MFCCs from Mel Spectrogram
    mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23})
    mfccs = mfcc_transform(waveform)

    return spectrogram, mel_spectrogram, mfccs

# Define the path to the directory
data_dir = Path("data")

# Read data from a jsonl file and reformat it
data = {'key': [], 'audio': [], 'transcript': []}
with jsonlines.open(data_dir / "asr.jsonl") as reader:
    for obj in reader:
        if len(data['key']) < 20 :  # Only keep the first 20 entries
            for key, value in obj.items():
                data[key].append(value)

# Convert to a Hugging Face dataset
dataset = Dataset.from_dict(data)

# Shuffle the dataset
dataset = dataset.shuffle(seed=42)

# Split the dataset into training, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, train_size + val_size))
test_dataset = dataset.select(range(train_size + val_size, train_size + val_size + test_size))


# Initially freeze all layers except the classifier layer
for param in model.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    param.requires_grad = True

# Function to load and preprocess audio
def preprocess_data(examples):
    input_values = []
    attention_masks = []
    labels = []
    clean_audio_dir = Path("cleaned_data")
    for audio_path, transcript in zip(examples['audio'], examples['transcript']):
        speech_array, sampling_rate = torchaudio.load(data_dir / audio_path)
        processed = processor(speech_array.squeeze(0), sampling_rate=sampling_rate, return_tensors="pt", padding=True)

        # Process labels with the same processor settings
        with processor.as_target_processor():
            label = processor(transcript, return_tensors="pt", padding=True)

        input_values.append(processed.input_values.squeeze(0))
        # Create attention masks based on the input values
        attention_mask = torch.ones_like(processed.input_values)
        attention_mask[processed.input_values == processor.tokenizer.pad_token_id] = 0  # Set padding tokens to 0
        attention_masks.append(attention_mask.squeeze(0))

        # Ensure labels are padded to the same length as inputs if needed
        padded_label = torch.full(processed.input_values.shape[1:], -100, dtype=torch.long)
        actual_length = label.input_ids.shape[1]
        padded_label[:actual_length] = label.input_ids.squeeze(0)
        labels.append(padded_label)

    # Concatenate all batches
    examples['input_values'] = torch.stack(input_values)
    examples['attention_mask'] = torch.stack(attention_masks)
    examples['labels'] = torch.stack(labels)

    return examples

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_data, batched=True, batch_size=1, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess_data, batched=True, batch_size=1, remove_columns=val_dataset.column_names)
test_dataset = test_dataset.map(preprocess_data, batched=True, batch_size=1, remove_columns=test_dataset.column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=1e-4,
    per_device_train_batch_size=1,  # Reduce to one for simplicity
    num_train_epochs=10,
    weight_decay=0.005,
    save_steps=500,
    eval_steps=500,
    logging_steps=10,
    load_best_model_at_end=True
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Use the validation dataset for evaluation
    tokenizer=processor.feature_extractor
)

# Train the model
trainer.train()

# The output directory where the best model is saved
best_model_dir = os.path.join(training_args.output_dir, "best_model")

os.makedirs(best_model_dir, exist_ok=True)

# Save the best model
model.save_pretrained(best_model_dir)
processor.save_pretrained(best_model_dir)