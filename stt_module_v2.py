import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import wave
import os

class SpeechToText:
    def __init__(self, model_name="small", device="cuda", language="tr", beam_size=10):
        """
        Initialize the Speech-to-Text system with specified Whisper model, device, language, and beam size for accuracy.
        :param model_name: The model to load (default is 'small' for faster performance with reasonable accuracy)
        :param device: Device to run the model on ('cuda' for GPU, 'cpu' for CPU)
        :param language: The language for transcription (default is 'tr' for Turkish)
        :param beam_size: Beam size for decoding (higher value for better accuracy)
        """
        # Use the specified model with float32 precision for increased accuracy
        self.model = WhisperModel(model_name, device=device, compute_type="float32")
        self.language = language
        self.is_recording = False
        self.recorded_audio = []
        self.sample_rate = 16000  # Sampling rate in Hz
        self.beam_size = beam_size  # Set higher beam size for improved accuracy

    def toggle_recording(self):
        """Toggle the recording state."""
        if not self.is_recording:
            print("Recording started... Press 'S' to stop.")
            self.recorded_audio = []
            self.is_recording = True
        else:
            print("Recording stopped.")
            self.is_recording = False

    def audio_callback(self, indata, frames, time, status):
        """Callback function to collect audio data while recording."""
        if self.is_recording:
            self.recorded_audio.append(indata.copy())

    def save_audio_to_wav(self, filename="output.wav"):
        """Save the recorded audio to a WAV file."""
        audio_data = np.concatenate(self.recorded_audio)
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono channel
            wav_file.setsampwidth(2)  # Sample width in bytes
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.astype(np.int16).tobytes())

    def transcribe_audio(self, audio_file, output_directory):
        """
        Transcribe the audio file using the Whisper model, print transcription and time in console,
        and save the transcription as a .txt file.

        :param audio_file: The path to the audio file to transcribe.
        :param output_directory: The directory where the transcription will be saved as a .txt file.
        """
        print("Transcribing audio...")

        # Perform transcription with a higher beam size for improved accuracy
        segments, _ = self.model.transcribe(audio_file, language=self.language, beam_size=self.beam_size)
        transcription = "".join([segment.text for segment in segments])

        # Print transcription and transcription time to console
        print(f"Transcription:\n{transcription}")

        # Prepare the output path
        output_path = os.path.join(output_directory, "transcription_output.txt")

        # Save the transcription as a text file
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(transcription)

        print(f"Transcription saved to {output_path}")

        return transcription
