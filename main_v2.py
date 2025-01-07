import keyboard
import sounddevice as sd
from stt_module_v2 import SpeechToText  # Importing the updated SpeechToText class

def main():
    # Initialize the SpeechToText system with a larger beam size for accuracy
    stt_system = SpeechToText(model_name="small", beam_size=10)  # Adjust beam_size here if needed

    # Set the hotkey to toggle recording
    keyboard.add_hotkey('s', stt_system.toggle_recording)

    # Specify the directory where you want to save the transcription
    output_directory = r"C:\Users\kutay\Documents"  # Change this to your desired path

    # Start the audio input stream
    with sd.InputStream(callback=stt_system.audio_callback, channels=1, samplerate=stt_system.sample_rate,
                        dtype='int16'):
        print("Press 'S' to start recording. Press 'S' again to stop.")

        while True:
            # Once recording is stopped, save and transcribe audio
            if not stt_system.is_recording and stt_system.recorded_audio:
                stt_system.save_audio_to_wav("output.wav")
                stt_system.transcribe_audio("output.wav", output_directory)
                stt_system.recorded_audio = []  # Reset for the next recording

                print("Press 'S' to start recording. Press 'S' again to stop.")

            # Exit the loop when 'esc' is pressed
            if keyboard.is_pressed('esc'):
                print("Exiting the program...")
                break

if __name__ == "__main__":
    main()
