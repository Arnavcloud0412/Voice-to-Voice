import gradio as gr
import assemblyai as aai
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import uuid
from pathlib import Path
import os

def voice_to_voice(audio_file):
    
    transciption_response = audio_transcription(audio_file)

    if transciption_response.status == aai.TranscriptStatus.error:
        raise gr.Error(transciption_response.error)

    else:
        text = transciption_response.text

    es_translation, ru_translation, de_translation = text_translation(text)

    es_audio_path = text_to_speech(es_translation)
    ru_audio_path = text_to_speech(ru_translation)
    de_audio_path = text_to_speech(de_translation)

    es_path = Path(es_audio_path)
    ru_path = Path(ru_audio_path)
    de_path = Path(de_audio_path)

    return es_path, ru_path, de_path

def audio_transcription(audio_file):
    
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    transcribe = aai.Transcriber()  
    transcription = transcribe.transcribe(audio_file)

    return transcription

def text_translation(text):
    
    translator_es = Translator(from_lang="en" , to_lang="es")
    es_text = translator_es.translate(text)

    translator_ru = Translator(from_lang="en" , to_lang="ru")
    ru_text = translator_ru.translate(text)

    translator_de = Translator(from_lang="en" , to_lang="de")
    de_text = translator_de.translate(text)

    return es_text, ru_text, de_text

def text_to_speech(text: str) -> str:
    elevenlabs = ElevenLabs(
        api_key = os.getenv("ELEVENLABS_API_KEY")
    )

    response = elevenlabs.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB", # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2", # Multilingual model
        # Optional voice settings that allow you to customize the output
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )
    save_file_path = f"{uuid.uuid4()}.mp3"
    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    print(f"{save_file_path}: A new audio file was saved successfully!")
    # Return the path of the saved audio file
    return save_file_path
        

audio_input = gr.Audio(
    sources=["microphone", "upload"],
    type="filepath"
)

demo = gr.Interface(
    fn=voice_to_voice,
    inputs= audio_input,
    outputs=[gr.Audio(label="Spanish"), gr.Audio(label="Russian"), gr.Audio(label="German")]
)

if __name__ == "__main__":
    demo.launch()