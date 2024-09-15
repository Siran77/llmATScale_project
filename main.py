import streamlit as st
import os
import requests
import google.generativeai as genai
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor
from gtts import gTTS
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_audioclips
import numpy as np
from IPython.display import display
from IPython.display import Markdown
import textwrap

# def to_markdown(text):
#   text = text.replace('•', '  *')
#   return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
def generate_story(items, target_duration=60):
    prompt3 = "Create a cohesive story based on the following plot points:\n" + "\n".join(items)+"shorten a story for a minute video"
    response4 = model.generate_content([prompt3], stream=True)
    response4.resolve()
    t=response4.text
    return t
# Generate text-to-speech and save audio
def generate_audio_from_text(text, output_audio_path):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(output_audio_path)
        return True
    except Exception as e:
        print(f"Failed to generate audio: {e}")
        return False


# Configure Google Generative AI
os.environ['GOOGLE_API_KEY'] = "AIzaSyB3tcyKxjewTaeNZOHEb6AXM9Pfpw5m6I4"
genai.configure(api_key="AIzaSyB3tcyKxjewTaeNZOHEb6AXM9Pfpw5m6I4")
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-1.5-flash')
# Streamlit app UI
st.title("AI-Powered Image Prompt Generation & Video Creation")

# Step 1: Input Proverb and Theme
st.header("Enter a Proverb")
prompt=[]
prompt.append(st.text_input("Enter the proverb"))
st.header("Enter a Theme")
prompt.append(st.text_input("Enter the Theme"))
video_duration = st.slider("Video Duration (seconds)", 10, 120, 60)
items=[]
if st.button("Generate Image Prompts"):
    if prompt:
        if len(prompt) == 2:
            prompt1 = (
    "Generate 20 distinct image prompts that visually represent the concept of '"
    + prompt[0]
    + "' in a single, continuous story. Each prompt should depict a different scene that builds upon the previous one, creating a cohesive narrative that emphasizes the central message of '"
    + prompt[1]
    + "'. Ensure that each prompt is formatted as a numbered list with a brief, vivid description of the scene, like so:\n\n"
    "## 20 Image Prompts for '"
    + prompt[1]
    + "':\n\n"
    "Scene description.\n"
    "Scene description.\n"
    "Scene description.\n"
    "...\n"
    "Scene description.\n\n"
    "The story should flow naturally from one scene to the next, with each image prompt contributing to the overarching narrative and illustrating how '"
    + prompt[1]
    + "' is developed and demonstrated throughout the story."
)
            st.write("Generating prompts...")
            response = model.generate_content([prompt1], stream=True)
            response_text = response.resolve()
            # Display the generated text
            st.text_area("Generated Image Prompts", response_text, height=300)
            text=response.text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            items = [line[line.find("**") + 2: line.rfind("**")].strip() for line in lines if '**' in line]
            # for i in items:
            #     st.write(i)
            st.session_state['items'] = items

        else:
            st.error("Please enter the input in thse format: Proverb - Theme")
    else:
        st.error("Please enter a proverb and theme.")

# Step 2: Generate Images from Prompts
if 'items' in st.session_state and st.session_state['items']:
    st.header("Generate Images from Prompts")
    if st.button("Generate Images"):
        API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
        headers = {"Authorization": "Bearer s"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.content

        def generate_and_save_image(prompt, file_name):
            image_bytes = query({"inputs": prompt})
            image = Image.open(io.BytesIO(image_bytes))
            image.save(os.path.join("images", file_name))
            return file_name

        st.write("Generating images...")
        os.makedirs("images", exist_ok=True)
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_image = {executor.submit(generate_and_save_image, scenario, f"image_{i+1}.png"): i for i, scenario in enumerate(st.session_state['items'])}

        st.success("Images generated successfully!")

# # Step 3: Create Video with Audio
if os.path.exists("images"):
    st.header("Create a Video with Audio")
    generated_story = generate_story(items)
    # text_to_speech=to_markdown(response4.text)
    # text_to_speech = st.text_area("Text for Speech Synthesis (will be used as video narration)")

    if st.button("Create Video"):
        image_folder = "images"
        audio_path = "temp_audio.mp3"
        video_path = "output_video_Final12.mp4"
        # Generate audio from text
        try:
            st.write("Generating audio...")
            tts = gTTS(text=generated_story, lang='en')
            tts.save(audio_path)
        except Exception as e:
            st.error(f"Failed to generate audio: {e}")

        # Get list of images
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()

        if not image_files:
            st.error("No images found in the specified folder.")
        else:
            st.write(f"Found {len(image_files)} images.")

        # Convert PIL Images to numpy arrays
        try:
            image_clips = [np.array(Image.open(img).convert('RGB')) for img in image_files]
        except Exception as e:
            st.error(f"Failed to process images: {e}")

        # Calculate FPS and create video
        fps = len(image_clips) / video_duration
        video_clip = ImageSequenceClip(image_clips, fps=fps)

        # Load audio and ensure it's long enough
        def extend_audio(audio_clip, target_duration):
            audio_duration = audio_clip.duration
            if audio_duration >= target_duration:
                return audio_clip.subclip(0, target_duration)
            else:
                loops = int(np.ceil(target_duration / audio_duration))
                extended_audio = concatenate_audioclips([audio_clip] * loops)
                return extended_audio.subclip(0, target_duration)

        try:
            st.write("Loading audio...")
            audio_clip = AudioFileClip(audio_path)
            audio_clip = extend_audio(audio_clip, video_duration)
            video_clip = video_clip.set_audio(audio_clip)
        except Exception as e:
            st.error(f"Failed to load or set audio: {e}")

        # Export final video
        try:
            st.write("Creating video...")
            video_clip.write_videofile(video_path, codec='libx264', audio_codec='aac')
            st.video(video_path)
        except Exception as e:
            st.error(f"Failed to write video file: {e}")
