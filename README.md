# 3Blue1Brown-Style Video Generator
A Streamlit application that automatically generates educational videos in the style of 3Blue1Brown using Google's Gemini AI. This tool creates complete educational content with script generation, visualizations, narration, and final video compilation.

✨ Features
AI-Powered Script Generation: Creates structured educational scripts with sections and visualization descriptions
Text-Free Visualization Creation: Generates custom visuals for each section with high-quality labels added programmatically
Automatic Narration: Converts script text to natural-sounding audio narration
Multiple Images Per Section: Creates varied visualizations for each section to enhance engagement
Complete Video Assembly: Combines images, narration, and transitions into a professional final video
Various Display Options: View content as video, gallery, or interactive HTML5 player
Rate Limiting Protection: Built-in safeguards to prevent API throttling

📋 Requirements
Python 3.8+
Google Gemini API key (Pro or higher tier recommended)
ffmpeg (for video processing)
Streamlit

🚀 Installation
# Clone repository
git clone https://github.com/username/3b1b-video-generator.git
cd 3b1b-video-generator

# Create conda environment
conda create -n 3b1b-env python=3.9
conda activate 3b1b-env

# Install required packages
pip install -r requirements.txt

# Install ffmpeg
conda install -c conda-forge ffmpeg

Requirements.txt contents:

streamlit>=1.30.0

google-generativeai>=0.3.0

pillow>=9.0.0

gtts>=2.3.2

moviepy>=1.0.3

requests>=2.28.0




streamlit run gemini3.py


 Open the app in your browser (usually at http://localhost:8501)
Enter your Google Gemini API key in the sidebar
Configure the video settings:
Topic to explain
Complexity level (1-5)
Target duration (in minutes)
Narration speed
Images per section (1-3)
Click "Generate Script & Media" to create the content
Once images and narration are generated, click "Create Final Video"
View the final video using the built-in player or download it for sharing

This tool requires a Google Gemini API key. You can obtain one by signing up at https://ai.google.dev/.
