#!/usr/bin/env python3
"""
3Blue1Brown-Style Video Generator using Google's Gemini API
Generates explanations, supporting images, narration audio, and a final video.
"""

import os
import json
import time
import urllib.request
import streamlit as st
import google.generativeai as genai
import requests
import base64
from io import BytesIO
from PIL import Image
from gtts import gTTS  # Google Text-to-Speech
from moviepy.editor import (
    ImageSequenceClip, TextClip, CompositeVideoClip,
    concatenate_videoclips, AudioFileClip, ColorClip, ImageClip
)
import math
import re

# Set page configuration
st.set_page_config(
    page_title="3Blue1Brown Video Generator (Gemini)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gemini API base URL for direct REST calls
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Fix PIL compatibility issue with ANTIALIAS
import PIL
if not hasattr(PIL.Image, 'ANTIALIAS'):
    # For Pillow >= 9.0.0
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS if hasattr(PIL.Image, 'LANCZOS') else PIL.Image.Resampling.LANCZOS

def setup_gemini(api_key):
    """Configure the Gemini API and return the model instance."""
    genai.configure(api_key=api_key)
    
    # Store the API key directly in the function result for REST API calls
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    # Add the API key as an attribute so we can extract it later for REST calls
    model.api_key = api_key
    return model

def generate_script(model, topic, complexity, duration):
    """Generate an educational script using Gemini."""
    prompt = f"""Create a 3Blue1Brown-style educational script explaining {topic}.

This explanation should be:
- Approximately {duration} minutes long
- At complexity level {complexity}/5 (where 1 is beginner, 5 is advanced)
- Focused on building intuition first, formalism second
- Structured into 5-7 clear sections

For each section, include:
1. A clear section title
2. Narration script for that section (100-150 words per section)
3. Description of what should be visualized

Format your response as a structured JSON with this format:
{{
  "title": "Main title of the explanation",
  "sections": [
    {{
      "section_title": "Section title",
      "narration": "What should be narrated in this section...",
      "visualization": "Description of what should be visualized..."
    }},
    ...more sections...
  ]
}}
"""
    try:
        response = model.generate_content(prompt)
        content = response.text

        # Extract JSON if wrapped in backticks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
                
        return json.loads(content)
    except json.JSONDecodeError:
        st.error("Failed to parse response as JSON. Showing raw response instead.")
        return {"title": topic, "raw_content": content, "sections": []}
    except Exception as e:
        st.error(f"Error generating content: {str(e)}")
        return None

def generate_image_for_section(model, section, topic, aspect_focus=None, image_index=0):
    """Generate an image for a section using direct Gemini REST API with retries.
    
    Parameters:
    - model: The Gemini model
    - section: The section data
    - topic: The overall topic
    - aspect_focus: A specific aspect to focus on in the visualization
    - image_index: Index of the image for this section, used for varied prompts
    """
    # Get API key from model (either from the client_options or our custom attribute)
    api_key = getattr(model, 'api_key', None)
    if not api_key:
        try:
            api_key = model._client_options.api_key  # Extract API key from the model
        except:
            st.error("Could not extract API key from model for REST API calls")
            return None
    
    model_name = "models/gemini-2.0-flash-exp"
    endpoint = f"{GEMINI_API_BASE_URL}/{model_name}:generateContent?key={api_key}"

    # Create aspect-specific prompts based on the index
    aspect_text = ""
    if aspect_focus:
        aspect_text = f" Focus specifically on {aspect_focus}."
    
    # Different prompt styles for each image index to maximize variety
    if image_index == 0:
        # First image - conceptual overview
        prompt_variations = [
            # Standard prompt - simple and direct
            f"""Create a COMPLETELY TEXT-FREE educational illustration in the style of 3Blue1Brown for:
Topic: {topic}
Section: "{section['section_title']}"
Visualization needs: {section['visualization']}{aspect_text}

!!CRITICAL INSTRUCTION!!
* DO NOT INCLUDE ANY TEXT WHATSOEVER in the image
* NO LABELS, NO TITLES, NO CAPTIONS, NO WORDS AT ALL
* I will add all necessary labels and text afterwards
* Focus on pure visual storytelling with shapes, colors, and graphics

Image style:
- Dark navy background (#162238)
- Vibrant, clear visual elements using 3Blue1Brown style
- Clean geometric shapes with sharp edges
- Use color coding and visual positioning to distinguish elements
- Focus on visual clarity - I'll add the text later""",

            # Detailed prompt - emphasizes the style more precisely
            f"""Generate a TEXT-FREE conceptual overview diagram for a 3Blue1Brown-style video.
Topic: {topic}
Section title: "{section['section_title']}"
What to visualize: {section['visualization']}{aspect_text}

IMPORTANT: Generate the image WITHOUT ANY TEXT ELEMENTS. I will add all necessary text later.

Style requirements:
- Dark navy background (like #162238)
- Clean geometric shapes with crisp edges
- Vibrant highlighting colors (teal #11ACCD, yellow #FF9F1C)
- NO TEXT WHATSOEVER - create a purely visual representation
- Use positioning, size and color to show relationships instead of labels
- Create a composition that leaves space for labels to be added later"""
        ]
    elif image_index == 1:
        # Second image - detailed visualization
        prompt_variations = [
            # Detailed visualization prompt
            f"""Create a TEXT-FREE detailed technical illustration in the style of 3Blue1Brown showing:
Topic: {topic}
Section: "{section['section_title']}"
Visualization needs: {section['visualization']}{aspect_text}

!!CRITICAL INSTRUCTION!!
* DO NOT INCLUDE ANY TEXT AT ALL in the image
* NO LABELS, NO TITLES, NO CAPTIONS, NO WORDS
* I will add all text in post-processing
* Focus only on visual representation

Image style:
- Dark background with precise, clear details
- Focus on mathematical or technical accuracy
- Use color strategically to highlight important elements
- Clean, sharp visuals with professional appearance
- Leave appropriate space for labels to be added later""",

            # Technical prompt - with specific formatting guidance
            f"""Create a TEXT-FREE technical diagram for an educational video on {topic}.
Section: "{section['section_title']}"
This image will accompany narration explaining: {section['visualization']}{aspect_text}

IMPORTANT: DO NOT INCLUDE ANY TEXT OR LABELS IN THE IMAGE. I will add all text myself.

Technical requirements:
1. Resolution: 1280x720px
2. Style: Minimalist, clean 3Blue1Brown aesthetic
3. Background: Dark navy or black (#162238 or similar)
4. Use clear, simple shapes and visual representations
5. Use positioning and visual hierarchy to show relationships
6. Create clear focal points where text labels will be added later"""
        ]
    else:
        # Third image - application or example
        prompt_variations = [
            # Application-focused prompt
            f"""Create a TEXT-FREE educational illustration showing a practical application or example of:
Topic: {topic}
Section: "{section['section_title']}"
Visualization needs: {section['visualization']}{aspect_text}

!!CRITICAL INSTRUCTION!!
* DO NOT INCLUDE ANY TEXT in the image
* NO WORDS, NO LABELS, NO ANNOTATIONS of any kind
* I will add all text elements in post-processing
* Focus on clear visual storytelling only

Image style:
- Dark blue background in the 3Blue1Brown style
- Focus on showing how this concept applies in practice
- Use vibrant colors to highlight key components
- Clean, professional visual design with sharp edges
- Leave appropriate blank areas where I can add labels later""",
            
            # Example-driven prompt
            f"""Generate a TEXT-FREE concrete example visualization for a 3Blue1Brown-style video.
Topic: {topic}
Section title: "{section['section_title']}"
What to visualize: {section['visualization']}{aspect_text}

IMPORTANT: DO NOT INCLUDE ANY TEXT OR LABELS IN THE IMAGE. Text will be added separately.

Style guidelines:
- Dark navy background
- Show a specific instance or application of the concept
- Use vibrant colors to highlight key elements
- NO TEXT of any kind - focus only on the visual elements
- Create a composition with space for labels to be added later
- Prioritize visual clarity and storytelling through graphics only"""
        ]
    
    # Number of retry attempts
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Choose prompt based on the current attempt
            prompt_index = min(attempt, len(prompt_variations) - 1)
            prompt_text = prompt_variations[prompt_index]
            
            st.write(f"Sending image generation request (attempt {attempt+1}/{max_retries})...")
            
            # Add progressive delay between retries to avoid rate limiting
            if attempt > 0:
                delay_seconds = 3 * attempt  # 3 seconds for first retry, 6 for second, etc.
                st.write(f"Waiting {delay_seconds} seconds before retry...")
                time.sleep(delay_seconds)
            
            body = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt_text}
                        ]
                    }
                ],
                "generationConfig": {
                    "response_modalities": ["TEXT", "IMAGE"],
                    "temperature": 0.2,  # Lower temperature for more precise outputs
                }
            }

            resp = requests.post(endpoint, json=body, timeout=180)  # Increased timeout
            resp.raise_for_status()
            data = resp.json()
            
            if "candidates" not in data or not data["candidates"]:
                st.warning(f"No candidates returned (attempt {attempt+1}). Trying again with different prompt...")
                time.sleep(2)  # Wait before retry
                continue

            parts = data["candidates"][0].get("content", {}).get("parts", [])
            
            for part in parts:
                inline_data = part.get("inline_data") or part.get("inlineData")
                if inline_data:
                    base64_data = inline_data.get("data")
                    if base64_data:
                        st.success(f"Image generated successfully on attempt {attempt+1}!")
                        return base64_data
            
            st.warning(f"Response didn't contain image data (attempt {attempt+1}). Trying again...")
            time.sleep(2)  # Wait before retry
                
        except Exception as e:
            last_error = e
            st.warning(f"Error on attempt {attempt+1}: {str(e)}. Retrying...")
            # Longer delay if we got a rate limit error (429)
            if "429" in str(e):
                delay_seconds = 10 * (attempt + 1)  # 10s, 20s, 30s
                st.warning(f"Rate limit hit. Waiting {delay_seconds} seconds before retry...")
                time.sleep(delay_seconds)
            else:
                time.sleep(3 * (attempt + 1))  # Progressive delay for other errors
    
    # If we're here, all attempts failed
    st.error(f"Failed to generate image after {max_retries} attempts. Last error: {str(last_error)}")
    
    # Generate a fallback image instead of returning None
    return generate_fallback_image(section["section_title"], topic, image_index=image_index)

def generate_fallback_image(title, topic, image_index=0):
    """Generate a text-based fallback image when API image generation fails.
    
    Parameters:
    - title: The section title
    - topic: The overall topic
    - image_index: Which image in the sequence (affects design/colors)
    """
    try:
        # Create a simple image with the section title and topic
        width, height = 1280, 720
        
        # Select different background colors based on image index for variety
        bg_colors = [
            (22, 28, 36),  # Dark blue
            (26, 32, 44),  # Darker blue
            (32, 22, 44)   # Dark purple
        ]
        bg_color = bg_colors[min(image_index, len(bg_colors)-1)]
        
        # Select different accent colors based on image index
        accent_colors = [
            (17, 172, 205),  # Light blue
            (255, 159, 28),  # Orange
            (120, 220, 120)  # Green
        ]
        accent_color = accent_colors[min(image_index, len(accent_colors)-1)]
        
        # Create a new image with dark background
        img = Image.new('RGB', (width, height), color=bg_color)
        
        # Get a drawing context
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font, fall back to default if not available
        try:
            # Try to use Arial or a system font
            title_font = ImageFont.truetype("Arial", 60)
            subtitle_font = ImageFont.truetype("Arial", 40)
        except:
            # If the specific font is not available, use default
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
        
        # Add suffix based on image index for subtitles
        subtitles = [
            "Concept overview",
            "Detailed visualization", 
            "Practical application"
        ]
        image_subtitle = subtitles[min(image_index, len(subtitles)-1)]
        
        # Draw section title
        title_text = title
        subtitle_text = f"{image_subtitle} • {topic}"
        
        # Get text size and position it in the center
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        
        subtitle_bbox = draw.textbbox((0, 0), subtitle_text, font=subtitle_font)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        
        # Position text in the center
        title_position = ((width - title_width) // 2, (height - title_height) // 2 - 50)
        subtitle_position = ((width - subtitle_width) // 2, (height - title_height) // 2 + 50)
        
        # Draw decorative elements based on image index
        if image_index == 0:
            # First image - horizontal line
            line_y = (height - title_height) // 2 + 20
            draw.rectangle([width//4, line_y, 3*width//4, line_y+2], fill=accent_color)
            
            # Circle with dot
            circle_radius = 60
            circle_x, circle_y = width//2, height//2 + 150
            draw.ellipse((circle_x-circle_radius, circle_y-circle_radius, 
                        circle_x+circle_radius, circle_y+circle_radius), 
                        outline=accent_color, width=3)
            draw.ellipse((circle_x-5, circle_y-5, circle_x+5, circle_y+5), 
                        fill=(255, 255, 255))
        
        elif image_index == 1:
            # Second image - grid pattern
            grid_spacing = 60
            for x in range(0, width+1, grid_spacing):
                # Draw vertical lines
                line_intensity = max(20, int(255 * (1 - abs(x - width/2) / (width/2)) * 0.3))
                line_color = (line_intensity, line_intensity, line_intensity)
                draw.line([(x, 0), (x, height)], fill=line_color, width=1)
            
            for y in range(0, height+1, grid_spacing):
                # Draw horizontal lines
                line_intensity = max(20, int(255 * (1 - abs(y - height/2) / (height/2)) * 0.3))
                line_color = (line_intensity, line_intensity, line_intensity)
                draw.line([(0, y), (width, y)], fill=line_color, width=1)
            
            # Highlight a specific point
            highlight_x, highlight_y = width//2, height//2 + 100
            draw.ellipse((highlight_x-8, highlight_y-8, highlight_x+8, highlight_y+8), 
                        fill=accent_color)
                        
        else:
            # Third image - interconnected dots
            dots = []
            for i in range(6):
                angle = i * (2 * 3.14159 / 6)
                radius = 150
                x = width//2 + int(radius * (0.8 + 0.2 * (i % 2)) * math.cos(angle))
                y = height//2 + 50 + int(radius * (0.8 + 0.2 * (i % 2)) * math.sin(angle))
                dots.append((x, y))
                draw.ellipse((x-6, y-6, x+6, y+6), fill=accent_color)
            
            # Connect the dots
            for i in range(len(dots)):
                for j in range(i+1, len(dots)):
                    # Vary line intensity based on distance
                    distance = ((dots[i][0] - dots[j][0])**2 + (dots[i][1] - dots[j][1])**2) ** 0.5
                    if distance < radius * 1.5:  # Only connect nearby dots
                        intensity = int(100 * (1 - distance / (radius * 1.5)))
                        line_color = (
                            min(255, accent_color[0] + intensity),
                            min(255, accent_color[1] + intensity),
                            min(255, accent_color[2] + intensity)
                        )
                        draw.line([dots[i], dots[j]], fill=line_color, width=1)
        
        # Draw the text on top
        draw.text(title_position, title_text, font=title_font, fill=(255, 255, 255))
        draw.text(subtitle_position, subtitle_text, font=subtitle_font, fill=(200, 200, 200))
        
        # Save to a bytes buffer
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        
        # Convert to base64 for the API return format
        img_bytes = buffer.getvalue()
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        st.info(f"Generated a fallback image ({image_subtitle}) with title and decorative elements")
        return base64_image
        
    except Exception as e:
        st.error(f"Error generating fallback image: {str(e)}")
        # Return a simple colored square if even the fallback fails
        try:
            # Vary the color based on image index
            colors = [(0, 0, 60), (0, 60, 0), (60, 0, 0)]
            color = colors[min(image_index, len(colors)-1)]
            
            simple_img = Image.new('RGB', (1280, 720), color=color)
            buffer = BytesIO()
            simple_img.save(buffer, format="JPEG")
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except:
            return None

def download_image(url_or_data, output_path):
    """Download an image from a URL or save image data to a local file.
    Returns True if successful, False otherwise."""
    try:
        # Check if we have a URL or direct image data
        if isinstance(url_or_data, str) and (url_or_data.startswith('http://') or url_or_data.startswith('https://')):
            # It's a URL, use urllib to download
            st.write(f"Downloading image from URL: {url_or_data[:50]}...")
            urllib.request.urlretrieve(url_or_data, output_path)
        else:
            # It's direct base64 image data from Gemini API
            if isinstance(url_or_data, str):
                # If it's a base64 string
                try:
                    # Direct base64 data from the REST API
                    st.write("Processing base64 image data...")
                    img_data = base64.b64decode(url_or_data)
                    with open(output_path, 'wb') as f:
                        f.write(img_data)
                except Exception as decode_err:
                    st.error(f"Error decoding base64 data: {decode_err}")
                    return False
            else:
                # Already binary data
                img_data = url_or_data
                with open(output_path, 'wb') as f:
                    f.write(img_data)
        
        # Verify the file was created successfully
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            # Try to open and validate the image
            try:
                with Image.open(output_path) as img:
                    # Get image dimensions
                    width, height = img.size
                    st.write(f"Image dimensions: {width}x{height}")
                    
                    # Check if image is unreasonably small (probably an error)
                    if width < 50 or height < 50:
                        st.warning(f"Image is too small ({width}x{height}). May not be usable.")
                        return False
                    
                    # Check image aspect ratio - should be close to 16:9 for videos
                    aspect = width / height
                    if not (1.4 < aspect < 2.0):  # reasonable range around 16:9 (1.78)
                        st.warning(f"Image has unusual aspect ratio ({aspect:.2f}). Resizing...")
                        # Resize to standard dimensions
                        target_width, target_height = 1280, 720
                        resized = img.resize((target_width, target_height), 
                                            PIL.Image.LANCZOS if hasattr(PIL.Image, 'LANCZOS') else PIL.Image.ANTIALIAS)
                        resized.save(output_path)
                        st.write(f"Resized image to {target_width}x{target_height}")
                    
                    # Ensure the image is RGB mode (not RGBA or grayscale)
                    if img.mode != 'RGB':
                        st.warning(f"Converting image from {img.mode} to RGB mode")
                        rgb_img = img.convert('RGB')
                        rgb_img.save(output_path)
                
                st.write(f"✅ Successfully saved and validated image to {output_path}")
                return True
            except Exception as img_err:
                st.error(f"Saved file is not a valid image: {img_err}")
                return False
        else:
            st.error(f"Image file was not created or is empty: {output_path}")
            return False
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return False

def generate_narration_audio(text, output_path, lang='en', slow=False):
    """Generate an audio file of the narration using Google TTS."""
    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(output_path)
        return True
    except Exception as e:
        st.error(f"Error generating narration audio: {str(e)}")
        return False

def create_video_from_images_and_narration(script, images_dir, audio_dir, output_path):
    """Create a video by combining section images with narration audio."""
    try:
        clips = []
        
        # Debug info
        st.write(f"Creating video from {len(script.get('sections', []))} sections")
        
        for i, section in enumerate(script["sections"]):
            section_id = f"section_{i+1}"
            audio_path = os.path.join(audio_dir, f"section_{i+1}.mp3")
            
            # Check for multiple images
            image_paths = section.get("image_paths", [])
            # For backward compatibility
            if not image_paths and "image_path" in section:
                image_paths = [section["image_path"]]
            
            # Default image path (old format) if none found in either list
            if not image_paths:
                # Try the old naming convention
                old_image_path = os.path.join(images_dir, f"{section_id}.jpg")
                if os.path.exists(old_image_path):
                    image_paths = [old_image_path]
            
            # Debug info
            image_count = len(image_paths)
            audio_exists = os.path.exists(audio_path)
            st.write(f"Section {i+1}: {image_count} images, Audio {'exists' if audio_exists else 'missing'}")
            
            try:
                # Always check for audio first as it's the basis for timing
                if os.path.exists(audio_path):
                    # Load narration audio to get duration
                    audio_clip = AudioFileClip(audio_path)
                    audio_duration = audio_clip.duration
                    
                    if image_paths:
                        # We have multiple images for this section
                        if len(image_paths) > 1:
                            try:
                                # Create a sequence of images with transitions
                                img_clips = []
                                img_count = len(image_paths)
                                # Calculate duration for each image
                                segment_duration = audio_duration / img_count
                                
                                st.write(f"Creating sequence of {img_count} images for section {i+1}")
                                
                                # Process existing images rather than generating new ones
                                for idx, img_path in enumerate(image_paths):
                                    if os.path.exists(img_path):
                                        # Calculate timings for smooth crossfade
                                        # First clip doesn't fade in, last doesn't fade out
                                        fadein_duration = 0.5 if idx > 0 else 0
                                        fadeout_duration = 0.5 if idx < img_count - 1 else 0
                                        
                                        # Adjust duration to account for overlapping fade transitions
                                        adjusted_duration = segment_duration
                                        if idx < img_count - 1:  # Not the last image
                                            adjusted_duration += fadeout_duration / 2
                                        if idx > 0:  # Not the first image
                                            adjusted_duration += fadein_duration / 2
                                        
                                        # Create clip from this image
                                        img = ImageClip(img_path).set_duration(adjusted_duration)
                                        
                                        # Add fades
                                        if fadein_duration > 0:
                                            img = img.fadein(fadein_duration)
                                        if fadeout_duration > 0:
                                            img = img.fadeout(fadeout_duration)
                                        
                                        img_clips.append(img)
                                        st.write(f"Added image {idx+1}/{img_count} to sequence")
                                    else:
                                        st.warning(f"Image file not found: {img_path}")
                                
                                if img_clips:
                                    # Combine the image clips with crossfades
                                    img_sequence = concatenate_videoclips(img_clips, method="compose")
                                    st.write(f"Created image sequence with {len(img_clips)} clips")
                                else:
                                    # Fallback to color clip if no images loaded
                                    img_sequence = ColorClip(size=(1280, 720), color=(22, 28, 32), duration=audio_duration)
                                    st.warning(f"No images loaded for section {i+1}, using ColorClip")
                            except Exception as seq_err:
                                st.error(f"Error creating image sequence: {str(seq_err)}")
                                # Fallback to a single image or color clip
                                if image_paths and os.path.exists(image_paths[0]):
                                    img_sequence = ImageClip(image_paths[0]).set_duration(audio_duration)
                                    st.warning(f"Using single image as fallback for section {i+1}")
                                else:
                                    img_sequence = ColorClip(size=(1280, 720), color=(22, 28, 32), duration=audio_duration)
                                    st.warning(f"Using ColorClip as fallback for section {i+1}")
                        else:
                            # Just one image, use the simpler approach
                            try:
                                img_sequence = ImageClip(image_paths[0]).set_duration(audio_duration)
                                st.write(f"Using single image for section {i+1}")
                            except Exception as img_err:
                                st.warning(f"Could not create image clip for section {i+1}: {img_err}")
                                # Fallback to ColorClip if image processing fails
                                img_sequence = ColorClip(size=(1280, 720), color=(22, 28, 32), duration=audio_duration)
                                st.write(f"Created fallback ColorClip for section {i+1}")
                    else:
                        # No images available, create a colored background
                        img_sequence = ColorClip(size=(1280, 720), color=(22, 28, 32), duration=audio_duration)
                        st.write(f"No images available for section {i+1}, using ColorClip")
                    
                    # Create text for the section title
                    section_title = section["section_title"]
                    text_clip = TextClip(
                        section_title,
                        fontsize=30,
                        color='white',
                        bg_color='rgba(0,0,0,0.5)',
                        font='Arial-Bold',
                        method='caption'
                    ).set_position(('center', 'bottom')).set_duration(audio_duration)
                    
                    # Combine the image sequence and text, add audio
                    video_clip = CompositeVideoClip([img_sequence, text_clip])
                    video_clip = video_clip.set_audio(audio_clip)
                    
                    # Add simple fade transitions between sections
                    if i > 0:
                        video_clip = video_clip.fadein(0.5)
                    video_clip = video_clip.fadeout(0.5)
                    
                    clips.append(video_clip)
                    st.write(f"Successfully added section {i+1} to video")
                    
                elif image_paths:
                    # Image exists but no audio - just use first image
                    try:
                        # Just use a static image for 5 seconds
                        default_duration = 5.0
                        img_clip = ImageClip(image_paths[0]).set_duration(default_duration)
                        text_clip = TextClip(
                            section["section_title"],
                            fontsize=30,
                            color='white',
                            bg_color='rgba(0,0,0,0.5)',
                            font='Arial-Bold',
                            method='caption'
                        ).set_position(('center', 'bottom')).set_duration(default_duration)
                        
                        video_clip = CompositeVideoClip([img_clip, text_clip])
                        clips.append(video_clip)
                        st.write(f"Added section {i+1} to video (image only, no audio)")
                    except Exception as img_only_err:
                        st.error(f"Error processing image-only section {i+1}: {str(img_only_err)}")
                else:
                    # Neither image nor audio available
                    st.warning(f"Skipping section {i+1} - missing both image and audio")
                    # Create a blank section with just text (if we want to include it anyway)
                    default_duration = 3.0
                    color_bg = ColorClip(size=(1280, 720), color=(22, 28, 32), duration=default_duration)
                    missing_text = TextClip(
                        f"Section {i+1}: {section['section_title']}\n(Content unavailable)",
                        fontsize=40,
                        color='white',
                        font='Arial-Bold',
                        method='caption'
                    ).set_position('center').set_duration(default_duration)
                    missing_clip = CompositeVideoClip([color_bg, missing_text])
                    clips.append(missing_clip)
                    
            except Exception as section_err:
                st.error(f"Error processing section {i+1}: {str(section_err)}")
                # Very basic emergency fallback
                try:
                    # Create a simple colored background with error text
                    emergency_duration = 5.0
                    color_bg = ColorClip(size=(1280, 720), color=(40, 0, 0), duration=emergency_duration)
                    error_text = TextClip(
                        f"Section {i+1}: {section['section_title']}\n(Error during processing)",
                        fontsize=30,
                        color='white',
                        font='Arial-Bold',
                        method='caption'
                    ).set_position('center').set_duration(emergency_duration)
                    emergency_clip = CompositeVideoClip([color_bg, error_text])
                    clips.append(emergency_clip)
                    st.write(f"Added section {i+1} to video (emergency fallback)")
                except Exception as e2:
                    st.error(f"Emergency fallback failed for section {i+1}: {str(e2)}")
        
        if clips:
            try:
                # Create an intro clip with the title
                intro_duration = 5
                intro_bg = ColorClip(size=(1280, 720), color=(22, 28, 32), duration=intro_duration)
                title_text = TextClip(
                    script["title"],
                    fontsize=60,
                    color='white',
                    font='Arial-Bold',
                    method='caption'
                ).set_position('center').set_duration(intro_duration)
                intro_clip = CompositeVideoClip([intro_bg, title_text])
                intro_clip = intro_clip.fadeout(1.0)
                
                # Add intro to the beginning of clips
                all_clips = [intro_clip] + clips
                st.write(f"Concatenating {len(all_clips)} clips (intro + {len(clips)} sections)")
                
                # Use method="compose" for smoother transitions
                final_video = concatenate_videoclips(all_clips, method="compose")
                
                st.write(f"Writing video to {output_path}")
                # Use web-optimized encoding configuration to ensure better compatibility with browsers
                try:
                    # First, create a regular MP4 for download
                    final_video.write_videofile(
                        output_path,
                        fps=24,
                        codec='libx264',
                        audio_codec='aac',
                        preset='medium',  # Balance between file size and quality
                        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"],  # Better web compatibility
                        verbose=False
                    )
                    
                    # Also create a web-optimized version for playback
                    web_output_path = output_path.replace('.mp4', '_web.mp4')
                    final_video.write_videofile(
                        web_output_path,
                        fps=24,
                        codec='libx264',
                        audio_codec='aac',
                        preset='fast',  # Faster encoding, potentially smaller file
                        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart", "-vf", "scale=1280:-2"],
                        verbose=False
                    )
                    
                    # Save both paths to session state
                    video_abs_path = os.path.abspath(output_path)
                    web_video_abs_path = os.path.abspath(web_output_path)
                    st.session_state["video_path"] = video_abs_path
                    st.session_state["web_video_path"] = web_video_abs_path
                    
                except Exception as write_err:
                    st.warning(f"Error with optimized encoding: {str(write_err)}. Trying simpler encoding...")
                    # Fallback to simpler encoding
                    try:
                        final_video.write_videofile(
                            output_path,
                            fps=24,
                            codec='libx264',
                            audio_codec='aac'
                        )
                        video_abs_path = os.path.abspath(output_path)
                        st.session_state["video_path"] = video_abs_path
                    except Exception as simple_write_err:
                        st.error(f"Error with simple encoding: {str(simple_write_err)}")
                        return False
                
                st.write("Video creation successful!")
                st.write(f"Saved video to: {st.session_state.get('video_path', output_path)}")
                return True
            except Exception as e:
                st.error(f"Error in final video creation: {str(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
                return False
        else:
            st.error("No video clips were created - nothing to compile")
            return False
    except Exception as e:
        st.error(f"Error creating video: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return False

# Add this method to show images as a gallery view
def display_video_as_image_gallery(script, images_dir):
    """Display a video as a gallery of section images with audio playback."""
    st.subheader("Section-by-Section Gallery View")
    st.write("Here's your video content displayed as individual sections:")
    
    # Display intro title
    st.markdown(f"## {script['title']}")
    st.markdown("---")
    
    # Display each section as an image + audio
    for i, section in enumerate(script.get("sections", [])):
        section_id = f"section_{i+1}"
        audio_path = os.path.join("outputs/audio", f"section_{i+1}.mp3")
        
        st.markdown(f"### {i+1}. {section['section_title']}")
        
        # Get all images for this section
        image_paths = section.get("image_paths", [])
        # For backward compatibility
        if not image_paths and "image_path" in section:
            image_paths = [section["image_path"]]
        
        # Try the old naming convention if no paths found
        if not image_paths:
            old_image_path = os.path.join(images_dir, f"{section_id}.jpg")
            if os.path.exists(old_image_path):
                image_paths = [old_image_path]
                
        # Check for multiple images using the new naming pattern
        if not image_paths:
            # Try to find images with the pattern section_X_img_Y.jpg
            potential_images = []
            for img_idx in range(1, 4):  # Check for up to 3 images
                img_path = os.path.join(images_dir, f"section_{i+1}_img_{img_idx}.jpg")
                if os.path.exists(img_path):
                    potential_images.append(img_path)
            
            if potential_images:
                image_paths = potential_images
        
        # Two columns: image and content
        if image_paths:
            # If multiple images, use a wider column for images
            col1, col2 = st.columns([4, 2]) if len(image_paths) > 1 else st.columns([3, 2])
            
            with col1:
                # Image(s)
                if len(image_paths) > 1:
                    # Multiple images - show in tabs
                    image_tabs = st.tabs([f"Image {j+1}" for j in range(len(image_paths))])
                    for j, (tab, img_path) in enumerate(zip(image_tabs, image_paths)):
                        with tab:
                            if os.path.exists(img_path):
                                st.image(img_path, use_column_width=True)
                                
                                # Image captions
                                subtitles = ["Concept overview", "Detailed visualization", "Practical application"]
                                if j < len(subtitles):
                                    st.caption(f"**{subtitles[j]}**")
                            else:
                                st.warning(f"Image {j+1} not available")
                else:
                    # Single image
                    if os.path.exists(image_paths[0]):
                        st.image(image_paths[0], use_column_width=True)
                    else:
                        st.warning("Image not available for this section")
        else:
            # No images found
            col1, col2 = st.columns([3, 2])
            with col1:
                st.warning("No images available for this section")
        
        with col2:
            # Narration text
            st.markdown("**Narration:**")
            st.markdown(f"_{section['narration']}_")
            
            # Audio player
            if os.path.exists(audio_path):
                st.audio(audio_path)
            else:
                st.warning("Audio not available")
        
        st.markdown("---")

def generate_labels_for_image(model, section, topic, image_index=0):
    """Generate appropriate text labels for an image based on section content."""
    
    prompt = f"""
I have an educational illustration for a 3Blue1Brown-style video with:
Topic: {topic}
Section: "{section['section_title']}"
Visualization: {section['visualization']}

I need 3-5 short, clear text labels for this image.
These labels should highlight the key concepts in this visualization.

For each label:
1. The exact text (1-3 words maximum)
2. A general position (top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right)

Format your response ONLY as a simple Python list of lists, like this exact format:
[
  ["Label 1", "position"],
  ["Label 2", "position"],
  ["Label 3", "position"]
]

DO NOT include any additional text, explanations, or formatting.
Just the list itself in the exact format shown.
"""

    # Add a delay before making the API request to avoid rate limiting
    st.write("Waiting 2 seconds before generating labels (to avoid rate limiting)...")
    time.sleep(2)

    # Default labels for fallback based on image index
    if image_index == 0:
        default_labels = [
            ["Concept Overview", "top-center"],
            ["Key Elements", "middle-right"],
            ["Relationship", "bottom-center"]
        ]
    elif image_index == 1:
        default_labels = [
            ["Technical Detail", "top-center"],
            ["Process", "middle-left"],
            ["Components", "bottom-right"]
        ]
    else:
        default_labels = [
            ["Application", "top-center"],
            ["Real-world Example", "middle-right"],
            ["Outcome", "bottom-center"]
        ]

    try:
        # Try up to 3 times with increasing delays
        max_attempts = 3
        label_text = None
        
        for attempt in range(max_attempts):
            try:
                response = model.generate_content(prompt)
                label_text = response.text
                break  # If successful, break out of retry loop
            except Exception as e:
                if "429" in str(e) and attempt < max_attempts - 1:  # Rate limit error
                    delay = 5 * (attempt + 1)  # 5s, 10s, 15s
                    st.warning(f"Rate limit hit while generating labels. Waiting {delay} seconds...")
                    time.sleep(delay)
                elif attempt < max_attempts - 1:  # Other error, but we can retry
                    st.warning(f"Error generating labels (attempt {attempt+1}): {str(e)}. Retrying...")
                    time.sleep(3)
                else:  # Last attempt failed
                    st.error(f"All attempts to generate labels failed. Using default labels.")
                    return default_labels
        
        if not label_text:
            st.warning("No response received for label generation. Using default labels.")
            return default_labels
            
        # Clean up the response text to extract just the list portion
        # First, check if we can find the list pattern [[ ... ]]
        if "```" in label_text:
            for block in label_text.split("```"):
                if "[" in block and "]" in block:
                    label_text = block.strip()
                    break
            
            # Try to find just the list part by locating outermost brackets
            start_idx = label_text.find("[")
            end_idx = label_text.rfind("]")
            if start_idx >= 0 and end_idx > start_idx:
                label_text = label_text[start_idx:end_idx+1]
        
        # Clean any non-list text and normalize formatting
        label_text = label_text.strip()
        
        # Fix common formatting issues
        label_text = label_text.replace("'", '"')  # Replace single quotes with double quotes
        label_text = re.sub(r'(\w+):', r'"\1":', label_text)  # Convert JSON-style keys to quoted strings

        # Try to parse the string as a Python list
        try:
            import ast
            labels_data = ast.literal_eval(label_text)
            
            # Validate the format
            if not isinstance(labels_data, list):
                st.warning("Response is not a list. Using default labels.")
                return default_labels
            
            for item in labels_data:
                if not isinstance(item, list) or len(item) != 2:
                    st.warning(f"Invalid item format: {item}. Using default labels.")
                    return default_labels
            
            # If we made it here, we have a valid list format
            if not labels_data:  # Empty list
                st.warning("Generated an empty list of labels. Using default labels.")
                return default_labels
                
            return labels_data
            
        except (SyntaxError, ValueError) as e:
            # If parsing fails, try a more manual approach
            st.warning(f"Failed to parse label data: {e}. Attempting manual extraction...")
            
            try:
                # Try to manually extract labels and positions
                manual_labels = []
                # Look for patterns like ["text", "position"] or similar
                pattern = r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]'
                matches = re.findall(pattern, label_text)
                
                if matches:
                    for label, position in matches:
                        manual_labels.append([label, position])
                    
                    if manual_labels:
                        st.info(f"Manually extracted {len(manual_labels)} labels.")
                        return manual_labels
                
                # If all else fails, use default labels
                st.warning("Manual extraction failed. Using default labels.")
                return default_labels
                
            except Exception as manual_err:
                st.error(f"Manual extraction error: {manual_err}. Using default labels.")
                return default_labels
    
    except Exception as e:
        st.error(f"Error generating labels: {str(e)}. Using default labels.")
        return default_labels

def add_clean_text_to_image(image_path, labels):
    """Add high-quality text labels to an image after generation."""
    from PIL import Image, ImageDraw, ImageFont
    import os
    
    try:
        # Open the generated image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Try to use a nice font, with fallbacks for different systems
        font_size = 28
        font = None
        
        # List of font possibilities in preference order
        font_options = [
            "Arial.ttf", 
            "Helvetica.ttf",
            "DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/Library/Fonts/Arial.ttf",  # macOS
            "/System/Library/Fonts/Supplemental/Arial.ttf"  # Alternative macOS path
        ]
        
        # Try each font until one works
        for font_name in font_options:
            try:
                font = ImageFont.truetype(font_name, font_size)
                st.write(f"Using font: {font_name}")
                break
            except:
                continue
        
        # If no font found, use default
        if font is None:
            font = ImageFont.load_default()
            st.write("Using default font")
        
        # Map text positions to pixel coordinates
        position_map = {
            "top-left": (width * 0.1, height * 0.1),
            "top-center": (width * 0.5, height * 0.1),
            "top-right": (width * 0.9, height * 0.1),
            "middle-left": (width * 0.1, height * 0.5),
            "center": (width * 0.5, height * 0.5),
            "middle-right": (width * 0.9, height * 0.5),
            "bottom-left": (width * 0.1, height * 0.9),
            "bottom-center": (width * 0.5, height * 0.9),
            "bottom-right": (width * 0.9, height * 0.9)
        }
        
        # Add each label with an attractive background for readability
        for text, position_desc in labels:
            # Get position coordinates
            position = position_map.get(position_desc.lower(), (width * 0.5, height * 0.5))
            
            # Calculate text size
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Adjust position to center text
            text_x = position[0] - text_width // 2
            text_y = position[1] - text_height // 2
            
            # Draw background with slight padding
            padding = 10
            draw.rectangle(
                [text_x - padding, text_y - padding, 
                 text_x + text_width + padding, text_y + text_height + padding],
                fill=(0, 0, 0, 200),  # Semi-transparent black
                outline=(255, 255, 255, 100),  # Subtle white outline
                width=1
            )
            
            # Draw text
            draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
        
        # Save the modified image
        img.save(image_path)
        return True
    except Exception as e:
        st.error(f"Error adding text to image: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    st.title("3Blue1Brown-Style Video Generator (Gemini)")
    st.markdown("""
    Create educational videos with explanations, visuals, and narration in the style of 3Blue1Brown.
    This app uses Google's Gemini API (model: gemini-2.0-flash-exp) for content and image generation,
    Google TTS for narration, and MoviePy to compile a final video.
    """)
    
    # Initialize session state variables if they don't exist
    if "processing_complete" not in st.session_state:
        st.session_state["processing_complete"] = False
    if "script_generated" not in st.session_state:
        st.session_state["script_generated"] = False
    if "images_audio_generated" not in st.session_state:
        st.session_state["images_audio_generated"] = False
    if "missing_images" not in st.session_state:
        st.session_state["missing_images"] = []
    
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Google Gemini API Key", type="password", key="api_key")
        topic = st.text_input("Topic to Explain", key="topic")
        complexity = st.slider("Complexity Level", 1, 5, 2, help="1 is beginner, 5 is advanced", key="complexity")
        duration = st.slider("Target Duration (minutes)", 5, 30, 10, key="duration")
        voice_speed = st.select_slider(
            "Narration Speed",
            options=["Slow", "Normal", "Fast"],
            value="Normal",
            key="voice_speed"
        )
        
        # Add configuration for multiple images per section
        images_per_section = st.slider(
            "Images per section", 
            min_value=1, 
            max_value=3, 
            value=2, 
            help="Number of different images to generate for each section"
        )
        
        generate_button = st.button("Generate Script & Media")
        
        if generate_button:
            if not api_key:
                st.error("Please enter your Google Gemini API Key")
            elif not topic:
                st.error("Please enter a topic")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create output directories
                os.makedirs("outputs", exist_ok=True)
                os.makedirs("outputs/images", exist_ok=True)
                os.makedirs("outputs/audio", exist_ok=True)
                os.makedirs("outputs/videos", exist_ok=True)
                
                # Store the configuration in session state
                st.session_state["images_per_section"] = images_per_section
                
                # Step 1: Generate Script
                status_text.info("Generating script...")
                model = setup_gemini(api_key)
                script = generate_script(model, topic, complexity, duration)
                if not script:
                    st.error("Script generation failed.")
                    return
                st.session_state["script"] = script
                st.session_state["script_generated"] = True
                progress_bar.progress(25)
                status_text.success("Script generated successfully!")
                
                # Step 2: Generate Images and Narration Audio
                status_text.info("Generating images and narration...")
                total_sections = len(script.get("sections", []))
                successful_images = 0
                fallback_images = 0
                images_per_section = st.session_state.get("images_per_section", 2)  # Default to 2 if not set
                
                for i, section in enumerate(script.get("sections", [])):
                    section_progress = int((i / total_sections) * 25)
                    progress_bar.progress(25 + section_progress)
                    
                    # Add delay between sections to avoid hitting rate limits
                    if i > 0:
                        section_delay = 5  # 5 seconds between sections
                        st.write(f"Waiting {section_delay} seconds before processing next section (to avoid rate limiting)...")
                        time.sleep(section_delay)
                    
                    status_text.info(f"Processing section {i+1}/{total_sections}: {section['section_title']}...")
                    
                    # Generate narration first (more reliable)
                    st.write(f"Generating narration for section {i+1}...")
                    slow_narration = (voice_speed == "Slow")
                    audio_path = os.path.join("outputs/audio", f"section_{i+1}.mp3")
                    if generate_narration_audio(section["narration"], audio_path, slow=slow_narration):
                        section["audio_path"] = audio_path
                        st.write(f"✅ Narration for section {i+1} generated and saved to {audio_path}")
                    else:
                        st.error(f"❌ Failed to generate narration for section {i+1}")
                    
                    # Generate multiple images for this section
                    section["image_paths"] = []  # Create a list to store paths to multiple images
                    
                    for img_idx in range(images_per_section):
                        st.write(f"Generating image {img_idx+1}/{images_per_section} for section {i+1}...")
                        
                        # Add delay between images to avoid rate limiting
                        if img_idx > 0:
                            delay_seconds = 3  # Wait 3 seconds between generating images for the same section
                            st.write(f"Waiting {delay_seconds} seconds before generating next image (to avoid rate limiting)...")
                            time.sleep(delay_seconds)
                        
                        # Different prompts for different images in the same section
                        aspect_prompts = [
                            f"the main concept of {section['section_title']}",
                            f"a detailed visualization of {section['section_title']}",
                            f"an application or example of {section['section_title']}"
                        ]
                        
                        # Select prompt aspect based on image index (or use default for images beyond our prepared aspects)
                        aspect = aspect_prompts[min(img_idx, len(aspect_prompts)-1)]
                        
                        # Generate image with custom aspect focus
                        image_data = generate_image_for_section(model, section, topic, 
                                                                  aspect_focus=aspect, 
                                                                  image_index=img_idx)
                        
                        if image_data:
                            # Save the image locally with index in filename
                            img_filename = f"section_{i+1}_img_{img_idx+1}.jpg"
                            standardized_img = os.path.join("outputs/images", img_filename)
                            st.write(f"Saving image to {standardized_img}...")
                            
                            if download_image(image_data, standardized_img):
                                section["image_paths"].append(standardized_img)
                                
                                # Check if it was a real generated image or a fallback
                                try:
                                    img = Image.open(standardized_img)
                                    # Display preview of the generated image
                                    st.image(img, caption=f"Image {img_idx+1} for section {i+1}", width=300)
                                    st.write(f"✅ Image {img_idx+1} saved successfully for section {i+1}")
                                    successful_images += 1
                                except Exception as img_err:
                                    st.error(f"Error verifying image: {str(img_err)}")
                                    fallback_images += 1
                            else:
                                st.error(f"❌ Failed to save image {img_idx+1} for section {i+1}")
                                fallback_path = os.path.join("outputs/images", img_filename)
                                try:
                                    generate_fallback_image(section["section_title"], topic, image_index=img_idx)
                                    section["image_paths"].append(fallback_path)
                                    st.warning(f"Created emergency fallback image {img_idx+1} for section {i+1}")
                                except:
                                    st.error(f"Could not create fallback image {img_idx+1} for section {i+1}")
                        else:
                            st.error(f"❌ Failed to generate image data {img_idx+1} for section {i+1}")
                    
                    # Ensure we have at least one image path for backwards compatibility
                    if section["image_paths"] and "image_path" not in section:
                        section["image_path"] = section["image_paths"][0]
                    
                    # Add clean text labels to each generated image
                    for img_idx, img_path in enumerate(section["image_paths"]):
                        if os.path.exists(img_path):
                            # Add delay before generating labels for each image
                            if img_idx > 0:
                                label_delay = 3  # 3 seconds between label generations
                                st.write(f"Waiting {label_delay} seconds before generating labels (to avoid rate limiting)...")
                                time.sleep(label_delay)
                                
                            st.write(f"Generating and adding text labels to image {img_idx+1}...")
                            
                            # Generate appropriate labels for this image
                            labels = generate_labels_for_image(model, section, topic, image_index=img_idx)
                            
                            # Add the labels to the image
                            if add_clean_text_to_image(img_path, labels):
                                st.success(f"✅ Added clean text labels to image {img_idx+1}")
                            else:
                                st.warning(f"⚠️ Failed to add text labels to image {img_idx+1}")
                
                # Summary of image generation
                total_expected_images = total_sections * images_per_section
                progress_bar.progress(50)
                if successful_images == total_expected_images:
                    status_text.success(f"✅ Successfully generated all {total_expected_images} images!")
                elif successful_images > 0:
                    status_text.warning(f"Generated {successful_images} real images and {fallback_images} fallback images for {total_expected_images} total expected images")
                else:
                    status_text.error(f"❌ Could not generate any real images. Using {fallback_images} fallback images.")
                
                st.success("Images and narration processing complete!")
                st.session_state["images_audio_generated"] = True
                st.session_state["processing_complete"] = True
    
    # Outside the sidebar - create video button
    if st.session_state.get("images_audio_generated", False):
        # Step 3: Create Final Video - Separate button outside the sidebar
        st.subheader("Video Creation")
        
        # Display information about missing images if any
        if st.session_state.get("missing_images", []):
            st.warning(f"Missing images for sections: {st.session_state['missing_images']}. Video may be incomplete.")
        
        # Create a button to generate the video
        if st.button("Create Final Video"):
            status_text = st.empty()
            progress_bar = st.progress(50)
            status_text.info("Creating final video...")
            
            script = st.session_state["script"]
            topic = st.session_state.get("topic", "unknown_topic")
            
            video_filename = f"{topic.lower().replace(' ', '_')}_3b1b_style.mp4"
            video_path = os.path.join("outputs/videos", video_filename)
            
            # Make sure we have all necessary data in the script
            # No need to pass model or generate new images at this point
            if create_video_from_images_and_narration(script, "outputs/images", "outputs/audio", video_path):
                st.session_state["video_path"] = video_path
                progress_bar.progress(100)
                status_text.success("Video created successfully!")
            else:
                progress_bar.progress(75)
                status_text.error("Failed to create video")
    
    # Display generated script, images, and narration
    if "script" in st.session_state:
        script = st.session_state["script"]
        st.header(script.get("title", "Untitled"))
        for i, section in enumerate(script.get("sections", [])):
            with st.expander(f"Section {i+1}: {section['section_title']}", expanded=(i==0)):
                st.subheader("Narration")
                st.markdown(section["narration"])
                
                # Get all images for this section
                image_paths = section.get("image_paths", [])
                if not image_paths and "image_path" in section and os.path.exists(section["image_path"]):
                    image_paths = [section["image_path"]]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Visualization Description")
                    st.markdown(section["visualization"])
                
                with col2:
                    st.subheader("Generated Images")
                    if image_paths:
                        if len(image_paths) > 1:
                            # Create tabs for multiple images
                            image_tabs = st.tabs([f"Image {j+1}" for j in range(len(image_paths))])
                            for j, (tab, img_path) in enumerate(zip(image_tabs, image_paths)):
                                with tab:
                                    if os.path.exists(img_path):
                                        st.image(img_path)
                                        
                                        # Add image captions based on index
                                        image_types = ["Concept overview", "Detailed visualization", "Example/Application"]
                                        if j < len(image_types):
                                            st.caption(f"**{image_types[j]}**")
                                    else:
                                        st.warning(f"Image {j+1} not available")
                        else:
                            # Single image
                            if os.path.exists(image_paths[0]):
                                st.image(image_paths[0])
                            else:
                                st.warning("Image not available for this section")
                        
                if "audio_path" in section and os.path.exists(section["audio_path"]):
                    st.subheader("Narration Audio")
                    st.audio(section["audio_path"])
                else:
                    st.warning("Audio not available for this section")
    
    # Display final video if available
    if "video_path" in st.session_state and os.path.exists(st.session_state["video_path"]):
        st.header("Final Generated Video")
        
        # Get file size for info
        video_path = os.path.abspath(st.session_state["video_path"])
        video_size_mb = os.path.getsize(video_path) / (1024*1024)
        video_filename = os.path.basename(video_path)
        
        # Format selection tabs
        display_format = st.radio(
            "Choose display format:",
            ["Video Player", "Image Gallery", "HTML5 Player"],
            horizontal=True
        )
        
        if display_format == "Video Player":
            st.info(f"Video file size: {video_size_mb:.2f} MB")
            
            # Check if we have a web-optimized version
            web_video_path = st.session_state.get("web_video_path")
            if web_video_path and os.path.exists(web_video_path):
                display_path = web_video_path
                st.success("Using web-optimized version for playback")
            else:
                display_path = video_path
            
            # Try Streamlit's native video player
            try:
                # Load and display directly from bytes
                with open(display_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                st.success("Video loaded in native Streamlit player")
            except Exception as e:
                st.error(f"Could not display video with Streamlit player: {e}")
                st.warning("Please try the HTML5 Player option or download the video to view it.")
        
        elif display_format == "HTML5 Player":
            st.info(f"Video file size: {video_size_mb:.2f} MB")
            
            # Use the web-optimized version if available
            display_path = st.session_state.get("web_video_path", video_path)
            if not os.path.exists(display_path):
                display_path = video_path
            
            try:
                video_bytes = open(display_path, "rb").read()
                video_b64 = base64.b64encode(video_bytes).decode()
                
                # Enhanced HTML5 video player
                html_code = f"""
                <div style="width:100%; max-width:1000px; margin:0 auto; padding:10px; background:#000; border-radius:5px;">
                    <video width="100%" height="auto" controls playsinline autoplay muted>
                        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    <p style="color:#fff; text-align:center;">If video doesn't play automatically, please click the play button.</p>
                </div>
                """
                st.markdown(html_code, unsafe_allow_html=True)
                st.success("Video loaded in HTML5 player")
            except Exception as html_err:
                st.error(f"HTML5 player failed: {html_err}")
                st.warning("Please try the Image Gallery option or download the video to view it.")
        
        elif display_format == "Image Gallery":
            # Display the video content as individual images
            if "script" in st.session_state:
                display_video_as_image_gallery(st.session_state["script"], "outputs/images")
            else:
                st.error("Script information not available for gallery view.")
        
        # Additional video information
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Video information")
            st.write(f"📁 Filename: {video_filename}")
            st.write(f"📏 Size: {video_size_mb:.2f} MB")
            st.write(f"🔍 Path: {os.path.dirname(video_path)}")
        
        # Always provide a download option
        with col2:
            st.subheader("Download Options")
            try:
                with open(video_path, "rb") as file:
                    video_bytes = file.read()
                    st.download_button(
                        label=f"⬇️ Download MP4 Video",
                        data=video_bytes,
                        file_name=video_filename,
                        mime="video/mp4"
                    )
                st.success(f"Video is ready for download")
                
                # Provide direct link to file
                if os.path.exists(video_path):
                    st.markdown(f"🔗 **[Open video in new tab]({video_path})**")
            except Exception as download_err:
                st.error(f"Could not prepare download button: {download_err}")
                st.info(f"Your video is saved at: {video_path}")
    
    elif "video_path" in st.session_state:
        st.warning(f"Video was generated but file not found at expected path: {st.session_state['video_path']}")
        # Help user find output files
        try:
            video_dir = os.path.dirname(st.session_state.get("video_path", "outputs/videos"))
            if os.path.exists(video_dir):
                st.subheader("Available Files")
                st.write(f"Files in video output directory ({video_dir}):")
                files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.webm'))]
                
                if files:
                    for file in files:
                        file_path = os.path.join(video_dir, file)
                        file_size_mb = os.path.getsize(file_path) / (1024*1024)
                        st.write(f"- {file} ({file_size_mb:.2f} MB)")
                        
                        # Offer to display this file
                        try:
                            st.video(file_path)
                            st.success(f"Displayed alternative video: {file}")
                        except:
                            st.info(f"Could not display {file} with native player")
                else:
                    st.info("No video files found in the output directory.")
            else:
                st.error(f"Output directory does not exist: {video_dir}")
        except Exception as dir_err:
            st.error(f"Error exploring output directory: {dir_err}")
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Python, Gemini, and Streamlit")

if __name__ == "__main__":
    main()