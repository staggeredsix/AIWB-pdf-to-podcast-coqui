

import gradio as gr
import uuid
import os
import json
import requests
from typing import List

def add_research_tab():
    """Add Research-to-Podcast tab with UI matching existing Gradio interface."""
    
    gr.Markdown("### Generate a podcast directly from web research on any topic.")
    
    with gr.Row():
        topic = gr.Textbox(
            label="Research Topic", 
            placeholder="Enter the topic you want to research",
            info="The main subject to research and create a podcast about"
        )
        
        focus = gr.Textbox(
            label="Focus Area (Optional)",
            placeholder="Specific aspect to focus on",
            info="E.g., 'environmental impact' for electric vehicles topic"
        )
    
    with gr.Row():
        settings = gr.CheckboxGroup(
            ["Monologue Only"], 
            label="Additional Settings", 
            info="Customize your podcast here"
        )
        
        depth = gr.Dropdown(
            ["shallow", "medium", "deep"],
            label="Research Depth",
            value="medium",
            info="How deeply to research the topic"
        )
    
    with gr.Row():
        speaker_1 = gr.Textbox(
            label="Host Name",
            value="Host",
            info="Name for the podcast host"
        )
        
        speaker_2 = gr.Textbox(
            label="Guest Name",
            value="Guest",
            info="Name for the podcast guest (ignored if Monologue Only is selected)"
        )
    
    duration = gr.Slider(
        minimum=1,
        maximum=15,
        value=5,
        step=1,
        label="Duration (minutes)"
    )
    
    with gr.Accordion("Optional: Email Details", open=False):
        gr.Markdown("Enter a recipient email here to receive your generated podcast in your inbox! \n\n**Note**: Ensure `SENDER_EMAIL` and `SENDER_EMAIL_PASSWORD` are configured in AI Workbench")
        recipient_email = gr.Textbox(
            label="Recipient email", 
            placeholder="Enter email here"
        )
                 
    generate_button = gr.Button("Generate Research Podcast")
    
    # Define the function to handle research podcast generation
    def generate_research_podcast(
        topic: str,
        focus: str,
        settings: List[str],
        depth: str,
        speaker_1: str,
        speaker_2: str,
        duration: int,
        recipient_email: str
    ):
        if not topic or len(topic.strip()) == 0:
            gr.Warning("Research topic is required. Please enter a topic and try again.")
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        # Check if monologue mode is selected
        monologue = True if "Monologue Only" in settings else False
        
        # Validate email configuration if provided
        sender_email = os.environ["SENDER_EMAIL"] if "SENDER_EMAIL" in os.environ else None
        sender_validation = validate_sender(sender_email)
        
        if not sender_validation and len(recipient_email) > 0:
            gr.Warning("SENDER_EMAIL not detected or malformed. Please fix or remove recipient email and try again. You may need to restart the container for Environment Variable changes to take effect.")
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        elif sender_validation and len(recipient_email) > 0 and "SENDER_EMAIL_PASSWORD" not in os.environ:
            gr.Warning("SENDER_EMAIL_PASSWORD not detected. Please fix or remove recipient email and try again. You may need to restart the container for Environment Variable changes to take effect.")
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        filename = str(uuid.uuid4())
        
        # Email will be used as filename prefix if provided and valid
        email = [recipient_email] if (sender_validation and len(recipient_email) > 0 and "SENDER_EMAIL_PASSWORD" in os.environ) else [filename + "@"]  # delimiter
        
        # Get API service URL
        base_url = os.environ["API_SERVICE_URL"]
        
        try:
            # Build research request
            research_request = {
                "topic": topic,
                "job_id": job_id,
                "style": "monologue" if monologue else "conversational",
                "duration": duration,
                "depth": depth,
                "focus": focus if focus else None,
                "speaker_1_name": speaker_1,
                "speaker_2_name": speaker_2 if not monologue else None,
                "userId": "test-userid"  # Match existing user ID pattern
            }
            
            print(f"Submitting research request: {research_request}")
            
            # Submit research request to API
            response = requests.post(
                f"{base_url}/research/generate",
                json=research_request
            )
            
            if response.status_code != 202:
                gr.Warning(f"Error starting research: {response.text}")
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            # Get job ID from response
            result = response.json()
            job_id = result.get("job_id")
            
            print(f"Research job submitted with ID: {job_id}")
            
            # Format is consistent with the existing email_demo.test_api function
            from frontend.utils import email_demo
            
            # Generate podcast using the research data
            # Note: This reuses the existing test_api function with modifications for research
            job_id = email_demo.test_api(base_url, [], [], email, monologue, False, job_id=job_id, research_mode=True)
            
            # Send email if configured
            if sender_validation and len(recipient_email) > 0 and "SENDER_EMAIL_PASSWORD" in os.environ:
                email_demo.send_file_via_email(
                    f"/project/frontend/demo_outputs/{recipient_email.split('@')[0]}-output.mp3",
                    sender_email,
                    recipient_email
                )
                return gr.update(
                    value=f"/project/frontend/demo_outputs/{recipient_email.split('@')[0]}-output.mp3",
                    label="podcast audio",
                    visible=True
                ), gr.update(
                    value=get_transcript(recipient_email.split('@')[0], job_id),
                    label="podcast transcript",
                    visible=True
                ), gr.update(
                    value=get_history(recipient_email.split('@')[0], job_id),
                    label="generation history",
                    visible=True
                )
            
            # Default return for locally saved files
            return gr.update(
                value=f"/project/frontend/demo_outputs/{filename}-output.mp3",
                label="podcast audio",
                visible=True
            ), gr.update(
                value=get_transcript(filename, job_id),
                label="podcast transcript",
                visible=True
            ), gr.update(
                value=get_history(filename, job_id),
                label="generation history",
                visible=True
            )
            
        except Exception as e:
            print(f"Error generating research podcast: {e}")
            gr.Warning(f"Error generating podcast: {str(e)}")
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    # Connect button to function
    generate_button.click(
        generate_research_podcast,
        inputs=[
            topic,
            focus,
            settings,
            depth,
            speaker_1,
            speaker_2,
            duration,
            recipient_email
        ],
        outputs=[
            gr.File(visible=False), 
            gr.File(visible=False), 
            gr.File(visible=False)
        ]
    )

# Helper functions from existing code
def validate_sender(sender):
    """Validate email format (reused from existing code)."""
    if sender is None:
        return False
    import re
    regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(regex, sender))

def get_transcript(filename, job_id):
    """Get transcript JSON (reused from existing code)."""
    service = os.environ["API_SERVICE_URL"]
    url = f"{service}/saved_podcast/{job_id}/transcript"
    params = {"userId": "test-userid"}
    filepath = f"/project/frontend/demo_outputs/transcript_{filename}.json"
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        json_data = response.json()
        with open(filepath, "w") as file:
            json.dump(json_data, file)
        print(f"JSON data saved to {filepath}")
        return filepath
    else:
        print(f"Error retrieving transcript: {response.status_code}")
        return filepath

def get_history(filename, job_id):
    """Get generation history JSON (reused from existing code)."""
    service = os.environ["API_SERVICE_URL"]
    url = f"{service}/saved_podcast/{job_id}/history"
    params = {"userId": "test-userid"}
    filepath = f"/project/frontend/demo_outputs/generation_history_{filename}.json"
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        json_data = response.json()
        with open(filepath, "w") as file:
            json.dump(json_data, file)
        print(f"JSON data saved to {filepath}")
        return filepath
    else:
        print(f"Error retrieving generation_history: {response.status_code}")
        return filepath
