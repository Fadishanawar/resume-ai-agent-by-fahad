import os
import json
import cohere
import gradio as gr
import pdfkit
import base64
from io import BytesIO
from PIL import Image
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("⚠️ COHERE_API_KEY missing in .env!")

# Initialize Cohere client
co = cohere.Client(cohere_api_key)

# Configure wkhtmltopdf path
wkhtmltopdf_path = os.getenv("WKHTMLTOPDF_PATH", r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
pdf_config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

# Set up Jinja2 environment and output folder
env = Environment(loader=FileSystemLoader("templates"))
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Convert image array to base64 string
def image_to_base64(img_array):
    if img_array is None:
        return ""
    image = Image.fromarray(img_array.astype("uint8"))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# Generate summary and skills from job description
def generate_summary_and_skills(name, job_description):
    summary_prompt = f"Write a 2-3 sentence professional summary for {name} based on this job description:\n{job_description}"
    skills_prompt = f"List 5 relevant skills for the following job:\n{job_description}"

    summary = co.generate(prompt=summary_prompt, max_tokens=150).generations[0].text.strip()
    skills_raw = co.generate(prompt=skills_prompt, max_tokens=60).generations[0].text.strip()
    skills = [s.strip() for s in skills_raw.replace("\n", ",").split(",") if s.strip()]
    return summary, skills

# Main function to generate resume
def build_resume(full_name, email, phone, address,
                 education, experience, job_description,
                 languages, references, resume_title,
                 template_name, profile_image_array):

    try:
        # Parse structured input
        education_list = json.loads(education)
        experience_list = json.loads(experience)
        references_list = json.loads(references)
        language_list = [lang.strip() for lang in languages.split(",") if lang.strip()]

        # Generate AI-powered content
        summary, skills = generate_summary_and_skills(full_name, job_description)
        profile_image_data = image_to_base64(profile_image_array)

        # Load and render the template
        template = env.get_template(template_name)
        html_content = template.render(
            full_name=full_name,
            email=email,
            phone=phone,
            address=address,
            education=education_list,
            experience=experience_list,
            references=references_list,
            languages=language_list,
            resume_title=resume_title,
            summary=summary,
            skills=skills,
            profile_image=profile_image_data
        )

        # Save and convert to PDF
        html_path = os.path.join(output_dir, "resume.html")
        pdf_path = os.path.join(output_dir, "resume.pdf")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        pdfkit.from_file(html_path, pdf_path, configuration=pdf_config)

        return html_path, pdf_path

    except json.JSONDecodeError:
        return "❌ Invalid JSON in education, experience, or references section.", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# Launch Gradio interface
demo = gr.Interface(
    fn=build_resume,
    inputs=[
        gr.Textbox(label="Full Name"),
        gr.Textbox(label="Email"),
        gr.Textbox(label="Phone"),
        gr.Textbox(label="Address"),
        gr.Textbox(label="Education (JSON list)", lines=4, placeholder='[{"degree": "BS CS", "institution": "ABC University", "year": "2023"}]'),
        gr.Textbox(label="Experience (JSON list)", lines=4, placeholder='[{"position": "Developer", "company": "XYZ Inc", "years": "2"}]'),
        gr.Textbox(label="Job Description", lines=4),
        gr.Textbox(label="Languages (comma-separated)", placeholder="English, Urdu, French"),
        gr.Textbox(label="References (JSON list)", lines=3, placeholder='[{"name": "John Doe", "contact": "john@example.com"}]'),
        gr.Textbox(label="Resume Title", placeholder="Software Engineer Resume"),
        gr.Dropdown(choices=os.listdir("templates"), label="Select Template"),
        gr.Image(type="numpy", label="Upload Profile Image (optional)")
    ],
    outputs=[
        gr.File(label="📄 Download HTML Resume"),
        gr.File(label="📄 Download PDF Resume")
    ],
    title="🧠 AI Resume Builder with Profile Image Support",
    description="Create an AI-generated resume from your inputs. Choose a template and optionally upload your image."
)

if __name__ == "__main__":
    demo.launch()
