from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import base64
import logging
import textwrap
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from anthropic import Anthropic
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize the Anthropic client
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Domains configuration
DOMAINS = ["healthcare", "psychology", "education", "undersea"]

# Domain-specific prompt templates
PROMPT_TEMPLATES = {
    "healthcare": {
        "base": """Analyze this medical image with high precision. Please structure your analysis in the following format:

1. OBJECT IDENTIFICATION
- List all medical objects, structures, or conditions visible in the image
- For each object, specify its approximate location (use terms like 'top-left', 'center', 'bottom-right', etc.)
- Indicate confidence level for each identification

2. DETAILED DESCRIPTION
- Size and characteristics of identified structures
- Notable medical features or anomalies
- Tissue characteristics and patterns
- Quality and clarity of the image

3. MEDICAL CONTEXT
- Type of medical imaging used (if apparent)
- Anatomical region shown
- Relevant medical context
- Image quality and positioning

4. TECHNICAL ASSESSMENT
- Medical significance of observed features
- Potential diagnostic indicators
- Anatomical relationships
- Quality factors affecting interpretation

5. CLINICAL CONSIDERATIONS
- Potential clinical implications
- Areas requiring attention
- Recommended follow-up if applicable

6. ANNOTATIONS
Please provide coordinates for detected objects in this format:
[Object1]: type, confidence_level, location_description
[Object2]: type, confidence_level, location_description
etc.

7. ADDITIONAL OBSERVATIONS
- Any other relevant medical details
- Image quality considerations
- Technical factors affecting interpretation

Please be as specific and detailed as possible in your medical analysis."""
    },
    "psychology": {
        "base": """Analyze this psychology-related image with high precision. Please structure your analysis in the following format:

1. OBJECT IDENTIFICATION
- List all relevant objects, expressions, or behavioral indicators in the image
- For each element, specify its location (use terms like 'top-left', 'center', 'bottom-right', etc.)
- Indicate confidence level for each identification

2. DETAILED DESCRIPTION
- Facial expressions and body language
- Environmental elements
- Interpersonal dynamics
- Visual patterns and arrangements

3. PSYCHOLOGICAL CONTEXT
- Setting and atmosphere
- Social dynamics present
- Environmental factors
- Mood and emotional tone

4. TECHNICAL ASSESSMENT
- Psychological significance of observed elements
- Behavioral patterns
- Interactive elements
- Environmental psychology factors

5. BEHAVIORAL CONSIDERATIONS
- Potential psychological implications
- Notable behavioral patterns
- Relevant psychological factors

6. ANNOTATIONS
Please provide coordinates for detected elements in this format:
[Element1]: type, confidence_level, location_description
[Element2]: type, confidence_level, location_description
etc.

7. ADDITIONAL OBSERVATIONS
- Contextual factors
- Cultural considerations
- Other relevant psychological elements

Please be as specific and detailed as possible in your psychological analysis."""
    },
    "education": {
        "base": """Analyze this educational content with high precision. Please structure your analysis in the following format:

1. OBJECT IDENTIFICATION
- List all educational elements, materials, or content visible
- For each element, specify its location (use terms like 'top-left', 'center', 'bottom-right', etc.)
- Indicate confidence level for each identification

2. DETAILED DESCRIPTION
- Content organization and layout
- Educational materials present
- Teaching tools and resources
- Visual aids and illustrations

3. EDUCATIONAL CONTEXT
- Subject matter and topic area
- Target educational level
- Learning objectives
- Pedagogical approach

4. TECHNICAL ASSESSMENT
- Educational effectiveness
- Teaching methodology
- Learning support elements
- Instructional design elements

5. PEDAGOGICAL CONSIDERATIONS
- Learning accessibility
- Student engagement factors
- Teaching effectiveness
- Areas for improvement

6. ANNOTATIONS
Please provide coordinates for detected elements in this format:
[Element1]: type, confidence_level, location_description
[Element2]: type, confidence_level, location_description
etc.

7. ADDITIONAL OBSERVATIONS
- Pedagogical effectiveness
- Accessibility considerations
- Educational best practices

Please be as specific and detailed as possible in your educational analysis."""
    },
    "undersea": {
        "base": """Analyze this underwater image with high precision. Please structure your analysis in the following format:

1. OBJECT IDENTIFICATION
- List all objects detected in the image
- For each object, specify its approximate location (use terms like 'top-left', 'center', 'bottom-right', etc.)
- Indicate confidence level for each identification

2. DETAILED DESCRIPTION
- Size and scale of objects
- Physical characteristics
- Notable features or markings
- Condition or state of objects

3. ENVIRONMENTAL CONTEXT
- Water conditions and visibility
- Depth indicators if apparent
- Surrounding elements or features
- Lighting conditions

4. TECHNICAL ASSESSMENT
- If military/industrial equipment: specifications, type, potential purpose
- If marine life: species, behavior, significance
- If natural phenomenon: type, characteristics, implications

5. SAFETY CONSIDERATIONS
- Potential hazards or risks
- Required precautions
- Safety recommendations

6. ANNOTATIONS
Please provide coordinates for detected objects in this format:
[Object1]: type, confidence_level, location_description
[Object2]: type, confidence_level, location_description
etc.

7. ADDITIONAL OBSERVATIONS
- Any other relevant details
- Unusual features or anomalies
- Historical or operational context if applicable

Please be as specific and detailed as possible, particularly when identifying military equipment, marine hazards, or significant marine phenomena."""
    }
}

def detect_domain(image_path):
    """Detect the appropriate domain for the image"""
    try:
        with open(image_path, 'rb') as img:
            image_data = img.read()
            base64_image = base64.b64encode(image_data).decode()
            
        prompt = """Please analyze this image and determine which single category it best fits into. Respond ONLY with one of these exact words:
- healthcare
- psychology
- education
- undersea

Choose the most appropriate category based on these criteria:
- healthcare: medical images, clinical photos, anatomical diagrams, health-related content
- psychology: behavioral studies, emotional expressions, psychological tests, therapy settings
- education: learning materials, classroom content, educational diagrams, teaching resources
- undersea: marine life, underwater equipment, oceanic phenomena, submarines, aquatic scenes

Respond with just the category name in lowercase, nothing else."""

        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=50,
            temperature=0,  # Added for more deterministic responses
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )
        
        detected_domain = response.content[0].text.strip().lower()
        logger.debug(f"Raw detected domain: {detected_domain}")
        
        # Clean up the response
        detected_domain = detected_domain.split()[0] if detected_domain else 'undersea'
        detected_domain = detected_domain.strip('.:!?')  # Remove any punctuation
        
        # Validate that it's one of our expected domains
        valid_domains = {'healthcare', 'psychology', 'education', 'undersea'}
        if detected_domain not in valid_domains:
            logger.warning(f"Unexpected domain detected: {detected_domain}, falling back to best match")
            # Map similar terms to valid domains
            domain_mapping = {
                'medical': 'healthcare',
                'health': 'healthcare',
                'clinical': 'healthcare',
                'psychological': 'psychology',
                'behavioral': 'psychology',
                'educational': 'education',
                'academic': 'education',
                'learning': 'education',
                'underwater': 'undersea',
                'marine': 'undersea',
                'aquatic': 'undersea'
            }
            detected_domain = domain_mapping.get(detected_domain, 'undersea')
        
        logger.debug(f"Final detected domain: {detected_domain}")
        return detected_domain
        
    except Exception as e:
        logger.error(f"Error in domain detection: {str(e)}")
        return 'undersea'  # Default fallback

def analyze_image_with_claude(image_path, domain):
    """Perform AI analysis using Claude"""
    try:
        logger.debug(f"Starting analysis for domain: {domain}")
        logger.debug(f"Image path: {image_path}")
        
        with open(image_path, 'rb') as img:
            image_data = img.read()
            base64_image = base64.b64encode(image_data).decode()
            
        logger.debug("Successfully read and encoded image")
        
        prompt = PROMPT_TEMPLATES[domain]["base"]
        logger.debug(f"Using prompt template for domain: {domain}")
        
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )
        
        logger.debug("Successfully received response from Claude")
        return response.content[0].text

    except Exception as e:
        logger.error(f"Error in analyze_image_with_claude: {str(e)}")
        raise

def parse_claude_response(response_text):
    """Parse Claude's response to extract structured data and annotations"""
    annotations = []
    analysis_sections = {}
    
    try:
        sections = response_text.split('\n\n')
        
        for section in sections:
            if '1. OBJECT IDENTIFICATION' in section:
                objects = section.split('\n-')[1:]
                for obj in objects:
                    if obj.strip():
                        obj_info = obj.strip().lower()
                        
                        confidence = 0.9 if 'high confidence' in obj_info else 0.7
                        
                        location_terms = {
                            'top-center': [0.3, 0.1, 0.7, 0.4],
                            'top-left': [0.1, 0.1, 0.4, 0.4],
                            'top-right': [0.6, 0.1, 0.9, 0.4],
                            'center': [0.3, 0.3, 0.7, 0.7],
                            'bottom-left': [0.1, 0.6, 0.4, 0.9],
                            'bottom-right': [0.6, 0.6, 0.9, 0.9],
                            'bottom-center': [0.3, 0.6, 0.7, 0.9]
                        }
                        
                        bbox = [0.3, 0.3, 0.7, 0.7]  # Default center
                        for term, coords in location_terms.items():
                            if term in obj_info:
                                bbox = coords
                                break
                        
                        label = obj_info.split(',')[0].strip()
                        label = label.split('(')[0].strip()
                        
                        annotations.append({
                            "label": label,
                            "confidence": confidence,
                            "bbox": bbox
                        })
            
            elif '2. DETAILED DESCRIPTION' in section:
                analysis_sections['detailed_description'] = section
            elif '3. ENVIRONMENTAL CONTEXT' in section or '3. MEDICAL CONTEXT' in section or '3. PSYCHOLOGICAL CONTEXT' in section or '3. EDUCATIONAL CONTEXT' in section:
                analysis_sections['context'] = section
            elif '4. TECHNICAL ASSESSMENT' in section:
                analysis_sections['technical_assessment'] = section
            elif '5. SAFETY CONSIDERATIONS' in section or '5. CLINICAL CONSIDERATIONS' in section or '5. BEHAVIORAL CONSIDERATIONS' in section or '5. PEDAGOGICAL CONSIDERATIONS' in section:
                analysis_sections['considerations'] = section
            elif '7. ADDITIONAL OBSERVATIONS' in section:
                analysis_sections['additional_observations'] = section
    
    except Exception as e:
        logger.error(f"Error parsing Claude response: {str(e)}")
    
    return annotations, analysis_sections

def annotate_image(image_path, annotations):
    """Annotate image with labels and bounding boxes"""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        colors = {
            'high': (255, 0, 0),    # Red
            'medium': (255, 165, 0), # Orange
            'low': (255, 255, 0)     # Yellow
        }
        
        for ann in annotations:
            bbox = ann["bbox"]
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
            
            color = colors['high'] if ann['confidence'] >= 0.8 else colors['medium']
            
            # Draw thick bounding box
            for i in range(4):
                draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color, width=2)
            
            # Prepare label text
            label = f"{ann['label'].title()} ({ann['confidence']:.0%})"
            
            # Calculate text size and background
            text_bbox = draw.textbbox((x1, y1-25), label, font=font)
            
            # Draw label background
            draw.rectangle(text_bbox, fill=color)
            
            # Draw white text
            draw.text((x1, y1-25), label, fill=(255, 255, 255), font=font)
        
        annotated_path = os.path.join('temp', f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        img.save(annotated_path, 'JPEG', quality=95)
        return annotated_path
    
    except Exception as e:
        logger.error(f"Error annotating image: {str(e)}")
        return image_path

def generate_pdf_report(analysis_results, analysis_sections, domain, image_path, annotations, annotated_path=None):
    """Generate PDF report with analysis results and annotated image"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{domain}_{timestamp}.pdf"
        filepath = os.path.join('reports', filename)
        
        c = canvas.Canvas(filepath)
        page_width = 595.27  # A4 width in points
        margin = 100  # Left margin
        text_width = page_width - (margin * 2)  # Available width for text
        
        # Starting position
        y = 750
        
        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y, f"Domain-Specific Image Analysis Report - {domain.title()}")
        y -= 30
        
        # Add timestamp
        c.setFont("Helvetica", 10)
        c.drawString(margin, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 40
        
        # Add original and annotated images side by side
        if os.path.exists(image_path) and os.path.exists(annotated_path):
            img_width = 220
            img_height = 160
            c.drawImage(ImageReader(image_path), margin, y - img_height, width=img_width, height=img_height)
            c.drawString(margin, y - img_height - 20, "Original Image")
            
            c.drawImage(ImageReader(annotated_path), margin + img_width + 30, y - img_height, width=img_width, height=img_height)
            c.drawString(margin + img_width + 30, y - img_height - 20, "Annotated Image")
            
            y -= (img_height + 40)
        
        # Add sections with improved text wrapping
        for section_title, content in analysis_sections.items():
            if y < 100:
                c.showPage()
                y = 750
                c.setFont("Helvetica", 10)  # Reset font after new page
            
            c.setFont("Helvetica-Bold", 12)
            title = section_title.replace('_', ' ').title()
            c.drawString(margin, y, title)
            y -= 20
            
            c.setFont("Helvetica", 10)
            
            # Process content line by line
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Wrap text to fit page width
                wrapped_lines = textwrap.fill(line, width=70).split('\n')
                
                for wrapped_line in wrapped_lines:
                    if y < 100:
                        c.showPage()
                        y = 750
                        c.setFont("Helvetica", 10)
                    
                    if wrapped_line.startswith('-'):
                        c.drawString(margin + 20, y, wrapped_line)
                    else:
                        c.drawString(margin, y, wrapped_line)
                    
                    y -= 15
            
            y -= 20  # Space between sections
        
        c.save()
        return filename
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        raise

@app.route('/reports/<path:filename>')
def download_report(filename):
    try:
        return send_from_directory('reports', filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        return jsonify({'error': 'Report not found'}), 404

@app.route('/analyze', methods=['POST'])
def analyze():
    temp_files = []
    try:
        logger.info("Starting analysis request")
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image = request.files['image']
        
        os.makedirs('temp', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        image_path = os.path.join('temp', f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        image.save(image_path)
        temp_files.append(image_path)
        
        # Detect the appropriate domain
        domain = detect_domain(image_path)
        logger.debug(f"Detected domain: {domain}")
        
        # Get domain-specific analysis
        analysis_results = analyze_image_with_claude(image_path, domain)
        
        annotations, analysis_sections = parse_claude_response(analysis_results)
        
        annotated_path = annotate_image(image_path, annotations)
        if annotated_path != image_path:
            temp_files.append(annotated_path)
        
        report_path = generate_pdf_report(
            analysis_results,
            analysis_sections,
            domain,
            image_path,
            annotations,
            annotated_path
        )
        
        # Clean up temporary files after PDF is generated
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Error cleaning up temp file {temp_file}: {str(e)}")
        
        return jsonify({
            'domain': domain,
            'analysis': analysis_results,
            'analysis_sections': analysis_sections,
            'annotations': annotations,
            'report_url': f"/reports/{report_path}"
        })
        
    except Exception as e:
        # Clean up temp files in case of error
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)