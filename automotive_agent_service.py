# automotive_agent_service.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
from supabase import create_client, Client
from datetime import datetime, date, timedelta, timezone
from typing import Optional, List
import json
import logging
import sys
import string  # ✅ NEW
import smtplib # Using smtplib for this service's emails as requested
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from markdown_it import MarkdownIt
md_converter = MarkdownIt()

# For IST timezone conversion for analytics
try:
    from zoneinfo import ZoneInfo
except ImportError:
    logging.warning("zoneinfo (or tzdata) not available. Using fixed offset for IST. Install 'tzdata' for full timezone support.")
    ZoneInfo = None # Fallback or handle differently


load_dotenv() # Load environment variables from .env file (for local testing)

# --- Logging Setup ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

# ─── GLOBAL VEHICLE DATA ────────────────────────────────────────────
# Used by both follow‑up and offer agents to look up AOE model specs
AOE_VEHICLE_DATA = {
    "AOE Apex": {
        "type": "Luxury Sedan",
        "powertrain": "Gasoline",
        "features": (
            "Premium leather interior, Advanced driver‑assistance systems (ADAS), "
            "Panoramic sunroof, Bose premium sound system, Adaptive cruise control, "
            "Lane‑keeping assist, Automated parking, Heated and ventilated seats."
        )
    },
    "AOE Volt": {
        "type": "Electric Compact",
        "powertrain": "Electric",
        "features": (
            "Long‑range battery (500 miles), Fast charging (80% in 20 min), "
            "Regenerative braking, Solar roof charging, Vehicle‑to‑Grid (V2G) capability, "
            "Digital cockpit, Over‑the‑air updates, Extensive charging network access."
        )
    },
    "AOE Thunder": {
        "type": "Performance SUV",
        "powertrain": "Gasoline",
        "features": (
            "V8 Twin‑Turbo Engine, Adjustable air suspension, Sport Chrono Package, "
            "High‑performance braking system, Off‑road capabilities, Torque vectoring, "
            "360‑degree camera, Ambient lighting, Customizable drive modes."
        )
    }
}
# ─────────────────────────────────────────────────────────────────────
# NEW: Store public URLs of consistent car images from Google Cloud Storage
AOE_VEHICLE_IMAGES = {
    "AOE Apex": "https://storage.googleapis.com/aoe-motors-images/AOE%20Apex.jpg",
    "AOE Thunder": "https://storage.googleapis.com/aoe-motors-images/AOE%20Thunder.jpg",
    "AOE Volt": "https://storage.googleapis.com/aoe-motors-images/AOE%20Volt.jpg"
}


# --- GLOBAL CONFIGURATIONS FOR ANALYTICS AND AUTOMATION SERVICE ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# --- LLM reasons toggles ---
LLM_REASONS_ENABLED = os.getenv("USE_LLM_REASONS", "false").strip().lower() == "true"
openai_client = None
if LLM_REASONS_ENABLED and OPENAI_API_KEY and OPENAI_API_KEY.strip():
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY.strip())
        logging.info("OpenAI client constructed for LLM reasons.")
    except Exception as e:
        logging.warning("Failed to construct OpenAI client for reasons: %s", e)
        openai_client = None

logging.info(
    "LLM reasons enabled? %s | OPENAI_API_KEY set? %s",
    LLM_REASONS_ENABLED,
    bool(OPENAI_API_KEY and OPENAI_API_KEY.strip()),
)

# Who gets the reminder
SALES_TEAM_EMAIL = "karthik.sundararaju@gmail.com"

# SMTP Credentials for this new service (using smtplib)
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT_STR = os.getenv("EMAIL_PORT")
EMAIL_PORT = int(EMAIL_PORT_STR) if EMAIL_PORT_STR else 0
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

ENABLE_SMTP_SENDING = all([EMAIL_HOST, EMAIL_PORT, EMAIL_ADDRESS, EMAIL_PASSWORD])
if not ENABLE_SMTP_SENDING:
    logging.error("SMTP credentials not fully configured for automotive_agent_service. Email sending will be disabled.")
else:
    logging.info("SMTP sending enabled for automotive_agent_service.")


if not SUPABASE_URL or not SUPABASE_KEY:
    logging.error("Supabase URL or Key environment variables are not set for automotive_agent_service.")
    raise ValueError("Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_KEY.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
SUPABASE_TABLE_NAME = "bookings"
EMAIL_INTERACTIONS_TABLE_NAME = "email_interactions" # To log agent's actions

if not OPENAI_API_KEY:
    logging.error("OpenAI API Key environment variable is not set for automotive_agent_service.")
    raise ValueError("OpenAI API Key not found. Please set OPENAI_API_KEY.")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

create_offer_fn = {
    "name": "create_offer",
    "description": "Generate a personalized offer email based on the lead profile.",
    "parameters": {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2–3 internal bullet points explaining the strategy."
            },
            "subject": {
                "type": "string",
                "description": "Email subject line (≤60 chars)."
            },
            "body": {
                "type": "string",
                "description": "Customer‑facing email body: 3 paragraphs of 2–3 sentences."
            }
        },
        "required": ["analysis", "subject", "body"]
    }
}

# --- Helper Function for Email Sending (using smtplib as requested for this service) ---
def send_email_via_smtp(recipient_email, subject, body, request_id=None, event_type="email_sent_agent"):
    if not ENABLE_SMTP_SENDING:
        logging.error("SMTP sending is disabled due to missing credentials.")
        return False
    
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html")) # Assuming body is already HTML from agent logic

    try:
        # Use SMTP_SSL for port 465 or SMTP + starttls for port 587
        # Assuming common setup for Render with Gmail is often SMTP_SSL on 465, or SMTP with STARTTLS on 587
        # Adjust based on your EMAIL_PORT set in environment variables
        if EMAIL_PORT == 465:
            server = smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT)
        elif EMAIL_PORT == 587:
            server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
            server.starttls()
        else:
            logging.error(f"Unsupported SMTP port: {EMAIL_PORT}. Email sending failed for {recipient_email}.")
            return False

        with server: # Use 'with' statement for automatic server closing
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logging.info(f"Email successfully sent via SMTP to {recipient_email}! Subject: {subject}")
        # Log the email sent event in email_interactions
        if request_id:
            try:
                data = {
                    "request_id": request_id,
                    "event_type": event_type,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                supabase.from_(EMAIL_INTERACTIONS_TABLE_NAME).insert(data).execute()
                logging.info(f"Logged email interaction: {event_type} for {request_id}.")
            except Exception as e:
                logging.error(f"Error logging email interaction for {request_id}: {e}", exc_info=True)
        return True
    except Exception as e:
        logging.error(f"Failed to send email via SMTP to {recipient_email}: {e}", exc_info=True)
        return False


# --- LLM Helper Functions (Copied from dashboard.py, now in this service) ---
def analyze_sentiment_llm(text):
    if not text.strip():
        return "NEUTRAL"
    prompt = f"""
    Analyze the following text and determine its overall sentiment. Respond only with 'POSITIVE', 'NEUTRAL', or 'NEGATIVE'.
    Text: "{text}"
    """
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "system", "content": "You are a sentiment analysis AI. Your only output is 'POSITIVE', 'NEUTRAL', or 'NEGATIVE'."}, {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=10
        )
        sentiment = completion.choices[0].message.content.strip().upper()
        return sentiment if sentiment in ["POSITIVE", "NEUTRAL", "NEGATIVE"] else "NEUTRAL"
    except Exception as e: logging.error(f"Error analyzing sentiment: {e}", exc_info=True); return "NEUTRAL"

def _norm_relevance(val: str) -> str:  # ✅ NEW
    cleaned = (val or "").strip().strip('"').strip("'").strip()  # ✅ NEW
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation)).upper()  # ✅ NEW
    if cleaned.startswith("REL") or cleaned == "YES":  # ✅ NEW
        return "RELEVANT"  # ✅ NEW
    return "IRRELEVANT"  # ✅ NEW

def check_notes_relevance_llm(sales_notes):
    if not sales_notes.strip(): return "IRRELEVANT"
    prompt = f"""
    Evaluate the following sales notes for their relevance and clarity in the context of generating a follow-up email for a vehicle test drive.
    Consider notes relevant if they provide *any* clear indication of the customer's experience, sentiment, questions, or specific interests related to the vehicle or the test drive, even if brief.
    Respond only with 'RELEVANT' or 'IRRELEVANT'.
    Sales Notes: "{sales_notes}"
    """
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "system", "content": "You are an AI assistant that evaluates the relevance of sales notes for email generation. Your only output is 'RELEVANT' or 'IRRELEVANT'."}, {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=10
        )
        raw = completion.choices[0].message.content or ""        # ✅ NEW fixed
        relevance = _norm_relevance(raw)                         # ✅ NEW
        return relevance  
    except Exception as e: logging.error(f"Error checking notes relevance: {e}", exc_info=True); return "IRRELEVANT"

# MODIFIED: generate_followup_email logic from dashboard.py, now in agent service
def generate_followup_email_llm(customer_name, customer_email, vehicle_name, sales_notes, vehicle_details, current_vehicle_brand=None, sentiment=None):
    features_str = vehicle_details.get("features", "cutting-edge technology and a luxurious experience.")
    vehicle_type = vehicle_details.get("type", "vehicle")
    powertrain = vehicle_details.get("powertrain", "advanced performance")
    comparison_context = ""
    
    prompt_instructions = """
    - Start with a polite greeting.
    - Acknowledge their test drive or recent interaction.
    - The entire email body MUST be composed of distinct **Markdown paragraphs**. Use a double newline (`\\n\\n`) to separate paragraphs.
    - For lists, use standard Markdown bullet points (e.g., `- Item 1\\n- Item 2`).
    - Each paragraph should be concise (typically 2-4 sentences maximum).
    - Aim for a total of 5-7 distinct Markdown paragraphs.
    - **DO NOT include any HTML tags** (like <p>, <ul>, <li>, <br>) directly in the output.
    - DO NOT include any section dividers (like '---').
    - Ensure there is no extra blank space before the first paragraph or after the last.
    - Output the email body in valid Markdown format.
    - Separate Subject and Body with "Subject: " at the beginning of the subject line.
    """

    ev_instructions = ""
    sales_notes_lower = sales_notes.lower()
    ev_cost_keywords = ["high cost", "expensive", "affordability", "price", "budget", "charging cost", "electricity bill", "cost effective"]
    charging_anxiety_keywords = ["charging", "range anxiety", "where to charge", "how long to charge", "charge time", "battery", "infrastructure"]

    if powertrain and powertrain.lower() == "electric":
        if any(keyword in sales_notes_lower for keyword in ev_cost_keywords):
            ev_instructions += """
            - Address any mentioned "high EV cost" or affordability concerns by focusing on long-term savings, reduced fuel costs, potential tax credits, Vehicle-to-Grid (V2G) capability, and the overall value proposition of electric ownership.
            """
        if any(keyword in sales_notes_lower for keyword in charging_anxiety_keywords):
            ev_instructions += """
            - Address any mentioned "charging anxiety" or range concerns by highlighting ultra-fast charging, solar integration (if applicable for the specific EV model), extensive charging network access, and impressive range.
            """
        if not ev_instructions.strip():
             ev_instructions = """
             - Briefly highlight general advantages of electric vehicles like environmental benefits, quiet ride, and low maintenance, if not specifically contradicted by sales notes.
             """
    else:
        if any(keyword in sales_notes_lower for keyword in ev_cost_keywords):
            ev_instructions += """
            - If customer mentioned 'high EV cost' in comparison, or general cost concerns, reframe to discuss the cost-effectiveness and efficiency of the gasoline/hybrid powertrain of the {vehicle_name}, highlighting its long-term value.
            """
        if any(keyword in sales_notes_lower for keyword in charging_anxiety_keywords):
            ev_instructions += """
            - If customer mentioned 'charging anxiety', emphasize the convenience and widespread availability of traditional fueling for the {vehicle_name}.
            """

    sales_notes_incorporation_instruction = """
    - Naturally incorporate the customer's experience, sentiment, questions, or interests directly into the email body, as if you learned them during a conversation, without explicitly stating "from our sales notes".
    - Address any *explicitly mentioned* concerns or questions from the sales notes directly.
    - If no specific negative feedback or concerns are explicitly stated in the 'Customer Issues/Comments (from sales notes)', *do not invent or assume any such concerns*. Instead, focus on reinforcing the positive aspects of their experience and the benefits of the vehicle.
    """

    # AOE_TYPE_TO_COMPETITOR_SEGMENT_MAP and COMPETITOR_VEHICLE_DATA needs to be passed or accessed here
    # For now, copying full logic, assuming AOE_VEHICLE_DATA is global or passed to agent service
    AOE_VEHICLE_DATA_LOCAL = {
        "AOE Apex": {"type": "Luxury Sedan", "powertrain": "Gasoline", "features": "Premium leather interior, Advanced driver-assistance systems (ADAS), Panoramic sunroof, Bose premium sound system, Adaptive cruise control, Lane-keeping assist, Automated parking, Heated and ventilated seats."},
        "AOE Volt": {"type": "Electric Compact", "powertrain": "Electric", "features": "Long-range battery (500 miles), Fast charging (80% in 20 min), Regenerative braking, Solar roof charging, Vehicle-to-Grid (V2G) capability, Digital cockpit, Over-the-air updates, Extensive charging network access."},
        "AOE Thunder": {"type": "Performance SUV", "powertrain": "Gasoline", "features": "V8 Twin-Turbo Engine, Adjustable air suspension, Sport Chrono Package, High-performance braking system, Off-road capabilities, Torque vectoring, 360-degree camera, Ambient lighting, Customizable drive modes."}
    }
    COMPETITOR_VEHICLE_DATA_LOCAL = {
        "Ford": {"Sedan": {"model_name": "Ford Sedan", "features": ""}, "SUV": {"model_name": "Ford SUV", "features": ""}, "EV": {"model_name": "Ford EV", "features": ""}}
    }
    AOE_TYPE_TO_COMPETITOR_SEGMENT_MAP_LOCAL = {
        "Luxury Sedan": "Sedan", "Electric Compact": "EV", "Performance SUV": "SUV"
    }


    if current_vehicle_brand and current_vehicle_brand.lower() == "ford":
        aoe_segment_key = AOE_TYPE_TO_COMPETITOR_SEGMENT_MAP_LOCAL.get(vehicle_type)
        if aoe_segment_key and aoe_segment_key in COMPETITOR_VEHICLE_DATA_LOCAL["Ford"]:
            ford_competitor = COMPETITOR_VEHICLE_DATA_LOCAL["Ford"][aoe_segment_key]
            comparison_context = f"""
            The customer's current vehicle brand is Ford. The {vehicle_name} falls into the {aoe_segment_key} segment.
            A representative Ford model in this segment is the {ford_competitor['model_name']} with features: {ford_competitor['features']}.
            """
            prompt_instructions += f"""
            - Start with a polite greeting.
            - Acknowledge their test drive.
            {sales_notes_incorporation_instruction}
                {ev_instructions}
            - Given the customer's interest in Ford, compare the {vehicle_name} with the representative Ford {aoe_segment_key} model ({ford_competitor['model_name']}) on 2-3 key differentiating features/specifications. Present this as a concise comparison in a clear, structured list format, under a heading like "Comparison: {vehicle_name} vs. {ford_competitor['model_name']}". For each feature, clearly state the feature name, then list the benefit/spec for {vehicle_name} and then for {ford_competitor['model_name']}.
              Example format:
              **Feature Name:**
              - {vehicle_name}: [Value/Description]
              - {ford_competitor['model_name']}: [Value/Description]
              Highlight where the {vehicle_name} excels or offers a distinct advantage. If a specific comparison point is not available for the Ford competitor from the provided features, infer a general or typical characteristic for that type of Ford vehicle, rather than stating 'not specified' or 'may vary'.
            - When highlighting features, be slightly technical to demonstrate the real value proposition, using terms from the '{vehicle_name} Key Features' list where appropriate. Ensure the benefit is clear and compelling.
            - Do NOT use bolding (e.g., `**text**`) in the email body except for section headings like "Comparison:" or feature names within the comparison.
            - If no specific issues are mentioned, write a general follow-up highlighting key benefits.
            - End with a low-pressure call to action. Instead of demanding a call or visit, offer to provide further specific information (e.g., a detailed digital brochure, a personalized feature comparison, or answers to any specific questions via email) that they can review at their convenience.
            - Maintain a professional, empathetic, and persuasive tone.
            - Output only the email content (Subject and Body), in plain text format. Do NOT use HTML.
            - Separate Subject and Body with "Subject: " at the beginning of the subject line.
            """
        else:
            prompt_instructions = f"""
            - Start with a polite greeting.
            - Acknowledge their test drive.
            {sales_notes_incorporation_instruction}
                {ev_instructions}
            - Position the {vehicle_name} as a compelling, modern alternative by focusing on clear, concise value propositions and AOE's distinct advantages (e.g., innovation, advanced technology, future-proofing) that might appeal to someone considering traditional brands like Ford.
            - When highlighting features, be slightly technical to demonstrate the real value proposition, using terms from the '{vehicle_name} Key Features' list where appropriate. Ensure the benefit is clear and compelling.
            - Do NOT use bolding (e.g., `**text**`) in the email body.
            - If no specific issues are mentioned, write a general follow-up highlighting key benefits.
            - End with a low-pressure call to action. Instead of demanding a call or visit, offer to provide further specific information (e.g., a detailed digital brochure, a personalized feature comparison, or answers to any specific questions via email) that they can review at their convenience.
            - Maintain a professional, empathetic, and persuasive tone.
            - Output only the email content (Subject and Body), in plain text format. Do NOT use HTML.
            - Separate Subject and Body with "Subject: " at the beginning of the subject line.
            """
    elif current_vehicle_brand and current_vehicle_brand.lower() in ["toyota", "hyundai", "chevrolet"]:
        prompt_instructions = f"""
        - Start with a polite greeting.
        - Acknowledge their test drive.
        {sales_notes_incorporation_instruction}
            {ev_instructions}
        - Position the {vehicle_name} as a compelling, modern alternative by focusing on clear, concise value propositions and AOE's distinct advantages (e.g., innovation, advanced technology, future-proofing) that might appeal to someone considering traditional brands like {current_vehicle_brand}.
        - When highlighting features, be slightly technical to demonstrate the real value proposition, using terms from the '{vehicle_name} Key Features' list where appropriate. Ensure the benefit is clear and compelling.
        - Do NOT use bolding (e.g., `**text**`) in the email body.
        - If no specific issues are mentioned, write a general follow-up highlighting key benefits.
        - End with a low-pressure call to action. Instead of demanding a call or visit, offer to provide further specific information (e.g., a detailed digital brochure, a personalized feature comparison, or answers to any specific questions via email) that they can review at their convenience.
        - Maintain a professional, empathetic, and persuasive tone.
        - Output only the email content (Subject and Body), in plain text format. Do NOT use HTML.
        - Separate Subject and Body with "Subject: " at the beginning of the subject line.
        """
    else:
        # Default prompt for no specific brand comparison or general case
        prompt_instructions = f"""
        - Start with a polite greeting.
        - Acknowledge their test drive.
        {sales_notes_incorporation_instruction}
            {ev_instructions}
        - When highlighting features, be slightly technical to demonstrate the real value proposition, using terms from the '{vehicle_name} Key Features' list where appropriate. Ensure the benefit is clear and compelling.
        - Do NOT use bolding (e.g., `**text**`) in the email body.
        - If no specific issues are mentioned, write a general follow-up highlighting key benefits.
        - End with a low-pressure call to action. Instead of demanding a call or visit, offer to provide further specific information (e.g., a detailed digital brochure, a personalized feature comparison, or answers to any specific questions via email) that they can review at their convenience.
        - Maintain a professional, empathetic, and persuasive tone.
        - Output only the email content (Subject and Body), in plain text format. Do NOT use HTML.
        - Separate Subject and Body with "Subject: " at the beginning of the subject line.
        """

    # --- Add positive sentiment instructions if applicable ---
    if sentiment == "POSITIVE":
        prompt_instructions += """
        - Since the customer expressed a positive experience, ensure the email reinforces this positive sentiment.
        - Highlight the exciting nature of the AOE brand and the community they would join.
        - Mention AOE's comprehensive support system, including guidance on flexible financing options, dedicated sales support for any questions, and robust long-term service contracts, ensuring peace of mind throughout their ownership journey.
        - Instead of directly mentioning discounts, subtly hint at "tailored offers" or "value packages" that can be discussed with a sales representative to maximize their value, encouraging them to take the next step.
        """
    elif sentiment == "NEGATIVE":
        prompt_instructions += """
        - The email must be highly empathetic, apologetic, and focused on resolution.
        - Acknowledge their specific frustration or concern (e.g., "frustrated with our process") directly and empathetically in the subject and opening.
        - Apologize sincerely for any inconvenience or dissatisfaction they experienced.
        - **CRITICAL: DO NOT include generic feature lists, technical specifications, or comparisons with other brands (like Ford, Toyota, etc.) in this email.** The primary goal is to address their negative experience, not to sell the car.
        - Offer a clear and actionable path to resolve their issue or address their concerns (e.g., "I'd like to personally ensure this is resolved," "Let's discuss how we can improve," "I'm here to clarify any confusion").
        - Reassure them that their feedback is invaluable and that AOE Motors is committed to an excellent customer experience.
        - Focus entirely on rebuilding trust and resolving the negative point.
        - Keep the tone professional, understanding, and solution-oriented throughout.
        - The call to action should be solely an invitation for a direct conversation to address and resolve the specific issue.
        """

    prompt = f"""
    Draft a polite, helpful, and persuasive follow-up email to a customer who recently test-drove an {vehicle_name}.

    **Customer Information:**
    - Name: {customer_name}
    - Email: {customer_email}
    - Vehicle of Interest: {vehicle_name} ({vehicle_type}, {powertrain} powertrain)
    - Customer Issues/Comments (from sales notes): "{sales_notes}"

    **{vehicle_name} Key Features:**
    - {features_str}

    {comparison_context}

    **Email Instructions:**
    {prompt_instructions}
    """

    try:
            # 👉 indent‐fix applied on 2025‑07‑26
            logging.info("Drafting follow‑up email with AI...") # This spinner is agent service
            raw_output = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful and persuasive sales assistant for AOE Motors."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=800
            )
            draft = raw_output.choices[0].message.content.strip()
            if "Subject:" in draft:
                parts = draft.split("Subject:", 1)
                subject_line = parts[1].split("\n", 1)[0].strip()
                body_content = parts[1].split("\n", 1)[1].strip()
            else:
                subject_line = f"Following up on your {vehicle_name} Test Drive"
                body_content = draft
            
            # Post-processing to convert Markdown to HTML for sending
            if body_content.strip() and not ("<p>" in body_content):
                paragraphs = body_content.split('\n\n')
                html_body_for_sending = "".join(f"<p>{p.strip()}</p>" for p in paragraphs if p.strip())
            else:
                html_body_for_sending = body_content
            
            logging.debug(f"Final Generated Body (Markdown for UI, partial): {body_content[:100]}...")
            logging.debug(f"Final HTML Body (for sending, partial): {html_body_for_sending[:100]}...")
            
            # Return both markdown and html content for the agent service
            return subject_line, body_content, html_body_for_sending # MODIFIED return tuple

    except Exception as e:
        logging.error(f"Error drafting email with AI: {e}", exc_info=True)
        return None, None, None

# NEW: Function to suggest offer for automation agent
def suggest_offer_llm(lead_details: dict, vehicle_data: dict) -> tuple:  # Returns (text_output, html_output)
    customer_name      = lead_details.get("customer_name", "customer")
    vehicle_name       = lead_details.get("vehicle_interested", "vehicle")
    current_vehicle    = lead_details.get("current_vehicle", "N/A")
    lead_score_text    = lead_details.get("lead_score_text", "New")
    numeric_lead_score = lead_details.get("numeric_lead_score", 0)
    sales_notes        = lead_details.get("sales_notes", "")
    vehicle_features   = vehicle_data.get("features", "excellent features")

    offer_prompt = f"""
    You are an expert automotive sales strategist at AOE Motors. Using the lead profile below, generate a JSON object with three keys:

    1. "analysis" (internal only): A list of 2–3 bullets covering:
       • Which single incentive you chose—0% APR financing, a cash rebate, an extended warranty, an accessories bundle, or a trade‑in bonus—and why.  
       • How the customer’s sales notes informed your choice.  
       • Which one key feature of the {vehicle_name} (from: {vehicle_features}) you’re highlighting.

    2. "subject": A benefit‑driven email subject ≤ 60 characters.

    3. "body": The customer‑facing email, plain text only, structured as **3 paragraphs** of **2–3 sentences** each:
       - **Paragraph 1**: Start with “Hi {customer_name}, ”. Mention their interest in the {vehicle_name} and spotlight **one** feature.  
       - **Paragraph 2**: State the exact incentive—e.g. “Enjoy 0% APR for 36 months,” “Receive a $1,200 trade‑in bonus,” or “Get a complimentary 5‑year warranty.”  
       - **Paragraph 3**: Close with a strong, outcome‑oriented CTA such as “Reply now to claim this offer” or “Call me at (91) 123‑4567 to secure your incentive.”  
       - After paragraph 3, include this signature block on its own line:

         AOE Motors

    **Do NOT** (under any circumstances):
    - Explicitly mention “pricing concerns,” “budget,” or “lead score.”  
    - State customer concerns or doubts in the email text.  
    - List multiple features—highlight only one.  
    - Use Markdown, HTML, or JSON in the body.  
    - Include analysis or internal rationale in the “body” field.

    **Lead Profile:**
    - Name: {customer_name}  
    - Current Vehicle: {current_vehicle or 'N/A'}  
    - Interested Model: {vehicle_name}  
    - Sales Notes: {sales_notes or 'None'}

    **Important:**  
    - Return strictly valid JSON with keys `"analysis"`, `"subject"`, and `"body"`.  
    - The `"body"` field must contain *only* the customer email copy—no analysis or internal notes.  
    """


    try:
        # 1) Call the LLM via function‐calling
        response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"You are an expert automotive sales strategist at AOE Motors."},
            {"role":"user","content":offer_prompt}
        ],
        functions=[create_offer_fn],
        function_call={"name":"create_offer"},
        temperature=0.7,
        )

        # 2) Parse the guaranteed JSON from the function call
        msg = response.choices[0].message
        payload = json.loads(msg.function_call.arguments)
        subject_txt = payload["subject"].removeprefix("Subject: ").strip()
        body_md     = payload["body"]

        # 3) Convert only the Markdown body to HTML
        html_body = md_converter.render(body_md)

        # 4) Return exactly what your SMTP caller expects
        return subject_txt, html_body

    except Exception as e:
        logging.error(f"Error suggesting offer: {e}", exc_info=True)
        return (
            "Error generating offer suggestion. Please try again.",
            "Error generating offer suggestion. Please try again."
        )
        # fixed inendation

# NEW: Function to suggest offer for automation agent
def suggest_offer_llm(lead_details: dict, vehicle_data: dict) -> tuple:
    # 1) Pull the real fields from your Supabase record
    customer_name      = lead_details.get("full_name", "Customer")
    vehicle_name       = lead_details.get("vehicle") or lead_details.get("vehicle_interested", "your selected model")
    current_vehicle    = lead_details.get("current_vehicle", "N/A")
    sales_notes        = lead_details.get("sales_notes", "")
    vehicle_features   = vehicle_data.get("features", "excellent features")

    # 2) Tactical‑advisor prompt
    offer_prompt = f"""
You are an expert automotive sales strategist at AOE Motors. Using the lead profile below, generate a JSON object with three keys:

1. "analysis" (internal only): 2–3 bullets explaining:
   • Which incentive you chose—0% APR, cash rebate, extended warranty, accessories bundle, or trade‑in bonus—and why.
   • How the customer’s notes (“{sales_notes or 'None'}”) guided your choice.
   • Which one key feature of the {vehicle_name} (e.g. {vehicle_features}) you’re highlighting.

2. "subject": A benefit‑driven email subject under 60 characters.

3. "body": The customer‑facing email copy only, plain text, formatted as **3 paragraphs** of **2–3 sentences** each:
   - **Paragraph 1**: Start with “Hi {customer_name}, ”, mention their interest in the {vehicle_name}, and spotlight one feature.
   - **Paragraph 2**: State the exact incentive—e.g. “Enjoy 0% APR for 36 months,” “Receive a $1,200 trade‑in bonus,” or “Get a complimentary 5‑year warranty.”
   - **Paragraph 3**: Close with a clear tactical CTA (“Reply now to claim this offer” or “Call me at (555) 123‑4567 to lock it in”).

After paragraph 3, on its own line put:

AOE Motors

**Do NOT** mention “pricing concerns,” “budget,” or lead scores.  
Return strictly valid JSON with keys "analysis", "subject", and "body".

**Lead Profile:**
- Name: {customer_name}
- Current Vehicle: {current_vehicle}
- Interested Model: {vehicle_name}
- Sales Notes: {sales_notes or 'None'}
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a highly analytical AI Sales Advisor."},
                {"role": "user",   "content": offer_prompt}
            ],
            functions=[create_offer_fn],
            function_call={"name": "create_offer"},
            temperature=0.7,
            max_tokens=300
        )
        msg = response.choices[0].message
        import json
        payload     = json.loads(msg.function_call.arguments)
        subject_txt = payload["subject"].removeprefix("Subject: ").strip()
        body_md     = payload["body"]
        html_body   = md_converter.render(body_md)
        return subject_txt, html_body

    except Exception as e:
        logging.error(f"Error in suggest_offer_llm: {e}", exc_info=True)
        return (
            "Error generating offer suggestion. Please try again.",
            "Error generating offer suggestion. Please try again."
        )
# ---- Helper: normalize action statuses (case/spacing tolerant)
def _norm_status(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    # unify common variants
    s = s.replace("the ", "")  # "Call the customer (AI)" -> "call customer (ai)"
    return s

# ---- Helper: score sales notes when Follow Up Required
def _score_from_sales_notes(notes: str) -> tuple[int, list[str]]:
    """
    Returns (score_delta, reason_bits) for Follow Up Required leads.
    We reward concrete concerns and time-bound callbacks.
    """
    if not notes:
        return 0, []

    n = notes.lower()
    bits = []
    score = 0

    # high-priority signals
    if any(k in n for k in ["call back", "callback", "call tomorrow", "call monday", "call next week", "reach me", "available after"]):
        score += 12
        bits.append("requested callback")
    if any(k in n for k in ["urgent", "asap", "immediate", "priority"]):
        score += 10
        bits.append("urgent tone")

    # concrete buying concerns (price/financing/availability/etc.)
    if any(k in n for k in ["price", "budget", "emi", "loan", "finance", "financing", "insurance"]):
        score += 9
        bits.append("financing/price concern")
    if any(k in n for k in ["delivery", "wait time", "availability"]):
        score += 6
        bits.append("delivery/availability concern")
    if any(k in n for k in ["range", "charging", "mileage", "battery"]):
        score += 6
        bits.append("range/efficiency concern")
    if any(k in n for k in ["test drive", "td", "schedule drive"]):
        score += 8
        bits.append("test drive interest")

    # deprioritizers
    if any(k in n for k in ["not interested", "no longer interested", "spam", "wrong number"]):
        score -= 20
        bits.append("low intent")

    return score, bits

import os, logging

logging.info(
    "Runtime check: raw USE_LLM_REASONS=%r | compiled=%s | key_set=%s",
    os.getenv("USE_LLM_REASONS"),
    USE_LLM_REASONS,
    bool(OPENAI_API_KEY and OPENAI_API_KEY.strip()),
)

# ---- Optional: batch LLM "reason" generator (fast, guarded)
LLM_REASONS_ENABLED = (
    os.getenv("USE_LLM_REASONS") 
    or os.getenv("ANALYTICS_USE_LLM_REASONS") 
    or "false"
).strip().lower() == "true"

def _llm_reasons(rows: list[str]) -> list[str | None]:
    # Must see both the toggle and a constructed client
    global LLM_REASONS_ENABLED, openai_client, OPENAI_API_KEY
    if not LLM_REASONS_ENABLED or openai_client is None or not (OPENAI_API_KEY and OPENAI_API_KEY.strip()):
        logging.info(
            "LLM reasons disabled or client missing: enabled=%s, client=%s, key_set=%s",
            LLM_REASONS_ENABLED,
            bool(openai_client),
            bool(OPENAI_API_KEY and OPENAI_API_KEY.strip()),
        )
        return [None] * len(rows)

    reasons: list[str | None] = []
    for r in rows:
        # r is a dict with keys Lead, Status, LeadScore, SalesNotes, AgeDays...
        brief = f"{r.get('Lead','')}; status={r.get('Status','')}; score={r.get('LeadScore',0)}; age_days={r.get('AgeDays',0)}; notes={ (r.get('SalesNotes') or '')[:160] }"
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",           # or "gpt-3.5-turbo" if you prefer
                temperature=0.2,
                max_tokens=24,
                messages=[
                    {"role": "system", "content": "Return a single short reason (max ~10 words) why this lead is priority to call. No punctuation heavy lines."},
                    {"role": "user", "content": brief}
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            reasons.append(text if text else None)
        except Exception as e:
            logging.warning("LLM reason failed for %s: %s", r.get("Lead"), e)
            reasons.append(None)

    logging.info("LLM reasons produced %d/%d lines.", sum(1 for x in reasons if x), len(reasons))
    return reasons

# --- ANALYTICS FUNCTION (MOVED FROM DASHBOARD.PY) ---
class AnalyticsQueryRequest(BaseModel):
    query_text: str
    start_date: str # Send as ISO format string
    end_date: str # Send as ISO format string

@app.post("/analyze-query")
async def analyze_query_endpoint(request_data: AnalyticsQueryRequest):
    query_text = (request_data.query_text or "").strip()
    start_date_str = request_data.start_date
    end_date_str = request_data.end_date

    try:
        # 1) Pull data
        response_data = supabase.from_(SUPABASE_TABLE_NAME).select(
            "request_id, full_name, email, vehicle, booking_date, current_vehicle, "
            "time_frame, action_status, sales_notes, lead_score, numeric_lead_score, booking_timestamp"
        ).order('booking_timestamp', desc=True).execute()

        if not response_data.data:
            return {"result_type": "TEXT", "result_message": "No bookings data available for analytics."}

        df = pd.DataFrame(response_data.data)

        # 2) Normalize timestamps → UTC (tolerant to tz-naive rows)
        #    We'll compare with half-open UTC window derived from sidebar dates.
        df["booking_timestamp"] = pd.to_datetime(df["booking_timestamp"], errors="coerce", utc=True)

        # 3) Sidebar date window (half-open range) in UTC
        #    Assume sidebar dates are in local (human) terms; treat them as all-day inclusive.
        start_dt_utc = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt_utc   = (datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)).replace(tzinfo=timezone.utc)

        df = df[(df["booking_timestamp"] >= start_dt_utc) & (df["booking_timestamp"] < end_dt_utc)].copy()

        # --- STRONG GUARDRails + NL intent router (prototype, no LLM) ---
        q = (query_text or "").lower().strip()

        # 4.1) Guardrails: only allow a small whitelist
        ALLOWED_TOKENS = {
            "lead","leads","total","hot","warm","cold",
            "converted","lost","chart","trend","distribution","score",
            "who","whom","call","rank","prioritize","list"
        }
        if not any(tok in q for tok in ALLOWED_TOKENS):
            return {
                "result_type": "TEXT",
                "result_message": (
                "🙅 Not relevant. Try: **'total leads'**, **'hot leads'**, "
                "**'trend conversions'**, **'lead score distribution'**, or **'who should I call'** "
                "(using the current date filters)."
               )    
            }


        # 4.3) Normalize commonly used columns
        df["lead_score"] = df["lead_score"].fillna("")
        df["numeric_lead_score"] = pd.to_numeric(df["numeric_lead_score"], errors="coerce").fillna(0).astype(int)

        # 4.4) Intent routing
        def wants_rank(s: str) -> bool:
            return any(k in s for k in ["who should i call","whom should i call","call list","calllist","prioritize","rank"])

     
        def wants_chart(s: str) -> bool:
            chart_terms = [
                "chart", "trend", "trends", "distribution", "histogram",
                "score distribution", "daily", "per day", "over time",
                "timeline", "time series", "time-series", "line","leads by status"
            ]
            return any(t in s for t in chart_terms)
 
        intent = "COUNT"
        if wants_rank(q):
            intent = "RANK"
        elif wants_chart(q):
            intent = "CHART"

        # 4.5) Helpers
        def _status_breakdown(_df):
            counts = (
                _df["lead_score"].str.lower().map({"hot":"Hot","warm":"Warm","cold":"Cold"}).fillna("Unknown")
                .value_counts().sort_index()
            )
            return {"kind":"bar","labels": list(counts.index), "values": counts.tolist()}

        def _score_distribution(_df):
            bands = pd.cut(
                _df["numeric_lead_score"],
                bins=[-1,5,10,15,20,100],
                labels=["0-5","6-10","11-15","16-20","21+"]
            )
            vc = bands.value_counts().sort_index()
            return {"kind":"bar","labels": list(vc.index.astype(str)), "values": vc.tolist()}
        
        def _converted_lost_trend(_df: pd.DataFrame) -> dict:
            """
            Returns a consistent time-series payload for Streamlit.
            - Uses DAILY frequency for short ranges (<= 21 days) and WEEKLY otherwise.
            - Ensures all series have the same length and share the same index.
            """
            if _df.empty:
                return {"kind": "line", "x": [], "series": {}}

    # Work entirely in a normalized daily timestamp column
            ts = pd.to_datetime(_df["booking_timestamp"], errors="coerce", utc=True)
            d  = _df.copy()
            d["day"] = ts.dt.floor("D")               # DatetimeIndex @ 00:00 UTC

            # Choose frequency based on span (weekly looks better for longer ranges)
            span_days = int((d["day"].max() - d["day"].min()).days) + 1
            freq = "D" if span_days <= 21 else "W-MON"

            # Build a complete date index for the chosen frequency
            date_index = pd.date_range(start=d["day"].min(), end=d["day"].max(), freq=freq)

            # Helper to make a series (aligned to date_index, filled with zeros)
            def _series(mask: pd.Series) -> pd.Series:
                if not mask.any():
                    return pd.Series(0, index=date_index)
                s = (
                    d.loc[mask]
                    .set_index("day")
                    .resample(freq)
                    .size()
                    .reindex(date_index, fill_value=0)
                )
                return s.astype(int)

            conv = _series(d["action_status"] == "Converted")
            lost = _series(d["action_status"] == "Lost")

            # If neither Converted nor Lost is present, fall back to total leads
            if conv.sum() == 0 and lost.sum() == 0:
                total = (
                    d.set_index("day")
                    .resample(freq)
                    .size()
                    .reindex(date_index, fill_value=0)
                    .astype(int)
                )
                series = {"Leads": total.tolist()}
            else:
                series = {
                    "Converted": conv.tolist(),
                    "Lost":      lost.tolist(),
               }

            x = [ts.isoformat() for ts in date_index]   # Streamlit will parse these fine
            return {"kind": "line", "x": x, "series": series}
      
        def _trend_for_status(_df, statuses):
            _df["date"] = pd.to_datetime(_df["booking_timestamp"], errors="coerce").dt.date
            label_map = {"hot":"Hot","warm":"Warm","cold":"Cold"}
            _df["__ls__"] = _df["lead_score"].str.lower()
            g = _df.groupby("date").apply(lambda d: pd.Series({
            lab: int((d["__ls__"]==s).sum()) for s,lab in [(k,label_map[k]) for k in statuses]
            }))
            g = g.sort_index().fillna(0)
            x = [d.isoformat() for d in g.index]
            series = {col: g[col].astype(int).tolist() for col in g.columns}
            return {"kind":"line","x": x, "series": series}

        
        def _rank_leads(_df: pd.DataFrame, top_n: int = 10, _status_bonus: dict | None = None) -> dict:
            """
            Build a Top-N call list with explicit priority rules:
            1) 'Call Customer (AI)' first
            2) 'Escalation Initiated' next
            3) Then score by lead score, recency, and Follow-Up notes
            Returns payload with columns + rows for the dashboard.
            """
            if _df.empty:
                return {"columns": ["Lead", "Vehicle", "Status", "LeadScore", "Reason"], "rows": []}

            # Strong status bonus so ordering is obvious
            if _status_bonus is None:
                _status_bonus = {
                    "call customer (ai)": 1000,
                    "escalation initiated": 800,
                    "call scheduled": 350,
                    "follow up required": 320,
                    "test drive due": 260,
                    "offer sent (ai)": 190,
                    "engaged - email": 170,
                    "personalized ad sent": 160,
                    "converted": -300,
                    "lost": -500,
                }

            now = pd.Timestamp.utcnow()
            rows = []
            for _, r in _df.iterrows():
            # Skip rows with no name or id to avoid weird blanks at the top
                if not r.get("full_name") or not r.get("request_id"):
                    continue

                stxt = _norm_status(r.get("action_status"))
                base = _status_bonus.get(stxt, 0)
                # Lead score + recency
                ls = int(pd.to_numeric(r.get("numeric_lead_score"), errors="coerce") or 0)
                score = base + 5 * ls

                ts = pd.to_datetime(r.get("booking_timestamp"), errors="coerce", utc=True)
                age_days = 0
                if isinstance(ts, pd.Timestamp):
                    try:
                        age_days = max(0, (now - ts).days)
                    except Exception:
                        age_days = 0

                # recency boost (max ~16 points in first few days)
                score += max(0, 8 - min(age_days, 8)) * 2

                # Sales notes weighting only when Follow Up Required
                notes_bits = []
                if stxt == "follow up required":
                    delta, bits = _score_from_sales_notes(r.get("sales_notes") or "")
                    score += delta
                    notes_bits.extend(bits)

                # Build a default reason (deterministic)
                reason_bits = []
                if base >= 1000:
                    reason_bits.append("explicit agent instruction")
                elif base >= 800:
                    reason_bits.append("escalation flagged")
                if ls >= 15:
                    reason_bits.append("high lead score")
                elif ls >= 10:
                    reason_bits.append("good lead score")
                if age_days <= 3:
                    reason_bits.append("recent lead")
                if notes_bits:
                    reason_bits.append("notes: " + ", ".join(notes_bits))

                reason_text = " • ".join(reason_bits) or "—"

                rows.append({
                    "Priority": int(score),
                    "Lead": r.get("full_name"),
                    "Vehicle": r.get("vehicle"),
                    "Status": r.get("action_status"),
                    "LeadScore": ls,
                    "SalesNotes": r.get("sales_notes") or "",
                    "AgeDays": age_days,
                    "RequestID": r.get("request_id"),
                    "Reason": reason_text,
                })

            if not rows:
                return {"columns": ["Lead", "Vehicle", "Status", "LeadScore", "Reason"], "rows": []}

            # Sort by (status priority first), then by Priority score
            def _status_rank(s):
                return -_status_bonus.get(_norm_status(s), 0)

            rows = sorted(rows, key=lambda x: (_status_rank(x["Status"]), -x["Priority"]))

            # Keep Top N
            rows = rows[: max(1, top_n)]

            # Optional: replace reason with a short LLM line if enabled
            rows_with_llm = _llm_reasons(rows)
            for i, txt in enumerate(rows_with_llm):
                if txt and isinstance(txt, str):
                    rows[i]["Reason"] = txt

            payload = {
                "columns": ["Lead", "Vehicle", "Status", "LeadScore", "Reason"],
                "rows": [{k: v for k, v in r.items() if k in ["Lead", "Vehicle", "Status", "LeadScore", "Reason"]} for r in rows],
            }
            return payload

        # 5) Execute intent (all within current FILTERED VIEW)
        s_label = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%b %d, %Y")
        e_label = datetime.strptime(end_date_str,   "%Y-%m-%d").strftime("%b %d, %Y")

        if intent == "RANK":
            payload = _rank_leads(df, top_n=10)
            msg = f"☎️ Recommended call list (using current filters: {s_label} → {e_label})"
            return {"result_type":"RANK", "result_message": msg, "payload": payload}

        if intent == "CHART":
    # Choose chart type by keywords
            # status breakdown (e.g., "leads by status", "status chart")
            if any(t in q for t in ["status", "by status", "leads by status"]):
                payload = _status_breakdown(df)
                msg = f"📊 Leads by status (using current filters: {s_label} → {e_label})"
                return {"result_type": "CHART", "result_message": msg, "payload": payload}

            if any(k in q for k in ["distribution","histogram","score"]):
                payload = _score_distribution(df)
                msg = f"📊 Lead score distribution (using current filters: {s_label} → {e_label})"
                return {"result_type":"CHART","result_message": msg, "payload": payload}
             # conversions/losses trend?
            conv_terms  = ["convert", "converted", "conversion", "conversions"]
            loss_terms  = ["lost", "loss", "losses"]
            trend_terms = ["trend", "trends", "daily", "per day", "over time", "timeline", "time series", "time-series", "chart", "line"]

            if (any(t in q for t in conv_terms + loss_terms)) and (any(t in q for t in trend_terms)):
                payload = _converted_lost_trend(df)
                msg = f"📈 Conversions vs. losses trend (using current filters: {s_label} → {e_label})"
                return {"result_type": "CHART", "result_message": msg, "payload": payload}
    
    # default chart: status breakdown
            payload = _status_breakdown(df)
            msg = f"📊 Leads by status (using current filters: {s_label} → {e_label})"
            return {"result_type":"CHART","result_message": msg, "payload": payload}

    # Fallback to basic COUNT (your earlier behavior), still filtered by dates
        lead_status_map = {
            "hot":"Hot","warm":"Warm","cold":"Cold",
            "converted":"Converted","lost":"Lost","total":"All"
        }
    # detect a status word; default is All
        selected = "All"
        for k,v in lead_status_map.items():
            if k in q:
                selected = v; break

        filtered_df = df.copy()
        if selected in ["Converted","Lost"]:
            filtered_df = filtered_df[filtered_df["action_status"] == selected]
        elif selected in ["Hot","Warm","Cold"]:
            filtered_df = filtered_df[filtered_df["lead_score"].str.lower() == selected.lower()]
    # else: All

        result_count = int(filtered_df.shape[0])
        label = f"{selected.lower()} leads" if selected!="All" else "total leads"
        msg = f"📊 {label.capitalize()}: **{result_count}** (using current filters: {s_label} → {e_label})"
        return {"result_type":"COUNT","result_message": msg}
          
    except Exception as e:#fixed inendation
        logging.error(f"Error processing analytics query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analytics error: {e}")

#Helper function for test drive due reminder emails
# Helper: format the “test drive due” reminder email for the sales team
def _format_testdrive_due_email(lead: dict) -> tuple[str, str]:
    """
    Returns (subject, html_body).
    Expects keys: full_name, email, vehicle, booking_date (YYYY-MM-DD), location,
                  numeric_lead_score, sales_notes, request_id
    """
    cust   = lead.get("full_name") or "Customer"
    email  = lead.get("email") or ""
    veh    = lead.get("vehicle") or "vehicle"
    # booking_date is a DATE in Supabase -> present it clearly
    when_d = lead.get("booking_date") or "—"
    loc    = lead.get("location") or "—"
    score  = lead.get("numeric_lead_score", 0)
    notes  = lead.get("sales_notes") or "—"
    req_id = lead.get("request_id") or "—"

    subject = f"[Reminder] Test drive due in <24h — {cust} ({veh})"
    body = f"""
    <p><b>Test drive due within 24 hours</b></p>
    <p><b>Customer:</b> {cust} &lt;{email}&gt;<br/>
       <b>Vehicle:</b> {veh}<br/>
       <b>Test drive date:</b> {when_d}<br/>
       <b>Location:</b> {loc}<br/>
       <b>Lead score:</b> {score}<br/>
       <b>Request ID:</b> {req_id}</p>

    <p><b>Sales notes</b><br/>{notes}</p>

    <p><b>Checklist</b></p>
    <ul>
      <li>Call/SMS customer to reconfirm slot</li>
      <li>Vehicle prepped & charged/fueled</li>
      <li>Route & paperwork ready</li>
      <li>Personalization talking points (features to highlight)</li>
    </ul>

    <p>— AOE Agent</p>
    """
    return subject, body


# --- BATCH AGENT ENDPOINTS (User-Triggered via Dashboard UI) ---

class BatchTriggerRequest(BaseModel):
    # This payload will come from dashboard.py, containing current filtered lead IDs
    lead_ids: list[str]
    selected_location: str
    start_date: str # ISO format
    end_date: str # ISO format


@app.post("/trigger-batch-followup-email-agent")
async def trigger_batch_followup_email_agent_endpoint(request_data: BatchTriggerRequest):
    logging.info(f"Received request to trigger batch follow-up email agent for {len(request_data.lead_ids)} leads.")
    
    processed_count = 0
    failed_count = 0
    
    for lead_id in request_data.lead_ids:
        try:
            # 1. Fetch full lead data from Supabase for this lead_id
            response = supabase.from_(SUPABASE_TABLE_NAME).select(
                "full_name, email, vehicle, current_vehicle, time_frame, action_status, sales_notes, lead_score, numeric_lead_score"
            ).eq('request_id', lead_id).eq('action_status', 'Follow Up Required').single().execute() # Ensure it's still 'Follow Up Required'
            
            if not response.data:
                logging.info(f"Lead {lead_id} not found or not 'Follow Up Required'. Skipping.")
                continue

            lead_data = response.data

            # 2. Re‑evaluate sales notes relevance/sentiment (similar to dashboard logic)
            sales_notes = lead_data.get('sales_notes', '')

            # ✅ NEW: default to IRRELEVANT to avoid UnboundLocalError
            notes_relevance = "IRRELEVANT"  # ✅ NEW

            # Only proceed if there are notes to evaluate
            if sales_notes.strip():
            # Call the relevance checker
                notes_relevance = check_notes_relevance_llm(sales_notes)
                logging.info(f"Sales notes relevance for lead {lead_id}: {notes_relevance}")

            # Skip if explicitly irrelevant
            if notes_relevance != "RELEVANT":
                logging.info(f"Skipping follow‑up for lead {lead_id} due to irrelevant notes.")
                continue
            # fixed inendation  

            # 3. Get vehicle details (hardcoded locally or fetched)
            vehicle_details = AOE_VEHICLE_DATA.get(lead_data['vehicle'], {})
            current_vehicle_brand_val = lead_data['current_vehicle'].split(' ')[0] if lead_data['current_vehicle'] else None

            if not vehicle_details:
                logging.warning(f"Vehicle details for {lead_data['vehicle']} not found. Skipping follow-up email for {lead_id}.")
                continue

            # 3) Analyze sentiment for use in the follow‑up email
            notes_sentiment = analyze_sentiment_llm(sales_notes)
            logging.info(f"Sales notes sentiment for lead {lead_id}: {notes_sentiment}")

            # fixed inendation

            # 4. Generate email content using LLM (reusing dashboard logic)
            subject, body_markdown, body_html = generate_followup_email_llm(
                lead_data['full_name'], lead_data['email'], lead_data['vehicle'], sales_notes, vehicle_details,
                current_vehicle_brand=current_vehicle_brand_val,
                sentiment=notes_sentiment
            )

            if not subject or not body_html:
                logging.error(f"Failed to generate email content for {lead_id}. Skipping.")
                failed_count += 1
                continue

            # 5. Send email via SMTP (using this service's SMTP function)
            email_sent = send_email_via_smtp(
                lead_data['email'], subject, body_html,
                request_id=lead_id, event_type="email_followup_agent_sent"
            )

            if email_sent:
                # 6. Update action status in database (using this service's supabase client)
                supabase.from_(SUPABASE_TABLE_NAME).update(
                    {'action_status': 'Follow-up Email Sent (AI)'}
                ).eq('request_id', lead_id).execute()
                logging.info(f"Updated action status for {lead_id} to 'Follow-up Email Sent (AI)'.")
                processed_count += 1
            else:
                failed_count += 1

        except Exception as e:
            logging.error(f"Error processing follow-up email for lead {lead_id}: {e}", exc_info=True)
            failed_count += 1

    return {
        "status": "success",
        "message": f"Batch follow-up email process completed. Sent: {processed_count}, Failed: {failed_count}.",
        "sent_count": processed_count,
        "failed_count": failed_count
    }


@app.post("/trigger-batch-offer-agent")
async def trigger_batch_offer_agent_endpoint(request_data: BatchTriggerRequest):
    logging.info(f"Received request to trigger batch offer agent for {len(request_data.lead_ids)} leads.")

    processed_count = 0
    failed_count = 0

    for lead_id in request_data.lead_ids:
        try:
            # 1. Fetch full lead data from Supabase for this lead_id
            response = supabase.from_(SUPABASE_TABLE_NAME).select(
                "full_name, email, vehicle, current_vehicle, time_frame, action_status, sales_notes, lead_score, numeric_lead_score"
            ).eq('request_id', lead_id).gte('numeric_lead_score', 12).not_.in_('action_status', ['Lost', 'Converted']).single().execute()
            
            if not response.data:
                logging.info(f"Lead {lead_id} not found, not 'score > 12', or already Lost/Converted. Skipping offer.")
                continue

            lead_data = response.data

            # 2. Get vehicle details
            vehicle_details = AOE_VEHICLE_DATA.get(lead_data['vehicle'], {})
            if not vehicle_details:
                logging.warning(f"Vehicle details for {lead_data['vehicle']} not found. Skipping offer for {lead_id}.")
                continue

            # 3. Generate offer content using LLM
            offer_subject, offer_text_html = suggest_offer_llm(lead_data, vehicle_details) # Reusing suggest_offer_llm
            
            if not offer_text_html:
                logging.error(f"Failed to generate offer content for {lead_id}. Skipping.")
                failed_count += 1
                continue

            
            # 4. Send offer email via SMTP
            email_sent = send_email_via_smtp(
                lead_data['email'], offer_subject, offer_text_html,
                request_id=lead_id, event_type="email_offer_agent_sent"
            )

            if email_sent:
                # 5. Update action status and log
                supabase.from_(SUPABASE_TABLE_NAME).update(
                    {'action_status': 'Offer Sent (AI)'}
                ).eq('request_id', lead_id).execute()
                logging.info(f"Updated action status for {lead_id} to 'Offer Sent (AI)'.")
                processed_count += 1
            else:
                failed_count += 1

        except Exception as e:
            logging.error(f"Error processing offer email for lead {lead_id}: {e}", exc_info=True)
            failed_count += 1

    return {
        "status": "success",
        "message": f"Batch offer process completed. Sent: {processed_count}, Failed: {failed_count}.",
        "sent_count": processed_count,
        "failed_count": failed_count
    }

from typing import List  # if not already imported
from datetime import datetime, timedelta, timezone

@app.post("/ops/mark-testdrives-due")
async def mark_testdrives_due():
    """
    Mark bookings as 'Test Drive Due' and notify the sales team when
    TODAY == (booking_date - 1 day).

    Idempotency: we skip rows already in 'Test Drive Due' or in 'Converted/Lost'.
    """
    now_utc = datetime.now(timezone.utc)
    target_date = (now_utc + timedelta(days=1)).date()  # booking_date to match (DATE)
    target_str  = target_date.isoformat()

    try:
        # Pull only the rows whose booking_date equals tomorrow’s date,
        # excluding Converted/Lost. (Idempotency guard handled below.)
        res = (
            supabase
            .from_(SUPABASE_TABLE_NAME)
            .select(
                "request_id, full_name, email, vehicle, booking_date, location, "
                "numeric_lead_score, sales_notes, action_status"
            )
            .eq("booking_date", target_str)
            .neq("action_status", "Converted")
            .neq("action_status", "Lost")
            .execute()
        )
        rows = res.data or []
    except Exception as e:
        logging.error(f"Supabase query failed in /ops/mark-testdrives-due: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Supabase query failed")

    updated, emailed, skipped = 0, 0, 0

    for lead in rows:
        current_status = (lead.get("action_status") or "").lower()
        if current_status == "Test Drive Due":
            # Idempotency guard: already marked; don’t update or resend
            skipped += 1
            continue

        # 1) Update status -> 'Test Drive Due'
        try:
            upd = (
                supabase
                .from_(SUPABASE_TABLE_NAME)
                .update({"action_status": "Test Drive Due"})
                .eq("request_id", lead["request_id"])
                .execute()
            )
            if upd.data:
                updated += 1
        except Exception as e:
            logging.error(f"Update failed for {lead.get('request_id')}: {e}", exc_info=True)
            continue  # still try next lead

        # 2) Email sales team
        try:
            subject, html_body = _format_testdrive_due_email(lead)
            ok = send_email_via_smtp(
                SALES_TEAM_EMAIL,
                subject,
                html_body,
                request_id=lead.get("request_id"),
                event_type="testdrive_due_notice"
            )
            if ok:
                emailed += 1
        except Exception as e:
            logging.error(f"Failed sending reminder for {lead.get('request_id')}: {e}", exc_info=True)

    return {
        "status": "ok",
        "target_booking_date": target_str,
        "found": len(rows),
        "updated_status_to_due": updated,
        "emails_sent": emailed,
        "skipped_already_due": skipped
    }

# --- ENDPOINTS FOR LEAD INSIGHTS (Future extension if needed from agent service) ---
# For now, Lead Insights remains an indicator in dashboard.py, but this service could
# analyze and store insights in a separate DB table if it ran autonomously.
# Example:
# @app.get("/get-lead-insights")
# async def get_lead_insights_endpoint():
#     # Logic to fetch pre-calculated insights from a database table
#     pass

# To run this service locally: uvicorn automotive_agent_service:app --reload --port 8001; done with indendation