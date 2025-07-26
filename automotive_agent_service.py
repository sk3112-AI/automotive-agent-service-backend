# automotive_agent_service.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
from supabase import create_client, Client
from datetime import datetime, date, timedelta, timezone
import json
import logging
import sys
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


# --- GLOBAL CONFIGURATIONS FOR ANALYTICS AND AUTOMATION SERVICE ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        relevance = completion.choices[0].message.content.strip().upper()
        return relevance if relevance in ["RELEVANT", "IRRELEVANT"] else "IRRELEVANT"
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

# MODIFIED: generate_lost_email to use HTML <p> tags
def generate_lost_email_html(customer_name, vehicle_name): # Renamed for clarity that it outputs HTML
    subject = f"We Miss You, {customer_name}!"
    body = f"""<p>Dear {customer_name},</p>
<p>We noticed you haven't moved forward with your interest in the {vehicle_name}. We understand circumstances change, but we'd love to hear from you if you have any feedback or if there's anything we can do to help.</p>
<p>Sincerely,</p>
<p>AOE Motors Team</p>
"""
    return subject, body

# MODIFIED: generate_welcome_email to use HTML <p> tags
def generate_welcome_email_html(customer_name, vehicle_name): # Renamed for clarity that it outputs HTML
    subject = f"Welcome to the AOE Family, {customer_name}!"
    body = f"""<p>Dear {customer_name},</p>
<p>Welcome to the AOE Motors family! We're thrilled you chose the {vehicle_name}.</p>
<p>To help you get started, here are some important next steps and documents:</p>
<ul>
    <li><b>Next Steps:</b> Our sales representative will be in touch shortly to finalize your delivery details and walk you through your new vehicle's features.</li>
    <li><b>Important Documents:</b> You'll find your purchase agreement, warranty information, and a quick-start guide for your {vehicle_name} attached to this email (or accessible via the link below).</li>
</ul>
<p>[Link to Digital Documents/Owner's Manual - e.g., www.aoemotors.com/your-vehicle-docs]</p>
<p>Should you have any questions before then, please don't hesitate to reach out to your sales representative or our customer support team at support@aoemotors.com.</p>
<p>We're excited for you to experience the AOE difference!</p>
<p>Sincerely,</p>
<p>The AOE Motors Team</p>
"""
    return subject, body

# NEW: Function to suggest offer for automation agent
def suggest_offer_llm(lead_details: dict, vehicle_data: dict) -> tuple:  # Returns (text_output, html_output)
    customer_name = lead_details.get("customer_name", "customer")
    vehicle_name = lead_details.get("vehicle_interested", "vehicle")
    current_vehicle = lead_details.get("current_vehicle", "N/A")
    lead_score_text = lead_details.get("lead_score_text", "New")
    numeric_lead_score = lead_details.get("numeric_lead_score", 0)
    sales_notes = lead_details.get("sales_notes", "")
    vehicle_features = vehicle_data.get("features", "excellent features")

    offer_prompt = f"""
    You are an expert automotive sales strategist at AOE Motors. Use the lead profile below to produce:

    1. **Strategic Rationale** (2–3 bullets):
       - Don’t call out “pricing,” “budget,” or any internal concerns—infer what matters most from the notes.
       - Identify the single most compelling incentive (e.g. “$1,000 rebate,” “0% APR for 36 months,” or “$1,200 trade‑in bonus”) that will drive immediate action.

    2. **Subject Line**:
       - Must start with “Subject:” and be under 60 characters.
       - Place the subject on its own line, followed by one blank line.

    3. **Email Body** (plain text only):
       - After the blank line following the subject, write exactly **3 paragraphs**, each **2–3 sentences** long.
       - Separate paragraphs with **one** blank line.
       - **Do not** indent paragraphs; start flush left.
       - Keep lines under **80 characters**.
       - **Paragraph 1**: Warm greeting by name and subtle reference to their interest in the {vehicle_name}.
       - **Paragraph 2**: Introduce the offer terms clearly (e.g. “I’m offering you a $1,000 rebate…”), phrased as a value‑add rather than a concession.
       - **Paragraph 3**: Close with an outcome‑oriented CTA (e.g. “Reply now to claim your rebate,” “Call me at (555) 123‑4567 to finalize this offer”).
       - After paragraph 3, add a signature block with your name on one line and “AOE Motors Sales Strategist” on the next.

    **Lead Profile:**
    - Name: {customer_name}
    - Current Vehicle: {current_vehicle or 'N/A'}
    - Interested Model: {vehicle_name}
    - Sales Notes: {sales_notes or 'None'}

    Respond in JSON with keys "analysis", "subject", and "body".
    """

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a highly analytical AI Sales Advisor. Provide concise, actionable offer suggestions."
                },
                {"role": "user", "content": offer_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        raw_output = completion.choices[0].message.content.strip()

        # Parse the JSON that the LLM returned
        import json
        try:
            parsed = json.loads(raw_output)
            # If the LLM included “Subject:” prefix, strip it off:
            subject_text = parsed.get("subject", "").removeprefix("Subject: ").strip()
            body_md     = parsed.get("body", "")
        except Exception as e:
            logging.error(f"JSON parse error in suggest_offer_llm: {e}", exc_info=True)
            # fall back to whole output as body
            subject_text = "Exclusive Offer from AOE Motors"
            body_md      = raw_output

        # Now convert **only** the Markdown body to HTML
        html_body = md_converter.render(body_md)
        return subject_text, html_body

# NEW: Function to generate call talking points for automation agent
def generate_call_talking_points_llm(lead_details: dict, vehicle_data: dict) -> str:
    customer_name = lead_details.get("customer_name", "customer")
    vehicle_name = lead_details.get("vehicle_interested", "vehicle")
    current_vehicle = lead_details.get("current_vehicle", "N/A")
    lead_score_text = lead_details.get("lead_score_text", "New")
    numeric_lead_score = lead_details.get("numeric_lead_score", 0)
    sales_notes = lead_details.get("sales_notes", "")
    vehicle_features = vehicle_data.get("features", "excellent features")

    prompt = f"""
    You are an AI Sales Advisor preparing talking points for a sales representative's call with {customer_name}.

    **Customer Profile**
    - Name: {customer_name}
    - Vehicle Interested: {vehicle_name}
    - Current Vehicle: {current_vehicle}
    - Lead Status: {lead_score_text} ({numeric_lead_score} points)
    - Sales Notes: {sales_notes or 'None'}
    - Key Features of {vehicle_name}: {vehicle_features}

    **Instructions**
    1. Start with **AI Talking Points:** as a header.
    2. Provide concise, actionable bullet points:
    - Acknowledge their interest in {vehicle_name}.
    - Address any concerns from the sales notes empathetically.
    - Highlight 2–3 top features of {vehicle_name} based on their profile.
    - Suggest strategic questions to uncover needs.
    - End with a clear, outcome‑oriented CTA (e.g., “Reply now to explore financing options”).
    3. Format your output as a Markdown list.
    4. If sales notes are empty or irrelevant, focus on re‑engagement and discovery.
    """

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI Sales Advisor that provides clear, actionable talking points for sales calls."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating talking points: {e}", exc_info=True)
        return "Error generating talking points. Please try again."

# --- ANALYTICS FUNCTION (MOVED FROM DASHBOARD.PY) ---
class AnalyticsQueryRequest(BaseModel):
    query_text: str
    selected_location: str
    start_date: str # Send as ISO format string
    end_date: str # Send as ISO format string

@app.post("/analyze-query")
async def analyze_query_endpoint(request_data: AnalyticsQueryRequest):
    query_text = request_data.query_text
    selected_location = request_data.selected_location 
    start_date_str = request_data.start_date
    end_date_str = request_data.end_date

    try:
        # Fetch all data from Supabase for analytics (this service fetches its own data)
        response_data = supabase.from_(SUPABASE_TABLE_NAME).select(
            "request_id, full_name, email, vehicle, booking_date, current_vehicle, location, time_frame, action_status, sales_notes, lead_score, numeric_lead_score, booking_timestamp"
        ).order('booking_timestamp', desc=True).execute()

        if not response_data.data:
            return {"result_message": "No bookings data available for analytics."}
        
        all_bookings_df = pd.DataFrame(response_data.data)
        all_bookings_df['booking_timestamp'] = pd.to_datetime(all_bookings_df['booking_timestamp'])
        
        if all_bookings_df['booking_timestamp'].dt.tz is None:
            all_bookings_df['booking_timestamp'] = all_bookings_df['booking_timestamp'].dt.tz_localize('UTC')
        else:
            all_bookings_df['booking_timestamp'] = all_bookings_df['booking_timestamp'].dt.tz_convert('UTC')

        # --- LLM PROMPT and FILTERING LOGIC (COPIED FROM DASHBOARD.PY'S INTERPRET_AND_QUERY) ---
        query = query_text.lower().strip()
        
        today_dt_ist = datetime.now(ZoneInfo('Asia/Kolkata')) if ZoneInfo else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=5, minutes=30) # Ensure IST reference
        today_dt_utc = today_dt_ist.astimezone(timezone.utc)
        yesterday_dt_utc = (today_dt_ist - timedelta(days=1)).astimezone(timezone.utc)
        last_week_start_dt_utc = (today_dt_ist - timedelta(days=7)).astimezone(timezone.utc)
        last_month_start_dt_utc = (today_dt_ist - timedelta(days=30)).astimezone(timezone.utc)
        last_year_start_dt_utc = (today_dt_ist - timedelta(days=365)).astimezone(timezone.utc)

        # The core LLM prompt for interpretation
        prompt = f"""
        Analyze the following user query about automotive leads.
        Extract the 'lead_status', 'time_frame', and optionally 'location'.
        Return a JSON object with 'lead_status', 'time_frame', and 'location'.
        If the query cannot be interpreted, return {{"query_type": "UNINTERPRETED"}}.

        Lead Statuses: "Hot", "Warm", "Cold", "Converted", "Lost", "All" (if no specific status is mentioned but asking for total leads, e.g., "total leads today").
        Time Frames: "TODAY", "YESTERDAY", "LAST_WEEK" (last 7 days), "LAST_MONTH" (last 30 days), "LAST_YEAR" (last 365 days), "ALL_TIME".
        Locations: "New York", "Los Angeles", "Chicago", "Houston", "Miami", "All Locations" (if no specific location is mentioned, e.g., "total leads in New York").

        Examples:
        - User: "how many hot leads last week in New York?"
        - Output: {{"lead_status": "Hot", "time_frame": "LAST_WEEK", "location": "New York"}}

        - User: "total leads today"
        - Output: {{"lead_status": "All", "time_frame": "TODAY", "location": "All Locations"}}

        - User: "cold leads from Houston"
        - Output: {{"lead_status": "Cold", "time_frame": "ALL_TIME", "location": "Houston"}}

        - User: "total conversions"
        - Output: {{"lead_status": "Converted", "time_frame": "ALL_TIME", "location": "All Locations"}}

        - User: "leads lost yesterday"
        - Output: {{"lead_status": "Lost", "time_frame": "YESTERDAY", "location": "All Locations"}}

        - User: "warm leads last month"
        - Output: {{"lead_status": "Warm", "time_frame": "LAST_MONTH", "location": "All Locations"}}

        User Query: "{query_text}"
        """
        
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "system", "content": "You are a helpful AI assistant that interprets user queries about sales data and outputs a JSON object. Only use the provided categories and timeframes."}, {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=150, response_format={"type": "json_object"}
        )
        response_json = json.loads(completion.choices[0].message.content.strip())
        
        lead_status_filter = response_json.get("lead_status")
        time_frame_filter = response_json.get("time_frame")
        location_filter_nlq = response_json.get("location")

        if response_json.get("query_type") == "UNINTERPRETED":
            return {"result_message": "This cannot be processed now - Restricted for demo. Please try queries about specific lead types (Hot, Warm, Cold, Converted, Lost), locations, or timeframes (today, last week, last month)."}

        filtered_df = all_bookings_df.copy()

        # Apply time filter based on the NLQ interpreted timeframe (using UTC dates)
        if time_frame_filter == "TODAY":
            filtered_df = filtered_df[filtered_df['booking_timestamp'].dt.date == today_dt_utc.date()]
        elif time_frame_filter == "YESTERDAY":
            filtered_df = filtered_df[filtered_df['booking_timestamp'].dt.date == yesterday_dt_utc.date()]
        elif time_frame_filter == "LAST_WEEK":
            filtered_df = filtered_df[filtered_df['booking_timestamp'] >= last_week_start_dt_utc]
        elif time_frame_filter == "LAST_MONTH":
            filtered_df = filtered_df[filtered_df['booking_timestamp'] >= last_month_start_dt_utc]
        elif time_frame_filter == "LAST_YEAR":
            filtered_df = filtered_df[filtered_df['booking_timestamp'] >= last_year_start_dt_utc]
        # "ALL_TIME" means no date filter applied here, it relies on sidebar filters

        # Apply lead status filter
        if lead_status_filter and lead_status_filter != "All":
            if lead_status_filter in ["Converted", "Lost"]:
                filtered_df = filtered_df[filtered_df['action_status'] == lead_status_filter]
            else:
                filtered_df = filtered_df[filtered_df['lead_score'].str.lower() == lead_status_filter.lower()]
            
        # Apply location filter
        if location_filter_nlq and location_filter_nlq != "All Locations":
            filtered_df = filtered_df[filtered_df['location'] == location_filter_nlq]
        
        result_count = filtered_df.shape[0]
        
        # --- Format the output message ---
        message_parts = []
        if lead_status_filter and lead_status_filter != "All":
            message_parts.append(f"{lead_status_filter.lower()} leads")
        else:
            message_parts.append("total leads")
        
        if time_frame_filter != "ALL_TIME":
            message_parts.append(f" {time_frame_filter.lower().replace('_', ' ')}")
        
        if location_filter_nlq and location_filter_nlq != "All Locations":
            message_parts.append(f" in {location_filter_nlq}")
        
        # Clarify that results are within the dashboard's sidebar's filters
        # The dashboard passes its filters, so we format the message to reflect them
        sidebar_context_str = ""
        if start_date_str and end_date_str:
            s_date = datetime.strptime(start_date_str, '%Y-%m-%d').strftime('%b %d, %Y')
            e_date = datetime.strptime(end_date_str, '%Y-%m-%d').strftime('%b %d, %Y')
            sidebar_context_str += f" (filtered from {s_date} to {e_date})"
        if selected_location != "All Locations":
            sidebar_context_str += f" in {selected_location}"

        result_message = f"📊 {' '.join(message_parts).capitalize()}: **{result_count}**{sidebar_context_str}"

        # "refine time period" message logic
        start_dt_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_dt_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        if result_count == 0 and (end_dt_date - start_dt_date).days < 7:
            result_message += "<br>Consider expanding the date range in the dashboard's sidebar filters if you expect more results."
        
        return {"result_message": result_message}

    except json.JSONDecodeError:
        logging.error("LLM did not return a valid JSON for /analyze-query.", exc_info=True)
        raise HTTPException(status_code=500, detail="LLM did not return a valid JSON for analytics query.")
    except Exception as e:
        logging.error(f"Error processing analytics query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your analytics query: {e}")


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

            # 2. Re-evaluate sales notes relevance/sentiment (similar to dashboard logic)
            sales_notes = lead_data.get('sales_notes', '')
            notes_relevance = check_notes_relevance_llm(sales_notes)

            if notes_relevance == "IRRELEVANT":
                logging.info(f"Sales notes for {lead_id} deemed irrelevant. Skipping follow-up email.")
                # Optionally, update sales_notes in DB to reflect this analysis
                continue

            notes_sentiment = analyze_sentiment_llm(sales_notes)

            # 3. Get vehicle details (hardcoded locally or fetched)
            vehicle_details = AOE_VEHICLE_DATA.get(lead_data['vehicle'], {})
            current_vehicle_brand_val = lead_data['current_vehicle'].split(' ')[0] if lead_data['current_vehicle'] else None

            if not vehicle_details:
                logging.warning(f"Vehicle details for {lead_data['vehicle']} not found. Skipping follow-up email for {lead_id}.")
                continue

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
            offer_text_markdown, offer_text_html = suggest_offer_llm(lead_data, vehicle_details) # Reusing suggest_offer_llm
            
            if not offer_text_html:
                logging.error(f"Failed to generate offer content for {lead_id}. Skipping.")
                failed_count += 1
                continue

            offer_subject = f"Exclusive Offer for You: Your New {lead_data['vehicle']}!" # Subject line for offer email

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


# --- ENDPOINTS FOR LEAD INSIGHTS (Future extension if needed from agent service) ---
# For now, Lead Insights remains an indicator in dashboard.py, but this service could
# analyze and store insights in a separate DB table if it ran autonomously.
# Example:
# @app.get("/get-lead-insights")
# async def get_lead_insights_endpoint():
#     # Logic to fetch pre-calculated insights from a database table
#     pass

# To run this service locally: uvicorn automotive_agent_service:app --reload --port 8001