
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# ==========================================
# 1. CONFIGURATION & API SETUP
# ==========================================

# Set the API key for Google Gemini
api_key = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_AI_STUDIO_KEY_HERE")
os.environ["GOOGLE_API_KEY"] = api_key

# Debug: Print if API key is loaded (mask for safety)
if api_key and api_key != "YOUR_GOOGLE_AI_STUDIO_KEY_HERE":
    print(f"Loaded GOOGLE_API_KEY: {api_key[:6]}...{api_key[-4:]}")
else:
    print("Warning: GOOGLE_API_KEY not set or using default placeholder!")

INPUT_FILE = "minutes.txt"
OUTPUT_FILE = "formatted_minutes.jsonl"


# ==========================================
# 2. DATA SCHEMA (Pydantic Model)
# ==========================================
class MeetingMinutes(BaseModel):
    date: str = Field(
        description="The date of the meeting, standardized to YYYY-MM-DD")
    location: str = Field(
        description="The physical location or home where the meeting occurred"
    )
    attendance_members: List[str] = Field(
        default=[], description="List of full names of members present"
    )
    attendance_guests: List[str] = Field(
        default=[], description="List of guests, prospective members, or pledges"
    )
    treasurer_report: Dict[str, Any] = Field(
        default={},
        description="Key-value pairs of financial data (e.g., 'checking': 200.00, 'savings': 150.00)",
    )
    motions: List[Dict[str, str]] = Field(
        default=[],
        description="List of motions including 'description', 'proposed_by', and 'result' (passed/failed/carried)",
    )
    key_events: List[str] = Field(
        default=[],
        description="Bullet points of important discussions, upcoming runs, or club decisions",
    )
    next_meeting_info: Optional[str] = Field(
        description="Details about when and where the next meeting is held"
    )


# ==========================================
# 3. CORE LOGIC
# ==========================================

def split_into_meetings(text: str) -> List[str]:
    """
    Splits the giant text file into chunks based on common separators.
    Handles both '***' and '---' as well as 'Meeting:'.
    """
    # Matches lines with only *** or --- (3 or more), or 'Meeting:' as a separator
    chunks = re.split(r"\n\s*(?:\*{3,}|-{3,}|Meeting:)\s*\n?", text)
    return [c.strip() for c in chunks if len(c.strip()) > 50]

def standardize_date(date_str: str) -> str:
    """
    Converts various date formats (e.g., 'January 21, 2004', 'Oct 6, 1971') to YYYY-MM-DD.
    Returns empty string if parsing fails.
    """
    import datetime
    import calendar

    # Try common formats
    formats = [
        "%B %d, %Y",   # January 21, 2004
        "%b %d, %Y",   # Jan 21, 2004
        "%B %d %Y",    # January 21 2004
        "%b %d %Y",    # Jan 21 2004
        "%m/%d/%Y",    # 01/21/2004
        "%Y-%m-%d",    # 2004-01-21
    ]
    date_str = date_str.strip()
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    # Try to extract month name, day, year manually
    match = re.match(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", date_str)
    if match:
        month = match.group(1)
        day = int(match.group(2))
        year = int(match.group(3))
        try:
            month_num = list(calendar.month_name).index(month) if month in calendar.month_name else list(calendar.month_abbr).index(month)
            dt = datetime.date(year, month_num, day)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return ""



def run_conversion():
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    structured_llm = llm.with_structured_output(MeetingMinutes)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert historical archivist for a motorcycle club. "
                "Extract all possible structured data from the following meeting minutes. "
                "Output a JSON object with these fields: "
                "date (YYYY-MM-DD), location, attendance_members (list of names), attendance_guests (list of names), "
                "treasurer_report (dictionary of fund names and amounts), motions (list of objects with description, proposed_by, result), "
                "key_events (list of strings), next_meeting_info (string), and original_text (the full input text). "
                "If a field is missing or not found, use an empty value (empty string, empty list, or empty dict as appropriate). "
                "Do not hallucinate data. Only use what is present in the text. "
                "Always standardize dates to YYYY-MM-DD. "
                "Always include the full input text in the 'original_text' field.",
            ),
            ("human", "Extract the data from these minutes:\n\n{minutes}"),
        ]
    )

    chain = prompt | structured_llm

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please create it.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        full_text = f.read()

    meetings = split_into_meetings(full_text)
    print(f"Found {len(meetings)} meeting entries. Starting conversion...")

    for i, meeting_chunk in enumerate(meetings):
        print(f"[{i+1}/{len(meetings)}] Processing...")
        try:
            response = chain.invoke({"minutes": meeting_chunk})
            meeting_dict = response.model_dump()
            meeting_dict["original_text"] = meeting_chunk
            # Standardize date if present
            if "date" in meeting_dict and meeting_dict["date"]:
                meeting_dict["date"] = standardize_date(meeting_dict["date"])
            with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(meeting_dict) + "\n")
            print(f"      Success: Saved meeting from {meeting_dict['date']}")
            time.sleep(6)
        except Exception as e:
            print(f"      Error on entry {i+1}: {e}")
            with open("errors.log", "a") as err_log:
                err_log.write(
                    f"Entry {i+1} failed: {str(e)}\n---\n{meeting_chunk}\n\n")
            time.sleep(10)
            continue
    print(f"\nProcessing Complete! Your data is in {OUTPUT_FILE}")


if __name__ == "__main__":
    run_conversion()
