# Python + Image and Import + FastAPI + MongoDB + Remark.
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import tempfile
import io
import pymongo
from datetime import datetime
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import speech_recognition as sr
import uuid
import uvicorn

# ==========================
# CONFIGURATION
# ==========================
load_dotenv()

# MongoDB Configuration for Visiting Cards
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/Visiting_Card_DB")
DB_NAME = "Visiting_Card_DB"
CARDS_COLLECTION_NAME = "Visiting_Card"
REMARKS_COLLECTION_NAME = "user_remarks"

# ==========================
# FASTAPI APP
# ==========================
app = FastAPI(
    title="Visiting Card Scanner API",
    description="AI-Powered Visiting Card Data Extraction with Voice Feedback",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# MONGODB SETUP - VISITING CARDS
# ==========================
def get_mongo_client():
    """Get MongoDB client connection."""
    try:
        client = pymongo.MongoClient(MONGODB_URI)
        # Test connection
        client.admin.command('ping')
        return client
    except Exception as e:
        print(f"Failed to connect to MongoDB: {str(e)}")
        return None

def init_database():
    """Initialize database and collection if they don't exist."""
    client = get_mongo_client()
    if client is not None:
        db = client[DB_NAME]
        cards_collection = db[CARDS_COLLECTION_NAME]
        remarks_collection = db[REMARKS_COLLECTION_NAME]
        return client, cards_collection, remarks_collection
    return None, None, None

def save_to_mongodb(data: dict, mobile_numbers: list, image_name: str, image_bytes: bytes = None):
    """Save extracted visiting card data to MongoDB."""
    client, collection, _ = init_database()
    if collection is not None:
        try:
            # Add mobile numbers to data
            for i, mobile_num in enumerate(mobile_numbers, 1):
                data[f"Mobile Number {i}"] = mobile_num
            
            # Create document
            document = {
                "_id": str(uuid.uuid4()),
                "file_name": image_name,
                "extracted_data": data,
                "timestamp": datetime.now(),
                "processed": True
            }
            
            # If image bytes are provided, store them as well (optional)
            if image_bytes:
                document["image_data"] = image_bytes
            
            result = collection.insert_one(document)
            return True
        except Exception as e:
            print(f"Failed to save to MongoDB: {str(e)}")
            return False
    else:
        print("Could not connect to MongoDB. Data not saved.")
        return False

def save_remarks_to_mongodb(remarks_data):
    """Save user remarks/feedback to MongoDB."""
    client, _, remarks_collection = init_database()
    if remarks_collection is not None:
        try:
            # Add timestamp
            remarks_data['submission_timestamp'] = datetime.now()
            result = remarks_collection.insert_one(remarks_data)
            print(f"✅ Remarks saved to MongoDB with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            print(f"❌ Failed to save remarks to MongoDB: {str(e)}")
            return None
    else:
        print("❌ Could not connect to MongoDB. Remarks not saved.")
        return None

# ==========================
# SPEECH TO TEXT FUNCTIONALITY
# ==========================
def record_audio(duration=5):
    """Record audio and return the transcribed text."""
    try:
        print(f"🎤 Recording audio for {duration} seconds...")
        
        # Record audio
        fs = 16000
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            write(tmpfile.name, fs, recording)
            tmp_path = tmpfile.name

        # Convert to text
        r = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            
        # Clean up
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        print(f"📝 Transcribed text: {text}")
        return text
            
    except sr.UnknownValueError:
        print("❌ No speech detected")
        return ""
    except Exception as e:
        print(f"❌ Recording error: {e}")
        return ""

# ==========================
# VISITING CARD DATA EXTRACTION
# ==========================
def extract_data_from_gemini_response(response_text):
    """Extract structured data from Gemini response text"""
    data = {
        "Person Name": "",
        "Company Name": "", 
        "Email": "",
        "Phone Numbers": "",
        "Address": "",
        "Website": "",
        "Country Code": ""
    }
    
    # Initialize mobile numbers
    mobile_numbers = []
    
    if not response_text:
        return data, mobile_numbers

    lines = response_text.strip().split('\n')
    
    for line in lines:
        if ': ' in line:
            key, value = line.split(': ', 1)
            key = key.strip()
            value = value.strip()
            
            # Handle different field names
            if key.lower() in ['person name', 'name']:
                data['Person Name'] = value
            elif key.lower() in ['company name', 'company']:
                data['Company Name'] = value
            elif key.lower() in ['email', 'email address']:
                data['Email'] = value
            elif key.lower() in ['phone numbers', 'phone number']:
                data['Phone Numbers'] = value
            elif key.lower() in ['address']:
                data['Address'] = value
            elif key.lower() in ['website', 'website url', 'url']:
                data['Website'] = value
            elif key.lower() in ['country code']:
                data['Country Code'] = value
            elif key.lower().startswith('mobile number'):
                # Extract mobile numbers
                mobile_numbers.append(value)
    
    return data, mobile_numbers

def extract_visiting_card_with_gemini(file_data, file_type):
    """Extract visiting card information using Gemini AI"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not found in .env file."}

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Enhanced prompt for better extraction
    prompt = """
    Analyze this visiting card image and extract ALL contact information with high accuracy.

    CRITICAL INSTRUCTIONS FOR PHONE NUMBERS:
    - PHONE NUMBERS: Extract ALL 8-digit numbers. If multiple found, combine them comma-separated in one field.
    - MOBILE NUMBERS: Extract ALL 10-digit numbers. List each separately as Mobile Number 1, Mobile Number 2, etc.
    - COUNTRY CODE: Extract the international dialing code (like +91, +1, +44, etc.)

    REQUIRED FIELDS (format exactly as shown):
    Person Name: [Full name of the person]
    Company Name: [Company or organization name]
    Email: [Email address]
    Phone Numbers: [8-digit numbers only, comma separated if multiple. Example: 12345678, 87654321]
    Address: [Complete physical address]
    Website: [Website URL]
    Country Code: [International dialing code like +91, +1, +44]

    FOR MOBILE NUMBERS (10-digit numbers):
    Mobile Number 1: [First 10-digit mobile number]
    Mobile Number 2: [Second 10-digit mobile number]
    Mobile Number 3: [Third 10-digit mobile number]

    IMPORTANT RULES:
    1. Phone Numbers field should ONLY contain 8-digit numbers (comma separated if multiple)
    2. Mobile Number fields should ONLY contain 10-digit numbers (each in separate field)
    3. Extract ALL numbers you find - don't miss any
    4. Be very precise with digit counts
    5. Include country code separately
    6. Only include fields where you find actual information

    Example output format:
    Person Name: John Doe
    Company Name: ABC Corporation
    Email: john.doe@abccorp.com
    Phone Numbers: 22334455, 66778899
    Mobile Number 1: 9876543210
    Mobile Number 2: 9123456789
    Address: 123 Business Park, Sector 25, New Delhi, India
    Website: www.abccorp.com
    Country Code: +91

    Now extract information from this visiting card:
    """

    try:
        if file_type == 'application/pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_data)
                temp_path = temp_file.name
            uploaded = genai.upload_file(temp_path, mime_type="application/pdf")
            response = model.generate_content([uploaded, prompt])
            genai.delete_file(uploaded.name)
            os.unlink(temp_path)
        else:
            image = Image.open(BytesIO(file_data))
            response = model.generate_content([image, prompt])

        if response and hasattr(response, 'text') and response.text.strip():
            # Extract structured data from Gemini response
            data, mobile_numbers = extract_data_from_gemini_response(response.text)
            
            # Add mobile numbers to the main data
            for i, mobile_num in enumerate(mobile_numbers, 1):
                data[f"Mobile Number {i}"] = mobile_num
            
            return {
                "success": True,
                "data": data
            }
        else:
            return {"success": False, "error": "No response from Gemini AI"}
            
    except Exception as e:
        return {"success": False, "error": f"Gemini extraction failed: {str(e)}"}

# ==========================
# PYDANTIC MODELS
# ==========================
class UnifiedRemarksRequest(BaseModel):
    text_remarks: Optional[str] = None
    record_audio: bool = False
    audio_duration: int = 5
    submission_type: str = "general"

# ==========================
# API ROUTES
# ==========================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Visiting Card Scanner API",
        "version": "1.0.0",
        "endpoints": {
            "extract_cards": "/extract_cards (POST)",
            "submit_remarks": "/submit_remarks (POST)",
            "database_status": "/database_status (GET)"
        }
    }

@app.post("/extract_cards")
async def extract_cards(files: List[UploadFile] = File(...)):
    """
    Extract data from visiting card files.
    Supports single or multiple files.
    Automatically saves to database.
    """
    try:
        results = []
        
        for file in files:
            # Read file data
            file_data = await file.read()
            
            # Extract data using Gemini
            extraction_result = extract_visiting_card_with_gemini(file_data, file.content_type)
            
            # Always save to MongoDB (automatically set to True)
            mongo_success = False
            if extraction_result.get("success"):
                mongo_success = save_to_mongodb(
                    extraction_result["data"],
                    [],  # Mobile numbers are now included in the data
                    file.filename,
                    file_data
                )
            
            # Simplified response structure
            file_result = {
                "file_name": file.filename,
                "file_type": file.content_type,
                "file_size": len(file_data),
                "extraction_result": {
                    "success": extraction_result.get("success", False),
                    "data": extraction_result.get("data", {})
                },
                "database_saved": mongo_success
            }
            
            # Add error if extraction failed
            if not extraction_result.get("success"):
                file_result["extraction_result"]["error"] = extraction_result.get("error", "Unknown error")
            
            results.append(file_result)
        
        return {
            "success": True,
            "total_files": len(results),
            "processed_files": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

@app.post("/submit_remarks")
async def submit_remarks(request: UnifiedRemarksRequest):
    """
    Unified remarks endpoint - handles both text and voice remarks
    Returns the remarks data back to user
    """
    try:
        final_remarks = ""
        remarks_type = ""
        
        # Case 1: Record audio if requested
        if request.record_audio:
            print("🎤 Recording audio for remarks...")
            transcribed_text = record_audio(request.audio_duration)
            
            if transcribed_text:
                final_remarks = transcribed_text
                remarks_type = "audio"
                print(f"✅ Voice remarks recorded: {final_remarks}")
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": "No speech detected during audio recording",
                        "remarks_text": "",
                        "remarks_type": "",
                        "remarks_id": None
                    }
                )
        
        # Case 2: Use text remarks if provided
        elif request.text_remarks and request.text_remarks.strip():
            final_remarks = request.text_remarks.strip()
            remarks_type = "text"
            print(f"✅ Text remarks received: {final_remarks}")
        
        # Case 3: Neither text nor successful audio recording
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Please provide either text remarks or enable audio recording",
                    "remarks_text": "",
                    "remarks_type": "",
                    "remarks_id": None
                }
            )
        
        # Prepare remarks data for database
        remarks_data = {
            'remarks': final_remarks,
            'submission_type': request.submission_type,
            'remarks_type': remarks_type,
            'timestamp': datetime.now()
        }
        
        # Save to MongoDB
        remarks_id = save_remarks_to_mongodb(remarks_data)
        
        if remarks_id:
            return {
                "success": True,
                "remarks_id": str(remarks_id),
                "remarks_text": final_remarks,  # Return the remarks back to user
                "remarks_type": remarks_type,   # Return the type (text/audio)
                "submission_type": request.submission_type,
                "message": "Remarks submitted successfully"
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to save remarks to database",
                    "remarks_text": final_remarks,  # Still return the remarks even if DB fails
                    "remarks_type": remarks_type,
                    "remarks_id": None
                }
            )
            
    except Exception as e:
        print(f"❌ Error in submit_remarks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit remarks: {str(e)}"
        )

@app.get("/database_status")
async def get_database_status():
    """Get database connection status and statistics"""
    try:
        client = get_mongo_client()
        if client is not None:
            db = client[DB_NAME]
            cards_collection = db[CARDS_COLLECTION_NAME]
            remarks_collection = db[REMARKS_COLLECTION_NAME]
            
            cards_count = cards_collection.count_documents({})
            remarks_count = remarks_collection.count_documents({})
            
            client.close()
            
            return {
                "connected": True,
                "total_cards": cards_count,
                "total_remarks": remarks_count,
                "message": "Database connected successfully"
            }
        else:
            return {
                "connected": False,
                "total_cards": 0,
                "total_remarks": 0,
                "message": "Database connection failed"
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking database status: {str(e)}"
        )

# ==========================
# MAIN APPLICATION
# ==========================
if __name__ == "__main__":
    print("🚀 Starting Visiting Card Scanner API...")
    print("📝 Unified remarks endpoint available at: POST /submit_remarks")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )