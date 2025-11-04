# Updated code with python + Streamlit + Image and Import + MongoDB + Speech to Text.
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import re
import json
import tempfile
import time
import io
import pymongo
from datetime import datetime
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import speech_recognition as sr
import uuid

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
        st.error(f"Failed to connect to MongoDB: {str(e)}")
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
            st.error(f"Failed to save to MongoDB: {str(e)}")
            return False
    else:
        st.warning("Could not connect to MongoDB. Data not saved.")
        return False

def save_remarks_to_mongodb(remarks_data):
    """Save user remarks/feedback to MongoDB."""
    client, _, remarks_collection = init_database()
    if remarks_collection is not None:
        try:
            # Add timestamp
            remarks_data['submission_timestamp'] = datetime.now()
            result = remarks_collection.insert_one(remarks_data)
            return result.inserted_id
        except Exception as e:
            st.error(f"Failed to save remarks to MongoDB: {str(e)}")
            return None
    else:
        st.warning("Could not connect to MongoDB. Remarks not saved.")
        return None

# ==========================
# SPEECH TO TEXT FUNCTIONALITY
# ==========================
def record_audio(duration=10):
    """Record audio and return the transcribed text."""
    try:
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
            
        return text
            
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        st.error(f"Recording error: {e}")
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

def format_data_as_text(data: dict, mobile_numbers: list):
    """Format extracted data as plain text for display"""
    text_output = "📇 EXTRACTED INFORMATION:\n\n"
    found_data = False
    
    # Base fields
    base_fields = ["Person Name", "Company Name", "Email", "Phone Numbers", "Address", "Website", "Country Code"]
    
    for key in base_fields:
        value = data.get(key, "")
        if value and value.strip():
            text_output += f"• {key}: {value}\n"
            found_data = True
    
    # Mobile numbers
    for i, mobile_num in enumerate(mobile_numbers, 1):
        if mobile_num and mobile_num.strip():
            text_output += f"• Mobile Number {i}: {mobile_num}\n"
            found_data = True
    
    if not found_data:
        text_output += "No information could be extracted from this card.\n"
    
    text_output += "\n" + "="*50 + "\n"
    return text_output

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
            
            # Format as text for display
            text_output = format_data_as_text(data, mobile_numbers)
            
            return {
                "success": True,
                "data": data,
                "mobile_numbers": mobile_numbers,
                "text_output": text_output,
                "formatted_data": {
                    "person_name": data.get("Person Name", ""),
                    "company_name": data.get("Company Name", ""),
                    "email": data.get("Email", ""),
                    "phone_numbers": data.get("Phone Numbers", ""),
                    "address": data.get("Address", ""),
                    "website": data.get("Website", ""),
                    "country_code": data.get("Country Code", ""),
                    "mobile_numbers_list": mobile_numbers
                }
            }
        else:
            return {"error": "No response from Gemini AI"}
            
    except Exception as e:
        return {"error": f"Gemini extraction failed: {str(e)}"}

# ==========================
# STREAMLIT UI - SAME AS SERVICE INVOICE
# ==========================
st.set_page_config(
    page_title="Visiting Card Scanner", 
    layout="centered",
    page_icon="📇",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling (same as service invoice)
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #6c757d;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .mode-button {
        height: 120px;
        border: none;
        border-radius: 15px;
        font-size: 1.3rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .mode-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .camera-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .import-btn {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .success-box {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .processing-box {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .file-dropdown {
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0;
    }
    .remarks-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        margin-top: 2rem;
    }
    .text-output {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        font-family: monospace;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'current_file_index' not in st.session_state:
    st.session_state.current_file_index = 0
if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = []
if 'processing_started' not in st.session_state:
    st.session_state.processing_started = False
if 'file_data_cache' not in st.session_state:
    st.session_state.file_data_cache = {}
if 'user_remarks' not in st.session_state:
    st.session_state.user_remarks = ""
if 'show_remarks_section' not in st.session_state:
    st.session_state.show_remarks_section = False
if 'just_recorded' not in st.session_state:
    st.session_state.just_recorded = False
if 'camera_extraction_result' not in st.session_state:
    st.session_state.camera_extraction_result = None
if 'camera_image_data' not in st.session_state:
    st.session_state.camera_image_data = None

# Main header
st.markdown('<div class="main-header">📇 Visiting Card Scanner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Visiting Card Data Extraction with Voice Feedback</div>', unsafe_allow_html=True)

# Show mode selection if no mode is selected
if st.session_state.current_mode is None:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "📷 **Click Visiting Card**\n\nUse camera to capture card", 
            key="camera_mode", 
            use_container_width=True,
            help="Take a photo of your visiting card using your camera"
        ):
            st.session_state.current_mode = "camera"
            st.rerun()
    
    with col2:
        if st.button(
            "📁 **Import Visiting Card**\n\nUpload PDF or image files", 
            key="import_mode", 
            use_container_width=True,
            help="Upload multiple visiting card files for batch processing"
        ):
            st.session_state.current_mode = "import"
            st.rerun()
    
    # Add feature highlights
    st.markdown("""
    <div style='text-align: center; margin-top: 3rem; padding: 2rem; background: #f8f9fa; border-radius: 12px;'>
        <h4 style='color: #495057; margin-bottom: 1rem;'>✨ Key Features</h4>
        <div style='display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;'>
            <div style='flex: 1; min-width: 150px;'>
                <h5 style='color: #667eea;'>👤 Contact Extraction</h5>
                <p style='font-size: 0.9rem; color: #6c757d;'>Name, company, email</p>
            </div>
            <div style='flex: 1; min-width: 150px;'>
                <h5 style='color: #667eea;'>📞 Phone Detection</h5>
                <p style='font-size: 0.9rem; color: #6c757d;'>Mobile & phone numbers</p>
            </div>
            <div style='flex: 1; min-width: 150px;'>
                <h5 style='color: #667eea;'>💾 Database Storage</h5>
                <p style='font-size: 0.9rem; color: #6c757d;'>Secure MongoDB storage</p>
            </div>
            <div style='flex: 1; min-width: 150px;'>
                <h5 style='color: #667eea;'>🎤 Voice Feedback</h5>
                <p style='font-size: 0.9rem; color: #6c757d;'>Speech-to-text remarks</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Camera Mode
elif st.session_state.current_mode == "camera":
    st.header("📷 Capture Visiting Card")
    
    # Back button
    if st.button("← Back", key="camera_back"):
        st.session_state.current_mode = None
        st.session_state.show_remarks_section = False
        st.session_state.camera_extraction_result = None
        st.session_state.camera_image_data = None
        st.rerun()
    
    # Show camera extraction result if available
    if st.session_state.camera_extraction_result:
        st.success("✅ Data extracted successfully!")
        
        # Display results in the same format as import mode
        st.subheader("📊 Extracted Data")
        
        if st.session_state.camera_extraction_result.get("success"):
            result = st.session_state.camera_extraction_result
            
            # Display in the same format as import mode
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Original Image:**")
                if st.session_state.camera_image_data:
                    st.image(st.session_state.camera_image_data, caption="Captured Visiting Card", use_container_width=True)
            
            with col2:
                st.write("**Extracted Data:**")
                st.markdown(f'<div class="text-output">{result["text_output"]}</div>', unsafe_allow_html=True)
                
                with st.expander("📋 Detailed View", expanded=False):
                    st.json(result["formatted_data"])
            
            # Show remarks section
            if not st.session_state.show_remarks_section:
                if st.button("💬 Add Remarks", type="primary", key="add_remarks_camera"):
                    st.session_state.show_remarks_section = True
                    st.rerun()
        
        # Reset button to capture another card
        if st.button("🔄 Capture Another Card", key="capture_another"):
            st.session_state.camera_extraction_result = None
            st.session_state.camera_image_data = None
            st.session_state.show_remarks_section = False
            st.rerun()
    
    else:
        # Camera input when no extraction result is available
        st.markdown("Position your visiting card clearly in the camera frame and capture the image.")
        
        camera_image = st.camera_input("Take a picture of your visiting card")
        
        if camera_image:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(camera_image, caption="Captured Visiting Card", use_container_width=True)
            
            with col2:
                if st.button("🔍 Extract Data", type="primary", use_container_width=True):
                    with st.spinner("🤖 Analyzing visiting card with AI..."):
                        result = extract_visiting_card_with_gemini(camera_image.getvalue(), "image/jpeg")
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        # Store the result in session state for consistent display
                        st.session_state.camera_extraction_result = result
                        st.session_state.camera_image_data = camera_image
                        
                        # Save to MongoDB
                        if result.get("success"):
                            # Convert image to bytes for storage
                            img_byte_arr = io.BytesIO()
                            Image.open(BytesIO(camera_image.getvalue())).save(img_byte_arr, format='PNG')
                            img_byte_arr = img_byte_arr.getvalue()
                            
                            mongo_success = save_to_mongodb(
                                result["data"], 
                                result["mobile_numbers"], 
                                "camera_capture.png", 
                                img_byte_arr
                            )
                            
                            if mongo_success:
                                st.success("✅ Data saved to database")
                        
                        st.rerun()

# Import Mode
elif st.session_state.current_mode == "import":
    st.header("📁 Import Visiting Cards")
    
    # Back button
    if st.button("← Back", key="import_back"):
        st.session_state.current_mode = None
        st.session_state.uploaded_files = []
        st.session_state.current_file_index = 0
        st.session_state.extraction_results = []
        st.session_state.processing_started = False
        st.session_state.file_data_cache = {}
        st.session_state.show_remarks_section = False
        st.rerun()
    
    uploaded_files = st.file_uploader(
        "Select visiting card files", 
        type=["pdf", "jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        help="Supported formats: PDF, JPG, JPEG, PNG"
    )
    
    if uploaded_files and not st.session_state.uploaded_files:
        # Store files in session state for sequential processing
        st.session_state.uploaded_files = uploaded_files
        st.session_state.current_file_index = 0
        st.session_state.extraction_results = []
        st.session_state.processing_started = False
        
        # Cache file data for later display
        for file in uploaded_files:
            st.session_state.file_data_cache[file.name] = file.getvalue()
        
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
    
    # Show extract button if files are uploaded but processing hasn't started
    if (st.session_state.uploaded_files and 
        not st.session_state.processing_started):
        
        if st.button("🚀 Start Processing All Files", type="primary", use_container_width=True):
            st.session_state.processing_started = True
            st.rerun()
    
    # Auto-process files when processing is started
    if (st.session_state.processing_started and 
        st.session_state.current_file_index < len(st.session_state.uploaded_files)):
        
        current_index = st.session_state.current_file_index
        current_file = st.session_state.uploaded_files[current_index]
        total_files = len(st.session_state.uploaded_files)
        
        # Progress bar and status
        progress = current_index / total_files
        st.progress(progress)
        st.write(f"**Processing:** {current_index + 1} of {total_files} files - {current_file.name}")
        
        # Process current file
        with st.spinner(f"🤖 Analyzing {current_file.name}..."):
            file_data = st.session_state.file_data_cache[current_file.name]
            result = extract_visiting_card_with_gemini(file_data, current_file.type)
        
        # Store result
        if "error" in result:
            status = "❌ Failed"
            st.error(f"Error processing {current_file.name}: {result['error']}")
        else:
            status = "✅ Success"
            # Save to MongoDB
            mongo_success = False
            if result.get("success"):
                # Convert image to bytes for storage
                img_byte_arr = io.BytesIO()
                if current_file.type.startswith('image'):
                    Image.open(BytesIO(file_data)).save(img_byte_arr, format='PNG')
                else:
                    # For PDF, we'll store a placeholder
                    img_byte_arr = None
                
                mongo_success = save_to_mongodb(
                    result["data"], 
                    result["mobile_numbers"], 
                    current_file.name, 
                    img_byte_arr.getvalue() if img_byte_arr else None
                )
        
        st.session_state.extraction_results.append({
            'file_name': current_file.name,
            'file_type': current_file.type,
            'file_data': st.session_state.file_data_cache[current_file.name],
            'result': result,
            'mongo_success': mongo_success if 'error' not in result else False,
            'status': status
        })
        
        # Move to next file
        st.session_state.current_file_index += 1
        
        # Auto-refresh to process next file
        if st.session_state.current_file_index < len(st.session_state.uploaded_files):
            st.rerun()
        else:
            # All files processed
            st.session_state.processing_started = False
            st.session_state.show_remarks_section = True
            st.rerun()
    
    # Show results when all files are processed
    if (st.session_state.uploaded_files and 
        st.session_state.current_file_index >= len(st.session_state.uploaded_files) and
        len(st.session_state.extraction_results) > 0):
        
        st.success(f"🎉 Processing completed! {len(st.session_state.extraction_results)} files processed.")
        
        # Show dropdowns for each file's results
        st.subheader("📊 Extraction Results")
        
        for i, result_data in enumerate(st.session_state.extraction_results):
            with st.expander(f"{result_data['status']} - {result_data['file_name']}", expanded=False):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Original File:**")
                    if result_data['file_type'].startswith('image'):
                        st.image(result_data['file_data'], caption=result_data['file_name'], use_container_width=True)
                    else:
                        st.info(f"📘 PDF Document: {result_data['file_name']}")
                
                with col2:
                    st.write("**Extracted Data:**")
                    if "error" in result_data['result']:
                        st.error(result_data['result']['error'])
                    else:
                        if result_data['result'].get("success"):
                            st.markdown(f'<div class="text-output">{result_data["result"]["text_output"]}</div>', unsafe_allow_html=True)
                            
                            with st.expander("📋 Detailed View", expanded=False):
                                st.json(result_data["result"]["formatted_data"])
                            
                            if result_data['mongo_success']:
                                st.success("✅ Saved to database")
                            else:
                                st.warning("⚠️ Data extracted but not saved to database")
                        else:
                            st.error("Failed to extract data")

# ==========================
# REMARKS/FEEDBACK SECTION - SAME AS SERVICE INVOICE
# ==========================
if st.session_state.show_remarks_section:
    st.markdown("---")
    st.markdown('<div class="remarks-section">', unsafe_allow_html=True)
    st.header("💬 User Remarks & Feedback")
    st.markdown("Please provide your feedback or remarks about the extraction results. You can type your message or use the microphone to speak.")
    
    # Handle recording if it just happened
    if st.session_state.just_recorded:
        st.session_state.just_recorded = False
        # Force update the text area
        st.rerun()
    
    # Text input area - use a unique key
    remarks_key = "remarks_textarea_" + str(hash(st.session_state.user_remarks))
    remarks_text = st.text_area(
        "🗒️ Your Remarks:",
        value=st.session_state.user_remarks,
        height=150,
        placeholder="Type your feedback here or use the microphone below to speak...",
        key=remarks_key
    )
    
    # Update session state with text input
    if remarks_text != st.session_state.user_remarks:
        st.session_state.user_remarks = remarks_text
    
    # Speech to text section
    st.subheader("🎤 Voice Input")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        recording_duration = st.slider(
            "🎧 Recording Duration (seconds)",
            min_value=2,
            max_value=10,
            value=5,
            key="recording_duration"
        )
    
    with col2:
        # Use a form with a unique key for the record button
        record_form_key = "record_form_" + str(hash(st.session_state.user_remarks))
        with st.form(key=record_form_key):
            record_submitted = st.form_submit_button("🎙️ Record Voice", use_container_width=True)
            
            if record_submitted:
                with st.spinner("🎤 Recording... Speak now!"):
                    transcribed_text = record_audio(recording_duration)
                
                if transcribed_text:
                    # Update the remarks directly
                    if st.session_state.user_remarks:
                        st.session_state.user_remarks += " " + transcribed_text
                    else:
                        st.session_state.user_remarks = transcribed_text
                    
                    st.session_state.just_recorded = True
                    st.success("✅ Voice recorded and text added!")
                    st.rerun()
    
    with col3:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_remarks"):
            st.session_state.user_remarks = ""
            st.rerun()
    
    # Submit button
    submit_col1, submit_col2 = st.columns([1, 1])
    with submit_col1:
        if st.button("✅ Submit Remarks", type="primary", use_container_width=True, key="submit_remarks"):
            if st.session_state.user_remarks.strip():
                # Prepare remarks data
                remarks_data = {
                    'remarks': st.session_state.user_remarks.strip(),
                    'submission_type': 'camera' if st.session_state.current_mode == 'camera' else 'import',
                    'file_count': 1 if st.session_state.current_mode == 'camera' else len(st.session_state.extraction_results),
                    'extraction_results_count': len([r for r in st.session_state.extraction_results if r['result'].get('success')]) if st.session_state.current_mode == 'import' else 1
                }
                
                # Save to MongoDB
                remarks_id = save_remarks_to_mongodb(remarks_data)
                if remarks_id:
                    st.success("✅ Your remarks have been submitted successfully!")
                    st.session_state.user_remarks = ""
                    st.session_state.show_remarks_section = False
                    st.rerun()
                else:
                    st.error("❌ Failed to save remarks to database.")
            else:
                st.warning("⚠️ Please enter some remarks before submitting.")
    
    with submit_col2:
        if st.button("❌ Cancel", use_container_width=True, key="cancel_remarks"):
            st.session_state.user_remarks = ""
            st.session_state.show_remarks_section = False
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Reset buttons for import mode after processing
if (st.session_state.current_mode == "import" and 
    st.session_state.uploaded_files and 
    st.session_state.current_file_index >= len(st.session_state.uploaded_files) and
    not st.session_state.show_remarks_section):
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Process More Files", use_container_width=True, key="process_more"):
            st.session_state.uploaded_files = []
            st.session_state.current_file_index = 0
            st.session_state.extraction_results = []
            st.session_state.processing_started = False
            st.session_state.file_data_cache = {}
            st.session_state.show_remarks_section = False
            st.rerun()
    with col2:
        if st.button("🏠 Back to Main Menu", use_container_width=True, key="back_main"):
            st.session_state.current_mode = None
            st.session_state.uploaded_files = []
            st.session_state.current_file_index = 0
            st.session_state.extraction_results = []
            st.session_state.processing_started = False
            st.session_state.file_data_cache = {}
            st.session_state.show_remarks_section = False
            st.rerun()

# MongoDB connection status in sidebar
with st.sidebar:
    st.header("🔧 System Status")
    client = get_mongo_client()
    if client is not None:
        st.success("✅ **Database Connected**")
        try:
            db = client[DB_NAME]
            cards_collection = db[CARDS_COLLECTION_NAME]
            count = cards_collection.count_documents({})
            st.info(f"📇 Total Cards: {count}")
            
            remarks_collection = db[REMARKS_COLLECTION_NAME]
            remarks_count = remarks_collection.count_documents({})
            st.info(f"💬 Total Remarks: {remarks_count}")
        except:
            pass
        client.close()
    else:
        st.error("❌ **Database Disconnected**")
    
    st.header("ℹ️ About")
    st.info("""
    **Visiting Card Scanner** uses advanced AI to automatically extract and validate data from visiting cards.
    
    **Extracts:**
    • Person & Company Names
    • Email Addresses  
    • Phone & Mobile Numbers
    • Address Information
    • Website URLs
    • Country Codes
    
    **New Features:**
    • 🎤 Voice-to-text remarks
    • 💬 User feedback system
    • 💾 Secure database storage
    • 📱 Mobile number detection
    """)
