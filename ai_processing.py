import cv2
import time
import base64
import os
import sys
import traceback
import json
from openai import OpenAI
from dotenv import load_dotenv
from ultralytics import YOLO

# Set up detailed logging
def log_debug(message):
    """Print a debug message with timestamp that will appear in Android logs"""
    print(f"[DEBUG][{time.strftime('%H:%M:%S')}] {message}", flush=True)

def log_info(message):
    """Print an info message with timestamp that will appear in Android logs"""
    print(f"[INFO][{time.strftime('%H:%M:%S')}] {message}", flush=True)

def log_error(message):
    """Print an error message with timestamp that will appear in Android logs"""
    print(f"[ERROR][{time.strftime('%H:%M:%S')}] {message}", flush=True)

# Log every import to help diagnose import errors
log_info("Successfully imported cv2, time, base64, os, sys, traceback, json")
log_info("Imported OpenAI, dotenv and YOLO modules")

# Load environment variables from .env file
log_info("Loading environment variables from .env file")
load_dotenv()
log_info("Environment variables loaded")

# --- Configuration ---
# Make sure to set the OPENAI_API_KEY environment variable
# You can get one from https://platform.openai.com/account/api-keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# What do you want to ask the AI about the image?
OPENAI_PROMPT = os.getenv("OPENAI_PROMPT", "What is in this image?") # Default prompt if not set
# OpenAI Analysis Condition
OPENAI_CONDITION_PERSON_STR = os.getenv("OPENAI_CONDITION_PERSON", "true") # Default to true
# New: Option to bypass YOLO
PROCESS_WITH_YOLO_FIRST_STR = os.getenv("PROCESS_WITH_YOLO_FIRST", "true") # Default to true

log_info(f"OPENAI_PROMPT: {OPENAI_PROMPT}")
log_info(f"OPENAI_CONDITION_PERSON: {OPENAI_CONDITION_PERSON_STR}")
log_info(f"PROCESS_WITH_YOLO_FIRST: {PROCESS_WITH_YOLO_FIRST_STR}")

OPENAI_CONDITION_PERSON = OPENAI_CONDITION_PERSON_STR.lower() == 'true'
PROCESS_WITH_YOLO_FIRST = PROCESS_WITH_YOLO_FIRST_STR.lower() == 'true'

if not OPENAI_API_KEY:
    log_error("OPENAI_API_KEY environment variable not set.")
    log_info("Current environment variables:")
    for key, value in os.environ.items():
        if "KEY" in key:
            log_info(f"  {key}: {'*' * 5}")
        else:
            log_info(f"  {key}: {value}")
    exit()
else:
    # Mask key for logging
    masked_key = OPENAI_API_KEY[:4] + "..." + OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 8 else "***"
    log_info(f"OpenAI API key found (masked): {masked_key}")

try:
    log_info("Creating OpenAI client")
    client = OpenAI(api_key=OPENAI_API_KEY)
    log_info("OpenAI client created successfully")
except Exception as e:
    log_error(f"Failed to create OpenAI client: {e}")
    log_error(traceback.format_exc())

# --- YOLO Setup ---
# Load a pretrained YOLO model globally to avoid reloading on each call
# Ensure 'yolov8n.pt' (or your chosen model) is accessible
log_info("Starting YOLO model initialization")
try:
    log_info("Attempting to load YOLO model (yolov8n.pt)")
    start_time = time.time()
    yolo_model = YOLO("yolov8n.pt") # Using yolov8n.pt as a common default, adjust if needed
    load_time = time.time() - start_time
    log_info(f"YOLO model loaded successfully in {load_time:.2f} seconds")
    
    # Log model details for debugging
    model_info = {
        "model_type": "yolov8n.pt",
        "task": yolo_model.task,
        "device": str(yolo_model.device),
        "names": yolo_model.names
    }
    log_info(f"YOLO model info: {json.dumps(model_info)}")
except Exception as e:
    log_error(f"Error loading YOLO model: {e}")
    log_error("Please ensure the YOLO model file (e.g., 'yolov8n.pt') is available.")
    log_error(traceback.format_exc())
    exit()

def encode_image_to_base64(image_frame):
    """Encodes a numpy array image to a base64 string."""
    log_debug("Starting image encoding to base64")
    try:
        start_time = time.time()
        height, width, channels = image_frame.shape
        log_debug(f"Image dimensions: {width}x{height}x{channels}")
        
        _, buffer = cv2.imencode(".jpg", image_frame)
        base64_str = base64.b64encode(buffer).decode("utf-8")
        
        # Compute size for logging
        original_size = len(buffer)
        encoded_size = len(base64_str)
        encoding_time = time.time() - start_time
        
        log_info(f"Image encoded: {original_size} bytes -> {encoded_size} bytes base64 in {encoding_time:.2f} seconds")
        return base64_str
    except Exception as e:
        log_error(f"Error during image encoding: {e}")
        log_error(traceback.format_exc())
        raise e

def analyze_image_with_openai(base64_image):
    """Sends the image to OpenAI Vision API and returns the description."""
    log_info("Starting OpenAI image analysis")
    try:
        log_info(f"Using model: gpt-4o with prompt: '{OPENAI_PROMPT}'")
        log_debug(f"Base64 image length: {len(base64_image)} characters")
        
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4o", # Or use the latest vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OPENAI_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        api_time = time.time() - start_time
        log_info(f"OpenAI API call completed in {api_time:.2f} seconds")
        
        if response.choices:
            result = response.choices[0].message.content
            log_info(f"OpenAI result: {result[:100]}...")
            return result
        else:
            log_error("No choices in OpenAI response")
            log_debug(f"Full response: {response}")
            return "No description returned from API."
    except Exception as e:
        log_error(f"Error calling OpenAI API: {e}")
        log_error(traceback.format_exc())
        
        # Check for common OpenAI errors
        error_msg = str(e)
        if "API key" in error_msg:
            return "Error: Invalid or missing OpenAI API key. Please update your .env file."
        elif "timeout" in error_msg.lower():
            return "Error: OpenAI API request timed out. Please try again."
        elif "rate limit" in error_msg.lower():
            return "Error: OpenAI rate limit exceeded. Please try again in a minute."
        
        return f"Error analyzing image: {str(e)}"

def analyze_image_with_yolo(image_frame):
    """Analyzes an image frame using YOLO and prints results."""
    log_info("Starting YOLO object detection")
    try:
        # Log image information for debugging
        height, width, channels = image_frame.shape
        log_debug(f"Input image for YOLO: {width}x{height}x{channels}")
        
        # Run YOLO inference
        start_time = time.time()
        results = yolo_model(image_frame)
        detection_time = time.time() - start_time
        log_info(f"YOLO detection completed in {detection_time:.2f} seconds")

        if results:
            # Count detected objects
            total_objects = len(results[0].boxes)
            log_info(f"YOLO detected {total_objects} objects")
            
            # Log object classes and confidence
            detected_classes = {}
            for box in results[0].boxes:
                class_id = int(box.cls[0])  # Get class ID
                class_name = yolo_model.names[class_id]
                confidence = float(box.conf[0])  # Get confidence
                
                # Count occurrences of each class
                if class_name in detected_classes:
                    detected_classes[class_name].append(confidence)
                else:
                    detected_classes[class_name] = [confidence]
            
            # Log found objects and their average confidence
            for class_name, confidences in detected_classes.items():
                avg_conf = sum(confidences) / len(confidences)
                log_info(f"Detected {len(confidences)}x {class_name} (avg conf: {avg_conf:.2f})")
            
            # Disabled showing results as this is running in a background thread on Android
            # results[0].show()
            return results  # Return the full results object
        else:
            log_info("No objects detected by YOLO")
            return None
    except Exception as e:
        log_error(f"Error during YOLO analysis: {e}")
        log_error(traceback.format_exc())
        return None

def check_for_person(results, model_names):
    """Checks YOLO results for the presence of a 'person' class."""
    log_debug("Checking for person in YOLO results")
    if not results or not results[0].boxes:
        log_info("No detection boxes in YOLO results")
        return False  # No results or no boxes means no person

    # Count total boxes
    total_boxes = len(results[0].boxes)
    log_debug(f"Found {total_boxes} detection boxes in results")
    
    # Extract person instances
    person_confidences = []
    person_boxes = []
    
    for i, box in enumerate(results[0].boxes):
        class_id = int(box.cls[0])  # Get class ID
        class_name = model_names[class_id].lower()
        confidence = float(box.conf[0])
        
        if class_name == 'person':
            person_confidences.append(confidence)
            box_coords = box.xyxy[0].tolist()  # Get box coordinates
            person_boxes.append(box_coords)
            log_debug(f"Person #{len(person_confidences)} detected with confidence {confidence:.2f} at {box_coords}")
    
    if person_confidences:
        avg_confidence = sum(person_confidences) / len(person_confidences)
        log_info(f"✓ Found {len(person_confidences)} persons with avg confidence {avg_confidence:.2f}")
        return True  # Person found
    else:
        log_info("✗ No persons detected in image")
        return False  # No person found

def process_image_for_analysis(image_frame):
    """
    Processes a single image frame (numpy array) with YOLO and conditionally with OpenAI.
    This is the primary function to call for analyzing an image.
    
    Returns the analysis result as a string.
    """
    # Track overall processing time
    overall_start_time = time.time()
    log_info(f"===== Starting AI image analysis process at {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    
    if image_frame is None:
        log_error("Error: Input image_frame is None. Cannot process.")
        return "Error: Could not process the image. The provided image data is invalid."

    try:
        # Log image details
        try:
            height, width, channels = image_frame.shape
            log_info(f"Processing image: {width}x{height}x{channels} pixels")
        except Exception as e:
            log_error(f"Error getting image dimensions: {e}")
    
        # --- Direct OpenAI Analysis (if YOLO is skipped) ---
        if not PROCESS_WITH_YOLO_FIRST:
            log_info("PROCESS_WITH_YOLO_FIRST is false. Skipping YOLO and proceeding directly to OpenAI analysis.")
            # Encode image for OpenAI
            log_info("Encoding image for OpenAI analysis")
            base64_image = encode_image_to_base64(image_frame)
            
            log_info("Sending image to OpenAI for analysis (YOLO skipped)")
            openai_start_time = time.time()
            description = analyze_image_with_openai(base64_image)
            openai_time = time.time() - openai_start_time

            if description:
                log_info(f"OpenAI analysis completed in {openai_time:.2f} seconds")
                log_info(f"OpenAI result: {description[:150]}...")
                total_time = time.time() - overall_start_time
                log_info(f"===== Total processing time (OpenAI only): {total_time:.2f} seconds =====")
                return description
            else:
                log_error("No description received from OpenAI (YOLO skipped path)")
                return "No description could be generated for the image (YOLO skipped). Please check your API key and try again."

        # --- YOLO Analysis (if enabled) ---
        log_info("Starting YOLO object detection phase") # This will only be reached if PROCESS_WITH_YOLO_FIRST is true
        yolo_start_time = time.time()
        yolo_results = analyze_image_with_yolo(image_frame)
        yolo_time = time.time() - yolo_start_time
        log_info(f"YOLO analysis completed in {yolo_time:.2f} seconds")
    
        # --- Check for Person ---
        person_detected = False # Default to false
        if yolo_results:
            log_info("Checking for persons in YOLO results")
            person_detected = check_for_person(yolo_results, yolo_model.names) # Check results for a person
        else:
            log_info("No YOLO results to check for persons")
    
        # --- OpenAI Analysis ---
        # Condition to send to OpenAI: EITHER the condition is disabled OR a person was detected
        if not OPENAI_CONDITION_PERSON or person_detected:
            # Encode image for OpenAI
            log_info("Encoding image for OpenAI analysis")
            base64_image = encode_image_to_base64(image_frame)
    
            # Analyze image
            if OPENAI_CONDITION_PERSON and person_detected:
                log_info("Sending image to OpenAI for analysis (person detected)")
            else:
                log_info("Sending image to OpenAI for analysis (condition disabled)")
            
            openai_start_time = time.time()
            description = analyze_image_with_openai(base64_image)
            openai_time = time.time() - openai_start_time
            
            if description:
                log_info(f"OpenAI analysis completed in {openai_time:.2f} seconds")
                log_info(f"OpenAI result: {description[:150]}...")
                
                # Calculate and log total processing time
                total_time = time.time() - overall_start_time
                log_info(f"===== Total processing time: {total_time:.2f} seconds =====")
                
                return description
            else:
                log_error("No description received from OpenAI")
                return "No description could be generated for the image. Please check your API key and try again."
        elif OPENAI_CONDITION_PERSON: # Only execute if the condition is enabled but no person found
            log_info("Skipping OpenAI analysis: No person detected by YOLO")
            return "No person detected in the image. Analysis skipped as per settings. Change OPENAI_CONDITION_PERSON in .env file to analyze all images."
    except Exception as e:
        log_error(f"Unhandled exception in process_image_for_analysis: {e}")
        log_error(traceback.format_exc())
        return f"Error during image analysis: {str(e)}"