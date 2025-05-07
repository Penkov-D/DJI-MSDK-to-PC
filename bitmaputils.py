import numpy as np
import cv2
import time
import traceback

# Set up detailed logging
def log_debug(message):
    """Print a debug message with timestamp that will appear in Android logs"""
    print(f"[BITMAP][DEBUG][{time.strftime('%H:%M:%S')}] {message}", flush=True)

def log_info(message):
    """Print an info message with timestamp that will appear in Android logs"""
    print(f"[BITMAP][INFO][{time.strftime('%H:%M:%S')}] {message}", flush=True)

def log_error(message):
    """Print an error message with timestamp that will appear in Android logs"""
    print(f"[BITMAP][ERROR][{time.strftime('%H:%M:%S')}] {message}", flush=True)

def bitmap_to_cv2(bitmap):
    """
    Convert an Android Bitmap object to an OpenCV image (numpy array)
    
    Args:
        bitmap: Android Bitmap object
    
    Returns:
        numpy.ndarray: OpenCV format image
    """
    start_time = time.time()
    
    try:
        # Get bitmap info
        width = bitmap.getWidth()
        height = bitmap.getHeight()
        config = bitmap.getConfig().name() if bitmap.getConfig() is not None else "null"
        log_info(f"Converting bitmap {width}x{height} with config {config}")
        
        # Create a numpy array to hold the pixel data
        log_debug("Creating numpy arrays for pixel data")
        pixels = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Create a 1D array to store the pixels temporarily
        # This makes the conversion much faster than accessing pixels individually
        buffer_size = width * height * 4
        log_debug(f"Allocating buffer of size {buffer_size} bytes")
        buffer = np.zeros(buffer_size, dtype=np.uint8)
        
        # Get pixels to buffer
        log_debug("Copying pixels from bitmap to buffer")
        pixel_copy_start = time.time()
        bitmap.copyPixelsToBuffer(buffer)
        pixel_copy_time = time.time() - pixel_copy_start
        log_debug(f"Pixel copy completed in {pixel_copy_time:.4f} seconds")
        
        # Reshape to 2D array with 4 channels (RGBA)
        log_debug("Reshaping buffer to 2D array")
        reshape_start = time.time()
        pixels = buffer.reshape(height, width, 4)
        reshape_time = time.time() - reshape_start
        log_debug(f"Buffer reshape completed in {reshape_time:.4f} seconds")
        
        # Check for any all-zero or all-black areas that might indicate problems
        zero_pixels = np.sum(pixels == 0)
        total_pixels = width * height * 4
        zero_percentage = (zero_pixels / total_pixels) * 100
        log_debug(f"Image contains {zero_percentage:.2f}% zero values")
        
        if zero_percentage > 90:
            log_error(f"Warning: Image appears to be mostly black/empty ({zero_percentage:.2f}% zeros)")
        
        # Convert from RGBA to BGR (which is what OpenCV uses)
        # Drop the alpha channel
        log_debug("Converting from RGBA to BGR format for OpenCV")
        conversion_start = time.time()
        cv2_image = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
        conversion_time = time.time() - conversion_start
        log_debug(f"Color conversion completed in {conversion_time:.4f} seconds")
        
        # Get image statistics for debugging
        mean_val = cv2.mean(cv2_image)
        log_debug(f"Image mean values (BGR): {mean_val[:3]}")
        
        total_time = time.time() - start_time
        log_info(f"Bitmap conversion completed in {total_time:.4f} seconds")
        
        return cv2_image
    
    except Exception as e:
        log_error(f"Error converting bitmap: {str(e)}")
        log_error(traceback.format_exc())
        raise e