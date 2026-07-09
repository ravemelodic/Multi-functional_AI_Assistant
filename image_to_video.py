"""
Image to Video Generator using Wan-AI/Wan2.2-I2V-A14B API
This module provides functionality to convert images to videos using external API
"""

import requests
import configparser
import logging
import time
import os
import base64
from typing import Optional


class ImageToVideoGenerator:
    """
    A client for the Wan-AI Image-to-Video API
    Converts static images into animated videos
    """
    
    def __init__(self, config):
        """
        Initialize the Image-to-Video generator with API configuration
        
        Args:
            config: ConfigParser object containing API settings
        """
        # Read API configuration from config file
        self.api_key = config['WAN_AI']['API_KEY']
        self.base_url = config['WAN_AI']['BASE_URL']  # https://api.siliconflow.cn/v1
        self.model = config['WAN_AI']['MODEL']  # Wan-AI/Wan2.2-I2V-A14B
        
        # Set up HTTP headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # API endpoints
        self.submit_url = f"{self.base_url}/video/submit"
        self.status_url = f"{self.base_url}/video/status"
        
        # Configure logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
    
    def image_to_base64(self, image_path: str) -> Optional[str]:
        """
        Convert local image file to base64 data URI
        
        Args:
            image_path: Path to local image file
            
        Returns:
            str: Base64 data URI (e.g., "data:image/png;base64,XXX") or None if failed
        """
        try:
            # Determine image format from file extension
            ext = os.path.splitext(image_path)[1].lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_types.get(ext, 'image/jpeg')
            
            # Read and encode image
            with open(image_path, 'rb') as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Return data URI
            return f"data:{mime_type};base64,{encoded}"
            
        except Exception as e:
            self.logger.error(f"Failed to encode image: {str(e)}")
            return None
    
    def submit_video_task(self, image_base64: str, prompt: Optional[str] = None, 
                         image_size: str = "1280x720") -> dict:
        """
        Submit a video generation task to SiliconFlow API
        
        Args:
            image_base64: Base64 encoded image data URI (e.g., "data:image/png;base64,XXX")
            prompt: Text prompt to guide video generation
            image_size: Output video resolution (default: "1280x720")
            
        Returns:
            dict: Response containing requestId for status checking
        """
        self.logger.info(f"Submitting video generation task")
        
        # Prepare the request payload according to SiliconFlow API
        payload = {
            "model": self.model,
            "image": image_base64,
            "prompt": prompt or "smooth natural animation",
            "image_size": image_size
        }
        
        try:
            # Send request to SiliconFlow API
            response = requests.post(
                self.submit_url,
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"Task submitted successfully: {result}")
                return result
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {"error": error_msg}
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
    
    def generate_from_local_image(self, image_path: str, output_path: str,
                                  prompt: Optional[str] = None,
                                  image_size: str = "1280x720",
                                  max_wait_time: int = 3000) -> bool:
        """
        Generate video from a local image file (converts to base64)
        
        Args:
            image_path: Path to local image file
            output_path: Path to save output video
            prompt: Optional generation prompt
            image_size: Video resolution (default: "1280x720")
            max_wait_time: Maximum time to wait in seconds (default: 3000)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Convert local image to base64
        self.logger.info(f"Converting local image to base64: {image_path}")
        image_base64 = self.image_to_base64(image_path)
        
        if not image_base64:
            self.logger.error("Failed to convert image to base64")
            return False
        
        # Use the base64 data to generate video
        return self.generate_and_wait(image_base64, output_path, prompt, image_size, max_wait_time)
    
    def check_video_status(self, request_id: str) -> dict:
        """
        Check the status of a video generation task
        
        Args:
            request_id: The request ID returned from submit_video_task()
            
        Returns:
            dict: Task status and video URL if completed
        """
        self.logger.info(f"Checking status for request: {request_id}")
        
        # Prepare payload with requestId
        payload = {
            "requestId": request_id
        }
        
        try:
            # Send POST request to status endpoint
            response = requests.post(
                self.status_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"Status result: {result}")
                return result
            else:
                error_msg = f"Status check failed: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {"error": error_msg}
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Status check failed: {str(e)}")
            return {"error": str(e)}
    
    def download_video(self, video_url: str, output_path: str = None) -> bytes:
        """
        Download the generated video from URL
        
        Args:
            video_url: URL of the generated video
            output_path: Optional local path to save the video (if None, return bytes)
            
        Returns:
            bytes: Video content if output_path is None
            bool: True if saved successfully (when output_path is provided)
        """
        self.logger.info(f"Downloading video from: {video_url}")
        
        try:
            response = requests.get(video_url, stream=True, timeout=60)
            
            if response.status_code == 200:
                video_bytes = b''
                for chunk in response.iter_content(chunk_size=8192):
                    video_bytes += chunk
                
                if output_path:
                    # Save to file
                    with open(output_path, 'wb') as f:
                        f.write(video_bytes)
                    self.logger.info(f"Video saved to: {output_path}")
                    return True
                else:
                    # Return bytes directly
                    self.logger.info(f"Video downloaded to memory ({len(video_bytes)} bytes)")
                    return video_bytes
            else:
                self.logger.error(f"Download failed: {response.status_code}")
                return False if output_path else None
                
        except Exception as e:
            self.logger.error(f"Download error: {str(e)}")
            return False if output_path else None
    
    def generate_and_wait(self, image_base64: str, output_path: str, 
                         prompt: Optional[str] = None, 
                         image_size: str = "1280x720",
                         max_wait_time: int = 3000,
                         status_callback=None) -> bool:
        """
        Generate video and wait for completion (blocking operation)
        
        Args:
            image_base64: Base64 encoded image data URI (e.g., "data:image/png;base64,XXX")
            output_path: Path to save output video
            prompt: Optional generation prompt
            image_size: Video resolution (default: "1280x720")
            max_wait_time: Maximum time to wait in seconds (default: 3000)
            status_callback: Optional async callback function(status, position) to report status updates
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Submit the task
        result = self.submit_video_task(image_base64, prompt, image_size)
        
        if "error" in result:
            self.logger.error(f"Task submission failed: {result['error']}")
            return False
        
        # Get requestId from response (try multiple possible field names)
        request_id = result.get("requestId") or result.get("request_id") or result.get("id")
        
        if not request_id:
            self.logger.error(f"No requestId in response: {result}")
            return False
        
        self.logger.info(f"Task submitted with requestId: {request_id}")
        
        # Poll for completion
        start_time = time.time()
        check_interval = 30  # Check every 30 seconds
        last_status = None
        
        while time.time() - start_time < max_wait_time:
            status_result = self.check_video_status(request_id)
            
            if "error" in status_result:
                self.logger.error(f"Error checking status: {status_result['error']}")
                time.sleep(check_interval)
                continue
            
            status = status_result.get("status")
            position = status_result.get("position", 0)
            
            # Only log and callback if status changed
            if status != last_status:
                self.logger.info(f"Task status: {status} (position: {position})")
                last_status = status
                
                # Call status callback if provided
                if status_callback:
                    import asyncio
                    try:
                        asyncio.create_task(status_callback(status, position))
                    except:
                        pass  # Ignore callback errors
            
            # Check for completion
            if status == "Succeed":
                # Extract video URL from SiliconFlow response format
                video_url = None
                if "results" in status_result:
                    results = status_result["results"]
                    if isinstance(results, dict) and "videos" in results:
                        videos = results["videos"]
                        if isinstance(videos, list) and len(videos) > 0:
                            video_url = videos[0].get("url")
                
                if video_url:
                    self.logger.info(f"Found video URL: {video_url}")
                    return self.download_video(video_url, output_path)
                else:
                    self.logger.error(f"No video URL in completed task: {status_result}")
                    return False
            
            elif status in ["Failed", "Error"]:
                error_msg = status_result.get("reason") or status_result.get("error") or "Unknown error"
                self.logger.error(f"Task failed: {error_msg}")
                return False
            
            elif status in ["InProgress", "InQueue"]:
                # Still processing or queued, wait and check again
                time.sleep(check_interval)
            
            else:
                self.logger.warning(f"Unknown status: {status}, continuing to wait...")
                time.sleep(check_interval)
        
        self.logger.error(f"Timeout after {max_wait_time} seconds")
        return False


async def handle_image_to_video(update, context, config):
    """
    Telegram bot handler for image-to-video conversion
    This function should be called from chatbot.py
    
    Args:
        update: Telegram Update object
        context: Telegram Context object
        config: ConfigParser object with API settings
    """
    from telegram import Update
    from telegram.ext import ContextTypes
    import io
    
    # Send initial status message
    status_msg = await update.message.reply_text(
        "🎬 Received your image! Preparing for video generation...\n"
        "This may take 2-5 minutes, please wait..."
    )
    
    # Get the image file
    if update.message.photo:
        # User sent a photo
        photo = update.message.photo[-1]  # Get highest resolution
        file = await photo.get_file()
    elif update.message.document:
        # User sent an image as document
        file = await update.message.document.get_file()
    else:
        await status_msg.edit_text("❌ Please send an image file.")
        return
    
    # Prepare output path
    output_video = f"output_video_{update.effective_user.id}.mp4"
    
    try:
        # Extract prompt from caption if provided
        prompt = update.message.caption if update.message.caption else "smooth natural animation"
        
        # Initialize generator
        generator = ImageToVideoGenerator(config)
        
        # Update status
        await status_msg.edit_text(
            "🎬 Downloading image and converting to base64...\n"
            "⏳ This will take a moment..."
        )
        
        # Download image directly to memory (BytesIO)
        image_bytes = io.BytesIO()
        await file.download_to_memory(image_bytes)
        image_bytes.seek(0)  # Reset pointer to beginning
        
        # Convert to base64 directly from bytes
        image_base64_data = base64.b64encode(image_bytes.read()).decode('utf-8')
        
        # Determine MIME type from file extension or default to jpeg
        file_path = file.file_path or ""
        if file_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif file_path.lower().endswith('.gif'):
            mime_type = 'image/gif'
        elif file_path.lower().endswith('.webp'):
            mime_type = 'image/webp'
        else:
            mime_type = 'image/jpeg'
        
        # Create data URI
        image_base64 = f"data:{mime_type};base64,{image_base64_data}"
        
        # Update status
        await status_msg.edit_text(
            "🎬 Image converted! Submitting to video generation API...\n"
            "⏳ Processing... (this may take 2-5 minutes)"
        )
        
        # Generate video using base64 data
        success = generator.generate_and_wait(
            image=image_base64,
            output_path=output_video,
            prompt=prompt,
            image_size="1280x720",
            max_wait_time=3000  # 50 minutes timeout
        )
        
        if success and os.path.exists(output_video):
            # Update status
            await status_msg.edit_text("✅ Video generated! Uploading...")
            
            # Send the video back to user
            with open(output_video, 'rb') as video_file:
                await update.message.reply_video(
                    video=video_file,
                    caption="🎥 Your generated video is ready!",
                    supports_streaming=True
                )
            
            # Delete status message
            await status_msg.delete()
            
        else:
            await status_msg.edit_text(
                "❌ Video generation failed. Please try again later.\n"
                "Possible reasons:\n"
                "- API service is busy\n"
                "- Image format not supported\n"
                "- Network connection issue"
            )
    
    except Exception as e:
        logging.error(f"Image-to-video error: {str(e)}")
        await status_msg.edit_text(
            f"❌ An error occurred during video generation:\n{str(e)}"
        )
    
    finally:
        # Clean up temporary video file
        if os.path.exists(output_video):
            os.remove(output_video)


if __name__ == '__main__':
    # Entry point for standalone testing
    import configparser
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    generator = ImageToVideoGenerator(config)
    
    print("Image to Video Generator - Test Mode")
    print("Using test.png in current directory")
    
    if not os.path.exists("test.png"):
        print("Error: test.png not found")
        exit(1)
    
    prompt = input("Enter prompt (or press Enter for default): ").strip() or "smooth natural animation"
    output = input("Enter output path (or press Enter for output.mp4): ").strip() or "output.mp4"
    
    print(f"\nGenerating video...")
    success = generator.generate_from_local_image("test.png", output, prompt)
    
    print(f"\n{'✓ Success' if success else '✗ Failed'}: {output}")
