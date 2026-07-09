"""
Task definitions for Celery workers
"""
from celery import Celery
import os
import sys
import logging
import configparser

# Add current directory to Python path
sys.path.insert(0, '/comp7940-lab')

# Import project modules
from ChatGPT_HKBU import ChatGPT
from image_to_video import ImageToVideoGenerator

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Celery
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = os.getenv('REDIS_PORT', '6379')
celery_app = Celery(
    'chatbot_tasks',
    broker=f'redis://{redis_host}:{redis_port}/0',
    backend=f'redis://{redis_host}:{redis_port}/0'
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 60 minutes hard limit (3600 seconds)
    task_soft_time_limit=3540,  # 59 minutes soft limit (3540 seconds)
)


@celery_app.task(
    name='tasks.generate_video',
    bind=True,
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,      # 2s, 4s, 8s exponential backoff
    retry_backoff_max=60,    # max 60s between retries
    retry_jitter=True,       # add randomness to avoid thundering herd
)
def generate_video_task(self, image_base64, prompt, user_id, output_path):
    """
    Task to generate video from image
    
    Args:
        image_base64: Base64 encoded image
        prompt: Video generation prompt
        user_id: Telegram user ID
        output_path: Path to save output video
    
    Returns:
        dict: Result with success status and video path or error
    """
    logger.info(f"Starting video generation task for user {user_id}")
    
    try:
        # Load config
        config = configparser.ConfigParser()
        config.read('config.ini')
        
        # Initialize generator
        generator = ImageToVideoGenerator(config)
        
        # Define status callback to update task state
        def status_callback(status, position):
            self.update_state(
                state='PROGRESS',
                meta={
                    'status': status,
                    'position': position,
                    'user_id': user_id
                }
            )
        
        # Generate video
        success = generator.generate_and_wait(
            image_base64=image_base64,
            output_path=output_path,
            prompt=prompt,
            image_size="1280x720",
            max_wait_time=3000,
            status_callback=status_callback
        )
        
        if success:
            logger.info(f"Video generation completed for user {user_id}")
            return {
                'success': True,
                'video_path': output_path,
                'user_id': user_id
            }
        else:
            logger.error(f"Video generation failed for user {user_id}")
            return {
                'success': False,
                'error': 'Video generation failed',
                'user_id': user_id
            }
            
    except Exception as e:
        logger.error(f"Video generation error for user {user_id}: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'user_id': user_id
        }


@celery_app.task(
    name='tasks.analyze_document',
    bind=True,
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
)
def analyze_document_task(self, file_path, file_type, user_id):
    """
    Task to analyze document (PDF or image) with OCR
    
    Args:
        file_path: Path to the document file
        file_type: Type of file ('pdf' or 'image')
        user_id: Telegram user ID
    
    Returns:
        dict: Result with extracted text and summary
    """
    logger.info(f"Starting document analysis task for user {user_id}")
    
    try:
        import fitz  # PyMuPDF
        
        # Load config
        config = configparser.ConfigParser()
        config.read('config.ini')
        
        extracted_text = ""
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'extracting_text'})
        
        # Extract text based on file type
        if file_type == 'pdf':
            doc = fitz.open(file_path)
            # Extract ALL pages (not just first 5) so the summary is complete
            for page in doc:
                extracted_text += page.get_text()
            doc.close()
        else:  # image
            # Without EasyOCR, we can't process images
            return {
                'success': False,
                'error': 'Image OCR not available (EasyOCR not installed)',
                'user_id': user_id
            }
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'generating_summary'})
        
        # Generate summary with ChatGPT
        if len(extracted_text.strip()) > 50:
            gpt = ChatGPT(config)

            # For documents exceeding token limits, chunk and recursively summarise
            # using the same RecursiveCharacterTextSplitter strategy as RAG
            if len(extracted_text) > 8000:
                from langchain_text_splitters import RecursiveCharacterTextSplitter

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " "],
                )
                chunks = splitter.split_text(extracted_text)

                # First pass: summarise each chunk independently
                chunk_summaries = []
                for i, chunk in enumerate(chunks):
                    chunk_prompt = (
                        f"You are a professional academic assistant. "
                        f"Summarise the following excerpt from an educational document "
                        f"(part {i+1}/{len(chunks)}). Capture key points, deadlines, and requirements.\n\n"
                        f"Excerpt:\n{chunk}"
                    )
                    chunk_summary = gpt.submit_sync(chunk_prompt, max_tokens=300)
                    chunk_summaries.append(chunk_summary)

                # Second pass: synthesise the per-chunk summaries into one
                combined = "\n\n".join(chunk_summaries)
                synthesis_prompt = (
                    f"Below are summaries of different sections of an educational document. "
                    f"Synthesize them into one coherent summary covering:\n"
                    f"1. Core Course Objectives\n"
                    f"2. Key Deadlines/Tasks\n"
                    f"3. Important Requirements/Knowledge Points\n\n"
                    f"Summaries:\n{combined}"
                )
                summary = gpt.submit_sync(synthesis_prompt, max_tokens=500)
            else:
                # Short document: summarise directly
                prompt = (
                    f"You are a professional academic assistant. Summarise the following "
                    f"educational document covering:\n"
                    f"1. Core Course Objectives\n"
                    f"2. Key Deadlines/Tasks\n"
                    f"3. Important Requirements/Knowledge Points\n\n"
                    f"Content:\n{extracted_text}"
                )
                summary = gpt.submit_sync(prompt, max_tokens=500)
        else:
            summary = "Could not extract enough text from the document."
        
        logger.info(f"Document analysis completed for user {user_id}")
        
        return {
            'success': True,
            'summary': summary,
            'user_id': user_id
        }
        
    except Exception as e:
        logger.error(f"Document analysis error for user {user_id}: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'user_id': user_id
        }


@celery_app.task(
    name='tasks.analyze_image',
    bind=True,
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
)
def analyze_image_task(self, image_base64, user_id):
    """
    Task to analyze image with GPT and suggest video prompts
    
    Args:
        image_base64: Base64 encoded image
        user_id: Telegram user ID
    
    Returns:
        dict: Result with AI analysis and suggested prompts
    """
    logger.info(f"Starting image analysis task for user {user_id}")
    
    try:
        # Load config
        config = configparser.ConfigParser()
        config.read('config.ini')
        
        # Initialize ChatGPT
        gpt = ChatGPT(config)
        
        # Analyze image
        ai_prompt = (
            "Analyze this image and provide:\n"
            "1. A brief description of what you see (1-2 sentences)\n"
            "2. Three creative video animation prompt suggestions that would work well with this image\n\n"
            "Format your response as:\n"
            "Description: [your description]\n\n"
            "Suggested prompts:\n"
            "1. [prompt 1]\n"
            "2. [prompt 2]\n"
            "3. [prompt 3]"
        )
        
        # Use sync method in worker
        ai_response = gpt.submit_with_image_sync(ai_prompt, image_base64, use_image_analysis_prompt=True)
        
        # Extract suggested prompts
        suggested_prompts = []
        lines = ai_response.split('\n')
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.')):
                prompt_text = line.split('.', 1)[1].strip()
                if prompt_text:
                    suggested_prompts.append(prompt_text)
        
        logger.info(f"Image analysis completed for user {user_id}")
        
        return {
            'success': True,
            'analysis': ai_response,
            'suggested_prompts': suggested_prompts,
            'user_id': user_id
        }
        
    except Exception as e:
        logger.error(f"Image analysis error for user {user_id}: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'user_id': user_id
        }
