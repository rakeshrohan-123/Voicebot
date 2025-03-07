import os
import time
import logging
from typing import Dict, Optional
import numpy as np
import torch
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from melo.api import TTS
import nltk
import base64
import tempfile
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple API key handling without SecretStr
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logger.warning("OpenAI API key not found in environment variables. Please set OPENAI_API_KEY.")

# Initialize NLTK
nltk.data.path.append('/home/haloocom/rakesh/melotts/MeloTTS/nltk_data')
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# MeloTTS Configuration
ACCENT_MAP = {
    "us": "EN-US",
    "uk": "EN-BR",
    "india": "EN_INDIA",
    "australia": "EN-AU",
    "default": "EN-Default"
}

DEFAULT_ACCENT = "australia"
DEFAULT_SPEED = 1.0
OUTPUT_DIR = "/home/haloocom/rakesh/fillers/latest-fillers"

class MeloTTSHandler:
    def __init__(self):
        self.model = None
        self.speaker_ids = None
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory for TTS files: {self.temp_dir}")
        self.initialize_model()

    def initialize_model(self):
        """Initialize the MeloTTS model."""
        try:
            self.model = TTS(language='EN', device='auto')
            self.speaker_ids = self.model.hps.data.spk2id
            logger.info(f"Available speaker IDs: {self.speaker_ids}")
        except Exception as e:
            logger.error(f"Error initializing MeloTTS: {str(e)}")
            raise

    def get_speaker_id(self, accent: str) -> int:
        """Get speaker ID for given accent."""
        accent_name = ACCENT_MAP.get(accent, ACCENT_MAP["australia"])
        try:
            return self.speaker_ids[accent_name]
        except Exception as e:
            logger.error(f"Error getting speaker ID for accent {accent}: {str(e)}")
            return list(self.speaker_ids.values())[0]

    def cleanup_temp_file(self, filepath):
        """Safely cleanup temporary file."""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary file {filepath}: {str(e)}")

    async def text_to_speech_stream(self, text: str, accent: str = DEFAULT_ACCENT, speed: float = DEFAULT_SPEED) -> bytes:
        """Generate TTS audio stream from text using MeloTTS."""
        if not text or not text.strip():
            logger.warning("Empty text provided to TTS")
            return b""

        temp_path = None
        try:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', dir=self.temp_dir)
            os.close(temp_fd)  # Close file descriptor

            # Get speaker ID
            speaker_id = self.get_speaker_id(accent)
            logger.info(f"Using speaker ID {speaker_id} for accent {accent}")
            
            logger.info(f"Generating TTS for text: '{text[:50]}...' with accent: {accent}")
            
            # Generate speech using the correct method signature according to the API
            audio_data = self.model.tts_to_file(
                text=text.strip(),
                speaker_id=speaker_id,
                output_path=temp_path,
                speed=speed,
                quiet=True  # Suppress the progress bar output
            )

            # Check if file was created and has content
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise Exception("TTS failed to generate audio file")

            # Read the generated file
            with open(temp_path, 'rb') as f:
                audio_data = f.read()

            return audio_data

        except Exception as e:
            logger.error(f"Error in TTS generation: {str(e)}")
            return b""

        finally:
            # Cleanup temporary file
            if temp_path:
                self.cleanup_temp_file(temp_path)

    def __del__(self):
        """Cleanup temporary directory on object destruction."""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory: {str(e)}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model: Optional[WhisperForConditionalGeneration] = None
processor: Optional[WhisperProcessor] = None
vad_model = None
llm_store: Dict[tuple, ChatMessageHistory] = {}
tts_handler: Optional[MeloTTSHandler] = None
from pydantic import SecretStr

# api_key = SecretStr(os.getenv('OPENAI_API_KEY', 'sk-rc2VRoyuUS0L4OebQ8nmlQFTkg-EE3k-sFE1foNY65T3BlbkFJKSs9zJQ2yryVddzkgjNaI_yF3IWB8FcYDTxEGC14wA'))
# if not api_key.get_secret_value():
#     logger.warning("OpenAI API key not found in environment variables. Please set OPENAI_API_KEY.")

from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI LLM
chat = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.2,  
    max_tokens=150,  
    api_key=api_key 
    #api_key=api_key or "sk-rc2VRoyuUS0L4OebQ8nmlQFTkg-EE3k-sFE1foNY65T3BlbkFJKSs9zJQ2yryVddzkgjNaI_yF3IWB8FcYDTxEGC14wA"
    #api_key=api_key.get_secret_value() 
    #api_key='sk-rc2VRoyuUS0L4OebQ8nmlQFTkg-EE3k-sFE1foNY65T3BlbkFJKSs9zJQ2yryVddzkgjNaI_yF3IWB8FcYDTxEGC14wA'
)

# Updated Prompt Template with more nuanced instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly AI assistant a personalized companion driving service.

Service Focus:
- Specialize in elderly and senior transportation services
- Prioritize safety, comfort, and personalized care
- Offer compassionate and patient communication

Core Communication Guidelines:
- Speak in a warm, clear, and reassuring tone
- Use respectful language appropriate for senior clients
- Keep responses concise, helpful, and easy to understand

Key Service Areas:
1. Ride Bookings: 
   - Collect location details
   - Understand specific client needs
   - Guide through booking process

2. Franchise Inquiries:
   - Provide professional, welcoming information
   - Offer to connect with franchise team
   - Collect contact details for follow-up

3. General Assistance:
   - Answer questions about services
   - Address concerns with empathy
   - Provide clear, step-by-step guidance

Important Restrictions:
- Do NOT provide specific pricing
- Do NOT make commitments beyond initial information
- Always verify location first
- Redirect complex queries to human representative

Response Structure:
- make your responses precise on point Do Not respond more than 15 words
- Friendly opening
- Clear, actionable information
- Specific next step or question
- Warm closing

Example:
"Hello! I'd love to help you today. Could you please share your postcode so I can connect you with our nearest caring team?"
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

runnable = prompt | chat

def get_session_history(user_id: str, conversation_id: str) -> ChatMessageHistory:
    """Get or create a chat history for the given session."""
    key = (user_id, conversation_id)
    if key not in llm_store:
        llm_store[key] = ChatMessageHistory()
    return llm_store[key]

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

# The rest of the code remains the same as in the previous implementation
# Include the SileroVAD, TranscriptionFilter, process_audio_chunk, etc.
# ... (Keep all other existing code from the previous implementation)

async def get_llm_response(text: str, user_id: str, session_id: str) -> str:
    """Get response from LLM."""
    try:
        if not text.strip():
            return ""
            
        response = await with_message_history.ainvoke(
            {"input": text},
            config={"configurable": {"user_id": user_id, "conversation_id": session_id}},
        )
        
        # Extract content from AIMessage
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}")
        return "I apologize, but I couldn't process that properly."

async def get_transcription(audio: np.ndarray, sample_rate: int) -> str:
    """Get transcription from audio."""
    try:
        if model is None or processor is None:
            raise ValueError("Model or processor not initialized")
            
        with torch.cuda.amp.autocast():
            input_features = processor(
                audio, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_features.to(model.device)
            
            predicted_ids = model.generate(
                input_features,
                language="english",
                task="transcribe",
                temperature=0.2,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                max_new_tokens=128
            )
            
            transcription = processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            transcription_filter = TranscriptionFilter()
            filtered_transcription = transcription_filter.filter_transcription(
                transcription,
                confidence=0.8
            )
            
            return filtered_transcription
    except Exception as e:
        logger.error(f"Error getting transcription: {str(e)}")
        return ""

LOCAL_MODEL_PATH = "/home/haloocom/rakesh/whisper-asr-eng/whisper_medium"
LOCAL_PROCESSOR_PATH = "/home/haloocom/rakesh/whisper-asr-eng/whisper_processor"
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> str:
        """Connect a new client."""
        await websocket.accept()
        client_id = str(id(websocket))
        async with self._lock:
            self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
        return client_id

    async def disconnect(self, client_id: str) -> None:
        """Disconnect a client."""
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
        logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_message(self, client_id: str, message: dict) -> None:
        """Send a message to a client."""
        if client_id in self.active_connections:
            try:
                # Ensure all data is JSON serializable
                serializable_message = self.prepare_message(message)
                await self.active_connections[client_id].send_json(serializable_message)
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {str(e)}")
                await self.disconnect(client_id)

    def prepare_message(self, message: dict) -> dict:
        """Prepare message for JSON serialization."""
        prepared_message = {}
        for key, value in message.items():
            if isinstance(value, bytes):
                # Convert bytes to base64 string
                prepared_message[key] = base64.b64encode(value).decode('utf-8')
            elif isinstance(value, (int, float, str, bool, type(None))):
                # These types are JSON serializable
                prepared_message[key] = value
            elif isinstance(value, (list, tuple)):
                # Handle lists and tuples
                prepared_message[key] = [self.prepare_value(item) for item in value]
            elif isinstance(value, dict):
                # Handle nested dictionaries
                prepared_message[key] = self.prepare_message(value)
            else:
                # Convert other types to string
                prepared_message[key] = str(value)
        return prepared_message

    def prepare_value(self, value):
        """Helper method to prepare individual values."""
        if isinstance(value, bytes):
            return base64.b64encode(value).decode('utf-8')
        elif isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [self.prepare_value(item) for item in value]
        elif isinstance(value, dict):
            return self.prepare_message(value)
        else:
            return str(value)

class SileroVAD:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        self.reset_state()
        self.SILENCE_THRESHOLD = 1.5  # Increased silence threshold
        self.MIN_SPEECH_DURATION = 0.5  # Minimum speech duration in seconds
        self.speech_start_time = None
        logger.info(f"Initialized SileroVAD with device: {self.device}")
    
    def reset_state(self):
        self.speech_buffer = []
        self.speech_started = False
        self.silence_duration = 0
        self.last_speech_time = 0
        self.speech_start_time = None
        
    def load_model(self):
        try:
            if self.model is None:
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False
                )
                self.model = model.to(self.device)
                self.get_speech_timestamps = utils[0]
                if self.device == "cuda":
                    self.model = self.model.half()
        except Exception as e:
            logger.error(f"Error loading Silero VAD model: {str(e)}")
            raise

    def detect_speech(self, audio: np.ndarray, sample_rate: int) -> dict:
        try:
            with torch.cuda.amp.autocast():
                audio_tensor = torch.from_numpy(audio.copy()).float().to(self.device)
                if torch.abs(audio_tensor).max() > 1.0:
                    audio_tensor = audio_tensor / torch.abs(audio_tensor).max()
                
                rms_energy = torch.sqrt(torch.mean(audio_tensor**2))
                energy_threshold = 0.005  # Increased energy threshold
                
                speech_timestamps = self.get_speech_timestamps(
                    audio_tensor,
                    self.model,
                    threshold=0.5,  # Increased VAD threshold
                    sampling_rate=sample_rate,
                    min_speech_duration_ms=250,  # Increased minimum speech duration
                    return_seconds=True
                )
                
                current_time = time.time()
                speech_detected = len(speech_timestamps) > 0 and rms_energy > energy_threshold

                result = {
                    "speech_detected": False,
                    "status": "silence",
                    "energy_level": float(rms_energy),
                    "buffered_audio": None
                }

                if speech_detected:
                    if not self.speech_started:
                        self.speech_start_time = current_time
                        self.speech_started = True
                        result["status"] = "started"
                    else:
                        result["status"] = "ongoing"
                    
                    self.speech_buffer.append(audio)
                    self.silence_duration = 0
                    self.last_speech_time = current_time
                    result["speech_detected"] = True
                else:
                    if self.speech_started:
                        self.silence_duration = current_time - self.last_speech_time
                        if self.silence_duration > self.SILENCE_THRESHOLD:
                            speech_duration = current_time - (self.speech_start_time or current_time)
                            if speech_duration >= self.MIN_SPEECH_DURATION:
                                result["status"] = "ended"
                                result["buffered_audio"] = np.concatenate(self.speech_buffer) if self.speech_buffer else None
                            self.reset_state()
                
                return result
                
        except Exception as e:
            logger.error(f"Error in detect_speech: {str(e)}")
            raise

class TranscriptionFilter:
    def __init__(self):
        self.common_hallucinations = {
            "you", "thank you", "bye", "goodbye", "thanks",
            "okay", "um", "uh", "mm", "hmm"
        }
        self.min_confidence_threshold = 0.6
        self.min_text_length = 3

    def filter_transcription(self, text: str, confidence: float) -> str:
        try:
            if not text or confidence < self.min_confidence_threshold:
                return ""
            
            text = text.strip().lower()
            words = text.split()
            
            if (text in self.common_hallucinations or
                len(words) < self.min_text_length or
                all(word in self.common_hallucinations for word in words)):
                return ""
            
            return text
        except Exception as e:
            logger.error(f"Error in filter_transcription: {str(e)}")
            return ""


    
async def process_audio_chunk(audio_data: bytes, client_id: str, sample_rate: int = 16000) -> dict:
    """Process an audio chunk and return the result with TTS audio."""
    try:
        # ... existing code ...
        if not isinstance(audio_data, bytes):
            return {
                "status": "error", 
                "message": f"Invalid data format. Expected bytes, got {type(audio_data)}"
            }

        audio = np.frombuffer(audio_data, dtype=np.float32)
        if vad_model is None:
            return {"status": "error", "message": "VAD model not initialized"}

        vad_result = vad_model.detect_speech(audio, sample_rate)
        
        result = {
            "status": "success",
            "speech_status": vad_result["status"],
            "transcription": "",
            "llm_response": "",
            "tts_audio": None,
            "energy_level": float(vad_result["energy_level"])
        }

        if vad_result["status"] == "ended" and vad_result.get("buffered_audio") is not None:
            try:
                transcription = await get_transcription(vad_result["buffered_audio"], sample_rate)
                if transcription:
                    # Save ASR transcription to conversation log
                    conversation_log_filepath = save_conversation_log(client_id, "asr", transcription)
                    
                    llm_response = await get_llm_response(transcription, client_id, client_id)
                    if llm_response:
                        # Save TTS transcription to conversation log
                        save_conversation_log(client_id, "tts", llm_response)
                        
                        tts_audio = await tts_handler.text_to_speech_stream(llm_response)
                        if tts_audio and isinstance(tts_audio, bytes):
                            result.update({
                                "transcription": transcription,
                                "llm_response": llm_response,
                                "tts_audio": tts_audio,
                                "conversation_log_filepath": conversation_log_filepath
                            })
                        else:
                            logger.error(f"Invalid TTS audio format: {type(tts_audio)}")
                            result["status"] = "error"
                            result["message"] = "TTS generation failed"
            except Exception as e:
                logger.error(f"Error in audio processing pipeline: {str(e)}")
                result["status"] = "error"
                result["message"] = f"Error processing audio: {str(e)}"

        return result

    except Exception as e:
        logger.error(f"Error in process_audio_chunk: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
# Add this function to the existing code
def save_conversation_log(client_id: str, message_type: str, content: str) -> str:
    """
    Save conversation messages to a single log file.
    
    Args:
        client_id (str): Unique identifier for the conversation
        message_type (str): Type of message ('asr', 'tts', 'welcome')
        content (str): Message content
    
    Returns:
        str: Path to the conversation log file
    """
    # Create output directory if it doesn't exist
    output_dir = "/home/haloocom/rakesh/conversation_logs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"conversation_{client_id}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    try:
        # Append message to the file with timestamp and type
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{current_time}] [{message_type.upper()}] {content}\n"
        
        with open(filepath, 'a') as f:
            f.write(log_entry)
        
        logger.info(f"Conversation log updated: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving conversation log: {str(e)}")
        return ""
@app.on_event("startup")
async def startup_event():
    """Initialize models and resources on startup."""
    try:
        global model, processor, vad_model, tts_handler
        
        logger.info("Loading Whisper model and processor...")
        model = WhisperForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH)
        processor = WhisperProcessor.from_pretrained(LOCAL_PROCESSOR_PATH)
        
        if torch.cuda.is_available():
            model = model.to("cuda").half()
            torch.backends.cudnn.benchmark = True
            logger.info("Models loaded on CUDA with half precision")
        else:
            logger.info("Models loaded on CPU")
        
        vad_model = SileroVAD()
        tts_handler = MeloTTSHandler()
        logger.info("Startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

# [Previous ConnectionManager class and WebSocket endpoint remain the same]
# Include the ConnectionManager class and websocket_endpoint here


manager = ConnectionManager()
@app.websocket("/ws/stream/")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections."""
    client_id = await manager.connect(websocket)
    
    try:
        while True:
            try:
                data = await websocket.receive()
                
                if "bytes" in data:
                    result = await process_audio_chunk(data["bytes"], client_id)
                    await manager.send_message(client_id, result)
                
                elif "text" in data:
                    try:
                        msg = json.loads(data["text"])
                        if msg.get("type") == "end":
                            logger.info(f"Client {client_id} requested end of session")
                            break
                        elif msg.get("type") == "welcome":
                            # Handle welcome message
                            welcome_msg = msg["message"]
                            
                            # Save welcome message to conversation log
                            conversation_log_filepath = save_conversation_log(client_id, "welcome", welcome_msg)
                            
                            welcome_audio = await tts_handler.text_to_speech_stream(welcome_msg)
                            welcome_response = {
                                "status": "success",
                                "speech_status": "ended",
                                "transcription": "",
                                "llm_response": welcome_msg,
                                "tts_audio": welcome_audio,
                                "conversation_log_filepath": conversation_log_filepath
                            }
                            await manager.send_message(client_id, welcome_response)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse text message as JSON")
                        continue
                    
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await manager.send_message(client_id, {
                    "status": "error",
                    "message": str(e)
                })
                break

    finally:
        await manager.disconnect(client_id)
        if (client_id, client_id) in llm_store:
            del llm_store[(client_id, client_id)]
if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="192.168.3.151", port=9116, log_level="info")
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Listen on all network interfaces
        port=9117, 
        # ssl_keyfile="/path/to/your/ssl/key.pem",
        # ssl_certfile="/path/to/your/ssl/cert.pem",
        log_level="info"
    )

