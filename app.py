import asyncio
import json
import base64
import numpy as np
import logging
import sys
import io
from PIL import Image # Kept for potential future use
import time
import os
from datetime import datetime
import re

import google.genai as genai
from google.genai import types # Import necessary types
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status # Import status
from fastapi.middleware.cors import CORSMiddleware
# Import websockets exceptions if needed for state check, or use isinstance with WebSocketDisconnect
import websockets # Need this for State check

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Keep INFO level for general flow
    # level=logging.DEBUG, # Switch to DEBUG for more granular detail if needed
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', # Added funcName
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables (for API key)
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables or .env file.")
    sys.exit(1)

# Configure Gemini API Client
# Try adding v1alpha if connection still fails
try:
    # client = genai.Client(api_key=API_KEY)
    client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'}) # Trying v1alpha
    # Simple test (optional - remove if causing issues)
    # asyncio.run(genai.GenerativeModel("gemini-pro").generate_content_async("test"))
    logger.info("Gemini API Client configured (using v1alpha).")
except Exception as e:
    logger.error(f"Failed to initialize Gemini Client or test API key: {e}", exc_info=True)
    sys.exit(1)


# --- Configuration ---
GEMINI_LIVE_MODEL_NAME = "gemini-2.0-flash-live-001"
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2

# --- Audio Segment Detector ---
class AudioSegmentDetector:
    """Detects speech segments based on audio energy levels"""

    def __init__(self,
                 sample_rate=INPUT_SAMPLE_RATE,
                 energy_threshold=0.015, # Adjust based on microphone sensitivity
                 silence_duration=0.8, # Seconds of silence to trigger segment end
                 min_speech_duration=0.3, # Minimum duration for a valid speech segment (can be shorter for Live)
                 max_speech_duration=15): # Maximum duration before forced segmentation

        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_samples = int(silence_duration * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)
        self.sample_width = SAMPLE_WIDTH # Bytes per sample (for 16-bit PCM)

        # Internal state
        self.audio_buffer = bytearray()
        self.is_speech_active = False
        self.silence_counter = 0
        self.speech_start_idx = 0
        self.lock = asyncio.Lock()
        self.segment_queue = asyncio.Queue()

        # Counters
        self.segments_detected = 0

    async def add_audio(self, audio_bytes: bytes):
        """Add audio data to the buffer and check for speech segments to send to Live API"""
        async with self.lock:
            self.audio_buffer.extend(audio_bytes)

            num_new_samples = len(audio_bytes) // self.sample_width
            if num_new_samples == 0:
                return

            # Use only the new data for energy check
            new_audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            energy = np.sqrt(np.mean(new_audio_array**2)) if len(new_audio_array) > 0 else 0

            # --- Speech detection logic ---
            if not self.is_speech_active and energy > self.energy_threshold:
                self.is_speech_active = True
                self.speech_start_idx = max(0, len(self.audio_buffer) - len(audio_bytes))
                self.silence_counter = 0
                # logger.debug(f"Speech start detected (energy: {energy:.6f})") # Use debug for frequent logs

            elif self.is_speech_active:
                if energy > self.energy_threshold:
                    # Continued speech
                    self.silence_counter = 0
                else:
                    # Potential end of speech - accumulate silence samples
                    self.silence_counter += num_new_samples
                    if self.silence_counter >= self.silence_samples:
                        speech_end_idx = len(self.audio_buffer) - (self.silence_counter * self.sample_width)
                        segment_len_samples = (speech_end_idx - self.speech_start_idx) // self.sample_width

                        if segment_len_samples >= self.min_speech_samples:
                            speech_segment = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx])
                            self.segments_detected += 1
                            duration_sec = segment_len_samples / self.sample_rate
                            logger.info(f"Speech segment detected (silence): {duration_sec:.2f}s - Queuing for send")
                            await self.segment_queue.put(speech_segment) # Put segment in queue
                        # else: logger.debug(f"Detected segment too short ({segment_len_samples} samples), ignoring.")

                        # Reset after silence detection (segment queued or ignored)
                        self.is_speech_active = False
                        self.silence_counter = 0
                        self.audio_buffer = self.audio_buffer[speech_end_idx:] # Trim buffer
                        self.speech_start_idx = 0

                # Check max duration while still speaking
                current_speech_len_samples = (len(self.audio_buffer) - self.speech_start_idx) // self.sample_width
                if self.is_speech_active and current_speech_len_samples > self.max_speech_samples:
                    # Force segment end at max duration
                    forced_end_idx = self.speech_start_idx + (self.max_speech_samples * self.sample_width)
                    speech_segment = bytes(self.audio_buffer[self.speech_start_idx:forced_end_idx])
                    self.segments_detected += 1
                    duration_sec = self.max_speech_samples / self.sample_rate
                    logger.info(f"Speech segment detected (max duration): {duration_sec:.2f}s - Queuing for send")
                    await self.segment_queue.put(speech_segment) # Put segment in queue

                    # Reset buffer, keep speaking state active as it wasn't silence
                    self.audio_buffer = self.audio_buffer[forced_end_idx:]
                    self.speech_start_idx = 0
                    self.silence_counter = 0

            # Buffer trimming during silence
            MAX_BUFFER_SECONDS = 5 # Keep less buffer during silence
            max_buffer_len = MAX_BUFFER_SECONDS * self.sample_rate * self.sample_width
            if not self.is_speech_active and len(self.audio_buffer) > max_buffer_len:
                 self.audio_buffer = self.audio_buffer[-max_buffer_len:]

    async def get_next_segment(self):
        """Get the next available speech segment from the queue (non-blocking)"""
        try:
            return self.segment_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def clear_queue(self):
        """Clear any pending segments in the queue."""
        while not self.segment_queue.empty():
            try:
                self.segment_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

# --- FastAPI Application ---
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Your React app origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket client connection using Gemini Live API"""
    client_address = websocket.client.host if websocket.client else "Unknown"
    logger.info(f"Connection attempt received from {client_address}") # <<< LOG 1

    # --- Accept WebSocket Connection ---
    try:
        await websocket.accept()
        logger.info(f"WebSocket accepted for {client_address}") # <<< LOG 2
    except Exception as accept_err:
        # This should ideally not happen if the initial handshake reached here
        logger.error(f"Error during websocket.accept() for {client_address}: {accept_err}", exc_info=True)
        return # Exit if accept fails

    detector = AudioSegmentDetector()
    live_session = None
    receive_gemini_task = None
    send_gemini_task = None

    # --- Main Processing Block ---
    try:
        # --- Connect to Gemini Live Session ---
        logger.info(f"Attempting Gemini Live connect for {client_address}") # <<< LOG 3
        live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                )
            ),
        )
        # This is the most likely point to hang if there are API key/permission/network issues to Google
        live_session = await client.aio.live.connect(model=GEMINI_LIVE_MODEL_NAME, config=live_config)
        logger.info(f"Gemini Live session established for {client_address}") # <<< LOG 4

        # --- Define Concurrent Tasks ---

        # Task: Receive from Gemini -> Send to Client WS
        async def receive_from_gemini(ws: WebSocket, session: genai.live.AsyncLiveSession):
            logger.info(f"Starting Gemini receive task for {client_address}")
            try:
                async for response in session.receive():
                    # Check WebSocket state before sending
                    # Use isinstance check for broader compatibility if websockets library structure changes
                    # if not isinstance(ws.client_state, websockets.protocol.State) or ws.client_state != websockets.protocol.State.OPEN:
                    if ws.client_state != websockets.protocol.State.OPEN:
                         logger.warning(f"Client WS no longer open in receive_from_gemini for {client_address}, stopping task.")
                         break

                    if response.data is not None:
                        # logger.debug(f"Received audio data ({len(response.data)} bytes) from Gemini for {client_address}")
                        audio_data = response.data
                        base64_audio = base64.b64encode(audio_data).decode('utf-8')
                        try:
                            await ws.send_text(json.dumps({
                                "type": "audio",
                                "data": base64_audio,
                                "sample_rate": OUTPUT_SAMPLE_RATE
                            }))
                        except (WebSocketDisconnect, RuntimeError) as send_err:
                            logger.warning(f"Client WebSocket closed or error sending audio for {client_address}: {send_err}")
                            break # Stop receiving if cannot send
                    # Handle other responses... (interrupted, complete, tokens, etc.)
                    if response.server_content:
                         if response.server_content.interrupted: logger.info(f"Gemini interrupted for {client_address}.")
                         if response.server_content.generation_complete: logger.info(f"Gemini generation complete for {client_address}.")
                    if response.usage_metadata: logger.info(f"Tokens for {client_address}: {response.usage_metadata.total_token_count}")
                    if response.go_away: logger.warning(f"Gemini GoAway for {client_address}: {response.go_away.time_left}")
                    if response.session_resumption_update: logger.info(f"Gemini SessionResumptionUpdate for {client_address}.")

            except asyncio.CancelledError:
                 logger.info(f"Gemini receive task cancelled for {client_address}.")
            except Exception as e:
                 logger.error(f"Error in receive_from_gemini for {client_address}: {e}", exc_info=True)
                 try: await ws.send_text(json.dumps({"type": "error", "message": f"Gemini API Receive Error: {e}"}))
                 except: pass
            finally:
                 logger.info(f"Gemini receive task finished for {client_address}.")

        # Task: Get segments from Detector -> Send to Gemini Session
        async def send_to_gemini(det: AudioSegmentDetector, session: genai.live.AsyncLiveSession, ws: WebSocket):
            logger.info(f"Starting Gemini send task for {client_address}")
            try:
                while True:
                     # Check WebSocket state before proceeding
                    if ws.client_state != websockets.protocol.State.OPEN:
                        logger.warning(f"Client WS no longer open in send_to_gemini for {client_address}, stopping task.")
                        break
                    audio_segment = await det.get_next_segment()
                    if audio_segment:
                        # logger.debug(f"Sending audio segment ({len(audio_segment)} bytes) to Gemini for {client_address}.")
                        try:
                            await session.send_client_content(
                                parts=[{"inline_data": {"mime_type": "audio/pcm", "data": audio_segment}}],
                                turn_complete=True
                            )
                        except Exception as send_err:
                             logger.error(f"Error sending audio to Gemini for {client_address}: {send_err}", exc_info=True)
                             # Add a small delay before retrying or breaking?
                             await asyncio.sleep(0.1)
                    await asyncio.sleep(0.02) # Yield control
            except asyncio.CancelledError:
                logger.info(f"Gemini send task cancelled for {client_address}.")
            except Exception as e:
                 logger.error(f"Error in send_to_gemini task for {client_address}: {e}", exc_info=True)
            finally:
                 logger.info(f"Gemini send task finished for {client_address}.")

        # --- Start Background Tasks ---
        logger.info(f"Starting background tasks for {client_address}") # <<< LOG 5
        receive_gemini_task = asyncio.create_task(receive_from_gemini(websocket, live_session))
        send_gemini_task = asyncio.create_task(send_to_gemini(detector, live_session, websocket))
        logger.info(f"Background tasks started for {client_address}") # <<< LOG 6

        # --- Main Loop: Receive messages from Client WebSocket ---
        logger.info(f"Starting main client receive loop for {client_address}") # <<< LOG 7
        while True:
            message_text = await websocket.receive_text()
            # logger.debug(f"Received message from {client_address}: {message_text[:100]}...") # Use debug for frequent logs
            data = json.loads(message_text)

            if "realtime_input" in data:
                for chunk in data["realtime_input"]["media_chunks"]:
                    mime_type = chunk.get("mime_type", "")
                    chunk_data_b64 = chunk.get("data", "")
                    if not chunk_data_b64: continue
                    decoded_data = base64.b64decode(chunk_data_b64)
                    if mime_type == "audio/pcm" or mime_type == "audio/l16":
                         await detector.add_audio(decoded_data)
                    # Ignore image data for now
            # Ignore standalone image data for now

    except WebSocketDisconnect:
        logger.info(f"Client {client_address} disconnected.")
    except json.JSONDecodeError as json_err:
        logger.error(f"Received invalid JSON from {client_address}. Error: {json_err}. Closing connection.")
        # Attempt to close with unsupported data code
        # await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA) # Might fail if already closing
    except Exception as e:
        logger.error(f"Unhandled exception during WebSocket processing for {client_address}: {e}", exc_info=True)
        # Attempt to close with an internal error code
        try: await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except: pass # Ignore errors during close after exception
    finally:
        logger.info(f"Cleaning up resources for client {client_address}")
        # Cancel running asyncio tasks
        tasks_to_cancel = [t for t in [receive_gemini_task, send_gemini_task] if t and not t.done()]
        if tasks_to_cancel:
            logger.info(f"Cancelling {len(tasks_to_cancel)} background tasks for {client_address}...")
            for task in tasks_to_cancel: task.cancel()
            try: await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            except asyncio.CancelledError: pass # Expected
            logger.info(f"Background tasks cancelled for {client_address}.")

        # Close the Gemini Live session gracefully
        if live_session:
            logger.info(f"Closing Gemini Live session for {client_address}...")
            try: await live_session.aclose()
            except Exception as close_err: logger.error(f"Error closing Gemini Live session for {client_address}: {close_err}")
            logger.info(f"Gemini Live session closed for {client_address}.")

        # Check state before attempting to close WebSocket (FastAPI might handle this)
        if websocket.client_state == websockets.protocol.State.OPEN:
             logger.info(f"Closing WebSocket connection for {client_address} from finally block.")
             try: await websocket.close()
             except: pass # Ignore potential errors if already closing

        logger.info(f"Finished cleanup for {client_address}")


# --- Run Server (if executed directly) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server with Uvicorn...")
    # Ensure host is 0.0.0.0 to be accessible, port 9073
    uvicorn.run(
        "app:app", # Use the correct format: 'filename:instance_name'
        host="0.0.0.0",
        port=9073,
        log_level="info", # Uvicorn's log level
        reload=True # Keep reload for development
        # reload_dirs=["."] # Optional: Specify directories to watch
        )