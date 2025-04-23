import os
import io
import base64
import asyncio
import logging
from collections import deque

import cv2
import mss
import PIL.Image
import pyaudio
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room

from google import genai
from google.ai.generativelanguage import Part

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Constants ---
MODEL = "models/gemini-2.0-flash-live-001"
# Rate and size constants
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
BROWSER_AUDIO_CHUNKS_PER_SEC = 5

# Audio capture constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK_SIZE = 1024

# --- Configure GenAI Client ---
# Ensure GOOGLE_API_KEY is set in environment
API_KEY = "AIzaSyCDvSi6OVlgdODnPmHmIBcc5UylRH0CvB8"
if not API_KEY:
    log.error("🔴 GOOGLE_API_KEY environment variable not set.")
    raise SystemExit("Please set the GOOGLE_API_KEY environment variable.")

client = genai.Client(
    api_key=API_KEY,
    http_options={"api_version": "v1beta"}
)

# --- Flask & SocketIO Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, async_mode='eventlet', logger=True, engineio_logger=True)

# Store active sessions
client_sessions = {}

# --- Audio/Video Capture Class ---
class AVCapture:
    def __init__(self, mode="camera"):
        self.mode = mode
        self.pya = pyaudio.PyAudio()
        self.audio_stream = None

    async def init_audio(self):
        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info['index'],
            frames_per_buffer=CHUNK_SIZE
        )

    async def read_audio(self):
        data = await asyncio.to_thread(
            self.audio_stream.read,
            CHUNK_SIZE,
            exception_on_overflow=False
        )
        return {"mime_type": "audio/pcm", "data": data}

    async def read_camera(self):
        # Capture one frame from default camera
        cap = cv2.VideoCapture(0)
        ret, frame = await asyncio.to_thread(cap.read)
        cap.release()
        if not ret:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(rgb)
        img.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return {"mime_type": "image/jpeg", "data": buffer.getvalue()}

    async def read_screen(self):
        sct = mss.mss()
        shot = sct.grab(sct.monitors[0])
        img = PIL.Image.frombytes("RGB", shot.size, shot.rgb)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return {"mime_type": "image/jpeg", "data": buffer.getvalue()}

# --- Gemini Session Handler ---
class GeminiSessionHandler:
    def __init__(self, sid, mode="camera"):
        self.sid = sid
        self.mode = mode
        self.session = None
        self.running = False
        self.tasks = []
        self.text_buffer = deque(maxlen=5)
        self.cap = None

    async def start(self):
        log.info(f"[{self.sid}] Starting Gemini session (mode: {self.mode})")
        self.running = True
        config = {
            "response_modalities": ["AUDIO", "TEXT"],
            "audio_input": {"sample_rate": SEND_SAMPLE_RATE, "channels": AUDIO_CHANNELS},
            "audio_output": {"sample_rate": RECEIVE_SAMPLE_RATE, "channels": AUDIO_CHANNELS}
        }
        try:
            aio_client = client.aio
            self.session = await aio_client.live.connect(model=MODEL, config=config)
            log.info(f"[{self.sid}] Gemini session connected.")

            # Setup local audio/video capture
            self.cap = AVCapture(self.mode)
            await self.cap.init_audio()

            # Start capture/send loop
            self.tasks.append(socketio.start_background_task(self._capture_and_send))
            # Start receive loop
            self.tasks.append(socketio.start_background_task(self._receive_audio_task))

            socketio.emit('session_started', {'sid': self.sid}, room=self.sid)
        except Exception as e:
            log.error(f"[{self.sid}] Error starting session: {e}")
            self.running = False
            socketio.emit('error', {'message': str(e)}, room=self.sid)
            await self.stop()

    async def stop(self):
        log.info(f"[{self.sid}] Stopping Gemini session...")
        self.running = False
        for task in self.tasks:
            try:
                task.cancel()
            except Exception:
                pass
        self.tasks.clear()
        if self.session:
            # Cleanup if explicit close exists
            try:
                if hasattr(self.session, 'close'):
                    await self.session.close()
            except Exception:
                pass
        self.session = None
        if self.sid in client_sessions:
            del client_sessions[self.sid]
        socketio.emit('session_stopped', {'sid': self.sid}, room=self.sid)

    async def _capture_and_send(self):
        """Continuously capture audio/video and send to Gemini."""
        while self.running and self.session:
            # Audio chunk
            audio = await self.cap.read_audio()
            await self.session.send(input=Part(inline_data=audio))
            # Video frame if applicable
            video = None
            if self.mode == 'camera':
                video = await self.cap.read_camera()
            elif self.mode == 'screen':
                video = await self.cap.read_screen()
            if video:
                await self.session.send(input=Part(inline_data=video))
            # Pace the loop
            await asyncio.sleep(1.0 / BROWSER_AUDIO_CHUNKS_PER_SEC)

    async def _receive_audio_task(self):
        """Receive and forward audio/text responses from Gemini."""
        interim_text = ""
        try:
            while self.running and self.session:
                turn = self.session.receive()
                async for response in turn:
                    if data := response.data:
                        socketio.emit('audio_chunk', data, room=self.sid)
                    if text := response.text:
                        if response.end_of_turn:
                            self.text_buffer.append(f"Model: {text}")
                            socketio.emit('text_message', {'text': text, 'is_final': True, 'interim': ""}, room=self.sid)
                            interim_text = ""
                        else:
                            interim_text = text
                            socketio.emit('text_message', {'text': "", 'is_final': False, 'interim': interim_text}, room=self.sid)
                    if error := getattr(response, 'error', None):
                        socketio.emit('error', {'message': str(error)}, room=self.sid)
                if interim_text:
                    socketio.emit('text_message', {'text': "", 'is_final': False, 'interim': ""}, room=self.sid)
                    interim_text = ""
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"[{self.sid}] Receive task error: {e}")
        finally:
            if interim_text:
                socketio.emit('text_message', {'text': "", 'is_final': False, 'interim': ""}, room=self.sid)

# --- Flask Routes & SocketIO Events ---
@app.route('/')
def index():
    return render_template('index.html', send_sample_rate=SEND_SAMPLE_RATE,
                           receive_sample_rate=RECEIVE_SAMPLE_RATE,
                           browser_chunks_per_sec=BROWSER_AUDIO_CHUNKS_PER_SEC)

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    log.info(f"Client connected: {sid}")
    emit('connection_ack', {'sid': sid})

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    log.info(f"Client disconnecting: {sid}")
    if sid in client_sessions:
        socketio.start_background_task(client_sessions[sid].stop)

@socketio.on('start_session')
@socketio.on('start_session')
def handle_start_session(data):
    sid  = request.sid
    mode = data.get('mode', 'camera')
    if sid in client_sessions:
        emit('error', {'message': 'Session already active. Stop first.'}, room=sid)
        return

    handler = GeminiSessionHandler(sid, mode)
    client_sessions[sid] = handler

    # wrap the coroutine in asyncio.run() so it actually executes:
    def _run():
        asyncio.run(handler.start())

    socketio.start_background_task(_run)


@socketio.on('stop_session')
def handle_stop_session():
    sid = request.sid
    if sid in client_sessions:
        socketio.start_background_task(client_sessions[sid].stop)
    else:
        emit('session_stopped', {'sid': sid}, room=sid)

@socketio.on('text_message')
def handle_text_message(data):
    sid = request.sid
    text = data.get('message')
    if sid in client_sessions and text:
        client_sessions[sid].text_buffer.append(f"User: {text}")
        socketio.start_background_task(client_sessions[sid]._send_text_async, text)

# --- Entry Point ---
if __name__ == '__main__':
    log.info("Starting Flask-SocketIO server...")
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)
