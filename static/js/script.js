document.addEventListener('DOMContentLoaded', () => {
    const socket = io(); // Connect to Socket.IO server

    // --- Config ---
    const configDiv = document.getElementById('config');
    const SEND_SAMPLE_RATE = parseInt(configDiv.dataset.sendSampleRate, 10);
    const RECEIVE_SAMPLE_RATE = parseInt(configDiv.dataset.receiveSampleRate, 10);
    const BROWSER_AUDIO_CHUNKS_PER_SEC = parseInt(configDiv.dataset.browserChunksPerSec, 10);
    const AUDIO_CHUNK_INTERVAL_MS = 1000 / BROWSER_AUDIO_CHUNKS_PER_SEC; // Interval to send audio chunks

    // --- UI Elements ---
    const modeSelect = document.getElementById('mode-select');
    const startButton = document.getElementById('start-button');
    const stopButton = document.getElementById('stop-button');
    const statusSpan = document.getElementById('status');
    const mediaPreviewDiv = document.getElementById('media-preview');
    const videoPreview = document.getElementById('video-preview');
    const logDiv = document.getElementById('log');
    const interimLogDiv = document.getElementById('interim-log');
    const textInput = document.getElementById('text-input');
    const sendButton = document.getElementById('send-button');

    // --- State Variables ---
    let localStream = null;
    let audioContext = null;
    let audioProcessorNode = null;
    let audioSourceNode = null;
    let videoIntervalId = null;
    let isSessionActive = false;
    let clientSid = null;

    // Audio Playback Queue
    let audioQueue = [];
    let isPlaying = false;
    let playbackAudioContext = null; // Separate context for playback


    // --- Helper Functions ---
    function logMessage(sender, message, isFinal = true) {
        const messageDiv = document.createElement('div');
        messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
        if (!isFinal) {
            messageDiv.style.color = 'gray';
            messageDiv.style.fontStyle = 'italic';
        }
        logDiv.appendChild(messageDiv);
        logDiv.scrollTop = logDiv.scrollHeight; // Auto-scroll
    }

    function setStatus(message, isError = false) {
        statusSpan.textContent = `Status: ${message}`;
        statusSpan.style.color = isError ? 'red' : 'black';
        console.log(`Status: ${message}`);
    }

    function updateUIState(sessionRunning) {
        isSessionActive = sessionRunning;
        startButton.disabled = sessionRunning;
        stopButton.disabled = !sessionRunning;
        textInput.disabled = !sessionRunning;
        sendButton.disabled = !sessionRunning;
        modeSelect.disabled = sessionRunning;
        if (!sessionRunning) {
             mediaPreviewDiv.style.display = 'none';
             interimLogDiv.textContent = ''; // Clear interim on stop
        } else {
             const mode = modeSelect.value;
             mediaPreviewDiv.style.display = (mode === 'camera' || mode === 'screen') ? 'block' : 'none';
        }
    }

    // --- Audio Processing (using AudioWorklet) ---
    async function setupAudioProcessing() {
        if (!localStream || !localStream.getAudioTracks().length) {
             throw new Error("No audio track available in the stream.");
        }
        if (audioContext && audioContext.state !== 'closed') {
            await audioContext.close(); // Close previous context if exists
        }
        audioContext = new AudioContext({ sampleRate: SEND_SAMPLE_RATE });

        try {
            // Dynamically import the worklet script
            await audioContext.audioWorklet.addModule('/static/js/audio-processor.js');
            console.log("AudioWorklet module added.");
        } catch (e) {
            console.error("Error adding AudioWorklet module:", e);
            setStatus("Error loading audio processor", true);
            throw e; // Re-throw to stop the process
        }

        audioSourceNode = audioContext.createMediaStreamSource(localStream);
        audioProcessorNode = new AudioWorkletNode(audioContext, 'audio-processor', {
             processorOptions: {
                 targetSampleRate: SEND_SAMPLE_RATE,
                 bufferSize: Math.floor(SEND_SAMPLE_RATE * AUDIO_CHUNK_INTERVAL_MS / 1000) // Calculate buffer size based on interval
             }
        });

        audioProcessorNode.port.onmessage = (event) => {
            if (event.data.type === 'audioData' && isSessionActive) {
                // event.data.buffer contains the Int16Array raw PCM data
                // Send ArrayBuffer directly
                socket.emit('audio_chunk', event.data.buffer.buffer);
            } else if (event.data.type === 'debug') {
                console.log('Audio Processor Debug:', event.data.message);
            }
        };

        audioSourceNode.connect(audioProcessorNode);
        // We don't need to connect to destination if we only process
        // Connecting the processor ensures it runs
        audioProcessorNode.connect(audioContext.destination); // Connect to keep graph alive, output will be silent
        audioContext.resume(); // Ensure context is running
        console.log("Audio processing pipeline setup complete.");
    }

    // --- Video Processing ---
    function startVideoProcessing() {
        if (!localStream || !localStream.getVideoTracks().length || modeSelect.value === 'none') {
            return; // No video track or mode is 'none'
        }
        console.log("Starting video processing.");
        videoPreview.srcObject = localStream;
        mediaPreviewDiv.style.display = 'block'; // Show preview area

        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        const trackSettings = localStream.getVideoTracks()[0].getSettings();
        const frameRate = trackSettings.frameRate || 1; // Default to 1 FPS if not available

        // Calculate interval based on desired frame rate (e.g., 1 frame per second)
        const videoInterval = 1000; // Send 1 frame per second

        videoIntervalId = setInterval(() => {
            if (!isSessionActive || videoPreview.readyState < videoPreview.HAVE_CURRENT_DATA) {
                return; // Don't send if session stopped or video not ready
            }

            // Adjust canvas size - keep aspect ratio, limit width
            const targetWidth = 320; // Smaller frame size
            const aspectRatio = videoPreview.videoWidth / videoPreview.videoHeight;
            canvas.width = targetWidth;
            canvas.height = targetWidth / aspectRatio;

            try {
                context.drawImage(videoPreview, 0, 0, canvas.width, canvas.height);
                // Get base64 JPEG data URL
                const dataUrl = canvas.toDataURL('image/jpeg', 0.7); // Quality 0.7
                socket.emit('video_frame', dataUrl);
            } catch (e) {
                 console.error("Error capturing or sending video frame:", e);
                 // Consider stopping video processing if errors persist
            }

        }, videoInterval);
    }

    function stopVideoProcessing() {
        if (videoIntervalId) {
            clearInterval(videoIntervalId);
            videoIntervalId = null;
        }
        videoPreview.srcObject = null; // Clear preview
        mediaPreviewDiv.style.display = 'none';
        console.log("Video processing stopped.");
    }

    // --- Audio Playback ---
    function initializePlaybackAudioContext() {
        if (!playbackAudioContext || playbackAudioContext.state === 'closed') {
            try {
                playbackAudioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: RECEIVE_SAMPLE_RATE // Use rate from server
                });
                console.log(`Playback AudioContext initialized with sample rate: ${playbackAudioContext.sampleRate}`);
            } catch (e) {
                console.error("Error creating playback AudioContext:", e);
                setStatus("Error initializing audio playback", true);
            }
        }
         // Resume context on user interaction if needed (browsers often require this)
        playbackAudioContext.resume();
    }

    function playAudioChunk(audioData) { // audioData is ArrayBuffer from server
        if (!playbackAudioContext || playbackAudioContext.state === 'closed') {
            console.warn("Playback AudioContext not ready or closed.");
            initializePlaybackAudioContext(); // Try to re-initialize
            if (!playbackAudioContext) return; // Still failed
        }
        playbackAudioContext.resume(); // Ensure it's running

        // Received data is raw PCM Int16 bytes. Convert to Float32 for Web Audio API.
        const int16Array = new Int16Array(audioData);
        const float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / 32768.0; // Convert to range [-1.0, 1.0]
        }

        if (float32Array.length === 0) return;

        audioQueue.push(float32Array);
        playQueue();
    }

    function playQueue() {
        if (isPlaying || audioQueue.length === 0 || !playbackAudioContext || playbackAudioContext.state !== 'running') {
            return;
        }
        isPlaying = true;

        const audioChunk = audioQueue.shift();
        try {
             const buffer = playbackAudioContext.createBuffer(
                1, // Number of channels
                audioChunk.length,
                playbackAudioContext.sampleRate
             );
            buffer.copyToChannel(audioChunk, 0); // Fill buffer

            const source = playbackAudioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(playbackAudioContext.destination);
            source.onended = () => {
                isPlaying = false;
                // Check immediately if more data is available to reduce gaps
                setTimeout(playQueue, 0); // Use setTimeout to avoid call stack overflow on rapid chunks
            };
            source.start(); // Play immediately
        } catch (e) {
             console.error("Error playing audio chunk:", e);
             isPlaying = false;
             // Consider clearing queue or handling error more gracefully
             setTimeout(playQueue, 10); // Try next chunk after short delay
        }
    }

     function clearAudioPlayback() {
        audioQueue = []; // Clear the queue
        isPlaying = false;
        // We don't necessarily need to stop the source node as it stops onended
        // If stopping mid-playback is needed, store the current source and call source.stop()
        console.log("Audio playback queue cleared.");
    }


    // --- Media Stream Handling ---
    async function startMediaCapture(mode) {
        try {
            // Stop existing stream first
            await stopMediaCapture();

            const constraints = {
                audio: {
                    sampleRate: SEND_SAMPLE_RATE, // Request desired rate if possible
                    channelCount: 1,
                    echoCancellation: true, // Enable echo cancellation
                    noiseSuppression: true  // Enable noise suppression
                },
                video: false // Default to no video
            };

            if (mode === 'camera') {
                constraints.video = { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } };
            }

            if (mode === 'screen') {
                 // Use getDisplayMedia for screen sharing
                localStream = await navigator.mediaDevices.getDisplayMedia({
                    video: { width: { ideal: 1280 }, height: { ideal: 720 } }, // Request screen resolution
                    audio: false // Typically capture mic separately for better control
                });
                // Need to get audio separately if screen share audio isn't captured or desired
                const audioStream = await navigator.mediaDevices.getUserMedia({ audio: constraints.audio });
                // Add the audio track to the screen stream
                audioStream.getAudioTracks().forEach(track => localStream.addTrack(track));
            } else {
                // Camera or Audio Only mode
                localStream = await navigator.mediaDevices.getUserMedia(constraints);
            }

            console.log("Media stream obtained.");
            // Setup audio processing pipeline
            await setupAudioProcessing();

            // Start video processing if applicable
            if (mode === 'camera' || mode === 'screen') {
                startVideoProcessing();
            }

        } catch (err) {
            console.error('Error accessing media devices.', err);
            setStatus(`Error accessing media: ${err.name} - ${err.message}`, true);
            await stopMediaCapture(); // Clean up partial streams
            updateUIState(false); // Ensure UI reflects stopped state
            throw err; // Propagate error to stop session start
        }
    }

    async function stopMediaCapture() {
        console.log("Stopping media capture...");
        stopVideoProcessing(); // Stop sending video frames first

        if (audioProcessorNode) {
            audioProcessorNode.disconnect();
            audioProcessorNode.port.close(); // Close the port
            audioProcessorNode = null;
            console.log("AudioProcessorNode disconnected.");
        }
        if (audioSourceNode) {
            audioSourceNode.disconnect();
            audioSourceNode = null;
            console.log("AudioSourceNode disconnected.");
        }

        if (audioContext && audioContext.state !== 'closed') {
            await audioContext.close();
            audioContext = null;
            console.log("AudioContext closed.");
        }

        if (localStream) {
            localStream.getTracks().forEach(track => {
                track.stop();
                console.log(`Track stopped: ${track.kind}`);
            });
            localStream = null;
            console.log("Local media stream stopped.");
        }
    }

    // --- Socket.IO Event Handlers ---
    socket.on('connect', () => {
        setStatus('Connected to server');
        console.log('Socket connected:', socket.id);
    });

    socket.on('disconnect', (reason) => {
        setStatus(`Disconnected: ${reason}`, true);
        console.error('Socket disconnected:', reason);
        stopMediaCapture(); // Clean up media
        clearAudioPlayback();
        updateUIState(false);
        clientSid = null;
    });

    socket.on('connection_ack', (data) => {
        clientSid = data.sid;
        setStatus(`Registered with SID: ${clientSid}`);
        console.log('Connection acknowledged by server:', clientSid);
        updateUIState(false); // Ready to start
    });

    socket.on('session_started', (data) => {
        if (data.sid === clientSid) {
             setStatus('Session Active');
             console.log('Session started successfully.');
             initializePlaybackAudioContext(); // Prepare for receiving audio
             updateUIState(true);
        }
    });

    socket.on('session_stopped', (data) => {
         if (data.sid === clientSid || !clientSid) { // Handle stop even if sid mismatch somehow
            setStatus('Session Stopped');
            console.log('Session stopped.');
            stopMediaCapture();
            clearAudioPlayback();
            updateUIState(false);
            // Don't nullify clientSid here, keep it until disconnect
         }
    });

    socket.on('audio_chunk', (chunk) => {
        // chunk is received as ArrayBuffer
        playAudioChunk(chunk);
    });

    socket.on('text_message', (data) => {
        // Handle both final and interim text
        if (data.is_final) {
            interimLogDiv.textContent = ''; // Clear interim when final arrives
            logMessage('Model', data.text);
        } else if (data.interim) {
            interimLogDiv.textContent = `Model (interim): ${data.interim}`;
        } else {
             // Clear interim if empty message received
             interimLogDiv.textContent = '';
        }
    });

    socket.on('error', (data) => {
        setStatus(`Server Error: ${data.message}`, true);
        console.error('Server Error:', data.message);
        // Decide if the session should be stopped based on error
        // For now, just log it. User might need to manually stop/start.
    });

    // --- UI Event Listeners ---
    startButton.addEventListener('click', async () => {
        if (isSessionActive) return;
        setStatus('Starting session...');
        startButton.disabled = true; // Prevent double clicks

        const selectedMode = modeSelect.value;
        try {
            await startMediaCapture(selectedMode); // Request permissions and setup streams
            socket.emit('start_session', { mode: selectedMode });
            // UI state updated via 'session_started' event from server
        } catch (error) {
            console.error("Failed to start session:", error);
            setStatus(`Failed to start: ${error.message || error}`, true);
            await stopMediaCapture(); // Ensure cleanup if start failed
            updateUIState(false); // Reset UI state
        }
    });

    stopButton.addEventListener('click', () => {
        if (!isSessionActive) return;
        setStatus('Stopping session...');
        stopButton.disabled = true; // Prevent double clicks
        socket.emit('stop_session');
        // UI state and media capture stopped via 'session_stopped' event
        // Explicitly stop media capture here too for faster cleanup
        stopMediaCapture();
        clearAudioPlayback();
        updateUIState(false); // Immediately update UI
    });

    sendButton.addEventListener('click', () => {
        const text = textInput.value.trim();
        if (text && isSessionActive) {
            logMessage('User', text);
            socket.emit('text_message', { message: text });
            textInput.value = '';
        }
    });

    textInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendButton.click(); // Trigger send on Enter key
        }
    });

    // Initial UI state
    updateUIState(false);
});