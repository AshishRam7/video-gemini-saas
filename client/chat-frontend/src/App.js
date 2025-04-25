import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

// --- Configuration ---
const WEBSOCKET_URL = 'ws://localhost:9073/ws'; // Ensure '/ws' path for FastAPI
const TARGET_SAMPLE_RATE = 16000; // Sample rate server expects
const OUTPUT_SAMPLE_RATE = 24000; // Sample rate received from server (Gemini Live API)
const BUFFER_SIZE = 4096; // Audio processing buffer size

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Disconnected');
  const [errorMessage, setErrorMessage] = useState('');

  const ws = useRef(null);
  const audioContext = useRef(null);
  const scriptProcessor = useRef(null);
  const mediaStream = useRef(null);
  const playbackAudioContext = useRef(null); // Separate context for playback
  const audioQueue = useRef([]); // Queue for incoming audio chunks
  const sourceNode = useRef(null); // To track the currently playing node


  // --- Audio Playback Logic ---
  // Define playback first as it doesn't depend on recording/websocket logic directly
  const playNextChunk = useCallback(async () => {
    if (isPlaying || audioQueue.current.length === 0) {
      return;
    }

    setIsPlaying(true);

    const { buffer: chunkBuffer, sampleRate } = audioQueue.current.shift(); // Get ArrayBuffer (Int16 data)

    try {
      if (!playbackAudioContext.current || playbackAudioContext.current.state === 'closed') {
        playbackAudioContext.current = new (window.AudioContext || window.webkitAudioContext)();
        console.log("Playback AudioContext created/resumed. Sample Rate:", playbackAudioContext.current.sampleRate);
      }
      if (playbackAudioContext.current.state === 'suspended') {
        await playbackAudioContext.current.resume();
      }

      const pcm16Data = new Int16Array(chunkBuffer);
      const pcmFloat32Data = new Float32Array(pcm16Data.length);
      for (let i = 0; i < pcm16Data.length; i++) {
        pcmFloat32Data[i] = pcm16Data[i] / 32768.0;
      }

      const audioBuffer = playbackAudioContext.current.createBuffer(
        1, pcmFloat32Data.length, sampleRate
      );
      audioBuffer.copyToChannel(pcmFloat32Data, 0);

      const source = playbackAudioContext.current.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(playbackAudioContext.current.destination);

      source.onended = () => {
          source.disconnect();
          sourceNode.current = null;
          setIsPlaying(false);
          setTimeout(playNextChunk, 0);
      };

      sourceNode.current = source;
      source.start();

    } catch (error) {
      console.error('Error playing audio chunk:', error);
      setErrorMessage(`Playback Error: ${error.message}`);
      setIsPlaying(false);
      setTimeout(playNextChunk, 50);
    }
  }, [isPlaying]); // Depends only on isPlaying state


  // --- Recording and WebSocket Logic ---
  // Need stopRecording defined before functions that use it.

  const stopRecording = useCallback(() => {
    if (!isRecording && !mediaStream.current && !audioContext.current) {
      return; // Nothing to stop
    }
     console.log("Stopping recording...");

    setIsRecording(false);
    // Only update status if it was actively 'Recording...' to avoid overwriting other messages
    if(statusMessage === 'Recording...') setStatusMessage('Recording stopped.');

    if (scriptProcessor.current) {
      scriptProcessor.current.disconnect();
      scriptProcessor.current.onaudioprocess = null;
      scriptProcessor.current = null;
    }
    if (mediaStream.current) {
      mediaStream.current.getTracks().forEach(track => track.stop());
      mediaStream.current = null;
    }
    if (audioContext.current && audioContext.current.state !== 'closed') {
      audioContext.current.close().then(() => {
        console.log("Recording AudioContext closed.");
        audioContext.current = null;
      }).catch(e => console.error("Error closing recording AudioContext:", e));
    } else {
         audioContext.current = null;
    }
  // Add statusMessage dependency as we read it
  }, [isRecording, statusMessage]);


  // Now define functions that depend on stopRecording
  const disconnectWebSocket = useCallback(() => {
    stopRecording(); // Stop recording before disconnecting
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
    setIsConnected(false);
    setStatusMessage('Disconnected');
  }, [stopRecording]); // Correctly depends on the memoized stopRecording


  const connectWebSocket = useCallback(() => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      setStatusMessage('Already connected.');
      return;
    }

    setStatusMessage('Connecting...');
    setErrorMessage('');
    ws.current = new WebSocket(WEBSOCKET_URL);

    ws.current.onopen = () => {
      console.log('WebSocket Connected');
      setIsConnected(true);
      setStatusMessage('Connected. Ready to record.');
    };

    ws.current.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'audio') {
          const audioData = message.data;
          const sampleRate = message.sample_rate || OUTPUT_SAMPLE_RATE;
          const byteString = atob(audioData);
          const byteArray = new Uint8Array(byteString.length);
          for (let i = 0; i < byteString.length; i++) {
            byteArray[i] = byteString.charCodeAt(i);
          }
          audioQueue.current.push({ buffer: byteArray.buffer, sampleRate });
          playNextChunk(); // Attempt to play

        } else if (message.type === 'error') {
          console.error('Server Error:', message.message);
          setErrorMessage(`Server Error: ${message.message}`);
        } else {
            console.log("Received unknown message type:", message.type);
        }
      } catch (error) {
        console.error('Failed to parse message or handle audio:', error);
        setErrorMessage(`Client Error: ${error.message}`);
      }
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket Error:', error);
      setStatusMessage('WebSocket error');
      setErrorMessage('WebSocket connection error. Check console.');
      setIsConnected(false);
      stopRecording(); // Call stopRecording here too
    };

    ws.current.onclose = (event) => {
      console.log('WebSocket Closed:', event.reason);
      setIsConnected(false);
      setStatusMessage(`Disconnected: ${event.reason || 'Connection closed'}`);
      stopRecording(); // Call stopRecording here
      ws.current = null;
    };
  // Add playNextChunk and stopRecording as dependencies
  }, [playNextChunk, stopRecording]);


  // Simple resampling function (nearest neighbor/average)
  const downsampleBuffer = (buffer, inputSampleRate, outputSampleRate) => {
    if (inputSampleRate === outputSampleRate) {
      return buffer;
    }
    const sampleRateRatio = inputSampleRate / outputSampleRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
      let accum = 0;
      let count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }
      result[offsetResult] = count > 0 ? accum / count : 0;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  };

  // Convert Float32 buffer to Int16 ArrayBuffer
  const float32ToInt16 = (buffer) => {
    const int16Buffer = new Int16Array(buffer.length);
    for (let i = 0; i < buffer.length; i++) {
      const val = Math.max(-1, Math.min(1, buffer[i]));
      int16Buffer[i] = val * 32767;
    }
    return int16Buffer.buffer;
  };

  const startRecording = useCallback(async () => {
    if (isRecording || !isConnected) {
        setStatusMessage(isConnected ? 'Already recording.' : 'Connect first.');
        return;
    }
    setStatusMessage('Initializing microphone...');
    setErrorMessage('');
    audioQueue.current = []; // Clear playback queue

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      mediaStream.current = stream;
      audioContext.current = new (window.AudioContext || window.webkitAudioContext)();
      setStatusMessage('Microphone access granted. Starting...');
      console.log("AudioContext Sample Rate:", audioContext.current.sampleRate);

      const source = audioContext.current.createMediaStreamSource(stream);
      scriptProcessor.current = audioContext.current.createScriptProcessor(BUFFER_SIZE, 1, 1);

      scriptProcessor.current.onaudioprocess = (event) => {
        // Check isRecording state directly inside the callback
        if (!isRecording || !ws.current || ws.current.readyState !== WebSocket.OPEN) {
          return;
        }

        const inputData = event.inputBuffer.getChannelData(0);
        const downsampledData = downsampleBuffer(
          inputData, audioContext.current.sampleRate, TARGET_SAMPLE_RATE
        );
        const pcm16Buffer = float32ToInt16(downsampledData);
        const base64String = btoa(String.fromCharCode(...new Uint8Array(pcm16Buffer)));

        const message = {
          realtime_input: {
            media_chunks: [{ mime_type: 'audio/pcm', data: base64String }]
          }
        };
        ws.current.send(JSON.stringify(message));
      };

      source.connect(scriptProcessor.current);
      scriptProcessor.current.connect(audioContext.current.destination);

      // Update state *after* setup is complete
      setIsRecording(true);
      setStatusMessage('Recording...');

    } catch (error) {
      console.error('Error starting recording:', error);
      setStatusMessage('Error starting recording');
      setErrorMessage(`Microphone Error: ${error.message}. Please grant permission.`);
      stopRecording(); // Use stopRecording for cleanup on error
    }
  // Add isRecording back as a dependency since it's checked in onaudioprocess
  }, [isConnected, isRecording, stopRecording]);


  // Cleanup effect on unmount
  useEffect(() => {
    // This function runs when the component unmounts
    return () => {
      console.log("App unmounting - cleaning up...");
      disconnectWebSocket(); // Disconnect WS (will also call stopRecording)
      // Ensure playback context is cleaned up too
      if (playbackAudioContext.current && playbackAudioContext.current.state !== 'closed') {
          playbackAudioContext.current.close().catch(e => console.error("Error closing playback AudioContext on unmount:", e));
          playbackAudioContext.current = null;
      }
      if (sourceNode.current) {
         try { sourceNode.current.stop(); sourceNode.current.disconnect(); } catch (e) { /* Ignore errors stopping already stopped node */ }
          sourceNode.current = null;
      }
      audioQueue.current = [];
    };
  // Add disconnectWebSocket as dependency
  }, [disconnectWebSocket]);

  return (
    <div className="App">
      <h1>Gemini Live API Voice Client</h1>
      <div className="controls">
        {!isConnected ? (
          <button onClick={connectWebSocket} disabled={isConnected}>Connect</button>
        ) : (
          <button onClick={disconnectWebSocket} disabled={!isConnected}>Disconnect</button>
        )}
        <button onClick={startRecording} disabled={!isConnected || isRecording}>Start Recording</button>
        <button onClick={stopRecording} disabled={!isRecording}>Stop Recording</button>
      </div>
      <div className="status">
        <p>Status: <span className={`status-text ${isConnected ? 'connected' : 'disconnected'}`}>{statusMessage}</span></p>
        {isRecording && <p className="recording-indicator">🎙️ Recording...</p>}
        {isPlaying && <p className="playing-indicator">🔊 Playing...</p>}
        {errorMessage && <p className="error-message">Error: {errorMessage}</p>}
      </div>
    </div>
  );
}

export default App;