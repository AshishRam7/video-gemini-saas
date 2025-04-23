// static/js/audio-processor.js

class AudioProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.targetSampleRate = options?.processorOptions?.targetSampleRate || 16000;
        // Calculate buffer size based on desired chunk interval/rate, passed from main script
        // e.g., 16000 samples/sec * 0.2 sec/chunk = 3200 samples per chunk
        this.bufferSize = options?.processorOptions?.bufferSize || 3200;
        this.internalBuffer = new Int16Array(this.bufferSize);
        this.bufferIndex = 0;
        this.lastSentTime = 0;
        this.chunkIntervalMs = 1000 / (this.targetSampleRate / this.bufferSize); // Calculated interval

        this.debugLog('Processor initialized.');
        this.debugLog(`Target Sample Rate: ${this.targetSampleRate}, Buffer Size: ${this.bufferSize}, Chunk Interval: ${this.chunkIntervalMs.toFixed(2)}ms`);

         this.port.onmessage = (event) => {
            // Handle messages from main thread if needed (e.g., config changes)
            this.debugLog(`Message from main thread: ${JSON.stringify(event.data)}`);
        };
    }

     debugLog(message) {
        // Send debug messages back to the main thread
        this.port.postMessage({ type: 'debug', message: `[AudioProcessor] ${message}` });
    }

    process(inputs, outputs, parameters) {
        // Assuming mono input channel 0
        const inputChannel = inputs[0]?.[0];

        if (!inputChannel || inputChannel.length === 0) {
            // No input data, keep processor alive
            return true;
        }

        // Input data is Float32Array, typically -1.0 to 1.0
        for (let i = 0; i < inputChannel.length; i++) {
            // Convert Float32 to Int16 (PCM)
            const sample = Math.max(-1, Math.min(1, inputChannel[i])); // Clamp
            const int16Sample = Math.round(sample * 32767); // Scale to Int16 range

            this.internalBuffer[this.bufferIndex++] = int16Sample;

            // Check if the buffer is full
            if (this.bufferIndex >= this.bufferSize) {
                this.sendBuffer();
            }
        }

        // Keep processor alive
        return true;
    }

    sendBuffer() {
         if (this.bufferIndex === 0) return; // Nothing to send

        // Create a copy of the filled part of the buffer to send
        // Important: Use slice() to create a copy, don't send the internal buffer directly
        //          as it might be overwritten before being processed by the main thread.
        //          PostMessage transfers ownership of ArrayBuffers efficiently.
        const bufferToSend = this.internalBuffer.slice(0, this.bufferIndex);

        // Post the Int16Array's underlying ArrayBuffer back to the main thread
        // The second argument [bufferToSend.buffer] marks it as transferable
        try {
             this.port.postMessage({ type: 'audioData', buffer: bufferToSend }, [bufferToSend.buffer]);
        } catch (error) {
             // This can happen if the buffer is detached or transfer fails
             this.debugLog(`Error posting message: ${error.message}. Buffer length: ${bufferToSend?.byteLength}`);
             // Try sending without transfer if it fails consistently
             // this.port.postMessage({ type: 'audioData', buffer: bufferToSend });
        }

        // Reset buffer index for the next chunk
        this.bufferIndex = 0;
        this.lastSentTime = currentTime; // currentTime is a global available in AudioWorkletProcessor
    }
}

// Register the processor
try {
    registerProcessor('audio-processor', AudioProcessor);
    console.log("AudioProcessor registered successfully."); // This log appears in the worklet's scope
} catch (error) {
    console.error("Failed to register AudioProcessor:", error);
}