import {
  TranscriptionProvider,
  TranscribeParams,
  TranscribeContext,
} from "../../core/pipeline-types";
import { logger } from "../../../main/logger";
import { SettingsService } from "../../../services/settings-service";
import {
  AppError,
  ErrorCodes,
  type ErrorCode,
} from "../../../types/error";

// Buffer is available globally in Node.js/Electron environment
declare const Buffer: any;

interface DeepgramResponse {
  metadata: {
    transaction_key: string;
    request_id: string;
    sha256: string;
    created: string;
    duration: number;
    channels: number;
    models: string[];
    model_info: Record<string, any>;
  };
  results: {
    channels: {
      alternatives: {
        transcript: string;
        confidence: number;
        words: any[];
      }[];
    }[];
  };
}

export class DeepgramProvider implements TranscriptionProvider {
  readonly name = "deepgram";

  private settingsService: SettingsService;
  private apiEndpoint = "https://api.deepgram.com/v1/listen";

  // Frame aggregation state
  private frameBuffer: Float32Array[] = [];
  private currentSilenceFrameCount = 0;
  
  // Configuration
  private readonly FRAME_SIZE = 512; // 32ms at 16kHz
  private readonly MIN_AUDIO_DURATION_MS = 500; // Minimum buffered audio duration before silence-based transcription
  private readonly MAX_SILENCE_DURATION_MS = 1000; // Max silence before cutting (Deepgram is fast, we can cut sooner)
  private readonly SAMPLE_RATE = 16000;
  private readonly SPEECH_PROBABILITY_THRESHOLD = 0.2;

  constructor(settingsService: SettingsService) {
    this.settingsService = settingsService;
    logger.transcription.info("DeepgramProvider initialized");
  }

  /**
   * Process an audio chunk - buffers and conditionally transcribes
   */
  async transcribe(params: TranscribeParams): Promise<string> {
    try {
      const { audioData, speechProbability = 1 } = params;

      // Add frame to buffer
      this.frameBuffer.push(audioData);

      // Consider it speech if probability is above threshold
      const isSpeech = speechProbability > this.SPEECH_PROBABILITY_THRESHOLD;

      // Track speech and silence
      if (isSpeech) {
        this.currentSilenceFrameCount = 0;
      } else {
        this.currentSilenceFrameCount++;
      }

      // Only transcribe if speech/silence patterns indicate we should
      if (!this.shouldTranscribe()) {
        return "";
      }

      return this.doTranscription();
    } catch (error) {
      logger.transcription.error("Deepgram transcription error:", error);
      throw error;
    }
  }

  /**
   * Flush any buffered audio and return transcription
   */
  async flush(context: TranscribeContext): Promise<string> {
    try {
      // flush() is called at session end, so this is the final call
      return this.doTranscription();
    } catch (error) {
      logger.transcription.error("Deepgram transcription error:", error);
      throw error;
    }
  }

  /**
   * Shared transcription logic - aggregates buffer, calls Deepgram API, clears state
   */
  private async doTranscription(): Promise<string> {
    // Combine all frames into a single Float32Array
    const totalLength = this.frameBuffer.reduce(
      (acc, frame) => acc + frame.length,
      0,
    );
    
    if (totalLength === 0) return "";

    const combinedAudio = new Float32Array(totalLength);
    let offset = 0;
    for (const frame of this.frameBuffer) {
      combinedAudio.set(frame, offset);
      offset += frame.length;
    }

    // Clear frame buffers
    this.frameBuffer = [];
    this.currentSilenceFrameCount = 0;

    // Make the API request
    return this.makeTranscriptionRequest(combinedAudio);
  }

  /**
   * Clear internal buffers without transcribing
   */
  reset(): void {
    this.frameBuffer = [];
    this.currentSilenceFrameCount = 0;
  }

  private shouldTranscribe(): boolean {
    const silenceDuration =
      ((this.currentSilenceFrameCount * this.FRAME_SIZE) / this.SAMPLE_RATE) *
      1000;
    const audioDuration =
      ((this.frameBuffer.length * this.FRAME_SIZE) / this.SAMPLE_RATE) * 1000;

    return (
      audioDuration >= this.MIN_AUDIO_DURATION_MS &&
      silenceDuration >= this.MAX_SILENCE_DURATION_MS
    );
  }

  private convertFloat32ToInt16(float32Array: Float32Array): Int16Array {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16Array;
  }

  private createWavHeader(dataLength: number): any {
    const buffer = Buffer.alloc(44);
    
    // RIFF identifier
    buffer.write("RIFF", 0);
    // file length
    buffer.writeUInt32LE(36 + dataLength, 4);
    // RIFF type
    buffer.write("WAVE", 8);
    // format chunk identifier
    buffer.write("fmt ", 12);
    // format chunk length
    buffer.writeUInt32LE(16, 16);
    // sample format (1 is PCM)
    buffer.writeUInt16LE(1, 20);
    // channel count
    buffer.writeUInt16LE(1, 22);
    // sample rate
    buffer.writeUInt32LE(this.SAMPLE_RATE, 24);
    // byte rate (sample rate * block align)
    buffer.writeUInt32LE(this.SAMPLE_RATE * 2, 28);
    // block align (channel count * bytes per sample)
    buffer.writeUInt16LE(2, 32);
    // bits per sample
    buffer.writeUInt16LE(16, 34);
    // data chunk identifier
    buffer.write("data", 36);
    // data chunk length
    buffer.writeUInt32LE(dataLength, 40);

    return buffer;
  }

  private async makeTranscriptionRequest(audioData: Float32Array): Promise<string> {
    // Get Deepgram config
    const config = await this.settingsService.getDeepgramConfig();
    if (!config?.apiKey) {
      throw new AppError(
        "Deepgram API key not configured",
        ErrorCodes.AUTH_REQUIRED,
      );
    }

    // Convert to WAV format
    // Deepgram supports raw audio but WAV is safer/easier with standard headers
    const int16Data = this.convertFloat32ToInt16(audioData);
    const wavHeader = this.createWavHeader(int16Data.length * 2);
    const wavBuffer = Buffer.concat([
      wavHeader,
      Buffer.from(int16Data.buffer)
    ]);

    logger.transcription.info("Sending audio to Deepgram API", {
      audioLength: audioData.length,
      byteLength: wavBuffer.length,
    });

    try {
      const url = new URL(this.apiEndpoint);
      url.searchParams.append("model", "nova-3");
      url.searchParams.append("smart_format", "true");
      // Add other params if needed, e.g. language

      const response = await fetch(url.toString(), {
        method: "POST",
        headers: {
          "Authorization": `Token ${config.apiKey}`,
          "Content-Type": "audio/wav",
        },
        body: wavBuffer,
      });

      if (!response.ok) {
        let errorMessage = `Deepgram API error: ${response.status} ${response.statusText}`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.err_msg || errorMessage;
        } catch {
          // Ignore json parse error
        }

        logger.transcription.error("Deepgram API error", {
          status: response.status,
          statusText: response.statusText,
        });

        if (response.status === 401) {
          throw new AppError(
            "Deepgram authentication failed. Please check your API key.",
            ErrorCodes.AUTH_REQUIRED,
          );
        }

        throw new AppError(
          errorMessage,
          ErrorCodes.INTERNAL_SERVER_ERROR,
        );
      }

      const result: DeepgramResponse = await response.json();
      
      const transcript = result.results?.channels[0]?.alternatives[0]?.transcript || "";
      
      logger.transcription.info("Deepgram transcription successful", {
        transcriptLength: transcript.length,
        transcript: transcript.substring(0, 50) + "...",
      });

      return transcript;

    } catch (error) {
      if (error instanceof AppError) throw error;
      
      throw new AppError(
        error instanceof Error ? error.message : "Deepgram transcription failed",
        ErrorCodes.NETWORK_ERROR,
      );
    }
  }
}
