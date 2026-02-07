package com.phonocardi.recording

/**
 * WAV file encoder.
 * Writes standard RIFF/WAV format with PCM audio data.
 * 44-byte header + raw PCM samples.
 */
class WavFileWriter(
    private val sampleRate: Int = 44100,
    private val bitsPerSample: Int = 16,
    private val channels: Int = 1
) {
    /**
     * Encode float samples to WAV byte array.
     * Samples should be in range [-1.0, 1.0].
     */
    fun encode(samples: FloatArray): ByteArray {
        val bytesPerSample = bitsPerSample / 8
        val dataSize = samples.size * bytesPerSample * channels
        val fileSize = 36 + dataSize

        val buffer = ByteArray(44 + dataSize)
        var offset = 0

        // RIFF header
        offset = writeString(buffer, offset, "RIFF")
        offset = writeInt32LE(buffer, offset, fileSize)
        offset = writeString(buffer, offset, "WAVE")

        // fmt sub-chunk
        offset = writeString(buffer, offset, "fmt ")
        offset = writeInt32LE(buffer, offset, 16) // sub-chunk size
        offset = writeInt16LE(buffer, offset, 1)  // PCM format
        offset = writeInt16LE(buffer, offset, channels)
        offset = writeInt32LE(buffer, offset, sampleRate)
        offset = writeInt32LE(buffer, offset, sampleRate * channels * bytesPerSample) // byte rate
        offset = writeInt16LE(buffer, offset, channels * bytesPerSample) // block align
        offset = writeInt16LE(buffer, offset, bitsPerSample)

        // data sub-chunk
        offset = writeString(buffer, offset, "data")
        offset = writeInt32LE(buffer, offset, dataSize)

        // Convert float samples to int16 PCM
        for (sample in samples) {
            val clamped = sample.coerceIn(-1f, 1f)
            val int16 = (clamped * 32767f).toInt().toShort()
            buffer[offset++] = (int16.toInt() and 0xFF).toByte()
            buffer[offset++] = ((int16.toInt() shr 8) and 0xFF).toByte()
        }

        return buffer
    }

    private fun writeString(buffer: ByteArray, offset: Int, value: String): Int {
        for (i in value.indices) {
            buffer[offset + i] = value[i].code.toByte()
        }
        return offset + value.length
    }

    private fun writeInt32LE(buffer: ByteArray, offset: Int, value: Int): Int {
        buffer[offset] = (value and 0xFF).toByte()
        buffer[offset + 1] = ((value shr 8) and 0xFF).toByte()
        buffer[offset + 2] = ((value shr 16) and 0xFF).toByte()
        buffer[offset + 3] = ((value shr 24) and 0xFF).toByte()
        return offset + 4
    }

    private fun writeInt16LE(buffer: ByteArray, offset: Int, value: Int): Int {
        buffer[offset] = (value and 0xFF).toByte()
        buffer[offset + 1] = ((value shr 8) and 0xFF).toByte()
        return offset + 2
    }
}
