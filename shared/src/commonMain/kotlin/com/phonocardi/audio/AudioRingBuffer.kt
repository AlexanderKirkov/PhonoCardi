package com.phonocardi.audio

/**
 * Lock-free ring buffer for audio samples.
 * Single-producer, single-consumer pattern for audio thread safety.
 */
class AudioRingBuffer(val capacity: Int) {
    private val buffer = FloatArray(capacity)
    private var writePos: Long = 0
    private var readPos: Long = 0

    val available: Int
        get() = (writePos - readPos).toInt()

    val totalWritten: Long
        get() = writePos

    fun write(samples: FloatArray, offset: Int = 0, count: Int = samples.size) {
        for (i in 0 until count) {
            buffer[((writePos + i) % capacity).toInt()] = samples[offset + i]
            
        }
        writePos += count
    }

    fun read(output: FloatArray, count: Int = output.size): Int {
        val toRead = minOf(count, available)
        for (i in 0 until toRead) {
            output[i] = buffer[((readPos + i) % capacity).toInt()]
        }
        readPos += toRead
        return toRead
    }

    fun peek(output: FloatArray, count: Int = output.size, offsetFromEnd: Int = 0): Int {
        val start = maxOf(0L, writePos - count - offsetFromEnd)
        val toRead = minOf(count, (writePos - start).toInt())
        for (i in 0 until toRead) {
            output[i] = buffer[((start + i) % capacity).toInt()]
        }
        return toRead
    }

    operator fun get(absoluteIndex: Long): Float {
        return buffer[(absoluteIndex % capacity).toInt()]
    }

    fun clear() {
        writePos = 0
        readPos = 0
    }
}
