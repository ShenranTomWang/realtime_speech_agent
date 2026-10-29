class Capture extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new Int16Array(2400);
        this.offset = 0;
    }
    process(inputs) {
        const samples = inputs[0]?.[0];
        if (!samples) return true;
        for (const value of samples) {
            this.buffer[this.offset++] = Math.round(Math.max(-1, Math.min(1, value)) * 32767);
            if (this.offset === this.buffer.length) {
                this.port.postMessage(this.buffer.buffer, [this.buffer.buffer]);
                this.buffer = new Int16Array(2400);
                this.offset = 0;
            }
        }
        return true;
    }
}
registerProcessor('capture', Capture);
