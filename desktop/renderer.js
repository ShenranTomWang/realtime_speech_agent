const $ = id => document.getElementById(id);
const token = location.hash.slice(1);
history.replaceState(null, '', '/');
let socket, context, stream, capture, micSource, silent;
let started = false, starting = false, muted = false, attached = false, ready = false;
let nextPlayback = 0;
const sources = new Set();
const pendingPermissions = new Set();

function send(message) {
    if (socket?.readyState === WebSocket.OPEN && attached) socket.send(JSON.stringify(message));
}
function transcript(role, text) {
    const item = document.createElement('div');
    item.className = `utterance ${role}`;
    item.textContent = `${role}: ${text}`;
    $('transcript').append(item);
    item.scrollIntoView({block: 'end'});
}
function stopSpeech() {
    for (const source of sources) { try { source.stop(); } catch {} }
    sources.clear();
    nextPlayback = 0;
}
async function audioContext() {
    if (!context) {
        context = new AudioContext({sampleRate: 24000});
        if (context.sampleRate !== 24000) throw new Error('This audio device cannot use 24 kHz.');
    }
    await context.resume();
}
function playback(encoded) {
    if (!context || context.state !== 'running') return;
    const binary = atob(encoded);
    const bytes = Uint8Array.from(binary, char => char.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    const buffer = context.createBuffer(1, bytes.length / 2, 24000);
    const output = buffer.getChannelData(0);
    for (let i = 0; i < output.length; i++) output[i] = view.getInt16(i * 2, true) / 32768;
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const when = Math.max(context.currentTime + 0.015, nextPlayback);
    source.start(when);
    nextPlayback = when + buffer.duration;
    sources.add(source);
    source.onended = () => sources.delete(source);
}
function button(label, action) {
    const node = document.createElement('button');
    node.textContent = label;
    node.onclick = action;
    return node;
}
function renderTasks(tasks) {
    $('tasks').replaceChildren();
    $('count').textContent = tasks.filter(item => item.status === 'running').length;
    for (const item of tasks) {
        const card = document.createElement('div');
        card.className = 'task';
        const badge = document.createElement('span');
        badge.className = 'badge';
        badge.textContent = item.status;
        const request = document.createElement('p');
        request.textContent = item.request;
        card.append(badge, request);
        if (item.result) {
            const result = document.createElement('p');
            result.textContent = item.result;
            card.append(result);
        }
        if (item.status === 'running') {
            card.append(button('Cancel task', () => send({type: 'task.cancel', id: item.id})));
            const form = document.createElement('form');
            const input = document.createElement('input');
            input.placeholder = 'Update instructions';
            const submit = document.createElement('button');
            submit.textContent = 'Steer';
            form.append(input, submit);
            form.onsubmit = event => {
                event.preventDefault();
                if (input.value.trim()) send({type: 'task.steer', id: item.id, instruction: input.value.trim()});
                input.value = '';
            };
            card.append(form);
        }
        $('tasks').append(card);
    }
}
function permission(message) {
    if (pendingPermissions.has(message.id)) return;
    pendingPermissions.add(message.id);
    const card = document.createElement('div');
    card.className = 'permission';
    card.id = `permission-${message.id}`;
    const title = document.createElement('strong');
    title.textContent = message.tool;
    const details = document.createElement('pre');
    details.textContent = JSON.stringify(message.input, null, 2);
    const answer = allow => {
        send({type: 'permission.answer', id: message.id, allow});
        card.remove();
        pendingPermissions.delete(message.id);
    };
    card.append(title, details, button('Allow once', () => answer(true)), button('Deny', () => answer(false)));
    $('permissions').append(card);
}
function connect() {
    socket = new WebSocket(`ws://${location.host}/ws`);
    socket.onopen = () => socket.send(JSON.stringify({token}));
    socket.onclose = () => {
        attached = false;
        ready = false;
        stopSpeech();
        $('start').disabled = true;
        $('status').textContent = 'Disconnected · reconnecting…';
        setTimeout(connect, 1500);
    };
    socket.onmessage = event => {
        const message = JSON.parse(event.data);
        switch (message.type) {
            case 'desktop.ready':
                attached = true;
                $('status').textContent = 'Connected · start microphone or type';
                $('start').disabled = false;
                if (started) send({type: 'session.start'});
                break;
            case 'session.ready':
                ready = true;
                $('status').textContent = 'Ready';
                break;
            case 'tasks.updated': renderTasks(message.tasks); break;
            case 'permission.request': permission(message); break;
            case 'permission.resolved':
                document.getElementById(`permission-${message.id}`)?.remove();
                pendingPermissions.delete(message.id);
                break;
            case 'input_audio_buffer.speech_started':
                stopSpeech(); $('status').textContent = 'Listening'; break;
            case 'input_audio_buffer.speech_stopped': $('status').textContent = 'Thinking'; break;
            case 'conversation.item.input_audio_transcription.completed': transcript('user', message.transcript); break;
            case 'response.output_audio_transcript.done': transcript('agent', message.transcript); break;
            case 'response.output_audio.delta': playback(message.delta); break;
            case 'response.cancelled': stopSpeech(); break;
            case 'response.done': $('status').textContent = 'Ready'; break;
            case 'error': transcript('error', message.error.message); break;
        }
    };
}
$('start').onclick = async () => {
    if (starting || started) return;
    starting = true;
    $('start').disabled = true;
    try {
        await audioContext();
        stream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true}});
        await context.audioWorklet.addModule('/capture.js');
        capture = new AudioWorkletNode(context, 'capture');
        micSource = context.createMediaStreamSource(stream);
        silent = context.createGain();
        silent.gain.value = 0;
        micSource.connect(capture).connect(silent).connect(context.destination);
        capture.port.onmessage = event => {
            if (!ready || socket.bufferedAmount > 240000) return;
            const bytes = muted ? new Uint8Array(event.data.byteLength) : new Uint8Array(event.data);
            let binary = '';
            for (const byte of bytes) binary += String.fromCharCode(byte);
            send({type: 'input_audio_buffer.append', audio: btoa(binary)});
        };
        send({type: 'session.start'});
        started = true;
        $('start').textContent = 'Microphone active';
        $('mute').disabled = false;
    } catch (error) {
        stream?.getTracks().forEach(track => track.stop());
        transcript('error', error.message);
        $('start').disabled = false;
    } finally { starting = false; }
};
$('mute').onclick = () => {
    muted = !muted;
    $('mute').textContent = muted ? 'Unmute' : 'Mute';
};
$('interrupt').onclick = () => { stopSpeech(); send({type: 'response.cancel'}); };
$('text').onsubmit = async event => {
    event.preventDefault();
    if (!attached) return;
    try {
        await audioContext();
        const text = $('message').value.trim();
        if (!text) return;
        if (!ready) {
            send({type: 'session.start'});
            transcript('error', 'Session is connecting. Send again once Ready.');
            return;
        }
        stopSpeech();
        send({type: 'response.cancel'});
        send({type: 'conversation.item.create', item: {type: 'message', role: 'user', content: [{type: 'input_text', text}]}});
        send({type: 'response.create'});
        transcript('user', text);
        $('message').value = '';
    } catch (error) { transcript('error', error.message); }
};
connect();
