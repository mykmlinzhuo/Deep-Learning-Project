import time, io, wave
import numpy as np
import gradio as gr

# --- Constants & Data ---
SAMPLE_RATE = 44100
VOLUME = 0.5
MAX_RECORD_TIME = 10.0

NOTE_FREQUENCIES = {
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13,
    'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00,
    'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
    'C5': 523.25
}
PRETTIFY_SCALE = ['C4','D4','E4','F4','G4','A4','B4','C5']
DEMO_SONGS = {
    'Happy Birthday': [('C4',0.5),('C4',0.5),('D4',1.0),('C4',1.0),('F4',1.0),('E4',2.0)],
    'Twinkle Twinkle': [('C4',0.5),('C4',0.5),('G4',1.0),('G4',1.0),('A4',1.0),('A4',1.0),('G4',2.0)],
}

# --- Helpers ---
def make_sine(freq, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(2*np.pi*freq*t) * VOLUME
    return wave.astype(np.float32)

def to_wav_bytes(buffer: np.ndarray):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        int_data = (buffer * (2**15-1)).astype(np.int16)
        wf.writeframes(int_data.tobytes())
    buf.seek(0)
    return buf

def prettify_events(events):
    if not events: return []
    events = sorted(events, key=lambda x: x[0])
    out = []
    for i,(t,note) in enumerate(events):
        next_t = events[i+1][0] if i+1<len(events) else t+0.5
        dur = max(next_t-t,0.25)
        dur_snapped = round(dur/0.5)*0.5
        # snap to nearest in PRETTIFY_SCALE
        base = NOTE_FREQUENCIES[note]
        nearest = min(PRETTIFY_SCALE, key=lambda k: abs(NOTE_FREQUENCIES[k]-base))
        out.append((nearest,dur_snapped))
    return out

def concat_sequence(seq):
    pieces = []
    for note,dur in seq:
        pieces.append(make_sine(NOTE_FREQUENCIES[note], dur))
    return np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)

# --- Gradio Callbacks ---
def toggle_record(state):
    if not state['recording'] and not state['waiting']:
        # start waiting
        state.update(recording=False, waiting=True, recorded=[])
        return "⏳ Waiting for first key..."
    elif state['recording']:
        # stop
        state['recording']=False
        buf = concat_sequence([(t,n) for t,n in state['recorded']])
        state['buffer']=buf
        state['save_ready']=True
        return "⏹️ Recording stopped."
    else:
        return ""

def key_press(note, state):
    t = time.time()
    # first key after waiting: start actual recording
    if state['waiting']:
        state['start_time']=t
        state['waiting']=False
        state['recording']=True
    if state['recording']:
        elapsed = t - state['start_time']
        if elapsed <= MAX_RECORD_TIME:
            state['recorded'].append((elapsed, note))
        else:
            # auto-stop
            state['recording']=False
            buf = concat_sequence([(t,n) for t,n in state['recorded']])
            state['buffer']=buf
            state['save_ready']=True
    # always return the note audio (so it plays immediately)
    return make_sine(NOTE_FREQUENCIES[note], 0.5)

def do_prettify(state):
    seq = prettify_events(state['recorded'])
    buf = concat_sequence(seq)
    return buf

def save_recording(state):
    if not state.get('save_ready'):
        return None
    return to_wav_bytes(state['buffer'])

def play_demo(name):
    seq = DEMO_SONGS[name]
    buf = concat_sequence(seq)
    return buf

# --- Build UI ---
css = """
<style>
.row { display: flex; justify-content: center; }
.white { width:60px; height:200px; background:white; border:1px solid #000; margin:1px; position:relative; }
.black { width:40px; height:120px; background:black; color:white; margin:-120px  -20px 0  -20px; z-index:2; }
.button-row { margin-top:20px; }
</style>
"""

with gr.Blocks(css=css) as demo:
    state = gr.State({
        'recording':False,'waiting':False,
        'start_time':None,'recorded':[],
        'buffer':None,'save_ready':False
    })

    gr.HTML(css + "<div class='row'>")
    # white keys
    for note in ['C4','D4','E4','F4','G4','A4','B4','C5']:
        gr.Button(note, elem_id=note+"_w").click(
            key_press, [gr.State(note), state], outputs=None, preprocess=False
        )
    gr.HTML("</div>")

    gr.HTML("<div class='row'>")
    # placeholders & black keys
    layout = ['C#4','D#4',None,'F#4','G#4','A#4',None]
    for note in layout:
        if note:
            gr.Button("", elem_id=note+"_b").click(
                key_press, [gr.State(note), state], outputs=None, preprocess=False
            )
        else:
            gr.HTML("<div style='width:60px;'></div>")
    gr.HTML("</div>")

    with gr.Row(elem_classes="button-row"):
        rec_btn = gr.Button("Record ⏺")
        rec_msg = gr.Textbox("", interactive=False)
        rec_btn.click(toggle_record, state, rec_msg)

        prettify_btn = gr.Button("Prettify ▶️")
        prettify_audio = gr.Audio()
        prettify_btn.click(do_prettify, state, prettify_audio)

        save_btn = gr.Button("Save 💾")
        save_file = gr.File()
        save_btn.click(save_recording, state, save_file)

    with gr.Row(elem_classes="button-row"):
        demo_hb = gr.Button("Demo: Happy Birthday")
        demo_tt = gr.Button("Demo: Twinkle Twinkle")
        demo_hb.click(play_demo, gr.State("Happy Birthday"), outputs=gr.Audio())
        demo_tt.click(play_demo, gr.State("Twinkle Twinkle"), outputs=gr.Audio())

    gr.Markdown("**Instructions:** Click keys to play. Hit **Record** then press your first key to start (max 10 s). **Prettify** snaps your melody to C-major. **Save** downloads your raw recording.")

demo.launch()
