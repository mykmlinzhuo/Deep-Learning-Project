import os, tempfile, subprocess
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import gradio as gr
from multitrack.musicgen import add_instrument_accompaniment
from multitrack.mix import mix_music


# 1) Create a FastAPI instance
app = FastAPI()

# 2) Mount your UI folder so that any request to /UI/* will serve files from ./UI/*
#    Make sure this path is correct relative to where you run `python app.py`
ui_dir = os.path.join(os.path.dirname(__file__), "UI")
app.mount("/UI", StaticFiles(directory=ui_dir), name="ui")


def generate_piano(webm_file, length):
    """
    Generate a piano melody from the uploaded WebM file.
    This function is a placeholder and should be replaced with actual piano generation logic.
    """
    # Convert .webm → .wav
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav"); os.close(wav_fd)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", webm_file.name,
        "-ac", "1", "-ar", "44100", wav_path
    ], check=True)

    # TODO:
    # Here you would normally call your piano generation model
    # For now, we just return the converted WAV path
    return wav_path


def generate_accompaniment(wav_path, instrument):
    # Run your existing accompaniment generator
    out_fd, out_path = tempfile.mkstemp(suffix="_acc.wav"); os.close(out_fd)
    add_instrument_accompaniment(wav_path, instrument, out_path)
    return out_path

# Mix original and accompaniment each with volume adjustments
def mix_music_wrapper(orig_path, accompany_wav_path, strength, accompany_strength):
    # 2. Mix with given strengths
    mix_fd, mix_path = tempfile.mkstemp(suffix="_mix.wav"); os.close(mix_fd)
    mix_music(
        original_path=orig_path,
        accompany_path=accompany_wav_path,
        output_path=mix_path,
        strength=strength,
        accompany_strength=accompany_strength
    )

    return mix_path


# 3) Build the Gradio UI
demo = gr.Blocks()

with demo:
    gr.Markdown("## 🎹 Piano + Accompaniment Demo")

    # Beautiful “Play Piano” button linking to your piano.html
    gr.HTML(
        """
        <a href="/UI/index.html" target="_blank" style="text-decoration:none">
          <button
            style="
              background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
              color: #333;
              border: none;
              border-radius: 12px;
              padding: 14px 28px;
              font-size: 18px;
              font-weight: bold;
              cursor: pointer;
              box-shadow: 0 6px 12px rgba(0,0,0,0.1);
              transition: transform 0.1s, box-shadow 0.1s;
            "
            onmouseover="this.style.transform='scale(1.03)'; this.style.boxShadow='0 8px 16px rgba(0,0,0,0.15)';"
            onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 6px 12px rgba(0,0,0,0.1)';"
          >
            🎼 Play Piano
          </button>
        </a>
        """
    )

    # Upload WebM and choose instrument
    gr.Markdown("### Generate Piano Melody from WebM")
    webm_input = gr.File(label="Upload `recording.webm`")
    length = gr.Slider(
        minimum=30, maximum=60, step=1, value=30, label="Melody Length (seconds)",
        info="Length of the generated piano melody in seconds."
    )

    # Generate Piano Melody
    piano_audio = gr.Audio(label="Generated Piano WAV", type="filepath")
    gen_piano_btn = gr.Button("Generate Piano Melody")
    gen_piano_btn.click(fn=generate_piano, inputs=[webm_input, length], outputs=piano_audio)

    # Generate Accompaniment
    gr.Markdown("### Generate Accompaniment for Piano")
    instrument = gr.Dropdown(choices=["violin", "guitar", "flute", "cello"], value="violin", label="Select Instrument")
    acc_audio = gr.Audio(label="Accompaniment WAV", type="filepath")
    gen_btn = gr.Button("Generate Accompaniment")
    gen_btn.click(fn=generate_accompaniment, inputs=[piano_audio, instrument], outputs=acc_audio)

    # Volume sliders for mixing
    gr.Markdown("### Mix Original and Accompaniment")
    strength_slider = gr.Slider(minimum=-20, maximum=20, step=1, value=3, label="Original Volume (dB)")
    accompany_slider = gr.Slider(minimum=-20, maximum=20, step=1, value=-6, label="Accompaniment Volume (dB)")

    # Mix Music button and output
    mix_audio = gr.Audio(label="Mixed Output WAV", type="filepath")
    mix_btn = gr.Button("Mix Music")
    mix_btn.click(
        fn=mix_music_wrapper,
        inputs=[piano_audio, acc_audio, strength_slider, accompany_slider],
        outputs=mix_audio
    )

# 4) Mount Gradio into the same FastAPI under the root path
app = gr.mount_gradio_app(app, demo, path="/")

# 5) Run with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
