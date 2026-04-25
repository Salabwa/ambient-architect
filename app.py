 import gradio as gr
import numpy as np
from scipy.io.wavfile import write
import tempfile, os

SR = 44100  # Slightly faster than 48k, still perfect for sub-20Hz
DURATION = 390

def generate_ambient(prompt: str, mode: str = "Preview (1:00)"):
    dur = 390 if mode == "Full (6:30)" else 60
    t = np.arange(dur * SR, dtype=np.float32) / SR
    omega = 2 * np.pi * t

    def env(start, end, shape="lin", phase="rise"):
        mask = (t >= start) & (t < end)
        frac = (t[mask] - start) / max(end - start, 1e-6)
        if phase == "rise": val = np.sqrt(frac) if shape!="lin" else frac
        elif phase == "fall": val = 1 - (np.sqrt(frac) if shape!="lin" else frac)
        elif phase == "hold": val = np.ones_like(frac)
        else: val = frac
        out = np.zeros(len(t), dtype=np.float32)
        out[mask] = val
        return out

    e = env(0, 60, "lin", "rise")
    s = env(60, 180, "lin", "hold")
    p = env(180, 300, "lin", "rise")
    d = env(300, 390, "lin", "fall")
    master = np.maximum(np.maximum(e, s), np.maximum(p, d))
    kernel = np.ones(2205, dtype=np.float32) / 2205
    master = np.convolve(master, kernel, mode='same')

    def osc(freq, amp, mod=None, gain=0.15):
        carrier = np.sin(omega * freq)
        if mod is not None: carrier *= mod
        return carrier * amp * gain

    pulse = 0.5 + 0.5 * np.sin(omega * 10)
    l1 = osc(199, master, mod=pulse, gain=0.18)
    l2 = osc(444, master * env(0, dur, "log"), gain=0.14)
    l3 = np.sin(omega * 777 + np.sin(omega * 0.3) * 5) * master * 0.12
    l4 = osc(999, master * env(0, min(300, dur), "exp"), gain=0.11)
    l5 = np.sin(omega * 12) * master * 0.4 + 0.1 * np.sin(omega * 24) * master
    l6 = osc(88, master * 0.7, gain=0.13)

    mix = l1 + l2 + l3 + l4 + l5 + l6
    peak = np.max(np.abs(mix))
    if peak > 0: mix /= peak
    mix *= 0.85

    if dur >= 95:
        vs = 90 * SR
        vd = 5 * SR
        vt = np.arange(vd, dtype=np.float32) / SR
        vw = (0.4*np.sin(2*np.pi*120*vt) + 0.2*np.sin(2*np.pi*240*vt) + 0.1*np.sin(2*np.pi*360*vt))
        ve = np.exp(-((vt - 2.0)**2) / 1.2)
        vw *= ve
        delay = int(0.4 * SR)
        delayed = np.zeros(vd, dtype=np.float32)
        delayed[delay:] = vw[:-delay] * 0.3
        vp = vw + delayed * 0.6
        end = min(vs + len(vp), len(mix))
        mix[vs:end] += vp[:end-vs] * 0.5

    peak = np.max(np.abs(mix))
    if peak > 0: mix /= peak

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        write(tmp.name, SR, np.int16(np.clip(mix, -1, 1) * 32767))
        return tmp.name

def ui():
    with gr.Blocks(title="Ambient Architect", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🌌 Ambient Sound Architect")
        gr.Markdown("Use **Preview** for fast testing. Switch to **Full** for final renders.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=4, 
                    value="Ambient. No melody, no drums. Six tonal layers: 199 Hz carrier + 10 Hz pulse + 444 Hz + 777 Hz shimmer + 999 Hz top tone + sub-20 Hz subsonic. Arc: emerge (0-1) → stillness (1-3) → peak (3-5) → dissolve (5-6:30) → silence. Voice at 1:30: 'Omnithral Vex\\'arion Tava\\'rel. Open. Expand. Become.'")
                mode = gr.Radio(["Preview (1:00)", "Full (6:30)"], value="Preview (1:00)", label="Render Mode")
                btn = gr.Button("✨ Generate", variant="primary")
            out = gr.Audio(label="Result", type="filepath")
        btn.click(generate_ambient, inputs=[prompt, mode], outputs=out)
    return demo

if __name__ == "__main__":
    ui().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
