import gradio as gr
import numpy as np
from scipy.io.wavfile import write
import tempfile
import os

SR = 48000
DURATION = 390  # 6:30

def generate_ambient(prompt: str):
    t = np.arange(DURATION * SR) / SR
    
    def envelope(start, end, shape="linear", phase="rise"):
        mask = (t >= start) & (t < end)
        frac = (t[mask] - start) / (end - start)
        if phase == "rise":
            val = frac if shape=="linear" else np.sqrt(frac)
        elif phase == "fall":
            val = 1 - frac if shape=="linear" else 1 - np.sqrt(frac)
        elif phase == "hold":
            val = np.ones_like(frac)
        else:
            val = frac
        env = np.zeros_like(t)
        env[mask] = val
        return env

    emerge = envelope(0, 60, "linear", "rise")
    still = envelope(60, 180, "hold")
    peak = envelope(180, 300, "linear", "rise")
    dissolve = envelope(300, 390, "linear", "fall")
    
    master_amp = np.maximum(emerge, np.maximum(still, np.maximum(peak, dissolve)))
    kernel = np.ones(4800) / 4800
    master_amp = np.convolve(master_amp, kernel, mode='same')

    def sine_layer(freq, amp_env, phase=0, mod=None, gain=0.15):
        carrier = np.sin(2 * np.pi * freq * t + phase)
        if mod is not None:
            carrier *= mod
        return carrier * amp_env * gain

    # Layer 1: 199 Hz + 10 Hz pulse
    pulse_10hz = 0.5 + 0.5 * np.sin(2 * np.pi * 10 * t)
    layer1 = sine_layer(199, master_amp, mod=pulse_10hz, gain=0.18)
    # Layer 2: 444 Hz
    layer2 = sine_layer(444, master_amp * envelope(0, 390, shape="log"), gain=0.14)
    # Layer 3: 777 Hz shimmer
    shimmer_mod = np.sin(2 * np.pi * 0.3 * t) * 5
    layer3 = np.sin(2 * np.pi * 777 * t + shimmer_mod) * master_amp * 0.12
    # Layer 4: 999 Hz
    layer4 = sine_layer(999, master_amp * envelope(0, 300, shape="exp"), gain=0.11)
    # Layer 5: Sub-20 Hz
    layer5 = np.sin(2 * np.pi * 12 * t) * master_amp * 0.4
    layer5 += 0.1 * np.sin(2 * np.pi * 24 * t) * master_amp
    # Layer 6: Low pad
    layer6 = sine_layer(88, master_amp * 0.7, gain=0.13)

    mix = layer1 + layer2 + layer3 + layer4 + layer5 + layer6
    peak_val = np.max(np.abs(mix))
    if peak_val > 0:
        mix = mix / peak_val * 0.85

    # Voice at 1:30 with reverb simulation
    voice_start = 90 * SR
    voice_dur = 5.0
    v_t = np.arange(int(voice_dur * SR)) / SR
    voice_wave = (
        0.4 * np.sin(2 * np.pi * 120 * v_t) +
        0.2 * np.sin(2 * np.pi * 240 * v_t) +
        0.1 * np.sin(2 * np.pi * 360 * v_t)
    )
    voice_env = np.exp(-((v_t - 2.0)**2) / 1.2)
    voice_wave = voice_wave * voice_env
    # Simple delay for space effect
    delayed = np.zeros_like(voice_wave)
    delay_samples = int(0.4 * SR)
    for i in range(delay_samples, len(voice_wave)):
        delayed[i] = voice_wave[i - delay_samples] * 0.3
    voice_processed = voice_wave + delayed * 0.6
    
    end_idx = min(voice_start + len(voice_processed), len(mix))
    mix[voice_start:end_idx] += voice_processed[:end_idx-voice_start] * 0.5

    peak_val = np.max(np.abs(mix))
    if peak_val > 0:
        mix = mix / peak_val * 0.9

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        write(tmp.name, SR, np.int16(mix * 32767))
        return tmp.name

def ui():
    with gr.Blocks(title="Ambient Architect", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🌌 Ambient Sound Architect")
        gr.Markdown("Generate precise ambient audio with exact frequency layers.")
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    value="Ambient. No melody, no drums. Six tonal layers: 199 Hz carrier + 10 Hz pulse (felt not heard) + 444 Hz + 777 Hz shimmer + 999 Hz top tone + sub-20 Hz subsonic. Arc: emerge lowest-first (0-1 min) → pressurized stillness (1-3 min) → expansion peak (3-5 min) → dissolve top-down (5-6:30) → silence. Voice at 1:30 deep reverb: 'Omnithral Vex\\'arion Tava\\'rel. Open. Expand. Become.' Mood: Vast, outward sphere, deep space resonance.",
                    lines=6
                )
                generate_btn = gr.Button("✨ Generate Audio", variant="primary")
            with gr.Column(scale=1):
                output_audio = gr.Audio(label="Result", type="filepath")
                download_btn = gr.File(label="Download WAV")
        
        generate_btn.click(
            fn=generate_ambient,
            inputs=prompt,
            outputs=output_audio
        ).then(
            fn=lambda x: x,
            inputs=output_audio,
            outputs=download_btn
        )
    return demo

if __name__ == "__main__":
    demo = ui()
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
