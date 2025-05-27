import pygame
import numpy as np
import time
import wave

# Constants
SAMPLE_RATE = 44100
VOLUME = 0.5
WHITE_KEY_WIDTH = 60
WHITE_KEY_HEIGHT = 200
BLACK_KEY_WIDTH = 40
BLACK_KEY_HEIGHT = 120
FPS = 60
MAX_RECORD_TIME = 10.0  # seconds

# Note frequencies (4th octave + C5)
NOTE_FREQUENCIES = {
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13,
    'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00,
    'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
    'C5': 523.25
}

# Scale for prettifier (C major)
PRETTIFY_SCALE = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']

# Keyboard bindings for piano keys
WHITE_KEYS = [
    (pygame.K_a, 'C4'), (pygame.K_s, 'D4'), (pygame.K_d, 'E4'), (pygame.K_f, 'F4'),
    (pygame.K_g, 'G4'), (pygame.K_h, 'A4'), (pygame.K_j, 'B4'), (pygame.K_k, 'C5')
]
BLACK_KEYS = [
    (pygame.K_w, 'C#4'), (pygame.K_e, 'D#4'), None,
    (pygame.K_t, 'F#4'), (pygame.K_y, 'G#4'), (pygame.K_u, 'A#4'), None
]

# Demo songs
DEMO_SONGS = {
    'Happy Birthday': [('C4', 0.5), ('C4', 0.5), ('D4', 1.0), ('C4', 1.0), ('F4', 1.0), ('E4', 2.0)],
    'Twinkle Twinkle': [('C4', 0.5), ('C4', 0.5), ('G4', 1.0), ('G4', 1.0), ('A4', 1.0), ('A4', 1.0), ('G4', 2.0)]
}


def generate_sound(freq, duration=1.0):
    """
    Generate a stereo sine-wave Sound object for a given frequency and duration.
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave_data = np.sin(2 * np.pi * freq * t) * VOLUME
    # Convert to 16-bit signed integers
    audio = (wave_data * (2**15 - 1)).astype(np.int16)
    # Expand to stereo if necessary
    if audio.ndim == 1:
        audio = np.repeat(audio[:, None], 2, axis=1)
    return pygame.sndarray.make_sound(audio)


def play_sequence(seq, sounds):
    for note, dur in seq:
        sounds[note].play()
        time.sleep(dur)


def prettify(recorded):
    if not recorded:
        return []
    rec = sorted(recorded, key=lambda x: x[0])
    seq = []
    for i, (t, note) in enumerate(rec):
        next_t = rec[i+1][0] if i+1 < len(rec) else t + 0.5
        dur = max(next_t - t, 0.25)
        dur_snapped = round(dur / 0.5) * 0.5
        freqs = {n: NOTE_FREQUENCIES[n] for n in PRETTIFY_SCALE}
        base = NOTE_FREQUENCIES[note]
        nearest = min(freqs.keys(), key=lambda n: abs(freqs[n] - base))
        seq.append((nearest, dur_snapped))
    return seq


def main():
    pygame.init()
    pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, 512)
    pygame.mixer.init()

    sounds = {note: generate_sound(freq) for note, freq in NOTE_FREQUENCIES.items()}
    width = WHITE_KEY_WIDTH * len(WHITE_KEYS)
    height = WHITE_KEY_HEIGHT + 180
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Python Piano')
    font = pygame.font.SysFont(None, 24)

    clock = pygame.time.Clock()
    waiting_for_first = False
    recording = False
    start_time = None
    recorded = []
    save_enabled = False
    audio_buffer = None

    demo_seq = None
    demo_mode = False
    demo_idx = 0
    demo_next_time = 0
    demo_highlight_note = None
    demo_highlight_end = 0

    # Build key rects for interaction
    white_key_rects = []
    for i, (key, note) in enumerate(WHITE_KEYS):
        rect = pygame.Rect(i * WHITE_KEY_WIDTH, 0, WHITE_KEY_WIDTH, WHITE_KEY_HEIGHT)
        white_key_rects.append((rect, key, note))
    black_key_rects = []
    for i, binding in enumerate(BLACK_KEYS):
        if binding:
            key, note = binding
            x = i * WHITE_KEY_WIDTH + WHITE_KEY_WIDTH - BLACK_KEY_WIDTH // 2
            rect = pygame.Rect(x, 0, BLACK_KEY_WIDTH, BLACK_KEY_HEIGHT)
            black_key_rects.append((rect, key, note))

    # Button layout
    button_y1 = WHITE_KEY_HEIGHT + 10
    button_y2 = WHITE_KEY_HEIGHT + 50
    record_rect = pygame.Rect(10, button_y1, 80, 30)
    prettify_rect = pygame.Rect(100, button_y1, 120, 30)
    save_rect = pygame.Rect(width - 100, button_y1, 90, 30)
    happy_rect = pygame.Rect(10, button_y2, 150, 30)
    twinkle_rect = pygame.Rect(170, button_y2, 150, 30)

    def prepare_buffer():
        nonlocal audio_buffer, save_enabled
        if not recorded:
            audio_buffer = None; save_enabled = False; return
        rec = sorted(recorded, key=lambda x: x[0])
        events = []
        for i, (t, note) in enumerate(rec):
            next_t = rec[i+1][0] if i+1 < len(rec) else t + 0.5
            dur = max(next_t - t, 0.25)
            events.append((t, dur, NOTE_FREQUENCIES[note]))
        total_time = events[-1][0] + events[-1][1]
        total_samples = int(total_time * SAMPLE_RATE)
        audio = np.zeros(total_samples, dtype=np.float32)
        for t, dur, freq in events:
            start = int(t * SAMPLE_RATE)
            length = int(dur * SAMPLE_RATE)
            t_arr = np.linspace(0, dur, length, False)
            audio[start:start+length] += np.sin(2 * np.pi * freq * t_arr) * VOLUME
        audio = np.clip(audio, -1.0, 1.0)
        audio_buffer = (audio * (2**15 - 1)).astype(np.int16)
        save_enabled = True

    def save_recording():
        if audio_buffer is None: return
        with wave.open('output.wav', 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_buffer.tobytes())
        print('Saved recording to output.wav')

    running = True
    while running:
        clock.tick(FPS)
        cur_time = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Record control
                if event.key == pygame.K_r:
                    if not recording and not waiting_for_first:
                        waiting_for_first = True; recorded.clear(); save_enabled = False; audio_buffer = None
                        print('Waiting for first key to start recording...')
                    elif recording:
                        recording = False; waiting_for_first = False
                        print('Recording stopped.')
                        prepare_buffer()
                # Prettify play
                elif event.key == pygame.K_p:
                    seq = prettify(recorded)
                    print('Playing prettified melody...')
                    play_sequence(seq, sounds)
                # Demo start (auto-record)
                elif event.key == pygame.K_1:
                    demo_seq = DEMO_SONGS['Happy Birthday']
                    demo_mode = True; demo_idx = 0; demo_next_time = cur_time; demo_highlight_note = None
                    waiting_for_first = False; recording = True; start_time = cur_time; recorded.clear(); save_enabled = False; audio_buffer = None
                    print('Starting demo: Happy Birthday (highlight only)')
                elif event.key == pygame.K_2:
                    demo_seq = DEMO_SONGS['Twinkle Twinkle']
                    demo_mode = True; demo_idx = 0; demo_next_time = cur_time; demo_highlight_note = None
                    waiting_for_first = False; recording = True; start_time = cur_time; recorded.clear(); save_enabled = False; audio_buffer = None
                    print('Starting demo: Twinkle Twinkle (highlight only)')
                else:
                    print("Detected key press:", pygame.key.name(event.key))
                    # Play or record on key press
                    for rect, key, note in white_key_rects + black_key_rects:
                        if event.key == key:
                            if waiting_for_first:
                                recording = True; start_time = cur_time; waiting_for_first = False
                            if recording and start_time:
                                elapsed = cur_time - start_time
                                if elapsed <= MAX_RECORD_TIME:
                                    recorded.append((elapsed, note))
                            sounds[note].play()
                            break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                # Button clicks mirror key events but no sound for demo buttons
                if record_rect.collidepoint(pos):
                    if not recording and not waiting_for_first:
                        waiting_for_first = True; recorded.clear(); save_enabled = False; audio_buffer = None
                        print('Waiting for first key to start recording...')
                    elif recording:
                        recording = False; waiting_for_first = False
                        print('Recording stopped.')
                        prepare_buffer()
                elif prettify_rect.collidepoint(pos):
                    seq = prettify(recorded)
                    print('Playing prettified melody...')
                    play_sequence(seq, sounds)
                elif happy_rect.collidepoint(pos):
                    demo_seq = DEMO_SONGS['Happy Birthday']
                    demo_mode = True; demo_idx = 0; demo_next_time = cur_time; demo_highlight_note = None
                    waiting_for_first = False; recording = True; start_time = cur_time; recorded.clear(); save_enabled = False; audio_buffer = None
                    print('Starting demo: Happy Birthday (highlight only)')
                elif twinkle_rect.collidepoint(pos):
                    demo_seq = DEMO_SONGS['Twinkle Twinkle']
                    demo_mode = True; demo_idx = 0; demo_next_time = cur_time; demo_highlight_note = None
                    waiting_for_first = False; recording = True; start_time = cur_time; recorded.clear(); save_enabled = False; audio_buffer = None
                    print('Starting demo: Twinkle Twinkle (highlight only)')
                elif save_enabled and save_rect.collidepoint(pos):
                    save_recording()
                else:
                    # Play or record on piano key click
                    for rect, key, note in white_key_rects + black_key_rects:
                        if rect.collidepoint(pos):
                            if waiting_for_first:
                                recording = True; start_time = cur_time; waiting_for_first = False
                            if recording and start_time:
                                elapsed = cur_time - start_time
                                if elapsed <= MAX_RECORD_TIME:
                                    recorded.append((elapsed, note))
                            sounds[note].play()
                            break

        # Auto-stop recording
        if recording and start_time and cur_time - start_time >= MAX_RECORD_TIME:
            recording = False; waiting_for_first = False
            print(f'Auto-stopped recording after {MAX_RECORD_TIME:.1f} seconds.')
            prepare_buffer()

        # Demo progression: highlight only
        if demo_mode:
            if demo_idx < len(demo_seq) and cur_time >= demo_next_time:
                note, dur = demo_seq[demo_idx]
                demo_highlight_note = note
                demo_highlight_end = cur_time + dur
                demo_next_time = cur_time + dur
                demo_idx += 1
            elif demo_idx >= len(demo_seq) and cur_time >= demo_next_time:
                demo_mode = False; demo_highlight_note = None
                # if recording:
                #     recording = False
                #     print('Recording stopped after demo.')
                #     prepare_buffer()

        # Draw UI
        screen.fill((200, 200, 200))
        # Keys
        for rect, key, note in white_key_rects:
            color = (255, 255, 0) if note == demo_highlight_note else (255, 255, 255)
            pygame.draw.rect(screen, color, rect); pygame.draw.rect(screen, (0, 0, 0), rect, 1)
        for rect, key, note in black_key_rects:
            color = (255, 255, 0) if note == demo_highlight_note else (0, 0, 0)
            pygame.draw.rect(screen, color, rect)
        # Labels
        for rect, key, note in white_key_rects:
            label = pygame.key.name(key).upper()
            text = font.render(label, True, (0, 0, 0))
            screen.blit(text, text.get_rect(center=(rect.x+WHITE_KEY_WIDTH//2, rect.y+WHITE_KEY_HEIGHT-10)))
        for rect, key, note in black_key_rects:
            label = pygame.key.name(key).upper()
            text = font.render(label, True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=(rect.x+BLACK_KEY_WIDTH//2, rect.y+BLACK_KEY_HEIGHT-10)))

        # draw buttons
        rec_color = (200, 0, 0) if recording else (100, 100, 100)
        pygame.draw.rect(screen, rec_color, record_rect)
        screen.blit(font.render('Record', True, (255, 255, 255)), (record_rect.x+10, record_rect.y+5))
        pygame.draw.rect(screen, (0, 200, 0), prettify_rect)
        screen.blit(font.render('Prettify', True, (255, 255, 255)), (prettify_rect.x+10, prettify_rect.y+5))
        pygame.draw.rect(screen, (0, 200, 0) if save_enabled else (100, 100, 100), save_rect)
        screen.blit(font.render('Save', True, (255, 255, 255)), (save_rect.x+20, save_rect.y+5))
        pygame.draw.rect(screen, (0, 0, 200), happy_rect)
        screen.blit(font.render('Happy Birthday', True, (255, 255, 255)), (happy_rect.x+5, happy_rect.y+5))
        pygame.draw.rect(screen, (0, 0, 200), twinkle_rect)
        screen.blit(font.render('Twinkle Twinkle', True, (255, 255, 255)), (twinkle_rect.x+5, twinkle_rect.y+5))

        # status
        sy = WHITE_KEY_HEIGHT + 100
        if waiting_for_first:
            screen.blit(font.render('Waiting for first key to start recording...', True, (0, 0, 0)), (10, sy)); sy += 20
        if recording:
            elapsed = min(cur_time - start_time, MAX_RECORD_TIME)
            screen.blit(font.render(f'Recording... {elapsed:.1f}s', True, (0, 0, 0)), (10, sy)); sy += 20
        screen.blit(font.render('Play: A-S-D-F-G-H-J-K (white), W-E-T-Y-U (black)', True, (0, 0, 0)), (10, WHITE_KEY_HEIGHT+120))
        screen.blit(font.render('Click keys or use keyboard.', True, (0, 0, 0)), (10, WHITE_KEY_HEIGHT+140))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
