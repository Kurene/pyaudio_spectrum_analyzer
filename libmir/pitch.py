import numpy as np
import librosa
import matplotlib.pyplot as plt

try:
    from libmir.utils import plot_hist, plot_specgram, compute_hist
    import libmir.structure as structure
except:
    from utils import plot_hist, plot_specgram
    import structure

def gen_pitch_windows(freqs, log_freqs, weight_on=True):
    length, log_length = len(freqs), len(log_freqs)
    windows = np.zeros((log_length, length))

    for k in range(0, log_length):
        win_sample = 0
        # Descending linear line
        if k < log_length - 1:
            cond = np.logical_and(log_freqs[k] < freqs, log_freqs[k+1] >= freqs)
        else:
            cond = np.logical_and(log_freqs[k] < freqs, log_freqs[k] + (log_freqs[k]-log_freqs[k-1]) >= freqs)
        
        indices = np.where(cond)[0]
        if len(indices) == 0:
            pass
        elif len(indices) == 1:
            windows[k,indices[0]] = 1.0
            win_sample += 1
        else:
            descend = np.linspace(1.0, 0.0, len(indices))
            windows[k, indices[0]-1:indices[-1]] = descend
            win_sample += len(descend)
        # Ascending linear line
        if k > 1:
            cond = np.logical_and(log_freqs[k-1] <= freqs, log_freqs[k] >= freqs)
        else:
            cond = np.array([])
        indices = np.where(cond)[0]
        if len(indices) == 0:
            pass
        elif len(indices) == 1:
            windows[k,indices[0]] = 1.0
            win_sample += 1
        else:
            ascend = np.linspace(0.0, 1.0, len(indices))
            windows[k, indices[0]:indices[-1]+1] = ascend
            win_sample += len(ascend)

        if weight_on and win_sample > 1:
            windows[k] /= np.sqrt(win_sample)

    return windows

def compute_log_spectrogram(spec, freqs, tempo_ref=440, oct_bin=12, oct_num=7):
    # log_freqs[0] is corresponds to C0 (65 Hz)
    offset = 4 * oct_bin - 3 
    indices = np.arange(0, oct_bin*oct_num) 
    log_freqs = tempo_ref * 2 ** ( (indices - offset) / oct_bin )

    windows = gen_pitch_windows(freqs, log_freqs)
    log_spec = np.dot(windows, spec)
    return log_spec, log_freqs

def compute_logspec_from_stft(y, sr, hop_length, n_bins, n_fft=2048):
    spec = np.abs(librosa.stft(y=y, hop_length=hop_length, n_fft=n_fft))
    freqs = np.linspace(0.0, 1.0, (n_fft//2+1)) * (sr / 2)
    log_spec, log_freqs = compute_log_spectrogram(spec, freqs)
    return log_spec, log_freqs

def compute_chroma_from_stft(y, sr, hop_length, n_bins, n_fft=2048):
    log_spec, log_freqs = compute_logspec_from_stft(y, sr, hop_length, n_bins, n_fft=n_fft)

    n_oct = 12
    n_pitch, n_t = log_spec.shape
    chroma = np.zeros((n_oct, n_t))
    for m in range(0, n_oct):
        chroma[m,:] = np.mean(log_spec[m:n_pitch:n_oct,:], axis=0)

    return chroma

def chroma(y, sr, hop_length, filepath=None, 
           n_bins=84, 
           feature_name_pitch="pitchgram", 
           feature_name_chroma="chromagram",
           context="",
           ):
    feature_name_pitch  = feature_name_pitch + context
    feature_name_chroma = feature_name_chroma + context
    chroma_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pitch_labels = [ c+str(p) for p in range(n_bins//12) for c in chroma_labels]

    pitch = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length, n_bins=n_bins))
    chroma= librosa.feature.chroma_cqt(C=pitch)
    plot_specgram(chroma, chroma_labels, feature_name_chroma, filepath=filepath)
    plot_specgram(pitch, pitch_labels, feature_name_pitch, filepath=filepath, ticks=n_bins//12)

    chroma_hist = compute_hist(chroma, axis=1)
    pitch_hist = compute_hist(pitch, axis=1)
    plot_hist(chroma_hist, chroma_labels, feature_name_chroma, filepath=filepath)
    plot_hist(pitch_hist, pitch_labels, feature_name_pitch, ticks=n_bins//12, filepath=filepath)

    structure.ssm(chroma, chroma_labels, feature_name_chroma, n_ma=50, filepath=filepath)
    structure.ssm(pitch, pitch_labels, feature_name_pitch, n_ma=50, filepath=filepath)

    return chroma, chroma_labels, chroma_hist



if __name__=='__main__':
    # Load music singal
    filepath = librosa.util.example_audio_file()
    filepath = "audio_example_pitch.mp3"
    y, sr = librosa.load(filepath, sr=16000, offset=95, duration=10)
    
    filepath=None
    n_bins = 85
    hop_length = 1024
    chroma_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pitch_labels = [ c+str(p+1) for p in range(n_bins//12) for c in chroma_labels]

    pitch, freqs = compute_logspec_from_stft(y, sr, hop_length, n_bins, n_fft=2048)
    chroma = compute_chroma_from_stft(y, sr, hop_length, n_bins, n_fft=2048)
    freqs = np.round(freqs)
    plot_specgram(pitch, pitch_labels, "pitchgram", filepath=filepath2)
    plot_specgram(chroma, chroma_labels, "chromagram", filepath=filepath)

    pitch = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length, n_bins=n_bins))
    chroma= librosa.feature.chroma_cqt(C=pitch)
    plot_specgram(pitch, pitch_labels, "pitchgram", filepath=filepath2)
    plot_specgram(chroma, chroma_labels, "chromagram", filepath=filepath)

    chroma_hist = np.mean(chroma, axis=1)
    pitch_hist = np.mean(pitch, axis=1)
    plot_hist(pitch_hist, pitch_labels, "pitchgram", filepath=filepath)
    plot_hist(chroma_hist, chroma_labels, "chromagram", filepath=filepath)

    plt.show()

    import code
    console = code.InteractiveConsole(locals=locals())
    console.interact()
