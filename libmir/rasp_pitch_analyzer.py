import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

try:
    from libmir.rasp_analyzer import SpectrumAnalyzer
    from libmir.pitch import gen_pitch_windows
except:
    from rasp_analyzer import SpectrumAnalyzer
    from pitch import gen_pitch_windows

class PitchAnalyzer(SpectrumAnalyzer):
    def set_plt(self):
        chroma_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        
        n_chroma = 12
        self.n_oct = 8
        self.min_An = 4 - 2 #A4 - n
        self.tempo_ref = 440
        

        freqs = np.linspace(0.0, 1.0, self.n_fft//2 + 1) * (self.sr / 2)
        offset = self.min_An * n_chroma - 3 
        indices = np.arange(0, n_chroma * self.n_oct) 
        log_freqs = self.tempo_ref * 2 ** ( (indices - offset) / n_chroma )
        self.n_pitch = len(log_freqs)
        self.pitch_windows = gen_pitch_windows(freqs, log_freqs)

        self.x = np.linspace(0.0, 1.0, self.n_pitch)
        x_labels = [ c+str(p+1) for p in range(0, self.n_oct) for c in chroma_labels]
        
        self.artist = plt.plot([], [], c="c")[0]
        plt.xlim(0, 1.0)
        plt.ylim(-120, 0)
        plt.xticks(self.x[::n_chroma], x_labels[::n_chroma])
        plt.xlabel('Frequency')
        plt.ylabel('dB')
        plt.title('Pitch Analyzer with pyaudio and matplotlib')
        plt.grid()
    
    def sig_proc(self, frame, spec):
        spec_mag = np.abs(spec)
        log_spec_mag = np.dot(self.pitch_windows, spec_mag)
        
        if frame == 0:
            self.msp = 1e-9
        else:
            self.msp = np.maximum(self.msp, np.max(log_spec_mag))
        
        y = 20 * np.log10(1e-9 + log_spec_mag / self.msp) # to dB
        return y
