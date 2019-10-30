import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

class SpectrumAnalyzer():
    def __init__(self, sr, n_chunk, n_ch):
        self.sr = sr
        self.n_chunk = n_chunk
        self.n_fft = n_chunk * 2
        self.n_freq = self.n_fft // 2 + 1
        self.n_ch = n_ch
        
        self.__set_mlp()
        self.set_plt()
        
    def __set_mlp(self):
        self.fig = plt.figure(figsize = (6, 4))
        ax = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor('gray')
        ax.patch.set_facecolor('black')
        win = plt.gcf().canvas.manager.window
        win.setWindowOpacity(0.85)
        
    def set_plt(self):
        self.x = np.linspace(0.0, 1.0, self.n_freq)
        x_labels = np.linspace(0.0, 1.0, self.n_freq) * (self.sr//2)
        
        self.artist = plt.plot([], [], c="c")[0]
        plt.xlim(0, 1.0)
        plt.ylim(-120, 0)
        plt.xticks(self.x[::self.n_freq//6], x_labels[::self.n_freq//6].astype(np.int))
        plt.xlabel('Frequency')
        plt.ylabel('dB')
        plt.title('Spectrum Analyzer with pyaudio and matplotlib')
        plt.grid()
      
    def set_data(self, y):
        self.artist.set_data(self.x, y)
    
    def sig_proc(self, frame, spec):
        spec_mag = np.abs(spec)
        
        if frame == 0:
            self.msp = 1e-9
        else:
            self.msp = np.maximum(self.msp, np.max(spec_mag))
        
        y = 20 * np.log10(1e-9 + spec_mag / self.msp) # to dB
        return y
        
    def run(self, stream):
        sig_block = np.zeros(self.n_fft)
        n_block = self.n_fft
        n_block_1_4 = n_block // 4
        n_block_2_4 = n_block // 2
        n_block_3_4 = 3 * n_block // 4
        fft_window = np.hanning(n_block)
        
        def update(frame):
            try:
                data = np.fromstring(stream.read(self.n_chunk), dtype=np.float32)
                # data: []L, R, L, R, ..., L, R] => data[n_fft, 2] (data[n_fft, 0] is Left channel)
                sig_tmp = np.reshape(data, (self.n_chunk, 2)).T
                # M/S processing
                if self.n_ch == 2:
                    sig_current = (sig_tmp[0] + sig_tmp[1]) # Mid             
                
                sig_block[n_block_3_4:n_block] = sig_current[0:n_block_1_4]
                
                # FFT
                spec = np.fft.rfft(fft_window * sig_block)
                # Signal proc
                y = self.sig_proc(frame, spec)
                
                sig_block[0:n_block_1_4] = sig_block[n_block_2_4:n_block_3_4]
                sig_block[n_block_1_4:n_block_3_4] = sig_current
                
            except IOError:
                pass
            self.set_data(y)
            return self.artist, # if blit is True, must return the list contains artist (Graph) objects
            
        anifunc = matplotlib.animation.FuncAnimation(self.fig, update, interval=0, blit=True)
        plt.show()
