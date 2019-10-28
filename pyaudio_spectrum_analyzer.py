import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

# Parameters:
RATE = 48000
CHUNK = 1024
FFTBIN = CHUNK // 2 + 1

p = pyaudio.PyAudio()

input_device_index = None
print(f"device_index\tmaxInputChannels\tmaxOutputChannels\tdevice_name")
for k in range(p.get_device_count()):
    dev = p.get_device_info_by_index(k)
    device_name = dev["name"]
    device_index = dev["index"]
    maxInputChannels = int(dev["maxInputChannels"])
    maxOutputChannels = int(dev["maxOutputChannels"])
    if type(device_name) is bytes:
        device_name = device_name.decode("cp932") 
    print(f"{device_index}\t{maxInputChannels}\t{maxOutputChannels}\t{device_name}")

    if "VoiceMeeter Output" in device_name and maxInputChannels==2:
        input_device_index = dev["index"]
        input_device_name = device_name

print("") 
print(f"Input device:  {input_device_name} is OK.")
print("")
        
# Set audio input stream with pyaudio
stream = p.open(
    format=pyaudio.paFloat32,
    channels=2,
    rate=RATE,
    input=True,
    output=False,
    frames_per_buffer=CHUNK,
    input_device_index=input_device_index,
)

x = np.linspace(0.0, 1.0, FFTBIN)
x_labels = np.linspace(0.0, 1.0, FFTBIN) * (RATE//2)
spec_zeros = np.zeros((2, FFTBIN))
sig_ms = np.zeros((2, CHUNK))

# Matplotlib setting
fig = plt.figure(figsize = (6, 4))
ax = fig.add_subplot(111)
fig.patch.set_facecolor('gray')
ax.patch.set_facecolor('black')
win = plt.gcf().canvas.manager.window
win.setWindowOpacity(0.85)

artist = plt.plot([], [], c="c")[0]
plt.ylim(-120,0)
plt.xlim(0, 1.0)
plt.xticks(x[::FFTBIN//6], x_labels[::FFTBIN//6].astype(np.int))
plt.xlabel('Frequency')
plt.ylabel('dB')
plt.title('Spectrum Analyzer with pyaudio and matplotlib')
plt.grid()

def update(frame):
    if frame == 0:
        update.msp = 1e-9
    try:
        data = np.fromstring(stream.read(CHUNK), dtype=np.float32)
        # data: []L, R, L, R, ..., L, R] => data[n_fft, 2] (data[n_fft, 0] is Left channel)
        sig = np.reshape(data, (CHUNK, 2)).T
        # M/S processing
        sig_ms[0] = (sig[0] + sig[1]) # Mid
        sig_ms[1] = (sig[0] - sig[1]) # Side
        # FFT
        spec = np.abs(np.fft.rfft(sig_ms))
        # to dB
        update.msp = np.maximum(update.msp, np.max(spec))
        spec = 20 * np.log10(1e-9 + spec / update.msp) 
        #spec = np.abs(spec) # to dB
    except IOError:
        spec = spec_zeros
    y = spec[0] # get mid signal
    artist.set_data(x, y)
    return artist, # if blit is True, must return the list contains artist (Graph) objects

anifunc = matplotlib.animation.FuncAnimation(fig, update, interval=0, blit=True)

plt.show()

stream.stop_stream()
stream.close()
p.terminate()
