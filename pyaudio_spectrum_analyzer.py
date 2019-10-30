import sys
from libmir.rasp_audio_stream import AudioInputStream
from libmir.rasp_analyzer import SpectrumAnalyzer
from libmir.rasp_pitch_analyzer import PitchAnalyzer

def test_callback_sigproc(sig, sr):
    print(sig.shape, sr)

ais = AudioInputStream()
mode = int(sys.argv[1])
if mode == 0:
    analyzer = SpectrumAnalyzer(ais.RATE, ais.CHUNK, ais.CHANNELS)
else:
    analyzer = PitchAnalyzer(ais.RATE, ais.CHUNK, ais.CHANNELS)
analyzer.run(ais.stream)
