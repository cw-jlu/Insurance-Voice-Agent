"""Actual output receipts bridge audio playback to conversation history."""
import threading
import time


def output_device(sd):
    """Choose an output-capable WASAPI peer of the current default output."""
    devices=sd.query_devices()
    default=sd.default.device[1]
    name=devices[default]['name'] if default is not None and default>=0 else None
    apis=sd.query_hostapis()
    candidates=[i for i,d in enumerate(devices) if d['max_output_channels']>=2
                and apis[d['hostapi']]['name']=='Windows WASAPI']
    matching=[i for i in candidates if devices[i]['name']==name]
    if matching:return matching[0]
    if candidates:return candidates[0]
    if default is not None and default>=0 and devices[default]['max_output_channels']>=2:return default
    raise RuntimeError('No stereo output device available')


def output_sample_rate(sd,device):
    rates=list(dict.fromkeys([48000,int(sd.query_devices(device)['default_samplerate'])]))
    for rate in rates:
        try:
            sd.check_output_settings(device=device,channels=2,dtype='int16',samplerate=rate)
            return rate
        except sd.PortAudioError:
            continue
    raise RuntimeError('No supported stereo output sample rate')


class PlaybackReceipts:
    def __init__(self, session, playback_id):
        self.session = session
        self.playback_id = playback_id
        self.lock = threading.RLock()
        self.fragments = []
        self.confirmed = 0
        self.heard = ''
        self.closed = False

    def enqueue(self, text):
        with self.lock:
            if self.closed:
                return None
            token = len(self.fragments)
            self.fragments.append({'text': text, 'dac_end': None, 'dac_first_audible':None})
            return token

    def scheduled_start(self,token,dac_first_audible):
        with self.lock:
            if not self.closed and self.fragments[token]['dac_first_audible'] is None:
                self.fragments[token]['dac_first_audible']=dac_first_audible

    def scheduled_end(self, token, dac_end):
        with self.lock:
            if not self.closed:
                self.fragments[token]['dac_end'] = dac_end

    def flush(self, now=None):
        now = time.perf_counter() if now is None else now
        with self.lock:
            if self.closed:
                return self.heard
            while self.confirmed < len(self.fragments):
                fragment = self.fragments[self.confirmed]
                if fragment['dac_end'] is None or fragment['dac_end'] > now:
                    break
                self.heard += fragment['text']
                self.confirmed += 1
            self.session.confirm_playback(self.playback_id, self.heard)
            return self.heard

    def interrupt(self, now=None):
        with self.lock:
            self.flush(now)
            self.closed = True
            return self.session.interrupt_playback(self.playback_id)

    def finish(self, now=None):
        with self.lock:
            self.flush(now)
            if self.confirmed != len(self.fragments):
                raise ValueError('Output still has unheard audio')
            if not self.session.confirm_playback(self.playback_id, self.heard, complete=True):
                return False
            self.closed = True
            return True


class TrackedDacPlayer:
    """WASAPI output with sentence-end DAC timestamps, no character/time guessing.

    Speech without word-alignment retains completed sentences only on interruption.
    Optional voice dependencies are imported only when creating the actual player.
    """
    def __init__(self, receipts, device=None):
        import numpy as np
        import sounddevice as sd
        from collections import deque
        self.np = np
        self.receipts = receipts
        self.device=output_device(sd) if device is None else device
        self.output_rate=output_sample_rate(sd,self.device)
        self.lock = threading.Lock()
        self.chunks = deque()
        self.offset = 0
        self.aborted = False
        self.first_dac = None
        self.last_dac = None
        self.current_token = None
        self.dropped_frames = 0
        self.statuses = []
        self.queue_empty = threading.Event()
        self.queue_empty.set()

        def callback(outdata, frames, info, status):
            buf = np.frombuffer(outdata, dtype=np.int16).reshape(-1, 2)
            buf[:] = 0
            dac = time.perf_counter() + info.outputBufferDacTime - info.currentTime
            with self.lock:
                if status:
                    self.statuses.append(str(status))
                pos = 0
                while self.chunks and not self.aborted:
                    current = self.chunks[0]
                    if isinstance(current, tuple):
                        if len(current)==2:
                            self.current_token=current[1]
                        else:
                            self.receipts.scheduled_end(current[0], dac + pos / self.output_rate)
                            self.current_token=None
                        self.chunks.popleft()
                        continue
                    if pos == frames:
                        break
                    n = min(frames - pos, len(current) - self.offset)
                    buf[pos:pos+n, 0] = current[self.offset:self.offset+n]
                    buf[pos:pos+n, 1] = current[self.offset:self.offset+n]
                    if self.first_dac is None or self.current_token is not None:
                        audible = np.flatnonzero(np.abs(buf[pos:pos+n, 0].astype(np.int32)) > 327)
                        if len(audible):
                            first=dac + (pos + int(audible[0])) / self.output_rate
                            if self.first_dac is None:self.first_dac=first
                            if self.current_token is not None:self.receipts.scheduled_start(self.current_token,first)
                    pos += n
                    self.offset += n
                    self.last_dac = dac + pos / self.output_rate
                    if self.offset == len(current):
                        self.chunks.popleft()
                        self.offset = 0
                if not self.chunks:
                    self.queue_empty.set()

        self.stream = sd.RawOutputStream(device=self.device, samplerate=self.output_rate, channels=2,
            dtype='int16', blocksize=round(self.output_rate*.02), latency='low', callback=callback)
        self.stream.start()

    def feed(self, pcm24):
        from scipy.signal import resample_poly
        from math import gcd
        np = self.np
        audio = np.frombuffer(pcm24, dtype=np.int16).astype(np.float32)
        divisor=gcd(self.output_rate,24000)
        data = np.clip(resample_poly(audio,self.output_rate//divisor,24000//divisor), -32768, 32767).astype(np.int16)
        with self.lock:
            if not self.aborted and len(data):
                self.chunks.append(data)
                self.queue_empty.clear()

    def sentence_end(self, token):
        with self.lock:
            if not self.aborted:
                self.chunks.append((token,))
                self.queue_empty.clear()

    def sentence_start(self,token):
        with self.lock:
            if not self.aborted:
                self.chunks.append(('start',token))
                self.queue_empty.clear()

    def abort(self):
        at = time.perf_counter()
        with self.lock:
            self.aborted = True
            self.dropped_frames = sum(len(x) for x in self.chunks if not isinstance(x, tuple)) - self.offset
            self.chunks.clear()
            self.offset = 0
            self.queue_empty.set()
            end = max(at, self.last_dac or at)
        history = self.receipts.interrupt(at)
        return {'request_wall': at, 'last_scheduled_dac_wall': end,
                'dropped_frames': self.dropped_frames, 'history': history}

    def drain(self, timeout=120):
        if not self.queue_empty.wait(timeout):
            raise TimeoutError('Playback did not drain')
        # A trailing sentence marker may be consumed in the next output callback
        # after the final PCM frame. Wait for BOTH timestamps before committing.
        marker_end=max((f['dac_end'] or 0 for f in self.receipts.fragments),default=0)
        end = max(self.last_dac or time.perf_counter(),marker_end)
        while end > time.perf_counter():
            time.sleep(max(0.0,min(.02,end-time.perf_counter())))
        self.receipts.finish()
        return end

    def close(self):
        self.stream.stop()
        self.stream.close()
