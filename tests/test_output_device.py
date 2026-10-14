import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent.playback import output_device,output_sample_rate
from agent.playback import TrackedDacPlayer
from unittest.mock import Mock,patch


class OutputDeviceTest(unittest.TestCase):
    def test_device_number_can_change_and_input_devices_are_never_selected(self):
        devices=[{'name':'Headset','max_output_channels':2,'hostapi':0},
                 {'name':'Mic','max_output_channels':0,'hostapi':1},
                 {'name':'Speakers','max_output_channels':2,'hostapi':1},
                 {'name':'Headset','max_output_channels':2,'hostapi':1}]
        sd=SimpleNamespace(default=SimpleNamespace(device=(1,0)),query_devices=lambda:devices,
                           query_hostapis=lambda:[{'name':'MME'},{'name':'Windows WASAPI'}])
        self.assertEqual(output_device(sd),3)

    def test_headset_native_sample_rate_fallback_is_checked(self):
        class Unsupported(Exception):pass
        tried=[]
        def check(**kwargs):
            tried.append(kwargs['samplerate'])
            if kwargs['samplerate']==48000:raise Unsupported()
        sd=SimpleNamespace(query_devices=lambda _: {'default_samplerate':44100},
                           check_output_settings=check,PortAudioError=Unsupported)
        self.assertEqual(output_sample_rate(sd,12),44100)
        self.assertEqual(tried,[48000,44100])

    def test_drain_waits_for_later_sentence_marker_not_only_last_audio_frame(self):
        player=TrackedDacPlayer.__new__(TrackedDacPlayer)
        player.queue_empty=Mock();player.queue_empty.wait.return_value=True
        player.last_dac=1.0
        player.receipts=Mock();player.receipts.fragments=[{'dac_end':1.02}]
        clock=[1.001]
        def finish():self.assertGreaterEqual(clock[0],1.02)
        player.receipts.finish.side_effect=finish
        with patch('agent.playback.time.perf_counter',side_effect=lambda:clock[0]),patch(
                'agent.playback.time.sleep',side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)):
            self.assertEqual(player.drain(),1.02)


if __name__=='__main__':unittest.main()
