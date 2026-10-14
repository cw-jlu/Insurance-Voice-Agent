import argparse,base64,io,json,sys,wave
from pathlib import Path
from playwright.sync_api import sync_playwright,expect

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'demo-runtime').mkdir(exist_ok=True)
parser=argparse.ArgumentParser(description='Paid-cloud synthetic duplex reproducibility check; run the demo server first.')
parser.add_argument('--provider',choices=['deepseek','qwen'],default='deepseek')
parser.add_argument('--browser-path',help='Optional Chromium executable; otherwise use Playwright-managed Chromium')
args=parser.parse_args()
sys.path.insert(0,str(ROOT/'src'))
from demo.app import capabilities
from agent import cloud

def speak_wav(tts,text):
    _,_,pcm=tts.synth(text);out=io.BytesIO()
    with wave.open(out,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(pcm)
    return base64.b64encode(out.getvalue()).decode()

tts=cloud.RealtimeTTS()
try:
    first=speak_wav(tts,'我三十五岁，有医保，首次投保必选计划个人单，年缴多少钱？')
    second=speak_wav(tts,'说错了，我没有医保，年缴重新算。')
finally:tts.close()

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=args.browser_path,args=['--autoplay-policy=no-user-gesture-required'])
    context=browser.new_context(viewport={'width':1440,'height':1000},bypass_csp=True)
    context.add_init_script('''
      navigator.mediaDevices.getUserMedia=async()=>{
        window.testMediaCalls=(window.testMediaCalls||0)+1;
        const context=new AudioContext();await context.resume();
        const destination=context.createMediaStreamDestination();
        window.testInput={context,destination,stream:destination.stream};return destination.stream;
      };
      window.feedSpeech=async(wav)=>{
        const bytes=Uint8Array.from(atob(wav),c=>c.charCodeAt(0));
        const input=window.testInput;const buffer=await input.context.decodeAudioData(bytes.buffer);
        const source=input.context.createBufferSource();source.buffer=buffer;
        source.connect(input.destination);source.start();
      };
    ''')
    page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8080');expect(page.locator('#connection')).to_have_text('会话已连接')
    expect(page.locator('#inspector')).to_be_hidden();expect(page.locator('#settings-panel')).to_be_hidden()
    page.locator('#open-settings').click();page.locator('#mode').select_option(args.provider);page.locator('#reset-confirm').click()
    expect(page.locator('#session-mode')).to_have_text('实时 · Qwen对照' if args.provider=='qwen' else '实时 · DeepSeek');page.locator('[data-close-panel]').first.click()
    page.locator('#microphone').click();expect(page.locator('#voice-state')).to_contain_text('持续聆听',timeout=20000)
    page.evaluate('wav=>feedSpeech(wav)',first)
    expect(page.locator('.message.user')).to_have_count(1,timeout=30000)
    page.wait_for_function('ui.metrics.first_business_sound_s !== undefined && ui.speaking && ui.audio && !ui.audio.paused',timeout=120000)
    page.evaluate('wav=>feedSpeech(wav)',second)
    expect(page.locator('.message.user')).to_have_count(2,timeout=30000)
    page.wait_for_function('ui.turn===2 && ui.done && !ui.speaking && ui.played.size===ui.sentenceCount',timeout=120000)
    expect(page.locator('#microphone')).to_have_text('结束语音')
    media_calls=page.evaluate('testMediaCalls');assert media_calls==1
    sid=page.evaluate('ui.session.id')
    data=page.request.get('http://127.0.0.1:8080/api/sessions/'+sid+'/export').json()
    report={'provider':'qwen-comparison' if args.provider=='qwen' else 'deepseek','synthetic_input':True,'real_asr_llm_tts':True,'human_aec_tested':False,
            'get_user_media_calls':media_calls,'page_errors':errors,'export':data}
    (ROOT/'demo-runtime/continuous-browser-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    assert data['turns'][0]['cancelled'],data['turns'][0]
    assert any(e['kind']=='confirmed_interrupt' for e in data['voice_events']),data['voice_events']
    assert '423' not in data['turns'][0]['heard']
    assert '971' in data['turns'][1]['response'],data['turns'][1]['response']
    assert data['session']['profile']['has_social'] is False
    assert not errors
    page.screenshot(path=str(ROOT/'demo-runtime/continuous-desktop.png'),full_page=True)
    desktop_overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth')
    page.locator('#open-inspector').click();expect(page.locator('#inspector')).to_be_visible()
    page.screenshot(path=str(ROOT/'demo-runtime/continuous-inspector.png'),full_page=True)
    page.keyboard.press('Escape');expect(page.locator('#inspector')).to_be_hidden()
    page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(ROOT/'demo-runtime/continuous-mobile.png'),full_page=True)
    mobile_overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth')
    assert not desktop_overflow and not mobile_overflow
    page.locator('#microphone').click();expect(page.locator('#microphone')).to_have_text('开启语音')
    assert page.evaluate('testInput.stream.getTracks().every(t=>t.readyState==="ended")')
    report.update(desktop_overflow=desktop_overflow,mobile_overflow=mobile_overflow,microphone_released=True)
    (ROOT/'demo-runtime/continuous-browser-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'get_user_media_calls':media_calls,'questions':[t['question'] for t in data['turns']],
                      'interrupt_events':data['voice_events'],'page_errors':errors,'microphone_released':True},ensure_ascii=False))
    browser.close()
