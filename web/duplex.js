/* Persistent capture and independent ASR. Playback never owns the microphone. */
const normalizeVoice = text => String(text||'').replace(/[\s\p{P}\p{S}]/gu,'').toLowerCase();
function voiceIntent(text, reference='', answering=false){
  const t=normalizeVoice(text);
  if(!t)return 'empty';
  if(answering && t.length>=6 && normalizeVoice(reference).includes(t))return 'echo';
  if(answering && /^(?:嗯+|啊+|哦+|好的|好吧|好|对的|对|是的|明白了|明白|知道了|原来如此|你说|你继续|继续|我在听|然后呢|收到){1,3}$/.test(t))return 'backchannel';
  if(/^(停|停止|暂停|停一下|停止回答|别说了|不要说了|先别说|等一下|等等)$/.test(t))return 'stop';
  return 'question';
}

class VoiceGate {
  constructor(){this.floor=.002;this.active=false;this.voicedSince=null;this.lastVoice=0;}
  frame(rms, now, answering=false){
    const threshold=Math.max(.009,this.floor*3.5);
    if(rms>threshold){
      this.lastVoice=now;
      if(this.voicedSince===null)this.voicedSince=now;
      if(!this.active && now-this.voicedSince>=120){this.active=true;return {type:'start',onset:this.voicedSince};}
    }else{
      if(!answering && !this.active)this.floor=Math.min(.004,this.floor*.98+rms*.02);
      if(!this.active)this.voicedSince=null;
      if(this.active && now-this.lastVoice>=520){this.active=false;this.voicedSince=null;return {type:'end'};}
    }
    return null;
  }
}

class ContinuousVoice {
  constructor(callbacks={}){
    this.cb=callbacks;this.running=false;this.ready=false;this.socket=null;
    this.gate=new VoiceGate();this.buffer=[];this.currentText='';this.confirmed=false;
    this.wasAnswering=false;this.onset=0;this.finalIndices=new Set();this.retryCount=0;
  }
  state(value,text=''){this.cb.state?.(value,text);}
  event(kind,text='',metrics={}){this.cb.event?.(kind,text,metrics);}
  answering(){return Boolean(this.cb.answering?.());}
  async start(){
    this.running=true;this.state('connecting');
    try{
      this.stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});
      if(!this.running){this.stream.getTracks().forEach(t=>t.stop());return;}
      this.context=new AudioContext();await this.context.resume();await this.context.audioWorklet.addModule('/assets/microphone.js');
      if(!this.running){this.stop();return;}
      this.source=this.context.createMediaStreamSource(this.stream);this.worklet=new AudioWorkletNode(this.context,'pcm-recorder');
      this.silence=this.context.createGain();this.silence.gain.value=0;
      this.source.connect(this.worklet);this.worklet.connect(this.silence);this.silence.connect(this.context.destination);
      this.worklet.port.onmessage=e=>this.frame(e.data);
      this.event('microphone_on');this.connect();
    }catch(error){this.cb.error?.(error.name==='NotAllowedError'?'请允许麦克风权限，或继续输入问题。':'语音未能开启，请检查麦克风与服务配置。');this.stop();}
  }
  frame(data){
    if(!this.running)return;
    const now=performance.now(),activity=this.gate.frame(data.rms,now,this.answering());
    if(activity?.type==='start'){
      this.onset=activity.onset;this.wasAnswering=this.answering();this.confirmed=false;
      this.state('hearing',this.currentText);
      if(this.wasAnswering){this.cb.pause?.();this.event('speech_pause','',{vad_pause_s:(now-this.onset)/1000});}
    }
    if(activity?.type==='end'){
      this.state('recognizing',this.currentText);
      clearTimeout(this.releaseTimer);
      this.releaseTimer=setTimeout(()=>{
        if(!this.confirmed&&!this.gate.active){this.wasAnswering=false;this.cb.resume?.();this.state('listening');}
      },1400);
    }
    if(this.ready&&this.socket?.readyState===WebSocket.OPEN)this.socket.send(data.pcm);
    else{this.buffer.push(data.pcm);if(this.buffer.length>40)this.buffer.shift();}
    // Renew the cloud connection between utterances; the media stream stays alive.
    if(this.ready&&now-this.readyAt>48000&&!this.gate.active&&!this.currentText)this.renew();
  }
  connect(){
    if(!this.running)return;
    this.connectAt=performance.now();this.ready=false;this.renewing=false;this.finalIndices.clear();
    const url=new URL('/api/asr',location.href);url.protocol=location.protocol==='https:'?'wss:':'ws:';
    const socket=this.socket=new WebSocket(url);socket.binaryType='arraybuffer';
    socket.onmessage=e=>{
      if(!this.running||socket!==this.socket)return;
      const data=JSON.parse(e.data);
      if(data.type==='ready'){
        this.ready=true;this.readyAt=performance.now();this.retryCount=0;
        this.connectSeconds=(this.readyAt-this.connectAt)/1000;
        for(const pcm of this.buffer)socket.send(pcm);this.buffer=[];this.state('listening');
      }else if(data.type==='partial'||data.type==='final')this.asr(data);
      else if(data.type==='done'){
        if(this.renewing){socket.close();this.connect();}
        else{socket.close();this.reconnect();}
      }else if(data.type==='error'){
        this.cb.error?.(data.message);this.stop();
      }
    };
    socket.onclose=()=>{if(this.running&&socket===this.socket)this.reconnect();};
    socket.onerror=()=>{};
  }
  reconnect(){
    if(!this.running||this.reconnectTimer)return;
    this.ready=false;
    if(++this.retryCount>3){this.cb.error?.('语音连接多次失败，已释放麦克风。可重新开启或输入问题。');this.stop();return;}
    this.state('connecting',this.currentText);
    this.reconnectTimer=setTimeout(()=>{this.reconnectTimer=null;this.connect();},Math.min(1500,300*this.retryCount));
  }
  renew(){
    if(this.renewing)return;this.renewing=true;this.ready=false;this.event('asr_renew');
    this.socket.send(JSON.stringify({type:'end'}));
    clearTimeout(this.renewTimer);
    this.renewTimer=setTimeout(()=>{if(this.running&&this.renewing){this.socket.close();}},5000);
  }
  asr(data){
    if(this.finalIndices.has(data.segment))return;
    const text=data.text?.trim();if(!text)return;
    this.currentText=text;this.state('hearing',text);
    const answering=this.wasAnswering||this.answering();
    const reference=this.cb.reference?.()||'';
    const kind=voiceIntent(text,reference,answering);
    const normalized=normalizeVoice(text);
    // Interim ASR text can be only the start of an acknowledgement or echo.
    // Wait for more words instead of irreversibly cancelling on that prefix.
    if(data.type==='partial'&&answering&&(
        (normalized.length>=2&&normalizeVoice(reference).includes(normalized))||
        ['原来如此','明白了','知道了','我在听','你继续','然后呢','好的','对的','是的'].some(t=>t.startsWith(normalized))))return;
    if((kind==='question'&&normalized.length>=2)||kind==='stop'){
      if(answering&&!this.confirmed){
        this.confirmed=true;this.cb.interrupt?.(text);
        this.event('confirmed_interrupt',text,{barge_confirm_s:this.onset?(performance.now()-this.onset)/1000:0});
      }
    }
    if(data.type!=='final')return;
    this.finalIndices.add(data.segment);clearTimeout(this.releaseTimer);
    if(kind==='echo'||kind==='backchannel'){
      this.event(kind,text);this.cb.resume?.();
    }else if(kind==='stop'){
      this.event('stop_only',text);if(!this.confirmed)this.cb.interrupt?.(text);
    }else if(kind==='question'){
      const now=performance.now();
      const metrics={asr_connect_s:this.connectSeconds||0,recording_s:this.onset?(now-this.onset)/1000:0,
                     asr_endpoint_s:this.gate.lastVoice?(now-this.gate.lastVoice)/1000:0};
      this.cb.utterance?.(text,metrics);
    }
    this.currentText='';this.confirmed=false;this.wasAnswering=false;this.onset=0;
    this.gate.active=false;this.gate.voicedSince=null;
    this.state('listening');
  }
  stop(){
    const active=this.running;this.running=false;this.ready=false;
    clearTimeout(this.releaseTimer);clearTimeout(this.renewTimer);clearTimeout(this.reconnectTimer);
    this.reconnectTimer=null;
    if(this.socket){this.socket.onclose=null;this.socket.close();this.socket=null;}
    this.stream?.getTracks().forEach(t=>t.stop());
    this.worklet?.disconnect();this.source?.disconnect();this.silence?.disconnect();this.context?.close();
    this.buffer=[];this.cb.resume?.();this.state('off');if(active)this.event('microphone_off');
  }
}
if(typeof module!=='undefined')module.exports={VoiceGate,voiceIntent,ContinuousVoice};
