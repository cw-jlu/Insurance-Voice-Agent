/* Plain browser client. No service keys, model prompts or credentials live here. */
const $ = id => document.getElementById(id);
const welcomeTemplate = $('welcome').cloneNode(true);
const ui = {session:null, socket:null, scenarios:[], selected:'replay', health:null,
  busy:false, turn:0, agent:null, speech:new Map(), queue:[], speaking:false,
  audio:null, audioUrl:null, speechGeneration:0, done:false, microphone:null, pendingMode:null};

function note(text=''){ $('notice').textContent=text; $('notice').hidden=!text; }
function connected(text,good=false){$('connection').textContent=text;$('connection').classList.toggle('connected',good);}
async function request(url,options={}){
  const r=await fetch(url,{...options,headers:{'Content-Type':'application/json',...options.headers}});
  const data=await r.json();if(!r.ok)throw new Error(data.detail||'服务暂不可用');return data;
}
function send(data){if(ui.socket?.readyState===WebSocket.OPEN)ui.socket.send(JSON.stringify(data));}
function element(tag,cls,text){const el=document.createElement(tag);if(cls)el.className=cls;if(text!==undefined)el.textContent=text;return el;}
function message(role,text){
  $('welcome')?.remove();const node=element('article',`message ${role}`);
  const header=element('div','message-header');header.append(element('span','speaker',role==='user'?'你':'知保 · AI 助理'));
  header.append(element('span','',new Date().toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'})));
  const body=element('div','message-body',text);const state=element('div','message-state');
  node.append(header,body,state);$('messages').append(node);scroll();return{node,body,state};
}
function scroll(){ $('messages').scrollTop=$('messages').scrollHeight; }
function busy(value){ui.busy=value;$('send').disabled=value||!ui.session;$('stop').disabled=!value&&!ui.speaking;$('question').readOnly=ui.session?.mode==='replay';}

function renderState(state){
  ui.session=state;$('stage').textContent=state.stage||'咨询中';
  $('session-mode').textContent=state.mode==='replay'?'历史回放':state.provider==='qwen'?'实时 · Qwen对照':'实时 · DeepSeek';
  $('profile').replaceChildren();const labels={age:'被保人年龄',has_social:'基本医保',relation:'被保人关系',gender:'性别',pay:'缴费方式',family_size:'家庭人数'};
  for(const [key,value] of Object.entries(state.profile||{})){
    const row=element('div');row.append(element('dt','',labels[key]||key),element('dd','',key==='has_social'?(value?'有医保':'无医保'):key==='age'?`${value}岁`:String(value)));$('profile').append(row);
  }
  if(!$('profile').children.length)$('profile').append(element('div','empty','随咨询补充年龄、医保等信息。'));
  $('tasks').replaceChildren();const tasks=state.consultation?.tasks||[];
  const status={answered:'已答复',needs_user:'待补条件',needs_review:'需人工核实',cancelled:'已取消',needs_answer:'待回答',pending:'需重新核对'};
  $('task-count').textContent=String(tasks.filter(t=>!['answered','cancelled'].includes(t.status)).length);
  for(const task of tasks){const row=element('div','task');row.append(element('div','task-title',task.question),element('span',`task-status ${task.status}`,status[task.status]||'待处理'));$('tasks').append(row);}
  if(!tasks.length)$('tasks').append(element('p','empty','问题和未确认条件会保留在这里。'));
  $('question').readOnly=state.mode==='replay';
  if(state.mode==='replay')$('question').value=state.next_question||'';
  renderSuggestions();
}
function renderSuggestions(){
  $('suggestions').replaceChildren();
  const scenario=ui.scenarios.find(s=>s.id===ui.selected);
  const choices=ui.session?.mode==='replay'?(ui.session.next_question?['播放下一轮录制问题']:[]):(scenario?.questions||[]);
  choices.forEach((text,index)=>{const b=element('button','suggestion',text);b.type='button';b.addEventListener('click',()=>{if(ui.session?.mode!=='replay')$('question').value=scenario.questions[index];$('question').focus();});$('suggestions').append(b);});
  if(ui.session?.mode==='replay')$('send').firstChild.textContent=ui.session.next_question?'播放本轮 ':'回放结束 ';
  else $('send').firstChild.textContent='发送问题 ';
}
function renderTrace(trace){
  $('events').textContent=(trace.events||[]).join('\n')||'本轮未记录额外流程事件。';
  if(trace.recorded_case)$('events').textContent+=`\n历史批次：${trace.recorded_batch} / ${trace.recorded_case}\n原实验整轮播完：${Number(trace.recorded_playback_end_s).toFixed(2)}秒`;
  $('citations').replaceChildren();const seen=new Set();
  for(const citation of trace.citations||[]){if(seen.has(citation.id))continue;seen.add(citation.id);const details=element('details','citation');details.append(element('summary','',citation.id),element('p','',citation.quote));$('citations').append(details);}
  if(!seen.size)$('citations').append(element('p','empty','本轮没有业务原文引用。流程承接与人工通知不等价于业务事实。'));
}
function stopSpeech(){
  ui.speechGeneration++;ui.queue=[];ui.speech.clear();
  if(ui.audio){ui.audio.pause();ui.audio.removeAttribute('src');ui.audio.load();ui.audio=null;}
  if(ui.audioUrl){URL.revokeObjectURL(ui.audioUrl);ui.audioUrl=null;}
  window.speechSynthesis?.cancel();ui.speaking=false;
}
function stopAnswer(){
  stopSpeech();send({type:'stop'});busy(false);$('seen').hidden=true;
  if(ui.agent){ui.agent.state.textContent='已停止；未播完的句子不计入已播历史。';ui.agent.node.classList.remove('pending');}
}
function playbackComplete(unit,generation){
  if(generation!==ui.speechGeneration)return;
  send({type:'receipt',turn:unit.turn,index:unit.index});ui.speaking=false;
  if(ui.agent)ui.agent.state.textContent=ui.done&&!ui.queue.length?'本轮已播完':'正在朗读';
  if(ui.done&&!ui.queue.length)$('stop').disabled=true;
  playNext();
}
function playbackFailed(unit,generation=ui.speechGeneration){
  if(generation!==ui.speechGeneration)return;
  ui.speaking=false;$('seen').hidden=false;note('浏览器未能朗读。可阅读文字后点击“已读完”，或重新开启朗读。');
  ui.queue=[];$('stop').disabled=true;
}
function playNext(){
  if(ui.speaking||!ui.queue.length)return;
  const unit=ui.queue.shift();if(unit.turn!==ui.turn)return playNext();
  if(!$('speak').checked){$('seen').hidden=false;ui.queue=[];return;}
  ui.speaking=true;$('stop').disabled=false;const generation=ui.speechGeneration;
  if(ui.agent)ui.agent.state.textContent='正在朗读';
  if(unit.wav){
    const bytes=Uint8Array.from(atob(unit.wav),c=>c.charCodeAt(0));const audioUrl=URL.createObjectURL(new Blob([bytes],{type:'audio/wav'}));ui.audioUrl=audioUrl;
    const audio=ui.audio=new Audio(audioUrl);audio.onended=()=>{URL.revokeObjectURL(audioUrl);if(ui.audio===audio)ui.audioUrl=null;playbackComplete(unit,generation);};
    audio.onerror=()=>playbackFailed(unit,generation);audio.play().catch(()=>playbackFailed(unit,generation));
  }else if('speechSynthesis' in window){
    const utterance=new SpeechSynthesisUtterance(unit.text);utterance.lang='zh-CN';utterance.rate=1.1;
    const voice=speechSynthesis.getVoices().find(v=>v.lang.startsWith('zh'));if(voice)utterance.voice=voice;
    utterance.onend=()=>playbackComplete(unit,generation);utterance.onerror=e=>{if(generation===ui.speechGeneration&&e.error!=='canceled'&&e.error!=='interrupted')playbackFailed(unit,generation);};
    speechSynthesis.speak(utterance);
  }else playbackFailed(unit,generation);
}
function audioReady(data){
  const unit=ui.speech.get(data.index);if(!unit||data.turn!==ui.turn)return;
  unit.wav=data.wav;ui.queue.push(unit);ui.speech.delete(data.index);if(data.message)note(data.message);playNext();
}
function socketEvent(data){
  if(data.type==='state'||data.type==='receipt'){renderState(data.state);return;}
  if(data.type==='turn_start'){
    ui.turn=data.turn;ui.done=false;stopSpeech();message('user',data.question);ui.agent=message('agent','');ui.agent.node.classList.add('pending');ui.agent.state.textContent='正在理解问题并核对材料';busy(true);$('seen').hidden=true;return;
  }
  if(data.turn!==undefined&&data.turn!==ui.turn)return;
  if(data.type==='sentence'){ui.agent.body.textContent+=data.text;ui.speech.set(data.index,{text:data.text,index:data.index,turn:data.turn});scroll();}
  if(data.type==='audio'||data.type==='audio_fallback')audioReady(data);
  if(data.type==='turn_done'){
    ui.done=true;busy(false);ui.agent.node.classList.remove('pending');ui.agent.state.textContent=ui.speaking?'正在朗读':`本轮文字已就绪 · ${data.elapsed_s.toFixed(2)}秒`;
    renderTrace(data.trace);renderState(data.state);if(!$('speak').checked)$('seen').hidden=false;
  }
  if(data.type==='stopped'){renderState(data.state);busy(false);}
  if(data.type==='error'){note(data.message);busy(false);if(data.state)renderState(data.state);if(ui.agent)ui.agent.state.textContent='本轮未完成';}
}
async function newSession(){
  stopSpeech();if(ui.microphone)await endMicrophone(false);
  const old=ui.session;ui.session=null;ui.socket?.close();ui.socket=null;
  if(old)await request(`/api/sessions/${old.id}`,{method:'DELETE'}).catch(()=>{});
  const mode=$('mode').value;
  ui.session=await request('/api/sessions',{method:'POST',body:JSON.stringify({mode:mode==='replay'?'replay':'live',provider:mode==='replay'?'deepseek':mode})});
  $('messages').replaceChildren(welcomeTemplate.cloneNode(true));ui.agent=null;ui.done=false;$('seen').hidden=true;note();
  $('citations').replaceChildren(element('p','empty','业务回答后可展开查看原文依据。'));$('events').textContent='开始咨询后显示处理过程。';
  const url=new URL(`/api/sessions/${ui.session.id}/ws`,location.href);url.protocol=location.protocol==='https:'?'wss:':'ws:';
  const socket=ui.socket=new WebSocket(url);
  socket.onopen=()=>{connected('会话已连接',true);busy(false);};
  socket.onmessage=e=>{if(ui.socket===socket)socketEvent(JSON.parse(e.data));};
  socket.onclose=()=>{if(ui.socket===socket){connected('连接已关闭');$('send').disabled=true;stopSpeech();}};
  socket.onerror=()=>note('连接失败，请确认演示服务正在运行。');
  renderState(ui.session);updateModeNote();
}
function updateModeNote(){
  const replay=$('mode').value==='replay';
  $('mode-note').textContent=replay?'回放来自exp44原始记录，不调用模型。自由提问请切换实时模式。':$('mode').value==='qwen'?'当前使用Qwen对照回答，不代表冻结版DeepSeek成绩。':'实时使用冻结咨询链路，外部服务仍可能失败。';
  $('microphone').disabled=replay||!ui.health?.capabilities.asr;
  $('cloud-voice').disabled=replay||!ui.health?.capabilities.cloud_tts;
  if(replay)$('cloud-voice').checked=false;
  $('composer-note').textContent=replay?'朗读使用浏览器语音，播放耗时不代表原实验时延。':'点击录音会显式停止当前播报；本页未宣称自然打断已达标。';
}
async function submit(){
  const text=$('question').value.trim();if(!text||ui.busy)return;
  if(ui.speaking)stopAnswer();note();send({type:'turn',text,cloud_voice:$('cloud-voice').checked&&$('speak').checked});busy(true);
  if(ui.session.mode!=='replay')$('question').value='';
}

async function startMicrophone(){
  if(!navigator.mediaDevices?.getUserMedia){note('当前环境无法使用麦克风。请用localhost或HTTPS打开，也可继续文字咨询。');return;}
  let stream,context;
  try{
    stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});
    stopAnswer();note('正在连接语音服务，请稍候再说话。');context=new AudioContext();await context.resume();await context.audioWorklet.addModule('/assets/microphone.js');
    const source=context.createMediaStreamSource(stream);const worklet=new AudioWorkletNode(context,'pcm-recorder');
    const silence=context.createGain();silence.gain.value=0;source.connect(worklet);worklet.connect(silence);silence.connect(context.destination);
    const url=new URL('/api/asr',location.href);url.protocol=location.protocol==='https:'?'wss:':'ws:';
    const socket=new WebSocket(url);socket.binaryType='arraybuffer';
    const mic=ui.microphone={stream,context,source,worklet,silence,socket,ready:false,finals:[],submit:true,timer:null};
    worklet.port.onmessage=e=>{if(mic.ready&&socket.readyState===WebSocket.OPEN)socket.send(e.data);};
    socket.onmessage=e=>{
      const data=JSON.parse(e.data);
      if(data.type==='ready'){mic.ready=true;note('正在听，讲完后点击“结束录音”。');}
      if(data.type==='partial')$('question').value=[...mic.finals,data.text].join('');
      if(data.type==='final'){mic.finals.push(data.text);$('question').value=mic.finals.join('');}
      if(data.type==='done'){const shouldSubmit=mic.submit;cleanupMicrophone(mic);note();if(shouldSubmit&&$('question').value.trim())submit();}
      if(data.type==='error'){note(data.message);cleanupMicrophone(mic);}
    };
    socket.onerror=()=>{note('语音连接失败，可继续文字咨询。');cleanupMicrophone(mic);};
    socket.onclose=()=>{if(ui.microphone===mic){cleanupMicrophone(mic);note('语音连接已结束，识别文本保留在输入框。');}};
    mic.timer=setTimeout(()=>endMicrophone(),55000);$('microphone').textContent='结束录音';
  }catch(error){note(error.name==='NotAllowedError'?'未获得麦克风权限，可继续文字咨询。':'录音未能启动，可继续文字咨询。');if(ui.microphone)cleanupMicrophone(ui.microphone);else{stream?.getTracks().forEach(t=>t.stop());context?.close();}}
}
function cleanupMicrophone(mic){
  clearTimeout(mic.timer);mic.stream.getTracks().forEach(t=>t.stop());mic.worklet.disconnect();mic.source.disconnect();mic.silence.disconnect();mic.context.close();
  if(mic.socket.readyState<2)mic.socket.close();if(ui.microphone===mic)ui.microphone=null;$('microphone').textContent='开始说话';
}
async function endMicrophone(shouldSubmit=true){
  const mic=ui.microphone;if(!mic)return;mic.submit=shouldSubmit;mic.ready=false;clearTimeout(mic.timer);mic.stream.getTracks().forEach(t=>t.stop());
  if(mic.socket.readyState===WebSocket.OPEN){mic.socket.send(JSON.stringify({type:'end'}));note('正在确认识别结果…');mic.timer=setTimeout(()=>{cleanupMicrophone(mic);note('识别确认超时，文字可编辑后发送。');},12000);}else cleanupMicrophone(mic);
}

$('composer').addEventListener('submit',e=>{e.preventDefault();submit();});
$('question').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();submit();}});
$('stop').addEventListener('click',stopAnswer);
$('microphone').addEventListener('click',()=>ui.microphone?endMicrophone():startMicrophone());
$('seen').addEventListener('click',()=>{stopSpeech();send({type:'seen'});$('seen').hidden=true;if(ui.agent)ui.agent.state.textContent='用户已确认读完文字；非声学播放验收。';});
$('speak').addEventListener('change',()=>{if(!$('speak').checked){stopSpeech();if(ui.agent)$('seen').hidden=false;}});
$('new-session').addEventListener('click',()=>$('reset-dialog').showModal());
$('reset-cancel').addEventListener('click',()=>{$('reset-dialog').close();if(ui.pendingMode){$('mode').value=ui.session.mode==='replay'?'replay':ui.session.provider;ui.pendingMode=null;}});
$('reset-dialog').addEventListener('cancel',()=>{if(ui.pendingMode&&ui.session){$('mode').value=ui.session.mode==='replay'?'replay':ui.session.provider;ui.pendingMode=null;}});
$('reset-confirm').addEventListener('click',async()=>{$('reset-dialog').close();ui.pendingMode=null;try{await newSession();}catch(e){note(e.message);busy(false);}});
$('mode').addEventListener('change',()=>{ui.pendingMode=$('mode').value;$('reset-dialog').showModal();});
$('export').addEventListener('click',async()=>{if(!ui.session)return;try{const data=await request(`/api/sessions/${ui.session.id}/export`);const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=element('a');a.href=url;a.download=`consultation-${ui.session.id.slice(0,8)}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){note(e.message);}});
window.addEventListener('beforeunload',()=>{stopSpeech();ui.socket?.close();if(ui.microphone)cleanupMicrophone(ui.microphone);});

async function init(){
  try{
    [ui.health,ui.scenarios]=await Promise.all([request('/api/health'),request('/api/scenarios')]);
    for(const provider of ['deepseek','qwen']){const option=$('mode').querySelector(`option[value="${provider}"]`);if(!ui.health.capabilities[provider]){option.disabled=true;option.textContent+=' · 未配置';}}
    for(const scenario of ui.scenarios){const b=element('button',`scenario${scenario.id===ui.selected?' active':''}`);b.append(element('strong','',scenario.name),element('span','',scenario.description));b.addEventListener('click',()=>{
      ui.selected=scenario.id;document.querySelectorAll('.scenario').forEach(n=>n.classList.remove('active'));b.classList.add('active');$('scenario-title').textContent=scenario.name;$('scenario-subtitle').textContent=scenario.description;
      if(scenario.mode==='live'&&ui.session.mode==='replay')note('这是自由提问场景，请先切换到已配置的实时模式。');renderSuggestions();
    });$('scenarios').append(b);}
    await newSession();
  }catch(error){connected('服务未连接');note(error.message);busy(false);}
}
init();
