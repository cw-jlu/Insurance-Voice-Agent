/* Plain browser client. No service keys, model prompts or credentials live here. */
const $ = id => document.getElementById(id);
const welcomeTemplate = $('welcome').cloneNode(true);
const ui = {session:null, socket:null, scenarios:[], selected:'free', health:null,
  busy:false, turn:0, agent:null, speech:new Map(), queue:[], speaking:false,
  audio:null, audioUrl:null, speechGeneration:0, done:false, microphone:null, pendingMode:null,
  turnStart:0, stages:new Map(), metrics:{}, inputMetrics:{}, finishedTrace:null, played:new Set(), sentenceCount:0,
  ducked:false, reference:'', cancelledTurn:null, stopWait:null, resolveStop:null, voiceSerial:Promise.resolve()};

function openPanel(name){
  $('settings-panel').hidden=name!=='settings';$('inspector').hidden=name!=='inspector';
  $('panel-backdrop').hidden=!name;
  if(name)document.body.dataset.panel=name;else delete document.body.dataset.panel;
  $('open-settings').setAttribute('aria-expanded',String(name==='settings'));
  $('open-inspector').setAttribute('aria-expanded',String(name==='inspector'));
}
function voiceState(state,text=''){
  const labels={off:'语音未开启',connecting:'正在连接麦克风',listening:'持续聆听 · 可以随时插话',hearing:'正在听你说',recognizing:'正在识别'};
  $('voice-state').textContent=labels[state]||state;$('voice-status').dataset.state=state;
  $('live-transcript').textContent=text;$('microphone').textContent=state==='off'?'开启语音':'结束语音';
}

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
  renderHistory(state.turns||[]);
  if(state.voice_events?.length){
    const names={microphone_on:'开启持续语音',microphone_off:'结束语音',speech_pause:'检测人声，暂停播报',confirmed_interrupt:'确认插话，取消旧回答',backchannel:'附和，继续回答',echo:'疑似回声，继续回答',stop_only:'语音叫停',asr_renew:'识别连接续期'};
    $('events').textContent=[...(ui.finishedTrace?.events||[]),...state.voice_events.map(e=>`${names[e.kind]||e.kind}${e.text?'：'+e.text:''}${e.metrics?.vad_pause_s!==undefined?' · VAD暂停 '+e.metrics.vad_pause_s.toFixed(3)+'s':''}${e.metrics?.barge_confirm_s!==undefined?' · 插话确认 '+e.metrics.barge_confirm_s.toFixed(3)+'s':''}`)].join('\n');
  }
  if(state.ended&&ui.microphone)endMicrophone(false);
}
function metric(label,value){const row=element('div');row.append(element('dt','',label),element('dd','',typeof value==='number'?`${value.toFixed(2)} s`:value));$('latency').append(row);}
function reportMetric(key,value){ui.metrics[key]=value;send({type:'browser_metrics',turn:ui.turn,metrics:{[key]:value}});renderLatency();}
function renderLatency(){
  $('latency').replaceChildren();
  const names={recording_s:'语音采集时长',asr_connect_s:'识别服务连接',asr_finalize_s:'结束到识别确认',asr_endpoint_s:'句尾到识别确认',first_sound_s:'回答首声（含承接）',first_business_sound_s:'业务正文首声',playback_end_s:'本轮完整播完'};
  for(const [key,label] of Object.entries(names))if(ui.metrics[key]!==undefined)metric(label,ui.metrics[key]);
  if(ui.finishedTrace){metric(ui.session?.mode==='replay'?'回放文本就绪':'生成与校验完成',ui.finishedTrace.text_ready_s);metric('全部音频就绪',ui.finishedTrace.audio_ready_s);}
  $('timeline').replaceChildren();
  for(const entry of ui.stages.values()){
    const row=element('div',`stage-row ${entry.status}`);
    const name=entry.model?`${entry.name} · ${entry.model}`:entry.sentence!==undefined?`${entry.name} · 第${entry.sentence+1}句`:entry.name;
    row.append(element('span','',name),element('strong','',entry.duration_s!==undefined?`${entry.duration_s.toFixed(2)} s`:entry.status==='running'?'进行中…':'未完成'));
    row.title=`相对本轮发送：${entry.start_s.toFixed(2)}s → ${entry.end_s?.toFixed(2)||'…'}s${entry.tokens?.total_tokens?'；'+entry.tokens.total_tokens+' tokens':''}`;
    $('timeline').append(row);
  }
}
function renderHistory(turns){
  $('history-count').textContent=`${turns.length}轮`;$('history-log').replaceChildren();
  for(const turn of turns){
    const d=element('details','history-turn');d.append(element('summary','',`${turn.turn}. ${turn.question}${turn.cancelled?' · 已停止':''}`));
    d.append(element('p','history-label','生成正文'),element('p','',turn.response||'生成未完成'),element('p','history-label','已播完句子'),element('p','',turn.heard||'尚无句尾播放回执'));
    if(turn.trace){const button=element('button','outline','查看本轮耗时与溯源');button.type='button';button.onclick=()=>{ui.stages=new Map((turn.stages||[]).map(s=>[s.id,s]));ui.metrics={...turn.browser_metrics};renderTrace(turn.trace);};d.append(button);}
    $('history-log').append(d);
  }
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
  ui.finishedTrace=trace;for(const entry of trace.stages||[])ui.stages.set(entry.id,entry);renderLatency();
  $('events').textContent=(trace.events||[]).join('\n')||'本轮未记录额外流程事件。';
  if(trace.recorded_case)$('events').textContent+=`\n历史批次：${trace.recorded_batch} / ${trace.recorded_case}\n原实验整轮播完：${Number(trace.recorded_playback_end_s).toFixed(2)}秒`;
  $('citations').replaceChildren();const seen=new Set();
  for(const citation of trace.citations||[]){
    if(seen.has(citation.id))continue;seen.add(citation.id);
    const details=element('details','citation');details.append(element('summary','',citation.id));
    const supported=(trace.claims||[]).filter(c=>(c.citations||[]).some(s=>s.id===citation.id)).map(c=>c.text).join('\n');
    if(supported)details.append(element('p','claim-text','引用该来源的答句：'+supported));
    details.append(element('p','',citation.quote),element('pre','source-location',JSON.stringify(citation.location,null,2)));
    const b=element('button','outline','核对完整原始证据');b.type='button';b.onclick=async()=>{
      b.disabled=true;try{const source=await request(`/api/sessions/${ui.session.id}/sources/${encodeURIComponent(citation.id)}`);details.append(element('p','source-label',source.source+'\n'+source.path+'\nSHA-256: '+source.sha256),element('p','',source.text));b.remove();}catch(e){note(e.message);b.disabled=false;}
    };details.append(b);$('citations').append(details);
  }
  if(!seen.size)$('citations').append(element('p','empty','本轮没有业务原文引用。流程承接与人工通知不等价于业务事实。'));
}
function stopSpeech(){
  ui.speechGeneration++;ui.queue=[];ui.speech.clear();ui.ducked=false;
  if(ui.audio){ui.audio.pause();ui.audio.removeAttribute('src');ui.audio.load();ui.audio=null;}
  if(ui.audioUrl){URL.revokeObjectURL(ui.audioUrl);ui.audioUrl=null;}
  window.speechSynthesis?.cancel();ui.speaking=false;
}
function stopAnswer(){
  if(!ui.session)return;
  ui.cancelledTurn=ui.turn;
  if(!ui.stopWait){
    ui.stopWait=new Promise(resolve=>{ui.resolveStop=resolve;});
    const wait=ui.stopWait;
    setTimeout(()=>{if(ui.stopWait===wait){ui.resolveStop?.(false);ui.stopWait=null;ui.resolveStop=null;}},5000);
  }
  stopSpeech();send({type:'stop'});busy(false);$('seen').hidden=true;
  if(ui.agent){ui.agent.state.textContent='已停止；未播完的句子不计入已播历史。';ui.agent.node.classList.remove('pending');}
}
function pauseForVoice(){
  ui.ducked=true;if(ui.audio)ui.audio.pause();else window.speechSynthesis?.pause();
}
function resumeAfterVoice(){
  if(!ui.ducked)return;ui.ducked=false;
  if(ui.audio&&ui.speaking)ui.audio.play().catch(()=>{});else window.speechSynthesis?.resume();
  playNext();
}
function playbackComplete(unit,generation){
  if(generation!==ui.speechGeneration)return;
  ui.played.add(unit.index);
  send({type:'receipt',turn:unit.turn,index:unit.index});ui.speaking=false;
  if(ui.agent)ui.agent.state.textContent=ui.done&&!ui.queue.length?'本轮已播完':'正在朗读';
  if(ui.done&&!ui.queue.length)$('stop').disabled=true;
  if(ui.done&&!ui.queue.length&&ui.played.size===ui.sentenceCount)reportMetric('playback_end_s',(performance.now()-ui.turnStart)/1000);
  playNext();
}
function playbackFailed(unit,generation=ui.speechGeneration){
  if(generation!==ui.speechGeneration)return;
  ui.speaking=false;$('seen').hidden=false;note('浏览器未能朗读。可阅读文字后点击“已读完”，或重新开启朗读。');
  ui.queue=[];$('stop').disabled=true;
}
function playNext(){
  if(ui.ducked||ui.speaking||!ui.queue.length)return;
  const unit=ui.queue.shift();if(unit.turn!==ui.turn)return playNext();
  if(!$('speak').checked){$('seen').hidden=false;ui.queue=[];return;}
  ui.speaking=true;$('stop').disabled=false;const generation=ui.speechGeneration;
  if(ui.agent)ui.agent.state.textContent='正在朗读';
  ui.reference=unit.text;
  const rate=Number($('playback-rate').value);
  const started=()=>{
    if(generation!==ui.speechGeneration)return;
    if(ui.metrics.first_sound_s===undefined)reportMetric('first_sound_s',(performance.now()-ui.turnStart)/1000);
    if(unit.kind==='business'&&ui.metrics.first_business_sound_s===undefined)reportMetric('first_business_sound_s',(performance.now()-ui.turnStart)/1000);
    reportMetric('rate',rate);
  };
  if(unit.wav){
    const bytes=Uint8Array.from(atob(unit.wav),c=>c.charCodeAt(0));const audioUrl=URL.createObjectURL(new Blob([bytes],{type:'audio/wav'}));ui.audioUrl=audioUrl;
    const audio=ui.audio=new Audio(audioUrl);audio.onended=()=>{URL.revokeObjectURL(audioUrl);if(ui.audio===audio)ui.audioUrl=null;playbackComplete(unit,generation);};
    audio.playbackRate=rate;audio.onplaying=started;audio.onerror=()=>playbackFailed(unit,generation);audio.play().catch(()=>playbackFailed(unit,generation));
  }else if('speechSynthesis' in window){
    const utterance=new SpeechSynthesisUtterance(unit.text);utterance.lang='zh-CN';utterance.rate=rate;utterance.onstart=started;
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
    ui.cancelledTurn=null;
    ui.played.clear();ui.sentenceCount=0;
    ui.stages.clear();ui.finishedTrace=null;ui.metrics={...ui.inputMetrics};ui.inputMetrics={};renderLatency();
    send({type:'browser_metrics',turn:data.turn,metrics:ui.metrics});
    ui.turn=data.turn;ui.done=false;stopSpeech();message('user',data.question);ui.agent=message('agent','');ui.agent.node.classList.add('pending');ui.agent.state.textContent='正在理解问题并核对材料';busy(true);$('seen').hidden=true;return;
  }
  if(data.type==='stopped'){
    renderState(data.state);busy(false);ui.resolveStop?.(true);ui.stopWait=null;ui.resolveStop=null;return;
  }
  if(data.turn!==undefined&&data.turn!==ui.turn)return;
  if(data.turn===ui.cancelledTurn)return;
  if(data.type==='stage'){ui.stages.set(data.stage.id,data.stage);renderLatency();}
  if(data.type==='sentence'){ui.sentenceCount++;ui.agent.body.textContent+=data.text;ui.speech.set(data.index,{text:data.text,index:data.index,turn:data.turn,kind:data.kind});scroll();}
  if(data.type==='audio'||data.type==='audio_fallback')audioReady(data);
  if(data.type==='turn_done'){
    ui.done=true;busy(false);ui.agent.node.classList.remove('pending');ui.agent.state.textContent=ui.speaking?'正在朗读':`生成与校验已完成 · ${data.trace.text_ready_s.toFixed(2)}秒`;
    renderTrace(data.trace);renderState(data.state);if(!$('speak').checked)$('seen').hidden=false;
    const inspect=element('button','quiet message-inspect','查看本轮');inspect.type='button';
    inspect.onclick=()=>{ui.stages=new Map((data.trace.stages||[]).map(s=>[s.id,s]));ui.metrics={...(ui.session?.turns?.find(t=>t.turn===data.turn)?.browser_metrics||{})};renderTrace(data.trace);openPanel('inspector');};ui.agent.node.append(inspect);
    if($('speak').checked&&!ui.speaking&&!ui.queue.length&&ui.metrics.first_sound_s!==undefined&&ui.played.size===ui.sentenceCount)reportMetric('playback_end_s',(performance.now()-ui.turnStart)/1000);
  }
  if(data.type==='error'){note(data.message);busy(false);if(data.state)renderState(data.state);if(ui.agent)ui.agent.state.textContent='本轮未完成';}
}
async function newSession(){
  stopSpeech();if(ui.microphone)await endMicrophone(false);
  const old=ui.session;ui.session=null;ui.socket?.close();ui.socket=null;
  ui.turn=0;
  ui.resolveStop?.(false);ui.stopWait=null;ui.resolveStop=null;ui.cancelledTurn=null;
  if(old)await request(`/api/sessions/${old.id}`,{method:'DELETE'}).catch(()=>{});
  const mode=$('mode').value;
  const selected=ui.scenarios.find(s=>s.id===ui.selected);
  if(selected?.mode!==(mode==='replay'?'replay':'live'))ui.selected=mode==='replay'?'replay':'free';
  const scenario=ui.scenarios.find(s=>s.id===ui.selected);
  $('scenario-title').textContent=mode==='replay'?'历史回放':'保险咨询';$('scenario-subtitle').textContent=mode==='replay'?'查看已保存实验；实时模式可以直接提问。':'保费、保障与理赔条件，都可以接着问。';
  document.querySelectorAll('.scenario').forEach(n=>n.classList.toggle('active',n.dataset.scenario===ui.selected));
  ui.session=await request('/api/sessions',{method:'POST',body:JSON.stringify({mode:mode==='replay'?'replay':'live',provider:mode==='replay'?'deepseek':mode})});
  $('messages').replaceChildren(welcomeTemplate.cloneNode(true));ui.agent=null;ui.done=false;$('seen').hidden=true;note();
  ui.metrics={};ui.inputMetrics={};ui.stages.clear();ui.finishedTrace=null;renderLatency();
  $('citations').replaceChildren(element('p','empty','业务回答后可展开查看原文依据。'));$('events').textContent='开始咨询后显示处理过程。';
  const url=new URL(`/api/sessions/${ui.session.id}/ws`,location.href);url.protocol=location.protocol==='https:'?'wss:':'ws:';
  const socket=ui.socket=new WebSocket(url);
  socket.onopen=()=>{connected('会话已连接',true);busy(false);};
  socket.onmessage=e=>{if(ui.socket===socket)socketEvent(JSON.parse(e.data));};
  socket.onclose=()=>{if(ui.socket===socket){connected('连接已关闭');$('send').disabled=true;stopSpeech();}};
  socket.onerror=()=>note('连接失败，请确认演示服务正在运行。');
  renderState(ui.session);updateModeNote();
  if(mode!=='replay')$('cloud-voice').checked=ui.health.capabilities.cloud_tts;
}
function updateModeNote(){
  const replay=$('mode').value==='replay';
  $('mode-note').textContent=replay?'回放来自exp44原始记录，不调用模型。自由提问请切换实时模式。':$('mode').value==='qwen'?'当前使用Qwen对照回答，不代表冻结版DeepSeek成绩。':'实时使用冻结咨询链路，外部服务仍可能失败。';
  $('microphone').disabled=!ui.health?.capabilities.asr;
  $('cloud-voice').disabled=replay||!ui.health?.capabilities.cloud_tts;
  if(replay)$('cloud-voice').checked=false;
  $('composer-note').textContent=replay?'当前为历史回放。开启实时语音，请先在设置中选择已配置的模型。':'开启一次即可持续对话；检测到插话先暂停，确认后停止旧回答。';
}
async function submit(){
  const text=$('question').value.trim();if(!text||ui.busy)return;
  if(ui.speaking)stopAnswer();
  if(ui.stopWait){const ok=await ui.stopWait;if(!ok){note('停止回执未到达，请检查连接后再发送。');return;}}
  note();send({type:'turn',text,cloud_voice:$('cloud-voice').checked&&$('speak').checked});busy(true);
  ui.turnStart=performance.now();
  if(ui.session.mode!=='replay')$('question').value='';
}

async function startMicrophone(){
  if(ui.session?.mode==='replay'){openPanel('settings');note('先选择实时模型，再开启持续语音。DeepSeek未配置时可主动选择Qwen对照。');return;}
  if(!navigator.mediaDevices?.getUserMedia){note('麦克风需要localhost或HTTPS，也可继续输入问题。');return;}
  const driver=ui.microphone=new ContinuousVoice({
    state:(state,text)=>{voiceState(state,text);if(state==='off'&&ui.microphone===driver)ui.microphone=null;},
    answering:()=>ui.busy||ui.speaking||ui.queue.length>0||ui.ducked,
    reference:()=>ui.reference,
    pause:pauseForVoice,resume:resumeAfterVoice,
    interrupt:()=>stopAnswer(),
    error:message=>{if(driver.currentText)$('question').value=driver.currentText;note(message);},
    event:(kind,text,metrics)=>send({type:'voice_event',turn:ui.turn,kind,text,metrics}),
    utterance:(text,metrics)=>{
      if(!$('auto-send').checked){$('question').value=text;note('识别完成，核对后可以发送。');return;}
      ui.voiceSerial=ui.voiceSerial.then(async()=>{
        if(ui.busy||ui.speaking)stopAnswer();
        if(ui.stopWait&&!await ui.stopWait){note('停止回执未到达，请确认连接。');return;}
        if(ui.microphone!==driver)return;
        $('question').value=text;ui.inputMetrics=metrics;await submit();
      }).catch(()=>note('本轮语音提交未完成，文字仍可编辑后发送。'));
    }
  });
  await driver.start();
}
function cleanupMicrophone(mic){mic.stop();}
async function endMicrophone(){ui.microphone?.stop();}
$('composer').addEventListener('submit',e=>{e.preventDefault();submit();});
$('open-settings').addEventListener('click',()=>openPanel(document.body.dataset.panel==='settings'?null:'settings'));
$('open-inspector').addEventListener('click',()=>openPanel(document.body.dataset.panel==='inspector'?null:'inspector'));
document.querySelectorAll('[data-close-panel]').forEach(b=>b.addEventListener('click',()=>openPanel(null)));
$('panel-backdrop').addEventListener('click',()=>openPanel(null));
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&document.body.dataset.panel){const name=document.body.dataset.panel;openPanel(null);$(name==='settings'?'open-settings':'open-inspector').focus();}});
$('question').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();submit();}});
$('stop').addEventListener('click',stopAnswer);
$('microphone').addEventListener('click',()=>ui.microphone?endMicrophone():startMicrophone());
$('playback-rate').addEventListener('change',()=>{const rate=Number($('playback-rate').value);if(ui.audio)ui.audio.playbackRate=rate;if(ui.turn)reportMetric('rate',rate);note(ui.audio?'已调整当前音频的播放速度。':'播放速度用于下一句浏览器朗读或云端音频。');});
$('seen').addEventListener('click',()=>{stopSpeech();send({type:'seen'});$('seen').hidden=true;if(ui.agent)ui.agent.state.textContent='用户已确认读完文字；非声学播放验收。';});
$('speak').addEventListener('change',()=>{if(!$('speak').checked){stopSpeech();if(ui.agent)$('seen').hidden=false;}});
$('new-session').addEventListener('click',()=>$('reset-dialog').showModal());
$('reset-cancel').addEventListener('click',()=>{$('reset-dialog').close();if(ui.pendingMode){$('mode').value=ui.session.mode==='replay'?'replay':ui.session.provider;ui.pendingMode=null;}});
$('reset-dialog').addEventListener('cancel',()=>{if(ui.pendingMode&&ui.session){$('mode').value=ui.session.mode==='replay'?'replay':ui.session.provider;ui.pendingMode=null;}});
$('reset-confirm').addEventListener('click',async()=>{$('reset-dialog').close();ui.pendingMode=null;try{await newSession();}catch(e){note(e.message);busy(false);}});
$('mode').addEventListener('change',()=>{ui.pendingMode=$('mode').value;$('reset-dialog').showModal();});
$('export').addEventListener('click',async()=>{if(!ui.session)return;try{const data=await request(`/api/sessions/${ui.session.id}/export`);const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=element('a');a.href=url;a.download=`consultation-${ui.session.id.slice(0,8)}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){note(e.message);}});
window.addEventListener('beforeunload',()=>{stopSpeech();ui.socket?.close();if(ui.microphone)cleanupMicrophone(ui.microphone);});
window.addEventListener('resize',()=>requestAnimationFrame(scroll));

async function init(){
  try{
    [ui.health,ui.scenarios]=await Promise.all([request('/api/health'),request('/api/scenarios')]);
    for(const provider of ['deepseek','qwen']){const option=$('mode').querySelector(`option[value="${provider}"]`);if(!ui.health.capabilities[provider]){option.disabled=true;option.textContent+=' · 未配置';}}
    $('mode').value=ui.health.capabilities.deepseek?'deepseek':'replay';
    if($('mode').value==='replay'){ui.selected='replay';$('scenario-title').textContent='完整对话回放';$('scenario-subtitle').textContent='DeepSeek尚未配置。补好本地密钥并重启后，可现场自由提问。';}
    for(const scenario of ui.scenarios){const b=element('button',`scenario${scenario.id===ui.selected?' active':''}`);b.dataset.scenario=scenario.id;b.append(element('strong','',scenario.name),element('span','',scenario.description));b.addEventListener('click',()=>{
      ui.selected=scenario.id;document.querySelectorAll('.scenario').forEach(n=>n.classList.remove('active'));b.classList.add('active');$('scenario-title').textContent=scenario.name;$('scenario-subtitle').textContent=scenario.description;
      if(scenario.mode==='live'&&ui.session.mode==='replay')note('这是自由提问场景，请先切换到已配置的实时模式。');renderSuggestions();
      if(scenario.questions?.length&&ui.session.mode!=='replay'){$('question').value=scenario.questions[0];openPanel(null);$('question').focus();}
    });$('scenarios').append(b);}
    await newSession();
  }catch(error){connected('服务未连接');note(error.message);busy(false);}
}
let rehearsalElapsed=0,rehearsalStart=null;
function drawTimer(){
  const elapsed=rehearsalElapsed+(rehearsalStart===null?0:(performance.now()-rehearsalStart)/1000);
  const stage=elapsed<180?'总结呈现':elapsed<600?'落地演示':elapsed<900?'评委提问':'演练结束';
  const left=Math.max(0,(elapsed<180?180:elapsed<600?600:900)-elapsed);
  $('rehearsal-time').textContent=`${stage} · ${String(Math.floor(left/60)).padStart(2,'0')}:${String(Math.floor(left%60)).padStart(2,'0')}`;
  $('rehearsal-time').classList.toggle('time-warning',left<=60&&elapsed<900);
  if(elapsed>=900&&rehearsalStart!==null){rehearsalElapsed=900;rehearsalStart=null;$('timer-toggle').textContent='演练完成';}
}
$('timer-toggle').onclick=()=>{if(rehearsalStart===null){if(rehearsalElapsed>=900)rehearsalElapsed=0;rehearsalStart=performance.now();$('timer-toggle').textContent='暂停演练';}else{rehearsalElapsed+=(performance.now()-rehearsalStart)/1000;rehearsalStart=null;$('timer-toggle').textContent='继续演练';}drawTimer();};
$('timer-reset').onclick=()=>{rehearsalElapsed=0;rehearsalStart=null;$('timer-toggle').textContent='开始15分钟演练';drawTimer();};
setInterval(drawTimer,1000);
init();
