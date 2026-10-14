const {test}=require('node:test');
const assert=require('node:assert/strict');
const {VoiceGate,voiceIntent,ContinuousVoice}=require('../web/duplex.js');

test('VAD pauses at sustained speech and ends after silence, not a single spike',()=>{
  const gate=new VoiceGate();
  assert.equal(gate.frame(.08,0),null);assert.equal(gate.frame(0,40),null);
  for(const t of [80,120,160])assert.equal(gate.frame(.03,t),null);
  assert.deepEqual(gate.frame(.03,200),{type:'start',onset:80});
  assert.equal(gate.frame(.03,240),null);assert.equal(gate.frame(0,700),null);
  assert.deepEqual(gate.frame(0,760),{type:'end'});
});

test('backchannels, corrections, direct stop and playback echoes remain distinct',()=>{
  assert.equal(voiceIntent('嗯，好的。','',true),'backchannel');
  assert.equal(voiceIntent('好的，那月缴多少？','',true),'question');
  assert.equal(voiceIntent('我没有医保。','',true),'question');
  assert.equal(voiceIntent('停止回答。','',true),'stop');
  assert.equal(voiceIntent('我帮您查一下。','我帮您查一下。',true),'echo');
  assert.equal(voiceIntent('好的。','',false),'question');
});

test('ASR final commits once per segment while the same microphone remains on',()=>{
  const heard=[];const driver=new ContinuousVoice({utterance:t=>heard.push(t)});
  driver.running=true;
  driver.asr({type:'partial',segment:0,text:'三十五岁'});
  driver.asr({type:'final',segment:0,text:'我三十五岁，有医保。'});
  driver.asr({type:'final',segment:0,text:'我三十五岁，有医保。'});
  driver.asr({type:'final',segment:1,text:'说错了，我没有医保。'});
  assert.deepEqual(heard,['我三十五岁，有医保。','说错了，我没有医保。']);
  assert.equal(driver.running,true);
});

test('partial correction interrupts once; final submits the corrected question',()=>{
  const heard=[],stops=[];
  const driver=new ContinuousVoice({answering:()=>true,interrupt:t=>stops.push(t),utterance:t=>heard.push(t)});
  driver.asr({type:'partial',segment:0,text:'我没有医保'});
  driver.asr({type:'partial',segment:0,text:'我没有医保，年缴多少钱'});
  driver.asr({type:'final',segment:0,text:'我没有医保，年缴多少钱？'});
  assert.equal(stops.length,1);assert.deepEqual(heard,['我没有医保，年缴多少钱？']);
});

test('a backchannel or playback echo resumes instead of becoming a new question',()=>{
  let stops=0,resumes=0,questions=0;
  const driver=new ContinuousVoice({answering:()=>true,reference:()=> '我帮您查一下。',interrupt:()=>stops++,resume:()=>resumes++,utterance:()=>questions++});
  driver.asr({type:'final',segment:0,text:'嗯，好的。'});
  driver.asr({type:'final',segment:1,text:'我帮您查一下。'});
  assert.equal(stops,0);assert.equal(questions,0);assert.equal(resumes,2);
});

test('unstable acknowledgement and echo prefixes do not cancel before final ASR',()=>{
  let stops=0;
  const driver=new ContinuousVoice({answering:()=>true,reference:()=> '我帮您查一下。',interrupt:()=>stops++});
  driver.asr({type:'partial',segment:0,text:'原来'});
  driver.asr({type:'final',segment:0,text:'原来如此。'});
  driver.asr({type:'partial',segment:1,text:'我帮'});
  driver.asr({type:'partial',segment:1,text:'我帮您查'});
  driver.asr({type:'final',segment:1,text:'我帮您查一下。'});
  assert.equal(stops,0);
});

test('voice stop stops the answer without starting a replacement generation',()=>{
  let stops=0,questions=0;
  const driver=new ContinuousVoice({answering:()=>true,interrupt:()=>stops++,utterance:()=>questions++});
  driver.asr({type:'partial',segment:0,text:'停止回答'});
  driver.asr({type:'final',segment:0,text:'停止回答。'});
  assert.equal(stops,1);assert.equal(questions,0);
});
