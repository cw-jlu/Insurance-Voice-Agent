/* Capture with browser echo cancellation; resample and stream 16kHz mono PCM. */
class PcmRecorder extends AudioWorkletProcessor {
  constructor(){super();this.samples=[];this.offset=0;this.output=[];}
  process(inputs){
    const input=inputs[0]?.[0];if(!input)return true;
    for(const sample of input)this.samples.push(sample);
    const ratio=sampleRate/16000;
    while(this.offset+ratio<=this.samples.length){
      const start=Math.floor(this.offset),end=Math.floor(this.offset+ratio);let value=0;
      for(let i=start;i<end;i++)value+=this.samples[i];value/=Math.max(1,end-start);
      this.output.push(Math.round(Math.max(-1,Math.min(1,value))*(value<0?32768:32767)));this.offset+=ratio;
      if(this.output.length===640){const pcm=new Int16Array(this.output);const rms=Math.sqrt(this.output.reduce((sum,v)=>sum+(v/32768)**2,0)/640);this.port.postMessage({pcm:pcm.buffer,rms},[pcm.buffer]);this.output=[];}
    }
    const consumed=Math.floor(this.offset);if(consumed){this.samples.splice(0,consumed);this.offset-=consumed;}
    return true;
  }
}
registerProcessor('pcm-recorder',PcmRecorder);
