const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { JSDOM } = require('jsdom');
async function app() {
  const dom = new JSDOM(readFileSync('demo/index.html', 'utf8'), {url:'https://demo.example/',runScripts:'outside-only'});
  const w=dom.window;
  w.fetch=async()=>({ok:true,json:async()=>JSON.parse(readFileSync('demo/data.json','utf8'))});
  w.HTMLElement.prototype.scrollIntoView=()=>{};
  w.Worker=class {postMessage(){ w.qaWorker=this; } terminate(){} };
  w.eval(readFileSync('demo/app.js','utf8'));
  for(let i=0;i<20&&!w.labReady;i++)await new Promise(r=>setTimeout(r,5));
  assert.equal(w.labReady,true);
  return {dom,w,d:w.document,$:s=>w.document.querySelector(s)};
}
test('rank filters, comparison metrics and navigation update the workspace',async()=>{
  const {dom,$}=await app();
  $('[data-tab="experiments"]').click();
  assert.equal($('#experiments').classList.contains('active'),true);
  $('[data-rank="8"]').click();
  assert.equal($('#trainable').textContent,'3,072');
  assert.equal($('#nll').textContent,'6.2549');
  $('#metric-choice').value='exact_match';$('#metric-choice').dispatchEvent(new dom.window.Event('change'));
  assert.equal($('#comparison strong').textContent,'0.0000');
  assert.equal($('#audit').children.length,6);dom.window.close();
});
test('interruption rolls back unsaved updates and incompatible contracts block resume',async()=>{
  const {dom,$}=await app();
  for(let i=0;i<3;i++)$('#advance').click();
  $('#interrupt').click();assert.equal($('#resume').disabled,false);
  $('#contract').value='data';$('#contract').dispatchEvent(new dom.window.Event('change'));
  assert.equal($('#resume').disabled,true);assert.match($('#contract-result').textContent,/rejected/);
  $('#contract').value='same';$('#contract').dispatchEvent(new dom.window.Event('change'));
  $('#resume').click();assert.equal([...dom.window.document.querySelectorAll('.step.done')].length,2);
  for(let i=0;i<6;i++)$('#advance').click();assert.equal($('#run-status').textContent,'COMPLETE');
  $('#restart').click();assert.equal($('#run-status').textContent,'READY');dom.window.close();
});
test('split viewer displays actual split records and architecture changes content',async()=>{
  const {dom,$}=await app();
  assert.equal($('#split-docs').children.length,16);
  $('[data-split="test"]').click();assert.equal($('#split-docs').children.length,4);
  $('[data-stage="serve"]').click();assert.match($('#stage-detail').textContent,/FastAPI/);dom.window.close();
});
test('QA form handles input validation, model responses and errors safely',async()=>{
  const {dom,w,$}=await app();
  $('#context').value='';$('#ask').click();assert.match($('#model-state').textContent,/Add both/);
  $('[data-example="lora"]').click();$('#ask').click();assert.equal($('#ask').disabled,true);
  const context=$('#context').value;const answer='weights';const start=context.indexOf(answer);
  w.qaWorker.onmessage({data:{type:'result',answer,score:.9,start,end:start+answer.length,seconds:.2}});
  assert.equal($('#ask').disabled,false);assert.equal($('#answer-text').textContent,answer);assert.equal($('#evidence mark').textContent,answer);
  $('#ask').click();w.qaWorker.onmessage({data:{type:'error',message:'Offline'}});assert.equal($('#ask').disabled,false);assert.equal($('#model-state').textContent,'Offline');dom.window.close();
});
