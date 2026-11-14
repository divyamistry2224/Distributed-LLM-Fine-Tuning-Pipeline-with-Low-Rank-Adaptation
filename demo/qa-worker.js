import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js';
env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = 1;
let model;
self.onmessage = async ({ data: { context, question } }) => {
  try {
    if (!model) {
      model = await pipeline('question-answering', 'Xenova/distilbert-base-cased-distilled-squad', {
        dtype: 'q8', device: 'wasm',
        progress_callback: p => {
          if (p.status === 'progress') self.postMessage({type:'progress',message:`Loading ${p.file} · ${Math.round(p.progress)}%`,progress:p.progress});
        },
      });
      self.postMessage({type:'ready'});
    }
    const encoded = model.tokenizer(question, {text_pair:context, truncation:false});
    if (encoded.input_ids.data.length > 512) throw Error('This QA model supports 512 tokens. Shorten the document or question.');
    const started=performance.now();
    const result=await model(question,context,{max_answer_len:60});
    // The question-answering pipeline returns text/score; locate the exact span for evidence.
    const start=context.indexOf(result.answer);
    if(start<0) throw Error('The answer could not be located in the source document.');
    self.postMessage({type:'result',answer:result.answer,score:result.score,start,end:start+result.answer.length,seconds:(performance.now()-started)/1000});
  } catch(error) {
    self.postMessage({type:'error',message:'Could not run QA. '+error.message+' Check your connection and try a shorter document.'});
  }
};
