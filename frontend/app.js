'use strict';
const $ = id => document.getElementById(id);
const state = {file:null, url:null, ready:false, busy:false, duration:null, lightweight:false, selection:0, timer:null};
const percent = value => Number.isFinite(Number(value)) ? `${(Number(value)*100).toFixed(3)}%` : 'Not available';
function showError(message){$('error').textContent=message;$('error').hidden=false;}
function updateButton(){ $('analyze').disabled = !state.file || !state.ready || state.busy || (state.lightweight && state.duration>60); }
function setBusy(busy){state.busy=busy;for(const id of ['choose','remove','again']) $(id).disabled=busy;updateButton();}
async function checkHealth(){
  $('retry').hidden=true;$('health').textContent='Connecting to analyzer…';
  try{
    const response=await fetch('/api/health',{signal:AbortSignal.timeout(90000)});
    if(!response.ok)throw new Error('Service unavailable');
    const data=await response.json();state.ready=Boolean(data.ready);state.lightweight=Boolean(data.lightweight);
    $('health').textContent=state.ready?(state.lightweight?'Analyzer ready · standard analysis':'Analyzer ready · detailed analysis'):'Model unavailable. Please try again later.';
    $('health-dot').className=state.ready?'ready':'offline';$('retry').hidden=state.ready;
    $('upload-hint').textContent=state.lightweight?'Up to 60 seconds. A short clip with a clear face works best.':'Choose a short clip with a clear, visible face.';
    const accuracy=data.evaluation_metrics?.accuracy;
    if(Number.isFinite(accuracy))$('accuracy').textContent=`${(accuracy*100).toFixed(0)}% accuracy on 300 test videos`;
    if(state.lightweight && state.duration>60)showError('Choose a clip that is 60 seconds or shorter for this server.');
  }catch(error){state.ready=false;$('health').textContent='Analyzer unavailable. The server may be waking up.';$('health-dot').className='offline';$('retry').hidden=false;}
  updateButton();
}
function clearFile(){
  if(state.busy)return;
  state.selection++;state.file=null;state.duration=null;
  $('preview').pause();$('preview').removeAttribute('src');$('preview').load();
  if(state.url)URL.revokeObjectURL(state.url);state.url=null;
  $('file').value='';$('selected').hidden=true;$('dropzone').hidden=false;$('error').hidden=true;$('results').hidden=true;$('empty').hidden=false;$('result-status').textContent='READY WHEN YOU ARE';updateButton();
}
function selectFile(file){
  if(!file || state.busy)return;
  if(!/\.(mp4|mov|avi|webm)$/i.test(file.name)){showError('Choose an MP4, MOV, AVI or WebM video.');return;}
  if(!file.size){showError('This file is empty. Please choose another video.');return;}
  if(file.size>20*1024*1024){showError('This video is larger than 20 MB. Trim it to a shorter clip and try again.');return;}
  clearFile();state.file=file;state.url=URL.createObjectURL(file);const selection=++state.selection;
  $('file-name').textContent=file.name;$('file-meta').textContent=`${(file.size/1024/1024).toFixed(2)} MB`;
  $('preview-note').hidden=true;$('selected').hidden=false;$('dropzone').hidden=true;
  $('preview').onloadedmetadata=()=>{if(selection!==state.selection)return;state.duration=$('preview').duration;
    if(Number.isFinite(state.duration))$('file-meta').textContent+=` · ${Math.ceil(state.duration)} seconds`;
    if(state.lightweight && state.duration>60)showError('Choose a clip that is 60 seconds or shorter for this server.');updateButton();};
  $('preview').onerror=()=>{if(selection===state.selection)$('preview-note').hidden=false;};
  $('preview').src=state.url;updateButton();
}
function analysisStage(){
  $('progress-stage').textContent='CHECKING YOUR VIDEO';$('progress-title').textContent='Looking beyond the surface';
  $('progress-description').textContent='The server is finding faces and checking appearance, texture and consistency. It does not report individual stage progress.';
  $('upload-progress').removeAttribute('value');$('upload-progress').setAttribute('aria-label','Analysis in progress');
}
function node(tag,className,text){const element=document.createElement(tag);if(className)element.className=className;if(text!==undefined)element.textContent=text;return element;}
function safePath(url){return typeof url==='string' && url.startsWith('/assets/web_analyses/') && !url.includes('..') ? url : null;}
function renderEvidence(data, report) {
  const gallery = $('sample-gallery');
  gallery.replaceChildren();
  for (const [index, url] of (report.sample_urls || []).entries()) {
    if (!safePath(url)) continue;
    const link = node('a', 'sample-card');
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener';
    const image = node('img');
    image.src = url;
    image.alt = `Face crop from sample ${index + 1}`;
    image.loading = 'lazy';
    link.append(image, node('span', '', `Sample ${index + 1}`));
    gallery.append(link);
  }
  if (!gallery.childElementCount) gallery.textContent = 'Face crops were not saved for this run.';
  const table = node('table');
  const headings = table.createTHead().insertRow();
  const rows = table.createTBody();
  const branches = Object.entries(data.suspicion_timeline || {}).map(
    ([key, values]) => [key, [values].flat(Infinity)]
  );
  headings.append(node('th', '', 'Sample'));
  const names = {frame: 'Appearance', freq: 'Texture', temporal: 'Consistency'};
  for (const [key] of branches) headings.append(node('th', '', names[key] || key));
  const length = Math.max(0, ...branches.map(([, values]) => values.length));
  for (let index = 0; index < length; index++) {
    const row = rows.insertRow();
    row.append(node('th', '', String(index + 1)));
    for (const [, values] of branches) {
      const score = values[index];
      row.insertCell().textContent = Number.isFinite(score) ? percent(score) : 'Unavailable';
    }
  }
  $('timeline').replaceChildren(length ? table : node('p', '', 'Timeline unavailable for this run.'));
  $('interpretation').textContent = report.interpretation || '';
  const facts = $('video-facts');
  facts.replaceChildren();
  for (const [label, value] of Object.entries({
    File: data.filename, Clip: $('file-meta').textContent, Model: data.model_run,
    Detector: data.face_detector, Device: data.device_name,
    'Predicted class score (uncalibrated)': percent(data.confidence)
  })) {
    if (value) facts.append(node('dt', '', label), node('dd', '', value));
  }
  $('indicators').replaceChildren(...(report.key_indicators || []).map(
    item => node('li', '', `${item.label} (${item.status}): ${item.description}`)
  ));
  const chart = $('contributions');
  chart.replaceChildren();
  if (safePath(data.branch_chart_url)) {
    const image = node('img');
    image.src = data.branch_chart_url;
    image.alt = 'Chart of branch contributions to the combined score';
    chart.append(node('h4', '', 'Branch contributions'), image);
    for (const [key, value] of Object.entries(data.branch_contributions || {})) {
      chart.append(node('p', 'small', `${key}: ${Number.isFinite(value) ? value.toPrecision(4) : value}`));
    }
  }
}
function renderResult(data){
  if(!['REAL','FAKE'].includes(data.label) || !Number.isFinite(data.fake_score))throw new Error('The server returned an incomplete result. Please try again.');
  $('prediction-label').textContent = data.label;
  $('prediction').classList.toggle('flagged', data.label === 'FAKE');
  const fake=data.label==='FAKE',report=data.forensic_report||{};
  $('assessment').classList.toggle('suspicious',fake);$('verdict-symbol').textContent=fake?'!':'✓';
  $('verdict-title').textContent=fake?'Possible manipulation detected':'No manipulation detected';
  $('verdict-description').textContent=fake?'The model found patterns that crossed its threshold for possible face manipulation. This is a reason to look closer, not proof that the video is fake.':'The model did not find enough evidence of face manipulation in the samples it checked. This does not prove that the video is authentic.';
  $('next-action').textContent=fake?'Check who originally published the video. Look for an original recording or reliable independent coverage before sharing a claim.':'Consider the source and context too. Audio edits, misleading captions and changes outside the face may not be detected.';
  renderEvidence(data, report);
  const count=report.sampled_frame_count;
  $('sample-count').textContent=count?`${count} face samples checked`:'Sampled-face analysis';
  $('processing-time').textContent=Number.isFinite(data.processing_seconds)?`${data.processing_seconds.toFixed(1)}s server processing`:'';
  $('score').textContent=percent(data.fake_score);$('threshold').textContent=percent(data.decision_threshold);
  const lightweight=report.attribution_method==='disabled_on_free_hosting';
  $('mode').textContent=lightweight?'Standard · CPU':'Detailed · '+(data.device_name||data.device||'server');
  $('signals').replaceChildren();
  const definitions=[['frame','Face appearance','Visual patterns in the face.'],['freq','Fine texture','Patterns in the image’s frequency information.'],['temporal','Consistency over time','How sampled face features change through the clip.']];
  for(const [key,title,description] of definitions){const value=data.branch_scores?.[key];if(!Number.isFinite(value))continue;const item=node('div','signal');const head=node('div','signal-head');head.append(node('strong','',title),node('span','',percent(value)));const track=node('div','signal-track'),fill=node('div','signal-fill');fill.style.width=`${Math.max(0,Math.min(1,value))*100}%`;track.append(fill);item.append(head,node('p','',description),track);$('signals').append(item);}
  const frames=(data.detected_frames||[]).filter(frame=>safePath(frame.url));$('frames-section').hidden=!frames.length;$('frames').replaceChildren();
  for(const frame of frames){const figure=node('figure'),image=node('img');image.src=frame.url;image.alt=`Model sensitivity for sampled face ${frame.frame_index+1}`;image.loading='lazy';figure.append(image,node('figcaption','',`Sample ${frame.frame_index+1} · score ${percent(frame.score)}`));$('frames').append(figure);}
  $('evidence-note').textContent=lightweight?'Standard mode shows sampled faces and scores. Heatmaps and branch attribution require detailed mode. The three-branch prediction still runs.':'These signals describe model behavior; they do not identify a proven edit. Face cropping can affect the result.';
  const url=safePath(data.report_url);$('download').hidden=!url;if(url)$('download').href=url;
  $('results').hidden=false;$('empty').hidden=true;$('result-status').textContent='ANALYSIS COMPLETE';$('results').focus({preventScroll:true});
}
function analyze(){
  if(!state.file || !state.ready || state.busy)return;
  setBusy(true);$('error').hidden=true;$('empty').hidden=true;$('results').hidden=true;$('loading').hidden=false;$('result-status').textContent='IN PROGRESS';
  $('progress-stage').textContent='UPLOADING YOUR VIDEO';$('progress-title').textContent='Getting your video ready';$('progress-description').textContent='Transferring the video to the analysis server.';$('upload-progress').value=0;$('upload-progress').setAttribute('aria-label','Upload progress');
  const started=performance.now();$('elapsed').textContent='0s elapsed';state.timer=setInterval(()=>{$('elapsed').textContent=`${Math.floor((performance.now()-started)/1000)}s elapsed`;},1000);
  const request=new XMLHttpRequest();request.open('POST','/api/analyze');request.timeout=15*60*1000;
  request.upload.onprogress=event=>{if(event.lengthComputable)$('upload-progress').value=event.loaded/event.total*100;};request.upload.onload=analysisStage;
  function finish(){clearInterval(state.timer);$('loading').hidden=true;setBusy(false);}
  function fail(message){finish();showError(message);$('empty').hidden=false;$('result-status').textContent='PLEASE TRY AGAIN';}
  request.onload=()=>{try{let data;try{data=JSON.parse(request.responseText);}catch{throw new Error('The server could not finish the analysis. Try a shorter video or retry when the service is ready.');}
    if(request.status<200||request.status>=300){const friendly={409:'Another video is being analyzed. Wait a moment, then try again.',413:'The video is too large. Choose a file under 20 MB.',503:'The analysis model is not ready. Please try again later.'};throw new Error(friendly[request.status]||(typeof data.detail==='string'?data.detail:'Analysis could not complete. Try a short clip with a clear face.'));}
    renderResult(data);finish();}catch(error){fail(error.message);}};
  request.onerror=()=>fail('Connection lost. The free server may have restarted. Check the connection and try a shorter clip.');
  request.ontimeout=()=>fail('The request took too long. The server may still be working; wait before trying a shorter clip.');
  const form=new FormData();form.append('file',state.file);request.send(form);
}
$('choose').addEventListener('click',()=>$('file').click());$('file').addEventListener('change',event=>selectFile(event.target.files[0]));$('remove').addEventListener('click',clearFile);$('again').addEventListener('click',()=>{clearFile();$('choose').focus();});$('analyze').addEventListener('click',analyze);$('retry').addEventListener('click',checkHealth);
for(const name of ['dragenter','dragover'])$('dropzone').addEventListener(name,event=>{event.preventDefault();if(!state.busy)$('dropzone').classList.add('dragging');});
for(const name of ['dragleave','drop'])$('dropzone').addEventListener(name,event=>{event.preventDefault();$('dropzone').classList.remove('dragging');});
$('dropzone').addEventListener('drop',event=>selectFile(event.dataTransfer.files[0]));
window.addEventListener('beforeunload',()=>{if(state.url)URL.revokeObjectURL(state.url);});checkHealth();
