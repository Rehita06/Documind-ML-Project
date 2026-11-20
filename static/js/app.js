const chat = document.getElementById('chat');
const form = document.getElementById('chatForm');
const question = document.getElementById('question');
const trace = document.getElementById('trace');
const sourceUsed = document.getElementById('sourceUsed');

function escapeHtml(s=''){return s.replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function formatText(s=''){return escapeHtml(s).replace(/\n/g,'<br>');}
function addMessage(role, text, source='', citations=[]){
  const wrap=document.createElement('div'); wrap.className=`message ${role}`;
  const citeHtml=citations.length?`<div class="citations"><strong>Sources</strong><br>${citations.map(c=>c.url?`<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>`:escapeHtml(c.title)).join('<br>')}</div>`:'';
  wrap.innerHTML=`<div class="avatar">AI</div><div class="bubble">${formatText(text)}${source?`<div class="answer-source">Source: ${escapeHtml(source)}</div>`:''}${citeHtml}</div>`;
  chat.appendChild(wrap); chat.scrollTop=chat.scrollHeight;
}
function renderTrace(items=[]){trace.innerHTML=items.length?items.map(x=>`<div class="trace-item">${escapeHtml(x)}</div>`).join(''):'<div class="empty">No trace.</div>';}
async function askAgent(q){
  addMessage('user',q); question.value=''; renderTrace(['Running LangGraph workflow...']); sourceUsed.textContent='Running';
  const btn=form.querySelector('button'); btn.disabled=true;
  try{
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
    const data=await res.json(); if(!res.ok) throw new Error(data.detail||'Request failed');
    addMessage('assistant',data.answer,data.source_used,data.citations||[]); renderTrace(data.trace||[]); sourceUsed.textContent=data.source_used;
  }catch(e){addMessage('assistant',`Error: ${e.message}`); renderTrace(['Request failed']); sourceUsed.textContent='Error';}
  finally{btn.disabled=false;}
}
form.addEventListener('submit',e=>{e.preventDefault(); const q=question.value.trim(); if(q) askAgent(q);});
document.querySelectorAll('.example').forEach(b=>b.addEventListener('click',()=>askAgent(b.textContent.trim())));

const modal=document.getElementById('uploadModal');
document.getElementById('openUpload').onclick=()=>modal.classList.remove('hidden');
document.getElementById('closeUpload').onclick=()=>modal.classList.add('hidden');
document.getElementById('uploadBtn').onclick=async()=>{
  const file=document.getElementById('fileInput').files[0]; const key=document.getElementById('adminKey').value; const status=document.getElementById('uploadStatus');
  if(!file){status.textContent='Choose a file first.'; return;}
  status.textContent='Indexing document...'; const fd=new FormData(); fd.append('file',file);
  try{const r=await fetch('/api/ingest',{method:'POST',headers:{'X-Admin-Key':key},body:fd}); const d=await r.json(); if(!r.ok) throw new Error(d.detail||'Upload failed'); status.textContent=d.duplicate
      ? `Duplicate detected. Existing version ${d.version_number} was retained.`
      : `Indexed ${d.file}: ${d.chunks} chunks. Version ${d.version_number} (${d.version_id}).`;}catch(e){status.textContent=`Error: ${e.message}`;}
};

// Document intelligence workspace: version comparison, ANN verification and human review.
const workspaceModal = document.getElementById('workspaceModal');
const workspaceKey = document.getElementById('workspaceAdminKey');
const workspaceStatus = document.getElementById('workspaceStatus');
const documentList = document.getElementById('documentList');
const versionCache = new Map();
let lastVerificationRun = null;

function adminHeaders(){
  const key = workspaceKey.value || document.getElementById('adminKey').value;
  return {'X-Admin-Key': key, 'Content-Type':'application/json'};
}
async function apiRequest(url, options={}){
  const response = await fetch(url, {...options, headers:{...adminHeaders(), ...(options.headers||{})}});
  const data = await response.json().catch(()=>({}));
  if(!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}
function optionMarkup(versions, placeholder='Choose version'){
  return `<option value="">${placeholder}</option>` + versions.map(v=>`<option value="${escapeHtml(v.version_id)}">v${v.version_number} · ${escapeHtml(v.original_filename||'document')} · ${escapeHtml(v.uploaded_at||'')}</option>`).join('');
}
async function loadDocuments(){
  workspaceStatus.textContent='Loading registered documents...';
  documentList.innerHTML='<div class="empty-state">Loading...</div>';
  try{
    const data=await apiRequest('/api/documents');
    versionCache.clear();
    const docs=data.documents||[];
    for(const doc of docs){
      try{
        const result=await apiRequest(`/api/documents/${encodeURIComponent(doc.document_id)}/versions`);
        versionCache.set(doc.document_id,(result.versions||[]).filter(v=>v.ingestion_status==='complete'));
      }catch(e){versionCache.set(doc.document_id,[]);}
    }
    if(!docs.length){documentList.innerHTML='<div class="empty-state">No registered documents yet. Upload a document first.</div>';}
    else{
      documentList.innerHTML=docs.map(doc=>{
        const versions=versionCache.get(doc.document_id)||[];
        const active=versions[0]?.version_id||'';
        const disabled=doc.status==='retired'||!versions.length;
        return `<div class="document-row" data-document-id="${escapeHtml(doc.document_id)}"><div><label><input class="verify-document-check" type="checkbox" ${disabled?'disabled':''}> <strong>${escapeHtml(doc.logical_name)}</strong></label><small>${escapeHtml(doc.status)} · ${versions.length} complete version(s) · ${escapeHtml(doc.original_filename||'')}</small></div><select class="verify-version-select" aria-label="Version for ${escapeHtml(doc.logical_name)}" ${disabled?'disabled':''}>${optionMarkup(versions,'Choose version')}</select></div>`;
      }).join('');
      documentList.querySelectorAll('.document-row').forEach(row=>{
        const versions=versionCache.get(row.dataset.documentId)||[];
        const select=row.querySelector('.verify-version-select');
        if(versions.length) select.value=versions[0].version_id;
      });
    }
    const compare=document.getElementById('compareDocument');
    compare.innerHTML='<option value="">Choose a document</option>'+docs.filter(d=>d.status!=='retired'&&(versionCache.get(d.document_id)||[]).length>1).map(d=>`<option value="${escapeHtml(d.document_id)}">${escapeHtml(d.logical_name)}</option>`).join('');
    document.getElementById('oldVersion').innerHTML='<option value="">Choose version</option>';
    document.getElementById('newVersion').innerHTML='<option value="">Choose version</option>';
    workspaceStatus.textContent=`Loaded ${docs.length} document(s).`;
  }catch(e){workspaceStatus.textContent=`Error: ${e.message}`;documentList.innerHTML='<div class="empty-state">Could not load documents.</div>';}
}
document.getElementById('openWorkspace').addEventListener('click',()=>{workspaceModal.classList.remove('hidden');loadDocuments();});
document.getElementById('closeWorkspace').addEventListener('click',()=>workspaceModal.classList.add('hidden'));
document.getElementById('refreshDocuments').addEventListener('click',loadDocuments);
document.getElementById('compareDocument').addEventListener('change',e=>{
  const versions=versionCache.get(e.target.value)||[];
  const options=optionMarkup(versions);
  document.getElementById('oldVersion').innerHTML=options;
  document.getElementById('newVersion').innerHTML=options;
  if(versions.length>1){document.getElementById('oldVersion').value=versions[versions.length-1].version_id;document.getElementById('newVersion').value=versions[0].version_id;}
});
document.getElementById('compareBtn').addEventListener('click',async()=>{
  const old_version_id=document.getElementById('oldVersion').value;
  const new_version_id=document.getElementById('newVersion').value;
  const output=document.getElementById('comparisonResults');
  if(!old_version_id||!new_version_id){output.textContent='Choose two versions of the same document.';return;}
  output.textContent='Comparing versions...';
  try{
    const data=await apiRequest('/api/documents/compare',{method:'POST',body:JSON.stringify({old_version_id,new_version_id})});
    const counts=data.counts||{};
    output.innerHTML=`<div class="result-card"><strong>Version comparison</strong><p class="result-meta">${escapeHtml(data.old_version?.logical_name||'Document')} · ${escapeHtml(data.old_version?.version_id||'')} → ${escapeHtml(data.new_version?.version_id||'')}</p><p>Added: <b>${counts.ADDED||0}</b> · Deleted: <b>${counts.DELETED||0}</b> · Modified: <b>${counts.MODIFIED||0}</b> · Unchanged: <b>${counts.UNCHANGED||0}</b></p><p class="result-meta">Alignment: ${escapeHtml(data.alignment_method||'unknown')}. Textual changes are not automatically policy contradictions.</p></div>`+(data.changes||[]).map(change=>`<div class="result-card"><span class="result-label">${escapeHtml(change.change_type)}</span><p><b>Old:</b> ${escapeHtml(change.old_text||'—')}</p><p><b>New:</b> ${escapeHtml(change.new_text||'—')}</p><p class="result-meta">${escapeHtml(change.old_location||'')} → ${escapeHtml(change.new_location||'')} · similarity ${change.similarity??'—'}</p></div>`).join('');
  }catch(e){output.textContent=`Comparison failed: ${e.message}`;}
});
document.getElementById('verifyBtn').addEventListener('click',async()=>{
  const version_ids=[];
  documentList.querySelectorAll('.document-row').forEach(row=>{
    if(row.querySelector('.verify-document-check')?.checked){const id=row.querySelector('.verify-version-select')?.value;if(id)version_ids.push(id);}
  });
  const output=document.getElementById('verificationResults');
  if(version_ids.length<2){output.textContent='Select at least two different documents and a version for each.';return;}
  output.textContent='Running ANN verification...';
  try{
    const report=await apiRequest('/api/verification/runs',{method:'POST',body:JSON.stringify({version_ids})});
    lastVerificationRun=report.run_id;
    renderVerificationReport(report,output);
  }catch(e){output.textContent=`Verification failed: ${e.message}`;}
});
function renderVerificationReport(report,output){
  const summary=report.summary||{};
  const header=`<div class="result-card"><strong>Verification run ${escapeHtml(report.run_id||'')}</strong><p class="result-meta">Status: ${escapeHtml(report.status||'')} · Model: ${escapeHtml(report.model_version||'')} · Evaluated pairs: ${summary.evaluated_pairs??0} / ${summary.candidate_pairs??0}</p><p>Contradictions: ${summary.label_counts?.CONTRADICTION||0} · Consistent: ${summary.label_counts?.CONSISTENT||0} · Uncertain: ${summary.label_counts?.UNCERTAIN||0}</p><p class="result-meta">${escapeHtml(summary.authority_warning||'ANN results are review-support signals, not authoritative policy decisions.')}</p></div>`;
  const cards=(report.findings||[]).map(f=>{
    const ann=f.ann||{};const probs=ann.probabilities||{};
    return `<div class="result-card"><span class="result-label">ANN: ${escapeHtml(ann.predicted_label||'UNKNOWN')}</span><span class="result-label">Confidence: ${Number(ann.confidence||0).toFixed(3)}</span><p><b>A:</b> ${escapeHtml(f.statement_a||'')}</p><p class="result-meta">${escapeHtml(f.source_a?.logical_name||'')} · ${escapeHtml(f.source_a?.location||'')} · v${escapeHtml(String(f.source_a?.version_number??''))}</p><p><b>B:</b> ${escapeHtml(f.statement_b||'')}</p><p class="result-meta">${escapeHtml(f.source_b?.logical_name||'')} · ${escapeHtml(f.source_b?.location||'')} · v${escapeHtml(String(f.source_b?.version_number??''))}</p><p class="result-meta">Probabilities: ${Object.entries(probs).map(([k,v])=>`${escapeHtml(k)} ${Number(v).toFixed(3)}`).join(' · ')} · ${f.review_status==='reviewed'?`Reviewed: ${escapeHtml(f.review_label||'')}`:'Human review: pending'}</p>${f.review_status==='reviewed'?'':`<div class="review-controls" data-pair-id="${escapeHtml(f.pair_id)}"><select class="review-label"><option>CONTRADICTION</option><option>CONSISTENT</option><option>UNCERTAIN</option></select><input class="reviewer-name" placeholder="Reviewer identifier" maxlength="128"><input class="review-notes" placeholder="Review notes (optional)" maxlength="2000"><button class="submit-review">Save review</button></div>`}</div>`;
  }).join('');
  output.innerHTML=header+(cards||'<div class="empty-state">No candidate statement pairs were produced.</div>');
}
document.getElementById('verificationResults').addEventListener('click',async e=>{
  const button=e.target.closest('.submit-review');if(!button)return;
  const controls=button.closest('.review-controls');const pairId=controls.dataset.pairId;
  const reviewer=controls.querySelector('.reviewer-name').value.trim();
  const label=controls.querySelector('.review-label').value;
  const notes=controls.querySelector('.review-notes').value;
  if(!reviewer){alert('Enter a reviewer identifier.');return;}
  button.disabled=true;
  try{
    await apiRequest(`/api/verification/pairs/${encodeURIComponent(pairId)}/review`,{method:'POST',body:JSON.stringify({reviewer,label,notes})});
    const report=await apiRequest(`/api/verification/runs/${encodeURIComponent(lastVerificationRun)}`);
    renderVerificationReport(report,document.getElementById('verificationResults'));
  }catch(err){alert(`Could not save review: ${err.message}`);button.disabled=false;}
});
