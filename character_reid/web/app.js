const state = {
  jobId: null, result: null, page: 0, image: null,
  selectedDialogue: null, selectedCluster: null,
  selectedClusters: new Set(), polling: null,
  phaseStartedAt: null, accumulatedMs: 0, timer: null,
};
const $ = id => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

function toast(message) {
  const node = $('toast');
  node.textContent = message;
  node.classList.add('show');
  setTimeout(() => node.classList.remove('show'), 2400);
}
function setStatus(status, text) {
  const pill = $('status-pill');
  pill.className = `status-pill ${status}`;
  pill.innerHTML = `<span></span>${text}`;
}
function formatElapsed(milliseconds) {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  return hours ? `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(rest).padStart(2, '0')}` : `${String(minutes).padStart(2, '0')}:${String(rest).padStart(2, '0')}`;
}
function renderElapsed() {
  const running = state.phaseStartedAt ? Date.now() - state.phaseStartedAt : 0;
  $('elapsed-time').textContent = formatElapsed(state.accumulatedMs + running);
}
function startTimer(reset = false) {
  if (reset) state.accumulatedMs = 0;
  if (!state.phaseStartedAt) state.phaseStartedAt = Date.now();
  clearInterval(state.timer);
  state.timer = setInterval(renderElapsed, 1000);
  renderElapsed();
}
function pauseTimer() {
  if (state.phaseStartedAt) state.accumulatedMs += Date.now() - state.phaseStartedAt;
  state.phaseStartedAt = null;
  clearInterval(state.timer);
  renderElapsed();
}
function updateProgress(progress, stage) {
  $('analysis-progress').classList.remove('hidden');
  document.body.classList.add('has-progress');
  $('progress-bar').style.width = `${Math.max(0, Math.min(100, Number(progress) || 0))}%`;
  $('progress-stage').textContent = stage || '处理中';
}
function fileUrl(path) {
  return `/files/${state.jobId}/${path.split('/').map(encodeURIComponent).join('/')}`;
}
function escapeHtml(value = '') {
  const node = document.createElement('div');
  node.textContent = value;
  return node.innerHTML;
}
function isReview() { return state.result?.stage === 'cluster_review'; }

async function responseBody(response) {
  const text = await response.text();
  if (!text) return {};
  try { return JSON.parse(text); }
  catch (_) { return { error: text.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim() || `HTTP ${response.status}` }; }
}

async function submitJob(event) {
  event.preventDefault();
  const files = $('pages').files;
  if (!files.length) return toast('请先选择漫画图片');
  const data = new FormData();
  [...files].forEach(file => data.append('pages', file));
  if ($('detections').files[0]) data.append('detections', $('detections').files[0]);
  data.append('tails_enabled', $('tails-enabled').checked ? '1' : '0');
  data.append('vlm_enabled', $('vlm-enabled').checked ? '1' : '0');
  data.append('vlm_model', $('vlm-model').value.trim());
  data.append('vlm_first_pass_threshold', $('vlm-first-pass-threshold').value);
  data.append('vlm_identity_threshold', $('vlm-identity-threshold').value);
  $('run-button').disabled = true;
  setStatus('running', '正在上传');
  startTimer(true);
  updateProgress(1, '正在上传漫画页面');
  try {
    const response = await fetch('/api/jobs', { method: 'POST', body: data });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || '上传失败');
    state.jobId = body.job_id;
    state.result = null;
    state.page = 0;
    state.selectedCluster = null;
    state.selectedClusters.clear();
    $('clusters').innerHTML = '';
    $('dialogues').innerHTML = '';
    $('save-names').disabled = true;
    $('merge-clusters').disabled = true;
    $('confirm-review').disabled = true;
    $('workspace').classList.remove('empty');
    pollJob();
  } catch (error) {
    pauseTimer();
    setStatus('failed', '启动失败');
    toast(error.message);
    $('run-button').disabled = false;
  }
}

async function pollJob() {
  clearTimeout(state.polling);
  const response = await fetch(`/api/jobs/${state.jobId}`);
  const job = await response.json();
  setStatus(job.status, job.stage || job.status);
  updateProgress(job.progress, job.message || job.stage);
  $('page-label').textContent = job.message || '';
  if (job.status === 'review') {
    pauseTimer();
    $('run-button').disabled = false;
    await loadResult();
    toast('角色聚类完成，请先审核角色簇');
    return;
  }
  if (job.status === 'completed') {
    pauseTimer();
    $('run-button').disabled = false;
    await loadResult();
    return;
  }
  if (job.status === 'failed') {
    pauseTimer();
    $('run-button').disabled = false;
    toast(job.message);
    await showLog();
    return;
  }
  state.polling = setTimeout(pollJob, 1500);
}

async function loadResult() {
  state.result = await (await fetch(`/api/jobs/${state.jobId}/result`)).json();
  state.page = Math.max(0, Math.min(state.page, (state.result.pages || []).length - 1));
  $('workspace').classList.toggle('final-stage', !isReview());
  $('save-names').disabled = !isReview();
  $('merge-clusters').disabled = !isReview() || state.selectedClusters.size < 2;
  $('confirm-review').disabled = !isReview();
  $('review-tip').textContent = isReview()
    ? '先删除误检、合并同一人物并为每个角色命名'
    : '角色库已确认，对白已经映射为角色名';
  renderClusters();
  renderThumbs();
  renderPage();
  const summary = state.result.summary || {};
  if (isReview()) {
    setStatus('review', `${summary.character_clusters} 个角色簇待审核`);
    updateProgress(65, '聚类完成，等待人工整理角色库');
  } else {
    setStatus('completed', `${summary.character_clusters} 个角色 · ${summary.dialogues} 条对白`);
    updateProgress(100, '全部人物检索与对白角色映射完成');
  }
}

function clusterMap() {
  const map = new Map();
  for (const item of state.result?.character_instances || []) {
    if (item.excluded) continue;
    if (!map.has(item.character_cluster_id)) {
      map.set(item.character_cluster_id, {
        id: item.character_cluster_id,
        name: item.character_name,
        items: [],
      });
    }
    map.get(item.character_cluster_id).items.push(item);
  }
  return map;
}

function renderClusters() {
  const groups = [...clusterMap().values()];
  const roleGroups = groups.filter(group => group.id !== 'unassigned');
  $('cluster-count').textContent = `${roleGroups.length} 个角色 · ${groups.reduce((n, g) => n + g.items.length, 0)} 个实例`;
  $('clusters').innerHTML = groups.map((group, groupIndex) => {
    const unassigned = group.id === 'unassigned';
    const shots = group.items.map(item => `
      <div class="review-shot" ${isReview() ? `draggable="true" data-drag-instance="${item.instance_id}"` : ''}>
        <img src="${fileUrl(`result/review_crops/${item.instance_id}.jpg`)}" alt="${escapeHtml(group.name)}">
        ${isReview() && !unassigned ? `<button class="delete-shot" data-delete-instance="${item.instance_id}" title="移出参考库，稍后自动检索">−</button>` : ''}
      </div>`).join('');
    const selected = state.selectedClusters.has(group.id);
    const reviewLabel = unassigned ? '待检索实例' : `角色簇 ${String(roleGroups.indexOf(group) + 1).padStart(2, '0')}`;
    return `<div class="cluster-card ${unassigned ? 'unassigned-card' : ''} ${state.selectedCluster === group.id ? 'active' : ''}" data-cluster="${group.id}">
      <div class="cluster-head">
        ${isReview() && !unassigned ? `<input class="cluster-select" type="checkbox" data-select-cluster="${group.id}" ${selected ? 'checked' : ''}>` : ''}
        <span class="cluster-id">${reviewLabel}</span>
        <input class="cluster-name" data-name="${group.id}" value="${escapeHtml(group.name)}" ${isReview() && !unassigned ? '' : 'readonly'}>
      </div>
      <div class="cluster-shots">${shots}</div>
      <div class="cluster-meta">${group.items.length} 个画面实例</div>
    </div>`;
  }).join('');

  document.querySelectorAll('.cluster-card').forEach(card => card.addEventListener('click', event => {
    if (['INPUT', 'BUTTON'].includes(event.target.tagName)) return;
    state.selectedCluster = state.selectedCluster === card.dataset.cluster ? null : card.dataset.cluster;
    renderClusters();
    drawCanvas();
  }));
  document.querySelectorAll('[data-select-cluster]').forEach(input => input.addEventListener('change', () => {
    input.checked ? state.selectedClusters.add(input.dataset.selectCluster) : state.selectedClusters.delete(input.dataset.selectCluster);
    $('merge-clusters').disabled = state.selectedClusters.size < 2;
  }));
  document.querySelectorAll('[data-delete-instance]').forEach(button => button.addEventListener('click', async event => {
    event.stopPropagation();
    await deleteInstance(button.dataset.deleteInstance);
  }));
  if (isReview()) setupDragAndDrop();
}

function setupDragAndDrop() {
  document.querySelectorAll('[data-drag-instance]').forEach(shot => {
    shot.addEventListener('dragstart', event => {
      event.dataTransfer.setData('text/plain', shot.dataset.dragInstance);
      event.dataTransfer.effectAllowed = 'move';
      shot.classList.add('dragging');
    });
    shot.addEventListener('dragend', () => shot.classList.remove('dragging'));
  });
  document.querySelectorAll('.cluster-card').forEach(card => {
    card.addEventListener('dragover', event => { event.preventDefault(); card.classList.add('drag-over'); });
    card.addEventListener('dragleave', () => card.classList.remove('drag-over'));
    card.addEventListener('drop', async event => {
      event.preventDefault(); card.classList.remove('drag-over');
      const instanceId = event.dataTransfer.getData('text/plain');
      if (instanceId && card.dataset.cluster !== 'unassigned') await moveInstance(instanceId, card.dataset.cluster);
    });
  });
  const newRole = $('new-role-dropzone');
  newRole.ondragover = event => { event.preventDefault(); newRole.classList.add('drag-over'); };
  newRole.ondragleave = () => newRole.classList.remove('drag-over');
  newRole.ondrop = async event => {
    event.preventDefault(); newRole.classList.remove('drag-over');
    const instanceId = event.dataTransfer.getData('text/plain');
    if (instanceId) await moveInstance(instanceId, '__new__');
  };
}

function renderThumbs() {
  $('thumbs').innerHTML = (state.result.pages || []).map((page, index) => `
    <button class="page-thumb ${index === state.page ? 'active' : ''}" data-page="${index}">
      <img src="${fileUrl(`input/${page.image}`)}"><span>${index + 1}</span>
    </button>`).join('');
  document.querySelectorAll('.page-thumb').forEach(node => node.onclick = () => {
    state.page = Number(node.dataset.page);
    state.selectedDialogue = null;
    renderThumbs();
    renderPage();
  });
}

function pageInstances(page) {
  return (state.result.character_instances || []).filter(item => item.image === page.image && !item.excluded);
}
function renderPage() {
  const pages = state.result.pages || [];
  if (!pages.length) return;
  const page = pages[state.page];
  $('page-label').textContent = `${page.image} · ${state.page + 1} / ${pages.length}`;
  $('prev-page').disabled = state.page === 0;
  $('next-page').disabled = state.page === pages.length - 1;
  const image = new Image();
  image.onload = () => {
    state.image = image;
    $('viewer').querySelector('.welcome').style.display = 'none';
    canvas.style.display = 'block';
    resizeCanvas();
  };
  image.src = fileUrl(`input/${page.image}`);
  renderDialogues(page);
}
function resizeCanvas() {
  if (!state.image) return;
  const viewer = $('viewer');
  const scale = Math.min((viewer.clientWidth - 36) / state.image.width, (viewer.clientHeight - 36) / state.image.height, 1);
  canvas.width = Math.max(1, Math.round(state.image.width * scale));
  canvas.height = Math.max(1, Math.round(state.image.height * scale));
  drawCanvas();
}
function boxCenter(box, sx, sy) { return [((box[0] + box[2]) / 2) * sx, ((box[1] + box[3]) / 2) * sy]; }
function strokeBox(box, color, width, sx, sy) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.strokeRect(box[0] * sx, box[1] * sy, (box[2] - box[0]) * sx, (box[3] - box[1]) * sy);
}
function drawCanvas() {
  if (!state.image || !state.result) return;
  const page = state.result.pages[state.page];
  const sx = canvas.width / state.image.width, sy = canvas.height / state.image.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  ctx.font = '600 12px sans-serif';
  for (const item of pageInstances(page)) {
    const active = !state.selectedCluster || item.character_cluster_id === state.selectedCluster;
    strokeBox(item.body_box, active ? '#2878d0' : '#91a6bc', active ? 2.4 : 1, sx, sy);
    if (active) {
      ctx.fillStyle = '#145a9b';
      ctx.fillText(item.character_name, item.body_box[0] * sx, Math.max(12, item.body_box[1] * sy - 3));
    }
  }
  if (isReview()) return;
  for (const [dialogueIndex, dialogue] of (page.dialogues || []).entries()) {
    const active = !state.selectedDialogue || state.selectedDialogue === dialogue.dialogue_id;
    const candidate = dialogue.top_candidates?.find(row => row.character_cluster_id === dialogue.character_cluster_id) || dialogue.top_candidates?.[0];
    if (!candidate) continue;
    strokeBox(dialogue.text_box, active ? '#e13b3b' : '#c58c8c', active ? 2.5 : 1, sx, sy);
    const [tx, ty] = boxCenter(dialogue.text_box, sx, sy), [bx, by] = boxCenter(candidate.body_box, sx, sy);
    ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = active ? 2.8 : 1.2;
    ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(bx, by); ctx.stroke();
    ctx.fillStyle = '#b32323';
    const displayId = dialogue.display_id || `D${dialogueIndex + 1}`;
    ctx.fillText(`${displayId} → ${dialogue.character_name}`, dialogue.text_box[0] * sx, Math.max(12, dialogue.text_box[1] * sy - 4));
    const tail = dialogue.tail_evidence;
    if (tail) {
      strokeBox(tail.tail_box, '#a32ac1', 2, sx, sy);
      const tip = tail.estimated_tail_tip, direction = tail.ray_direction, length = tail.ray_panel_exit;
      ctx.strokeStyle = '#a32ac1'; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(tip[0] * sx, tip[1] * sy);
      ctx.lineTo((tip[0] + direction[0] * length) * sx, (tip[1] + direction[1] * length) * sy); ctx.stroke();
    }
  }
}

function renderDialogues(page) {
  if (isReview()) {
    $('dialogue-count').textContent = '等待确认角色库';
    $('dialogues').innerHTML = '<div class="ocr">对白分析将在角色簇审核完成后运行。先在左侧删除误检、合并同一角色并填写角色名。</div>';
    return;
  }
  const groups = [...clusterMap().values()];
  $('dialogue-count').textContent = `${page.dialogues?.length || 0} 条对白`;
  $('dialogues').innerHTML = (page.dialogues || []).map((dialogue, dialogueIndex) => {
    const isUnknown = dialogue.character_cluster_id === 'unknown' || dialogue.character_name === 'unknown';
    const options = `<option value="" ${isUnknown ? 'selected' : ''} disabled>unknown</option>` + groups.map(group => `<option value="${group.id}" ${group.id === dialogue.character_cluster_id ? 'selected' : ''}>${escapeHtml(group.name)}</option>`).join('');
    const source = dialogue.speaker_source === 'gemini_panel_speaker' ? ['Gemini VLM', 'vlm'] : dialogue.speaker_source === 'v3_tail_fusion' ? ['尾巴 + V3', ''] : dialogue.speaker_source === 'manual_override' ? ['人工修正', 'manual'] : ['V3', 'fallback'];
    const confidence = dialogue.top_candidates?.[0]?.softmax_share;
    const speakerReview = dialogue.speaker_vlm_top5;
    const identityReview = dialogue.identity_vlm_top5;
    const vlmDetails = speakerReview || identityReview ? `<div class="vlm-details">${escapeHtml(`VLM speaker: ${speakerReview?.status || 'not reviewed'}${speakerReview?.confidence != null ? ` (${Math.round(speakerReview.confidence * 100)}%)` : ''}${speakerReview?.reason ? ` — ${speakerReview.reason}` : ''}`)}<br>${escapeHtml(`VLM identity: ${identityReview?.status || 'not reviewed'}${identityReview?.confidence != null ? ` (${Math.round(identityReview.confidence * 100)}%)` : ''}${identityReview?.reason ? ` — ${identityReview.reason}` : ''}`)}</div>` : '';
    const displayId = dialogue.display_id || `D${dialogueIndex + 1}`;
    return `<article class="dialogue-card ${state.selectedDialogue === dialogue.dialogue_id ? 'active' : ''}" data-dialogue="${dialogue.dialogue_id}">
      <div class="dialogue-top"><span class="dialogue-id">${displayId}</span><span class="source ${source[1]}">${source[0]}</span></div>
      <div class="ocr">${escapeHtml(dialogue.ocr_text || '（未读取到文字）')}</div>
      <div class="speaker-row"><label>说话角色</label><select data-dialogue-select="${dialogue.dialogue_id}">${options}</select><span class="confidence">${confidence != null ? `${Math.round(confidence * 100)}%` : ''}</span></div>${vlmDetails}
    </article>`;
  }).join('');
  document.querySelectorAll('.dialogue-card').forEach(card => card.addEventListener('click', event => {
    if (event.target.tagName === 'SELECT') return;
    state.selectedDialogue = state.selectedDialogue === card.dataset.dialogue ? null : card.dataset.dialogue;
    renderDialogues(page); drawCanvas();
  }));
  document.querySelectorAll('[data-dialogue-select]').forEach(select => select.onchange = () => {
    if (select.value) overrideDialogue(select.dataset.dialogueSelect, select.value);
  });
  if (state.selectedDialogue) {
    requestAnimationFrame(() => document.querySelector(`[data-dialogue="${state.selectedDialogue}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }));
  }
}

function selectDialogueFromCanvas(event) {
  if (!state.result || isReview()) return;
  const page = state.result.pages[state.page];
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * state.image.width / rect.width;
  const y = (event.clientY - rect.top) * state.image.height / rect.height;
  const matches = (page.dialogues || []).filter(dialogue => {
    const box = dialogue.text_box;
    return box[0] <= x && x <= box[2] && box[1] <= y && y <= box[3];
  }).sort((left, right) => {
    const leftArea = (left.text_box[2] - left.text_box[0]) * (left.text_box[3] - left.text_box[1]);
    const rightArea = (right.text_box[2] - right.text_box[0]) * (right.text_box[3] - right.text_box[1]);
    return leftArea - rightArea;
  });
  if (!matches.length) return;
  state.selectedDialogue = matches[0].dialogue_id;
  renderDialogues(page);
  drawCanvas();
}

async function saveNames(showToast = true) {
  const names = {};
  document.querySelectorAll('[data-name]').forEach(input => names[input.dataset.name] = input.value.trim());
  const response = await fetch(`/api/jobs/${state.jobId}/names`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ names }),
  });
  const body = await responseBody(response);
  if (!response.ok) { toast(body.error || `保存失败（HTTP ${response.status}）`); return false; }
  await loadResult();
  if (showToast) toast('角色命名已保存');
  return true;
}
async function deleteInstance(instanceId) {
  const response = await fetch(`/api/jobs/${state.jobId}/review/delete-instance`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ instance_id: instanceId }),
  });
  const body = await responseBody(response);
  if (!response.ok) return toast(body.error || `删除失败（HTTP ${response.status}）`);
  await loadResult(); toast('已移出参考库；确认后仍会自动检索角色');
}
async function moveInstance(instanceId, targetClusterId) {
  const response = await fetch(`/api/jobs/${state.jobId}/review/move-instance`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ instance_id: instanceId, target_cluster_id: targetClusterId }),
  });
  const body = await responseBody(response);
  if (!response.ok) return toast(body.error || '移动失败');
  await loadResult(); toast(targetClusterId === '__new__' ? '已创建新角色' : '人物参考图已移动');
}
async function mergeClusters() {
  const clusterIds = [...state.selectedClusters];
  if (clusterIds.length < 2) return toast('请至少选择两个角色簇');
  const response = await fetch(`/api/jobs/${state.jobId}/review/merge`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cluster_ids: clusterIds }),
  });
  const body = await responseBody(response);
  if (!response.ok) return toast(body.error || '合并失败');
  state.selectedClusters.clear(); await loadResult(); toast('角色簇已合并');
}
async function confirmReview() {
  if (!await saveNames(false)) return;
  const response = await fetch(`/api/jobs/${state.jobId}/review/confirm`, { method: 'POST' });
  const body = await responseBody(response);
  if (!response.ok) return toast(body.error || '无法确认角色库');
  $('confirm-review').disabled = true;
  setStatus('running', '正在分析对白人物');
  startTimer(false);
  updateProgress(70, '正在建立角色原型并检索全部人物');
  pollJob();
}
async function overrideDialogue(dialogueId, clusterId) {
  const group = clusterMap().get(clusterId);
  if (!group) return;
  await fetch(`/api/jobs/${state.jobId}/dialogues/${dialogueId}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ character_cluster_id: clusterId, character_name: group.name }),
  });
  await loadResult(); toast('对白角色已修正');
}
async function showLog() {
  if (!state.jobId) return toast('还没有运行任务');
  const log = $('log');
  if (!log.classList.contains('hidden')) {
    log.classList.add('hidden'); $('dialogues').classList.remove('hidden'); return;
  }
  const body = await (await fetch(`/api/jobs/${state.jobId}/log`)).json();
  log.textContent = body.log; log.classList.remove('hidden'); $('dialogues').classList.add('hidden'); log.scrollTop = log.scrollHeight;
}

$('upload-form').addEventListener('submit', submitJob);
$('save-names').onclick = () => saveNames(true);
$('merge-clusters').onclick = mergeClusters;
$('confirm-review').onclick = confirmReview;
$('toggle-log').onclick = showLog;
canvas.addEventListener('click', selectDialogueFromCanvas);
$('prev-page').onclick = () => { if (state.page > 0) { state.page--; renderThumbs(); renderPage(); } };
$('next-page').onclick = () => { if (state.page < state.result.pages.length - 1) { state.page++; renderThumbs(); renderPage(); } };
$('welcome-files').onchange = event => {
  const transfer = new DataTransfer();
  [...event.target.files].forEach(file => transfer.items.add(file));
  $('pages').files = transfer.files;
  toast(`已选择 ${transfer.files.length} 张图片，点击“开始分析”`);
};
window.addEventListener('resize', () => requestAnimationFrame(resizeCanvas));
