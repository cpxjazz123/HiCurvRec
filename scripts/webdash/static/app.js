// ===========================================
// GeneRec Web Dashboard - Frontend (vanilla JS)
// ===========================================

const state = {
  view: 'dashboard',
  tasks: [],
  tasksFiltered: [],
  verdicts: [],
  verdictsFiltered: [],
  loop: null,
};

// ---------- API ----------
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

function fmtTime(iso) {
  if (!iso) return '--';
  const d = new Date(iso);
  return d.toLocaleString('zh-CN', { hour12: false });
}

function ago(iso) {
  if (!iso) return '';
  const ms = Date.now() - new Date(iso).getTime();
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s 前`;
  if (s < 3600) return `${Math.floor(s/60)}m 前`;
  if (s < 86400) return `${Math.floor(s/3600)}h 前`;
  return `${Math.floor(s/86400)}d 前`;
}

function statusBadge(status) {
  const cls = ['GO', 'PASS'].includes(status) ? 'GO' :
              ['NO-GO', 'FAIL'].includes(status) ? 'NO-GO' : 'UNKNOWN';
  return `<span class="verdict-tag ${cls}">${status}</span>`;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ---------- Navigation ----------
function switchView(name) {
  state.view = name;
  document.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === name));
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === `view-${name}`));
  // 按需加载
  if (name === 'dashboard') loadDashboard();
  else if (name === 'tasks') loadTasks();
  else if (name === 'verdicts') loadVerdicts();
  else if (name === 'loop') loadLoop();
  else if (name === 'rules') loadRules();
}

document.querySelectorAll('.nav-item').forEach(n => {
  n.addEventListener('click', () => switchView(n.dataset.view));
});

// ---------- Dashboard ----------
async function loadDashboard() {
  const data = await api('/api/overview');
  // 统计卡片
  const cards = [
    { label: '任务总数', value: data.task_total, color: '', sub: 'tasks/ 目录' },
    { label: 'Verdicts', value: data.verdict_total, color: 'purple', sub: 'verdicts/ 目录' },
    { label: 'GO 数量', value: data.status_counts.GO || 0, color: 'green', sub: '判定通过' },
    { label: 'NO-GO 数量', value: data.status_counts['NO-GO'] || 0, color: 'red', sub: '判定不通过' },
  ];
  document.getElementById('dash-stats').innerHTML = cards.map(c => `
    <div class="stat-card ${c.color}">
      <div class="stat-label">${c.label}</div>
      <div class="stat-value">${c.value}</div>
      <div class="stat-sub">${c.sub}</div>
    </div>
  `).join('');

  // 当前活跃迭代
  const activeTbl = data.active_iterations.length ? `
    <table>
      <thead><tr><th>版本</th><th>关键改动</th><th>test_R@10</th><th>valid_R@10</th></tr></thead>
      <tbody>
        ${data.active_iterations.map(r => `
          <tr>
            <td class="mono">${escapeHtml(r.version)}</td>
            <td>${escapeHtml(r.change)}</td>
            <td class="mono">${escapeHtml(r.test_r10 || '—')}</td>
            <td class="mono">${escapeHtml(r.valid_r10 || '—')}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  ` : '<div class="empty">loop.md 无当前迭代表</div>';
  document.getElementById('dash-active').innerHTML = activeTbl;

  // 最近 verdict
  document.getElementById('dash-recent-verdicts').innerHTML = data.recent_verdicts.length
    ? data.recent_verdicts.map(v => `
        <div class="list-item" data-iid="${v.iid}">
          <span class="iid">#${v.iid}</span>
          ${statusBadge(v.agg_status)}
          <span class="name">${escapeHtml(v.top_file || '')}</span>
          <span class="ts">${ago(new Date(v.mtime * 1000).toISOString())}</span>
        </div>
      `).join('')
    : '<div class="empty">暂无 verdict</div>';

  document.getElementById('dash-recent-verdicts').querySelectorAll('.list-item').forEach(el => {
    el.addEventListener('click', () => openVerdict(el.dataset.iid));
  });

  // Stage 覆盖率
  const total = data.task_total || 1;
  document.getElementById('dash-stage-coverage').innerHTML = Object.entries(data.stage_coverage).map(([k, v]) => {
    const pct = Math.round(v / total * 100);
    return `
      <div class="bar-row">
        <div class="bar-label">${k}</div>
        <div class="bar-track">
          <div class="bar-fill" style="width: ${pct}%">${v} / ${total} (${pct}%)</div>
        </div>
      </div>
    `;
  }).join('');

  // badge
  document.getElementById('badge-tasks').textContent = data.task_total;
  document.getElementById('badge-verdicts').textContent = data.verdict_total;

  document.getElementById('ts-now').textContent = '刷新于 ' + fmtTime(data.generated_at);
}

// ---------- Tasks ----------
async function loadTasks() {
  const data = await api('/api/tasks');
  state.tasks = data.tasks;
  state.tasksFiltered = data.tasks;
  renderTasks();
  document.getElementById('badge-tasks').textContent = data.tasks.length;
}

function renderTasks() {
  const q = document.getElementById('task-search').value.toLowerCase();
  state.tasksFiltered = state.tasks.filter(t => t.name.toLowerCase().includes(q));
  const grid = document.getElementById('task-grid');
  if (!state.tasksFiltered.length) {
    grid.innerHTML = '<div class="empty" style="grid-column: 1/-1">无匹配任务</div>';
    return;
  }
  grid.innerHTML = state.tasksFiltered.map(t => {
    const stagePills = ['stage1', 'stage2', 'stage3', 'stage4'].map(s => {
      const exists = t.stages[s]?.exists;
      return `<div class="stage-pill ${exists ? 'exists' : ''}">${s.replace('stage', '')}</div>`;
    }).join('');
    const vTags = t.related_verdicts.map(v => statusBadge(v.status)).join(' ');
    return `
      <div class="task-card" data-name="${escapeHtml(t.name)}">
        <div class="task-card-header">
          <div class="task-name">${escapeHtml(t.name)}</div>
        </div>
        <div class="task-stages">${stagePills}</div>
        <div>${vTags || '<span class="ts" style="font-size:11px">无关联 verdict</span>'}</div>
        <div class="task-meta">
          <span>${t.stage_count}/4 stage</span>
          <span>${ago(t.mtime_iso)}</span>
        </div>
      </div>
    `;
  }).join('');
  grid.querySelectorAll('.task-card').forEach(el => {
    el.addEventListener('click', () => openTask(el.dataset.name));
  });
}

document.getElementById('task-search').addEventListener('input', renderTasks);
document.getElementById('btn-refresh-tasks').addEventListener('click', loadTasks);
document.getElementById('btn-refresh-dash').addEventListener('click', loadDashboard);

// ---------- Task Detail Panel ----------
async function openTask(name) {
  document.getElementById('task-panel-title').textContent = name;
  document.getElementById('task-panel-body').innerHTML = '<div class="empty loading">加载</div>';
  document.getElementById('task-panel-backdrop').classList.add('show');
  document.getElementById('task-panel').classList.add('show');

  const t = await api(`/api/tasks/${encodeURIComponent(name)}`);

  const stageHtml = ['stage1', 'stage2', 'stage3', 'stage4'].map(s => {
    const st = t.stages[s];
    if (!st.exists) {
      return `<div class="section-title">${s}</div><div class="empty">不存在</div>`;
    }
    return `
      <div class="section-title">${s} — <span class="script-name">${escapeHtml(st.file)}</span></div>
      <div class="script-meta">
        <span>${st.line_count} 行 · ${(st.size/1024).toFixed(1)} KB</span>
        <span>${fmtTime(new Date(st.mtime*1000).toISOString())}</span>
      </div>
      <pre class="script-block" data-stage="${s}">${escapeHtml(st.head)}…
<span style="color: var(--text-faint)">… (只显示前 25 行, 点击展开查看完整脚本)</span></pre>
      <button class="btn view-full" data-task="${escapeHtml(t.name)}" data-stage="${s}">📄 查看完整脚本</button>
    `;
  }).join('');

  const relatedHtml = t.related_verdicts.length ? `
    <div class="section-title">关联 Verdict</div>
    <div class="related-list">
      ${t.related_verdicts.map(v => `
        <div class="related-item" data-iid="${v.iid}">
          <span class="riid">#${v.iid}</span>
          ${statusBadge(v.status)}
          <span class="rname">${v.file ? '' : ''}</span>
          <span class="ts">${v.r10.test_r10 ? 'R@10=' + v.r10.test_r10 : ''}</span>
        </div>
      `).join('')}
    </div>
  ` : '';

  document.getElementById('task-panel-body').innerHTML = `
    <div class="script-meta">
      <span>路径: <code>${escapeHtml(t.path)}</code></span>
      <span>${t.stage_count}/4 stage 完成 · ${fmtTime(t.mtime_iso)}</span>
    </div>
    ${stageHtml}
    ${relatedHtml}
  `;

  // 关联 verdict 点击
  document.querySelectorAll('#task-panel-body .related-item').forEach(el => {
    el.addEventListener('click', () => {
      closeTaskPanel();
      openVerdict(el.dataset.iid);
    });
  });

  // 查看完整脚本
  document.querySelectorAll('.view-full').forEach(btn => {
    btn.addEventListener('click', () => viewFullScript(btn.dataset.task, btn.dataset.stage));
  });
}

function viewFullScript(task, stage) {
  api(`/api/tasks/${encodeURIComponent(task)}/script/${stage}`).then(d => {
    const pre = document.querySelector(`#task-panel-body .script-block[data-stage="${stage}"]`);
    if (pre) {
      pre.textContent = d.content;
      pre.style.maxHeight = '60vh';
    }
  });
}

function closeTaskPanel() {
  document.getElementById('task-panel').classList.remove('show');
  document.getElementById('task-panel-backdrop').classList.remove('show');
}
document.getElementById('task-panel-close').addEventListener('click', closeTaskPanel);
document.getElementById('task-panel-backdrop').addEventListener('click', closeTaskPanel);

// ---------- Verdicts ----------
async function loadVerdicts() {
  const data = await api('/api/verdicts');
  state.verdicts = data.verdicts;
  state.verdictsFiltered = data.verdicts;
  renderVerdicts();
  document.getElementById('badge-verdicts').textContent = data.verdicts.length;
}

function renderVerdicts() {
  const q = document.getElementById('verdict-search').value.toLowerCase();
  state.verdictsFiltered = state.verdicts.filter(v => {
    const haystack = `${v.iid} ${v.files.map(f => f.name).join(' ')}`.toLowerCase();
    return haystack.includes(q);
  });
  const list = document.getElementById('verdict-list');
  if (!state.verdictsFiltered.length) {
    list.innerHTML = '<div class="empty" style="grid-column: 1/-1">无匹配</div>';
    return;
  }
  list.innerHTML = state.verdictsFiltered.map(v => {
    const topR10 = v.files[0]?.r10?.test_r10;
    return `
      <div class="verdict-card ${v.agg_status}" data-iid="${v.iid}">
        <div class="verdict-iid">#${v.iid}</div>
        ${statusBadge(v.agg_status)}
        <div class="verdict-files">${v.file_count} 个文件</div>
        ${topR10 ? `<div class="verdict-r10">test_R@10 = ${topR10}</div>` : ''}
        <div class="ts" style="margin-top:6px">${ago(new Date((v.files[0]?.mtime || 0)*1000).toISOString())}</div>
      </div>
    `;
  }).join('');
  list.querySelectorAll('.verdict-card').forEach(el => {
    el.addEventListener('click', () => openVerdict(el.dataset.iid));
  });
}

document.getElementById('verdict-search').addEventListener('input', renderVerdicts);
document.getElementById('btn-refresh-verdicts').addEventListener('click', loadVerdicts);

// ---------- Verdict Detail ----------
async function openVerdict(iid) {
  document.getElementById('verdict-panel-title').textContent = `Verdict #${iid}`;
  document.getElementById('verdict-panel-body').innerHTML = '<div class="empty loading">加载</div>';
  document.getElementById('verdict-panel-backdrop').classList.add('show');
  document.getElementById('verdict-panel').classList.add('show');

  const v = await api(`/api/verdicts/${iid}`);
  document.getElementById('verdict-panel-body').innerHTML = `
    <div class="script-meta">
      <span>${v.file_count} 个文件</span>
      <span>${statusBadge(v.agg_status)}</span>
    </div>
    ${v.files.map(f => `
      <div class="section-title">${escapeHtml(f.name)} ${statusBadge(f.status)}</div>
      <div class="script-meta">
        <span>${(f.size/1024).toFixed(1)} KB</span>
        <span>${f.r10.test_r10 ? 'test_R@10=' + f.r10.test_r10 : ''}
              ${f.r10.valid_r10 ? ' valid_R@10=' + f.r10.valid_r10 : ''}</span>
        <span>${fmtTime(new Date(f.mtime*1000).toISOString())}</span>
      </div>
      <pre class="script-block">${escapeHtml(f.content)}</pre>
    `).join('')}
  `;
}

function closeVerdictPanel() {
  document.getElementById('verdict-panel').classList.remove('show');
  document.getElementById('verdict-panel-backdrop').classList.remove('show');
}
document.getElementById('verdict-panel-close').addEventListener('click', closeVerdictPanel);
document.getElementById('verdict-panel-backdrop').addEventListener('click', closeVerdictPanel);

// ---------- Loop ----------
async function loadLoop() {
  const data = await api('/api/loop');
  state.loop = data;
  document.getElementById('loop-mtime').textContent = '更新于 ' + fmtTime(data.mtime_iso);

  const activeTbl = data.active.length ? `
    <table>
      <thead><tr><th>版本</th><th>关键改动</th><th>test_R@10</th><th>valid_R@10</th></tr></thead>
      <tbody>
        ${data.active.map(r => `
          <tr>
            <td class="mono">${escapeHtml(r.version)}</td>
            <td>${escapeHtml(r.change)}</td>
            <td class="mono">${escapeHtml(r.test_r10 || '—')}</td>
            <td class="mono">${escapeHtml(r.valid_r10 || '—')}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  ` : '<div class="empty">无当前迭代表</div>';
  document.getElementById('loop-active').innerHTML = activeTbl;
  document.getElementById('loop-content').textContent = data.content;
}
document.getElementById('btn-refresh-loop').addEventListener('click', loadLoop);

// ---------- Rules ----------
async function loadRules() {
  const data = await api('/api/claudemd');
  document.getElementById('rules-content').textContent = data.content;
}
document.getElementById('btn-refresh-rules').addEventListener('click', loadRules);

// ---------- Init ----------
(async () => {
  // 启动时并行拉 overview + badge
  loadDashboard().catch(e => {
    document.getElementById('dash-stats').innerHTML = `<div class="empty">加载失败: ${e.message}</div>`;
  });
  // 时钟
  setInterval(() => {
    const el = document.getElementById('ts-now');
    if (el && el.textContent.startsWith('刷新于')) {
      el.textContent = '刷新于 ' + fmtTime(new Date().toISOString());
    }
  }, 1000);
})();

// ESC 关闭 panel
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    closeTaskPanel();
    closeVerdictPanel();
  }
});
