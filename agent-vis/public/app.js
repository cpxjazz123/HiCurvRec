import { createStore, applyChunk, roundEnded, handoffRequested } from './protocol.js?v=__ASSET_V__';

/**
 * 前端渲染器。
 *
 * 后端吐的是标准 AI SDK v7 UI message stream 分片，这里按 protocol.js 的
 * reducer 还原成时间轴模型再画到 DOM 上；协议解析与测试共用同一份实现。
 *
 * 页面上有左右两个 agent 面板，两边完全独立：各自的 store、DOM 节点、
 * 请求与统计，互不影响，可以同时跑。
 */

const buildTagEl = document.getElementById('buildTag');
const globalMetaEl = document.getElementById('globalMeta');
const settingsMetaEl = document.getElementById('settingsMeta');
const alertBarEl = document.getElementById('alertBar');
const newAgentBtnEl = document.querySelector('[data-role=newAgentBtn]');
const settingsBtnEl = document.querySelector('[data-role=settingsBtn]');
const settingsDialogEl = document.querySelector('[data-role=settingsDialog]');
const settingsHostEl = document.querySelector('[data-role=settingsHost]');
const settingsWorkdirEl = document.querySelector('[data-role=settingsWorkdir]');
const settingsStateEl = document.querySelector('[data-role=settingsState]');
const settingsSaveEl = document.querySelector('[data-role=settingsSave]');
const settingsResetEl = document.querySelector('[data-role=settingsReset]');
const settingsCloseEl = document.querySelector('[data-role=settingsClose]');
const settingsAgentEl = document.querySelector('[data-role=settingsAgent]');
const settingsResetSessionEl = document.querySelector('[data-role=settingsResetSession]');
const settingsGroupsEl = document.querySelector('[data-role=settingsGroups]');
const newGroupBtnEl = document.querySelector('[data-role=newGroupBtn]');
const newGroupDialogEl = document.querySelector('[data-role=newGroupDialog]');
const newGroupBodyEl = document.querySelector('[data-role=newGroupBody]');
const newGroupCloseEl = document.querySelector('[data-role=newGroupClose]');
const layoutEl = document.querySelector('.layout');
const agentListEl = document.querySelector('[data-role=agentList]');
const groupListEl = document.querySelector('[data-role=groupList]');

/**
 * 页面上所有面板。
 *
 * 运行时可以往里加 —— 「新建 Agent」造出来的面板也在这儿。
 * 首页模板里的那份不算：模板内容不在文档树里，querySelectorAll 看不到它。
 */
const panels = [];

/**
 * 服务端知道的 agent 身份。
 *
 * 交接目标只能指向这些身份 —— 指向一个已经不存在的 agent，交接根本发不出去。
 * 清单来自服务端的 /api/health：删掉的 agent 要从这里摘掉。
 */
const knownAgentIds = new Set();
/** 服务端上的组：id -> {id, name, members}。 */
const knownGroups = new Map();

/**
 * agent 身份 → 显示用的名字。
 *
 * 交接目标的勾选项按身份列出来：身份是服务端的，面板只是页面上挂的那一份，
 * 所以名字单独记一份，面板不在时也照样写得出来。
 */
const agentNames = new Map();

/**
 * 左侧栏里每个 agent 那一行的 DOM。
 *
 * key 是 agent 身份，值是这一行上的节点（状态点、名字按钮、配置按钮）。
 */
const sidebarRows = new Map();

/**
 * 右侧当前显示的是哪个 agent。
 *
 * null 表示还没定 —— 页面第一次起来时定成第一个 agent。
 */
let activeAgentId = null;
/** 服务端的 agent 清单到了没有：没到之前，地址栏指向还没挂出来的面板不算错。 */
let healthLoaded = false;
/** 地址栏指着、但面板还没挂出来的那个 agent。 */
let pendingRoute = null;

/**
 * 右侧当前显示的是聊天框还是配置页。
 *
 * 'chat' —— 完整聊天框；'config' —— 这个 agent 自己的配置页。
 */
let activeView = 'chat';


/**
 * 与 config.js 的 FRONTEND_VERSION 必须一致。
 * 服务端回报的号对不上，就说明浏览器里跑的是旧版 JS —— 直接提示硬刷新，
 * 而不是让旧版把新协议分片画成看不懂的卡片。
 */
const FRONTEND_VERSION = '2026-11-14.3';

/**
 * 接口调用的根前缀，由服务端在首页里注入。
 *
 * 本地部署是空串；对外暴露（挂路径前缀）时是 '/vis-xxxx' ——
 * 不带上它，请求会打到裸路径上，被门槛挡成 404。
 */
const API_BASE = window.__AGENT_VIS_API_BASE__ || '';

/**
 * 交接消息里默认写上游的哪个名字。
 *
 * 页面上每次交接都会带上来源的名字，这个常量只在直接调接口、
 * 没给来源名字时兜住 —— 与 config.js 的 HANDOFF_SOURCE_LABEL 一致。
 */
const DEFAULT_HANDOFF_SOURCE = 'AGENT A';

/**
 * 旧版把轮次步骤存在浏览器的 localStorage 里，键前缀如下。
 *
 * 现在配置存在服务端（agent-vis/.data/agents.json）：换个浏览器、换台机器、
 * 服务重启之后都还在。localStorage 里这两份只用来把老配置搬进服务端，
 * 搬完就把键删掉 —— 留着的话两处配置会各说各话。
 */
const LEGACY_STEP_STORAGE_PREFIX = 'agent-vis.steps:';
const LEGACY_PERMISSION_STORAGE_PREFIX = 'agent-vis.permissions:';

/**
 * 文字类改动停手多久之后存到服务端（毫秒）。
 *
 * 提示词与步骤说明是边打字边改的，每敲一下就发一次请求太吵；
 * 但存太慢的话，用户改完立刻跑一轮，那一轮用的还是旧配置。
 * 所以：停手 400 毫秒就存，跑之前再强制存一次。
 */
const CONFIG_SAVE_DELAY_MS = 400;

/**
 * 把一份配置改动存到服务端。
 *
 * 返回服务端合并后的生效配置（里面既有用户改过的，也有代码里的默认）。
 * 存失败直接抛错：调用方要拿这个错误告诉用户"没存上"，
 * 不能悄悄吞掉 —— 界面上留着没存上的改动，等于骗用户。
 */
async function putConfig(profileId, patch) {
  const res = await fetch(
    `${API_BASE}/api/agents/${encodeURIComponent(profileId)}/config`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    },
  );
  const text = await res.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch {
    // 响应体不是 JSON（网关错误之类），那就只报状态码。
    data = null;
  }
  if (!res.ok) {
    const detail = data && typeof data.error === 'string' ? data.error : '';
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return data;
}

/**
 * 读某个 agent 的对话记录（按时间顺序）。
 *
 * 记录读不出来时抛错：这一栏的历史读不出来，就该让用户知道，
 * 而不是给它画一个空白的过去。
 */
async function getChats(profileId) {
  const res = await fetch(`${API_BASE}/api/agents/${encodeURIComponent(profileId)}/chats`);
  const text = await res.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch {
    data = null;
  }
  if (!res.ok) {
    const detail = data && typeof data.error === 'string' ? data.error : '';
    throw new Error(detail || `HTTP ${res.status}`);
  }
  if (!data || !Array.isArray(data.records)) {
    throw new Error('服务端给的对话记录里没有 records 数组');
  }
  return data.records;
}

/** 清掉某个 agent 的对话记录。 */
async function deleteChats(profileId) {
  const res = await fetch(
    `${API_BASE}/api/agents/${encodeURIComponent(profileId)}/chats`,
    { method: 'DELETE' },
  );
  if (!res.ok) {
    const text = await res.text();
    let detail = '';
    try {
      const data = JSON.parse(text);
      if (data && typeof data.error === 'string') detail = data.error;
    } catch {
      detail = '';
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return true;
}

/**
 * 启动标记。首页的内联守卫在等它：
 * 等不到就说明浏览器跑的 JS 是缓存里的旧版，于是换地址重载一次。
 */
window.__AGENT_VIS_FRONTEND__ = FRONTEND_VERSION;

/**
 * 分片身份不用再加前缀了。
 *
 * 以前多步轮次要把好几次请求画进同一条时间轴，同一份 text id 会在两次
 * 请求里重复出现，后一个请求的文字会被并进前一个请求的卡片里 ——
 * 于是给每份分片的身份加了请求前缀。
 *
 * 现在每一次请求在时间轴上各占一个块，各有各的 store：
 * id 在自己那份 store 里唯一就行，不必再加前缀。
 * 而且服务端存的正是原样的分片，回放时和当时跑的一模一样。
 */

/** 页面级告警（版本不一致、协议不认识、agent 身份对不上），跨运行保留。 */
function showAlert(message) {
  if (!alertBarEl) return;
  alertBarEl.textContent = message;
  alertBarEl.classList.add('on');
}

/**
 * 把时间戳写成人看得懂的样子（本地时间）。
 *
 * 时间轴上每一轮的开头都写着它是什么时候跑的 —— 刷新页面回放时，
 * 光看内容认不出哪一轮是哪一轮。
 */
function formatTime(ts) {
  const date = new Date(ts);
  const pad = (n) => String(n).padStart(2, '0');
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
  );
}

function createPanel(root) {
  const pick = (role) => root.querySelector(`[data-role="${role}"]`);
  const els = {
    statusDot: pick('statusDot'),
    statusText: pick('statusText'),
    name: pick('name'),
    form: pick('form'),
    prompt: pick('prompt'),
    runBtn: pick('runBtn'),
    stopBtn: pick('stopBtn'),
    timeline: pick('timeline'),
    timelineInner: pick('timelineInner'),
    emptyHint: pick('emptyHint'),
    instructions: pick('instructions'),
    instructionsBox: pick('instructionsBox'),
    instructionsState: pick('instructionsState'),
    resetInstructions: pick('resetInstructions'),
    nameInput: pick('nameInput'),
    nameState: pick('nameState'),
    resetName: pick('resetName'),
    deleteBtn: pick('deleteBtn'),
    handoffBox: pick('handoffBox'),
    handoffSummary: pick('handoffSummary'),
    handoffTargets: pick('handoffTargets'),
    handoffDelay: pick('handoffDelay'),
    handoffDelayState: pick('handoffDelayState'),
    resetSessionHandoff: pick('resetSessionHandoff'),
    resetSessionHandoffState: pick('resetSessionHandoffState'),
    allowHandoffBack: pick('allowHandoffBack'),
    allowHandoffBackState: pick('allowHandoffBackState'),
    handoffConditionText: pick('handoffConditionText'),
    handoffConditionState: pick('handoffConditionState'),
    loopBox: pick('loopBox'),
    loopSummary: pick('loopSummary'),
    loopEnabled: pick('loopEnabled'),
    loopCount: pick('loopCount'),
    loopInterval: pick('loopInterval'),
    loopState: pick('loopState'),
    stepsBox: pick('stepsBox'),
    stepsSummary: pick('stepsSummary'),
    stepsList: pick('stepsList'),
    stepsHint: pick('stepsHint'),
    addStepBtn: pick('addStep'),
    clearStepsBtn: pick('clearSteps'),
    restoreStepsBtn: pick('restoreSteps'),
    permissionsBox: pick('permissionsBox'),
    permissionsSummary: pick('permissionsSummary'),
    permissionList: pick('permissionList'),
    permissionsNote: pick('permissionsNote'),
    metaList: pick('metaList'),
    stepBar: pick('stepBar'),
    configBtn: pick('configBtn'),
    clearHistoryBtn: pick('clearHistory'),
    resetSessionBtn: pick('resetSessionBtn'),
    backToChat: pick('backToChat'),
    configTitle: pick('configTitle'),
  };

  let controller = null;
  /** 跑完之后的回调（交接用）：收到本轮最后一段文字回答。 */
  let finishHandler = null;
  // 本轮如果有上游，记下它的 agent 身份：收尾时不把结果再交回给上游，
  // 否则 A → B → A → B 会互相触发成环。
  let handoffSourceId = '';
  /** 这一轮是不是正在跑（决定「结束」按钮该不该可点）。 */
  let running = false;
  /** 本轮是不是被用户点「结束」停掉的。 */
  let abortedByUser = false;
  /** 运行序号：只有最新那一轮的收尾才配改状态，旧轮的收尾不许覆盖新轮。 */
  let runSeq = 0;
  /** 本次手动运行配置的总轮数、已完成的轮数（0 = 不在循环里），以及下一轮的定时器。 */
  let loopTarget = 0;
  let loopDone = 0;
  let loopTimer = null;
  /** 时间轴上的块序号：给没有服务端记录 id 的块起个名字。 */
  let blockSeq = 0;
  /** 轮次序号：同一轮里的几步共用一个 roundId，靠它认得出来。 */
  let roundSeq = 0;
  /**
   * 服务端确认过的那份配置（生效值 + 代码里的默认 + 哪些字段是用户改过的）。
   *
   * 存失败的时候界面要退回这一份：界面上留着没存上的改动，
   * 等于让用户以为自己的配置已经生效了。
   */
  let profileState = null;
  /**
   * 服务端的第一份配置到了没有。
   *
   * 没到的时候用户就改了某一栏 —— 那一栏算用户的，服务端那份到了也不许盖掉。
   * 不护住这一层，"页面刚打开就点清空步骤"会被 /api/health 回来的默认步骤
   * 悄悄盖掉：用户以为自己改了，其实没改。
   */
  let profileApplied = false;
  /** 用户动过的栏位。动了就归用户管，服务端的第一份配置不许盖。 */
  const userEdited = {
    instructions: false,
    steps: false,
    permissions: false,
    handoff: false,
    handoffDelay: false,
    loop: false,
    name: false,
  };
  /**
   * 用户改过的系统提示词。
   *
   * null = 没改过，用代码里的默认。空串也算改过 ——
   * 那是明确要"没有系统提示词"，跟"没改过"是两件事。
   */
  let instructionsOverride = null;
  /** 代码里写死的默认系统提示词。用户没改就用它，改了就用用户那份。 */
  let defaultInstructions = '';
  /** 面板显示用的名字：交接配置的勾选项、交接卡片都要用它。 */
  let displayName = root.dataset.profile;
  /** 代码里写死的默认名字，用户点「恢复默认」回到它。 */
  let defaultName = root.dataset.profile;
  /** 名字是不是用户改过的（服务端 customized.name）。 */
  let nameCustomized = false;
  /**
   * 这一栏跑完后，结果要发给哪些 agent（可以多选）。
   *
   * 存的是 agent 身份，不是面板引用 —— 面板被删掉时要从所有地方把它摘掉。
   */
  const handoffTargets = new Set();
  /** 删除按钮的两步确认：第一次点只是进入确认状态，第二次才真删。 */
  let deleteArmed = false;
  /**
   * 延迟交接的定时器。
   *
   * 用户在这一栏重新跑一轮、或把面板删掉时，这些"还没到点"的交接必须取消 ——
   * 否则一条旧结果会在几秒后突然打到下游，看起来像鬼。
   */
  const pendingHandoffTimers = new Set();
  /**
   * 这一栏的轮次步骤：每一轮按哪几步做。
   *
   * 空数组 = 一次跑完（从前到后一直是这样）。有内容 = 每一步各跑一次，
   * 严格按顺序，前一步的产出接到后一步。
   */
  let steps = [];
  /**
   * 用户自己配过这一栏的步骤没有。
   *
   * 分得清这件事，才分得清"用户清空了步骤"与"用户还没配过步骤"：
   * 没配过的用服务端给的默认步骤（AGENT A / AGENT B 的提示词拆出来的），
   * 用户自己清空了就尊重用户的清空，不再把默认步骤铺回来。
   */
  let stepsConfigured = false;
  /**
   * 服务端给的默认步骤。
   *
   * AGENT A / AGENT B 的提示词里那份流程被拆成了固定的几步 ——
   * 页面在用户没配过之前就铺好它们。「恢复默认步骤」也是回到这份。
   */
  let defaultSteps = [];
  /**
   * 服务端给的默认步骤到了没有。
   *
   * 还没到就点「恢复默认步骤」，那一步要么什么都不做（把用户的配置清空），
   * 要么默默用空列表覆盖 —— 两种都是坑。所以先分清"还没取回来"和"确实没有"。
   */
  let defaultsLoaded = false;
  /**
   * 这个 agent 当前被授的权限（权限 id 的集合）。
   *
   * 这份东西是硬约束：服务端按它裁工具表 —— 没授的工具压根不会出现在
   * 这一轮的工具里，模型调不到。页面上勾的就是这一轮真正生效的。
   */
  let permissions = new Set();
  /**
   * 服务端给的权限目录（每一项：id、名称、说明、绑着的工具）。
   *
   * 目录没到之前不知道哪些 id 是合法的，所以先把用户存的原样留着，
   * 目录到了再过滤 —— 反过来做会把用户刚勾的权限弄丢。
   */
  let permissionCatalog = [];
  /**
   * 权限目录到了没有。
   */
  let permissionCatalogLoaded = false;
  /**
   * 当前这一次请求是不是多步轮次里的一步。
   *
   * 为真时这一次不清时间轴、不改面板状态、也不触发交接 ——
   * 整轮的收尾由跑整轮的那一段统一管。
   */
  let currentPartOfRound = false;
  /**
   * 请求计数：多步轮次里每一次请求都要有自己的一份分片身份
   * （不同请求会给出同样的 text id，不分开就会并进同一张卡）。
   */
  let requestSeq = 0;
  /**
   * 用户是不是想看最新内容。
   *
   * 贴底时为真（自动跟着新内容往下走）；用户往上翻看历史时为假 ——
   * 这时新内容不该把他拽回底部，否则窗口里的历史根本读不到。
   */
  let stickToBottom = true;
  /** 当前状态：状态点的样式与文字。左侧栏那一行的圆点跟着它走。 */
  let statusKind = '';
  let statusLabel = '待运行';

  /**
   * 时间轴上的每一轮。
   *
   * 一轮 = 一次请求。多步轮次里每一步各是一次请求，于是各有各的块 ——
   * 每块里是自己的分片、自己的卡片，不会跟别的轮次串在一起。
   *
   * 这一份结构跟服务端的对话记录一一对应：刷新页面时按记录重画，
   * 画出来的东西跟当时跑的时候一模一样。
   */
  const runs = [];
  // 每轮（roundId）的累计耗时：多步轮次里每一步是一次请求，
  // 「总耗时」要把这几步加起来，而不是每换一步就从头算。
  const roundTotals = new Map();
  /** 当前正在跑的那一轮（算耗时、看产出都看它）。 */
  let currentRun = null;
  /**
   * 配置改动里还没存到服务端的那一份（文字类改动会攒一会儿）。
   *
   * 存的时候按字段合并：先改提示词再改步骤，两次合成一次存。
   */
  let pendingPatch = null;
  /** 攒着这次改动的定时器。 */
  let saveTimer = null;

  /** 步骤进度条上的 chip：下标是步骤序号，值是那个 chip 元素。 */
  const planChips = new Map();

  function setStatus(kind, label) {
    statusKind = kind;
    statusLabel = label;
    els.statusDot.className = 'dot' + (kind ? ' ' + kind : '');
    els.statusText.textContent = label;
    // 左侧栏那一行的圆点跟着一起变。
    syncSidebar();
  }

  function setMeta(field, value) {
    const node = els.metaList.querySelector(`[data-field="${field}"]`);
    if (node) node.textContent = value;
  }

  // ---------- 配置：存在服务端 ----------

  /**
   * 把一份配置改动存到服务端。
   *
   * 存成功：服务端合并后的那份就是现在的生效配置，记下来（存失败时要知道退回哪一份）。
   *
   * 存失败：把原因说清楚，并把界面退回服务端最后确认的那一份 ——
   * 界面上留着没存上的改动，等于骗用户以为自己的配置生效了。
   */
  async function saveConfig(patch) {
    let merged = null;
    try {
      merged = await putConfig(root.dataset.profile, patch);
    } catch (err) {
      showAlert(
        `配置没能存到服务端：${(err && err.message) || String(err)}。` +
          '界面已退回服务端保存的那一份。',
      );
      revertView();
      return null;
    }
    profileState = merged;
    // 不重新渲染：用户可能正在提示词框或步骤框里打字，
    // 重画一遍会把正在打的字冲掉。界面上的内容就是刚存进去的那份。
    return merged;
  }

  /**
   * 攒一会儿再存：文字类改动（提示词、步骤标题与说明）边打字边改，
   * 每敲一下发一次请求太吵。停手 CONFIG_SAVE_DELAY_MS 毫秒才存。
   */
  function scheduleSave(patch) {
    pendingPatch = { ...(pendingPatch || {}), ...patch };
    if (saveTimer !== null) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      saveTimer = null;
      const next = pendingPatch;
      pendingPatch = null;
      void saveConfig(next);
    }, CONFIG_SAVE_DELAY_MS);
  }

  /**
   * 立刻把攒着的改动存掉。
   *
   * 开跑之前一定要调：用户改完配置马上点运行，那一轮必须用新配置。
   */
  async function flushConfigSave() {
    if (saveTimer !== null) {
      clearTimeout(saveTimer);
      saveTimer = null;
    }
    if (pendingPatch === null) return true;
    const next = pendingPatch;
    pendingPatch = null;
    const merged = await saveConfig(next);
    // 存失败时 saveConfig 已经把界面退回了服务端确认过的那一份，
    // 也把原因报给了用户。这里只回答一句：存上了没有。
    return merged !== null;
  }

  /**
   * 把服务端给的一份配置装到面板上。
   *
   * 这份配置里有三样东西：生效的值、代码里的默认值、哪些字段是用户改过的。
   * 三者必须分开 —— 用户清空过的步骤与"用户还没配过步骤"不是一回事。
   */
  function applyProfile(profile, options = {}) {
    const prune = options.prune !== false;
    if (!profile || typeof profile !== 'object') {
      throw new Error('服务端给的配置不是一个对象');
    }
    const customized = profile.customized || {};
    // 服务端的第一份配置：用户在这之前动过的栏位归用户管，不盖。
    const first = !profileApplied;
    profileApplied = true;

    // 留着的那一份配置可能是用户改之前取的（用户改完就存，/api/health 的
    // 响应里还是旧值）。用户动过的那几栏，这一栏保留服务端确认过的那份 ——
    // revertView 要靠它退回"服务端确认过的样子"。
    const previousState = profileState;
    profileState = { ...profile };
    if (previousState) {
      if (first && userEdited.instructions) {
        profileState.instructions = previousState.instructions;
      }
      if (first && userEdited.steps) profileState.steps = previousState.steps;
      if (first && userEdited.permissions) {
        profileState.permissions = previousState.permissions;
      }
      if (first && userEdited.handoff) {
        profileState.handoffTargets = previousState.handoffTargets;
      }
    }

    // 名字：生效值由服务端给；用户正在输入框里改名时，别把输入冲掉。
    defaultName = typeof profile.defaultName === 'string' ? profile.defaultName : profile.name;
    nameCustomized = Boolean(customized.name);
    if (typeof profile.name === 'string' && profile.name !== '') {
      applyNameToDom(profile.name);
      if (!(first && userEdited.name) && els.nameInput) els.nameInput.value = profile.name;
    }
    refreshNameState();

    // 提示词：生效值 = 用户改过的那份，没改过就是代码里的默认。
    defaultInstructions =
      typeof profile.defaultInstructions === 'string' ? profile.defaultInstructions : '';
    if (!(first && userEdited.instructions)) {
      instructionsOverride = customized.instructions
        ? typeof profile.instructions === 'string'
          ? profile.instructions
          : ''
        : null;
      const effectiveInstructions =
        instructionsOverride === null ? defaultInstructions : instructionsOverride;
      if (els.instructions && els.instructions.value !== effectiveInstructions) {
        els.instructions.value = effectiveInstructions;
      }
    }
    refreshInstructionsState();

    // 轮次步骤：用户配过就用用户那份（哪怕是清空），没配过才铺默认步骤。
    defaultSteps = normalizeStepList(profile.defaultSteps);
    defaultsLoaded = true;
    if (!(first && userEdited.steps)) {
      stepsConfigured = Boolean(customized.steps);
      steps = stepsConfigured
        ? normalizeUserStepList(profile.steps)
        : defaultSteps.map((step) => ({ ...step }));
      renderSteps();
    }

    // 权限：同理。
    if (!(first && userEdited.permissions)) {
      permissions = new Set(Array.isArray(profile.permissions) ? profile.permissions : []);
      renderPermissions();
    }

    // 交接目标：服务端存的那份。
    if (!(first && userEdited.handoff)) {
      handoffTargets.clear();
      for (const id of Array.isArray(profile.handoffTargets) ? profile.handoffTargets : []) {
        if (typeof id === 'string' && id !== '') handoffTargets.add(id);
      }
    }
    // 交接条件：一句话，由轮次最后一步的模型判定。跟着服务端那份走。
    if (els.handoffConditionText) {
      els.handoffConditionText.value =
        typeof profile.handoffCondition === 'string' ? profile.handoffCondition : '';
      refreshHandoffConditionState();
    }

    // 允不允许回传给上游（审查—返修循环）：跟着服务端那份走。
    if (els.allowHandoffBack) {
      els.allowHandoffBack.checked = Boolean(profile.allowHandoffBack);
      refreshAllowHandoffBackState();
    }

    // 交接时要不要先重置自己这一栏的会话：跟着服务端那份走。
    if (els.resetSessionHandoff) {
      els.resetSessionHandoff.checked = Boolean(profile.resetSessionOnHandoff);
      refreshResetSessionHandoffState();
    }

    // 交接延迟：0 = 立刻。输入框跟着服务端那份走。
    if (els.handoffDelay) {
      const delay =
        typeof profile.handoffDelaySeconds === 'number' ? profile.handoffDelaySeconds : 0;
      els.handoffDelay.value = String(delay);
      refreshHandoffDelayState();
    }

    // 循环：用户改过就不覆盖输入框，否则铺服务端那份（默认关闭）。
    if (!(first && userEdited.loop) && els.loopEnabled) {
      const loop = profile.loop && typeof profile.loop === 'object' ? profile.loop : null;
      const setting = loop || { enabled: false, count: 2, intervalSeconds: 0 };
      els.loopEnabled.checked = Boolean(setting.enabled);
      if (els.loopCount) els.loopCount.value = String(setting.count);
      if (els.loopInterval) els.loopInterval.value = String(setting.intervalSeconds);
      refreshLoopState();
    }

    // 存下来的目标里可能有已经被删掉的 agent：那些交不出去，摘掉并存回去。
    const dropped = pruneHandoffTargets();
    renderHandoffTargets();
    if (dropped.length > 0 && prune) {
      void saveConfig({ handoffTargets: Array.from(handoffTargets) });
    }
    // 服务端报上来的过期权限（目录里已经没有的 id）：去掉并存回去。
    const stale = Array.isArray(profile.stalePermissions) ? profile.stalePermissions : [];
    if (stale.length > 0) {
      showAlert(
        `存下来的权限里有一项已经不在服务端的权限目录里（${stale.join('、')}），已经去掉。`,
      );
      void saveConfig({ permissions: Array.from(permissions) });
    }
  }

  /** 界面退回服务端最后确认的那一份配置。存失败时用。 */
  function revertView() {
    if (!profileState) return;
    // 不再顺手去存"摘掉死目标"这件事：存失败时递归下去没完没了。
    applyProfile(profileState, { prune: false });
  }

  /**
   * 把交接目标里已经不存在的 agent 摘掉。
   *
   * 返回摘掉的那几个身份（没有就返回空数组）。
   *
   * 服务端的清单还没到的时候不动：那时候什么都会被当成"不存在"。
   */
  function pruneHandoffTargets() {
    if (knownAgentIds.size === 0) return [];
    const dropped = [];
    for (const id of Array.from(handoffTargets)) {
      // 目标是 agent 或者组（group:<id>）：两边都得是服务端还认识的。
      const isGroup = id.startsWith('group:');
      const alive = isGroup ? knownGroups.has(id.slice('group:'.length)) : knownAgentIds.has(id);
      if (!alive) {
        handoffTargets.delete(id);
        dropped.push(id);
      }
    }
    return dropped;
  }

  /**
   * 把服务端给的用户步骤整理成面板能用的形状。
   *
   * 与默认步骤不同：用户的步骤允许标题或指令是空串 ——
   * 用户正在打字时标题可能还没填上，那不该把整份配置拒掉。
   * 但形状坏了（不是数组、条目不是对象）照样报出来。
   */
  function normalizeUserStepList(list) {
    if (!Array.isArray(list)) {
      showAlert('服务端给的轮次步骤不是一个数组，这一栏按没有步骤开始。');
      return [];
    }
    const kept = [];
    list.forEach((entry, index) => {
      if (!entry || typeof entry !== 'object') {
        showAlert(`服务端给的轮次步骤第 ${index + 1} 条不是对象，已跳过这一条。`);
        return;
      }
      kept.push({
        title: typeof entry.title === 'string' ? entry.title : '',
        instruction: typeof entry.instruction === 'string' ? entry.instruction : '',
      });
    });
    return kept;
  }

  /**
   * 步骤进度条：显示配好的轮次步骤，不是协议层的 Step 0/1/2。
   *
   * 每个配好的步骤一个 chip，状态写在 data-state 上：
   *   pending —— 还没轮到；active —— 正在做；filled —— 做完了；failed —— 做砸了。
   * 没配步骤时进度条是空的 —— 没有步骤可显示，别拿别的东西凑数。
   */
  function renderPlanChips() {
    els.stepBar.innerHTML = '';
    planChips.clear();
    steps.forEach((step, index) => {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.setAttribute('data-step-chip', String(index));
      const title = step.title.trim() === '' ? `第 ${index + 1} 步` : step.title.trim();
      chip.textContent = `第 ${index + 1} 步 · ${title}`;
      chip.dataset.state = 'pending';
      els.stepBar.appendChild(chip);
      planChips.set(index, chip);
    });
  }

  /** 把某一步的进度状态写进 chip。 */
  function setPlanChipState(index, state) {
    const chip = planChips.get(index);
    if (!chip) return;
    chip.dataset.state = state;
  }

  /** 「运行」与「结束」两个按钮一起管：跑的时候只有结束可点，闲下来只有运行可点。 */
  function setRunning(isRunning) {
    els.runBtn.disabled = isRunning;
    els.stopBtn.disabled = !isRunning;
  }

  /** 名字那一行右侧的状态说明。 */
  function refreshNameState(extra = '') {
    if (!els.nameState) return;
    if (extra !== '') {
      els.nameState.textContent = extra;
      return;
    }
    els.nameState.textContent = nameCustomized ? '已修改（存在服务端，本次运行生效）' : '默认名字';
  }

  /** 提示词改没改：折叠栏上标出来，免得忘了自己改过什么。 */
  function refreshInstructionsState() {
    if (!els.instructionsState) return;
    const custom = instructionsOverride;
    let label;
    if (custom !== null) label = '已修改（存在服务端，本次运行生效）';
    else if (defaultInstructions.trim() !== '') label = '默认提示词';
    else label = '空（这个 agent 没有系统提示词）';
    els.instructionsState.textContent = label;
    els.instructionsState.classList.toggle('dirty', custom !== null);
  }

  /**
   * 把提示词框恢复成服务端给的默认。
   *
   * 恢复 = 抹掉服务端存的那份用户改动（存 null），生效值回到代码里的默认。
   */
  function resetInstructions() {
    if (!els.instructions) return;
    userEdited.instructions = true;
    instructionsOverride = null;
    els.instructions.value = defaultInstructions;
    refreshInstructionsState();
    void saveConfig({ instructions: null });
  }

  /** 交接条件那一行旁边的说明。 */
  function refreshHandoffConditionState() {
    if (!els.handoffConditionState) return;
    const condition = els.handoffConditionText ? els.handoffConditionText.value.trim() : '';
    els.handoffConditionState.textContent =
      condition === ''
        ? '没有条件：轮次最后会加一个「交接结论」步骤，无条件调用 requestHandoff。'
        : `有条件：最后一步按「${condition}」判断要不要调用 requestHandoff。`;
  }

  /** 「允许回传给上游」那个勾选旁边的说明。 */
  function refreshAllowHandoffBackState() {
    if (!els.allowHandoffBackState) return;
    const on = Boolean(els.allowHandoffBack && els.allowHandoffBack.checked);
    els.allowHandoffBackState.textContent = on
      ? '勾上了：上游把结果发过来时，这一栏可以把返修意见交回给它（循环有 8 跳上限）。'
      : '没勾：不把结果交回给刚把它发给我的那一栏。';
  }

  /** 交接时重置会话那个勾选旁边的说明。 */
  function refreshResetSessionHandoffState() {
    if (!els.resetSessionHandoffState) return;
    const on = Boolean(els.resetSessionHandoff && els.resetSessionHandoff.checked);
    els.resetSessionHandoffState.textContent = on
      ? '勾上了：交接时先清掉这一栏的会话，下游拿到消息时它已经从头开始。'
      : '没勾：交接不动这一栏的会话。';
  }

  /** 延迟输入框旁边的说明：写清"现在是多少"，以及是不是用户改过的。 */
  function refreshHandoffDelayState() {
    if (!els.handoffDelayState) return;
    const raw = els.handoffDelay ? Number(els.handoffDelay.value) : 0;
    const delay = Number.isInteger(raw) && raw > 0 ? raw : 0;
    els.handoffDelayState.textContent =
      delay === 0 ? '立即发送' : `${delay} 秒后发送`;
  }

  /**
   * 面板上的循环设置：读输入框，夹到合法区间。
   *
   * 页面上敲的中间态（空串、负数）按"最小合法值"解释，存盘时再由服务端复核。
   */
  function loopSetting() {
    const enabled = Boolean(els.loopEnabled && els.loopEnabled.checked);
    const countRaw = els.loopCount ? Number(els.loopCount.value) : 1;
    const intervalRaw = els.loopInterval ? Number(els.loopInterval.value) : 0;
    return {
      enabled,
      count: Number.isInteger(countRaw) ? Math.min(100, Math.max(1, countRaw)) : 1,
      intervalSeconds: Number.isInteger(intervalRaw) ? Math.min(3600, Math.max(0, intervalRaw)) : 0,
    };
  }

  /** 折叠栏与状态行：现在开没开、几轮、隔多久、还剩几轮。 */
  function refreshLoopState() {
    const setting = loopSetting();
    if (els.loopSummary) {
      els.loopSummary.textContent = setting.enabled
        ? `循环（${setting.count} 轮 · 间隔 ${setting.intervalSeconds}s）`
        : '循环（关闭）';
    }
    if (!els.loopState) return;
    if (!setting.enabled) {
      els.loopState.textContent = '关闭时点一次「运行」就跑一轮。';
      return;
    }
    if (loopTarget > 0 && loopDone < loopTarget) {
      els.loopState.textContent = `进行中：已完成 ${loopDone} 轮，共 ${loopTarget} 轮。`;
      return;
    }
    els.loopState.textContent = `共 ${setting.count} 轮，每轮间隔 ${setting.intervalSeconds} 秒。`;
  }

  /** 折叠栏上写明勾了几个交接目标。 */
  function refreshHandoffSummary() {
    if (!els.handoffSummary) return;
    const count = handoffTargets.size;
    els.handoffSummary.textContent =
      count === 0 ? '交接配置（未选目标）' : `交接配置（已选 ${count} 个）`;
  }

  /**
   * 交接配置：把服务端认识的、别的 agent 列成可勾选的复选框。
   *
   * 候选来自服务端的身份表，不是页面上挂着的面板：面板删了但身份还在的
   * agent 也还能被选 —— 交接是服务端那一头的事。
   * 勾完立刻存到服务端：刷新页面、服务重启之后还在。
   */
  function renderHandoffTargets() {
    if (!els.handoffTargets) return;

    els.handoffTargets.innerHTML = '';
    const others = [...knownAgentIds].filter((id) => id !== root.dataset.profile);
    if (others.length === 0) {
      const empty = document.createElement('span');
      empty.className = 'handoff-empty';
      empty.textContent = '服务端上没有别的 agent 可选。';
      els.handoffTargets.appendChild(empty);
    }
    // 组：整组发给一个人，组里每个人都收到同一条消息。
    for (const group of knownGroups.values()) {
      const target = `group:${group.id}`;
      const label = document.createElement('label');
      label.className = 'handoff-option handoff-group';

      const box = document.createElement('input');
      box.type = 'checkbox';
      box.value = target;
      box.checked = handoffTargets.has(target);
      box.addEventListener('change', () => {
        userEdited.handoff = true;
        if (box.checked) handoffTargets.add(target);
        else handoffTargets.delete(target);
        refreshHandoffSummary();
        void saveConfig({ handoffTargets: Array.from(handoffTargets) });
      });

      const text = document.createElement('span');
      const memberNames = group.members
        .map((id) => agentNames.get(id) || id)
        .join('、');
      text.textContent = `组：${group.name}（${group.members.length} 人：${memberNames}）`;

      label.append(box, text);
      els.handoffTargets.appendChild(label);
    }
    for (const id of others) {
      const label = document.createElement('label');
      label.className = 'handoff-option';

      const box = document.createElement('input');
      box.type = 'checkbox';
      box.value = id;
      box.checked = handoffTargets.has(id);
      box.addEventListener('change', () => {
        userEdited.handoff = true;
        if (box.checked) handoffTargets.add(id);
        else handoffTargets.delete(id);
        refreshHandoffSummary();
        void saveConfig({ handoffTargets: Array.from(handoffTargets) });
      });

      const text = document.createElement('span');
      text.textContent = agentNames.get(id) || id;

      label.append(box, text);
      els.handoffTargets.appendChild(label);
    }
    refreshHandoffSummary();
  }

  /**
   * 交接目标与服务端认识的 agent 对齐。
   *
   * 服务端上已经删掉的 agent 交不出去，从目标里摘掉并存回去。
   */
  function syncTargets() {
    const dropped = pruneHandoffTargets();
    if (dropped.length > 0) {
      void saveConfig({ handoffTargets: Array.from(handoffTargets) });
    }
    renderHandoffTargets();
  }

  /**
   * 删除按钮：两步确认。
   *
   * 第一次点只把按钮变成「确认删除」，第二次才真删 —— 一栏里可能刚写好
   * 一段系统提示词，点一下就没了不是好交互。
   */
  function armDelete() {
    if (!deleteArmed) {
      deleteArmed = true;
      els.deleteBtn.textContent = '确认删除';
      els.deleteBtn.classList.add('armed');
      return;
    }
    deleteArmed = false;
    els.deleteBtn.textContent = '删除';
    els.deleteBtn.classList.remove('armed');
    removePanel(api);
  }

  /** 窗口现在是不是贴着底部。 */
  function atBottom() {
    const el = els.timeline;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  }

  /**
   * 把窗口滚到最新一条。
   *
   * 滚的是时间轴自己的滚动容器，不再 scrollIntoView 页面 ——
   * 页面是锁在视口里的，页面级滚动会把整个布局顶走。
   * 用户往上翻看时（stickToBottom 为假）不动他的位置。
   */
  function scrollToBottom() {
    if (!stickToBottom) return;
    const el = els.timeline;
    el.scrollTop = el.scrollHeight;
  }

  function createCard(part) {
    const card = document.createElement('div');
    card.className = 'part';

    const head = document.createElement('div');
    head.className = 'part-head';

    const badge = document.createElement('span');
    badge.className = 'badge ' + (part.kind === 'tool' ? 'tool' : part.kind);
    badge.textContent = part.kind === 'tool' ? part.toolName : part.kind;
    head.appendChild(badge);

    const state = document.createElement('span');
    state.className = 'state';
    head.appendChild(state);

    card.appendChild(head);

    const body = document.createElement('div');
    body.className = 'part-body';
    card.appendChild(body);

    card._state = state;
    card._body = body;

    card.classList.add('is-active');
    setTimeout(() => card.classList.remove('is-active'), 700);

    return card;
  }

  function dropDetails(card) {
    if (card._details) {
      card._details.remove();
      card._details = null;
      card._detailsPre = null;
    }
  }

  /** 直接铺开一段短文本。 */
  function setPlainBody(card, text, dim) {
    dropDetails(card);
    card._body.hidden = false;
    card._body.className = 'part-body' + (dim ? ' dim' : '');
    card._body.textContent = text;
  }

  function updateCard(card, part) {
    if (part.kind === 'reasoning-file') {
      setPlainBody(card, `附件：${part.mediaType || '未知类型'}\n${part.url || ''}`, true);
      return;
    }

    if (part.kind === 'source-url' || part.kind === 'source-document' || part.kind === 'file') {
      const meta = part.mediaType ? ` (${part.mediaType})` : '';
      setPlainBody(card, `${part.title ? part.title + '\n' : ''}${part.url || ''}${meta}`, true);
      return;
    }

    if (part.kind === 'custom') {
      setPlainBody(
        card,
        `${part.providerKind || 'custom'}\n${JSON.stringify(part.providerMetadata ?? {}, null, 2)}`,
        true,
      );
      return;
    }

    dropDetails(card);
    card._body.hidden = false;
    card._body.className = 'part-body' + (part.kind === 'reasoning' ? ' reasoning' : '');
    card._body.textContent = part.text;
  }

  function updateMeta(run) {
    // 运行信息里只留总耗时：步骤数、工具调用数、token 计数都撤了，
    // 步骤的进展改由上面的「步骤进度」条显示（那边看的是配好的步骤）。
    // 说的是最新那一轮：时间轴上留着好几轮，总耗时看的是最近这一轮。
    //
    // 多步轮次里「总耗时」是这一轮**所有步骤**的耗时之和：每一步是一次请求，
    // 各有各的耗时，加起来才是这一轮真正花掉的时间。跑到一半时，
    // 已完成的步骤用记下来的耗时，正在跑的那一步用到目前为止的时间。
    if (!run) return;
    if (run.roundId) {
      const acc = roundTotals.get(run.roundId) || { doneMs: 0 };
      const live = run.durationMs === null ? Date.now() - run.startedAt : 0;
      const total = acc.doneMs + live;
      setMeta('duration', `${(total / 1000).toFixed(1)} s`);
      return;
    }
    const ms = run.durationMs === null ? Date.now() - run.startedAt : run.durationMs;
    setMeta('duration', `${(ms / 1000).toFixed(1)} s`);
  }

  /**
   * 时间轴上开一个新块：一次请求 = 一个块。
   *
   * 每个块有自己的 store：多步轮次的每一步各是一次请求，
   * 各自的文字不会串进别人那份卡片里。
   *
   * 块的结构跟服务端的对话记录一一对应：刷新页面时按记录重画，
   * 画出来的东西跟当时跑的时候一模一样。
   */
  function beginRun(meta = {}) {
    const run = {
      id: typeof meta.id === 'string' && meta.id !== '' ? meta.id : `run-${++blockSeq}`,
      // 多步轮次里的每一步都带着同一个 roundId：总耗时按它归堆。
      roundId: typeof meta.roundId === 'string' && meta.roundId !== '' ? meta.roundId : '',
      meta,
      store: createStore(),
      partNodes: new Map(),
      reasoningCard: null,
      startedAt: typeof meta.startedAt === 'number' ? meta.startedAt : Date.now(),
      finishedAt: null,
      durationMs: null,
      status: 'running',
    };

    const block = document.createElement('div');
    block.className = 'run-block';
    block.dataset.runId = run.id;

    const head = document.createElement('div');
    head.className = 'run-head';

    const time = document.createElement('span');
    time.className = 'run-time';
    time.textContent = formatTime(run.startedAt);
    head.appendChild(time);

    const status = document.createElement('span');
    status.className = 'run-status';
    status.textContent = '运行中';
    head.appendChild(status);

    const duration = document.createElement('span');
    duration.className = 'run-duration';
    head.appendChild(duration);

    block.appendChild(head);

    // 这一轮问的是什么：刷新页面之后回放，光看卡片认不出是哪一轮。
    const promptText = typeof meta.promptText === 'string' ? meta.promptText : '';
    if (promptText !== '') {
      const prompt = document.createElement('div');
      prompt.className = 'run-prompt';
      prompt.textContent = promptText;
      block.appendChild(prompt);
    }

    const body = document.createElement('div');
    body.className = 'run-body';
    block.appendChild(body);

    els.timelineInner.appendChild(block);
    els.timelineInner.classList.add('has-content');
    if (els.emptyHint) els.emptyHint.hidden = true;

    run.block = block;
    run.body = body;
    run.statusEl = status;
    run.durationEl = duration;
    runs.push(run);
    currentRun = run;
    return run;
  }

  /** 收尾：把这一块的状态与耗时写清楚。 */
  function finishRun(run, status, errorText) {
    if (!run) return;
    run.status = status;
    run.finishedAt = typeof run.finishedAt === 'number' ? run.finishedAt : Date.now();
    if (run.durationMs === null) run.durationMs = run.finishedAt - run.startedAt;
    // 这一步的耗时记进它所属的那一轮：下一步开始时，「总耗时」接着往上加。
    if (run.roundId) {
      const acc = roundTotals.get(run.roundId) || { doneMs: 0 };
      acc.doneMs += run.durationMs;
      roundTotals.set(run.roundId, acc);
    }
    const labels = { done: '完成', error: '出错', aborted: '已中止' };
    if (run.statusEl) run.statusEl.textContent = labels[run.status] || run.status;
    if (run.durationEl) {
      run.durationEl.textContent = `${(run.durationMs / 1000).toFixed(1)} s`;
    }
    run.block.dataset.status = run.status;
    if (run.status === 'error' && errorText) {
      const banner = document.createElement('div');
      banner.className = 'err-banner';
      banner.textContent = errorText;
      run.body.appendChild(banner);
    }
    updateMeta(run);
  }

  /**
   * 画一轮：按出现顺序把卡片平铺在块里。
   *
   * 不再按协议的 step 分段 —— 一轮里模型每回一次话就是一个 step，
   * 切出来只会得到一串 "Step 0/1/2…" 的空壳，读起来比平铺更碎。
   *
   * 推理（reasoning）例外：不管被切成几段，都并进块顶上的**同一个**推理块，
   * 它是模型的思考流，按段拆开只会让人以为模型反复重新想过。
   */
  function renderRun(run) {
    if (!run) return;
    const reasoningText = [];

    run.store.steps.forEach((step) => {
      step.parts.forEach((part) => {
        // 工具调用（名称、参数、执行结果）一律不进时间轴：这一页看的是模型的推理，
        // 一串"readFile 执行完成"只会把推理挤下去。协议里照样有这些分片，
        // 只是不画。
        if (part.kind === 'tool') return;
        if (part.kind === 'reasoning') {
          reasoningText.push(part.text);
          return;
        }
        const key = `${part.kind}:${part.key}`;
        let card = run.partNodes.get(key);
        if (!card) {
          card = createCard(part);
          run.partNodes.set(key, card);
          run.body.appendChild(card);
        }
        updateCard(card, part);
      });
    });

    const reasoning = reasoningText.join('');
    if (reasoning !== '') {
      if (!run.reasoningCard) {
        run.reasoningCard = createCard({ kind: 'reasoning', key: 'reasoning' });
        run.body.prepend(run.reasoningCard);
      }
      updateCard(run.reasoningCard, { kind: 'reasoning', text: reasoning });
    }

    updateMeta(run);
  }

  /** 时间轴上放东西的地方：正在跑的那一轮的块里，没有就放最外层。 */
  function timelineHost() {
    if (currentRun) return currentRun.body;
    return els.timelineInner;
  }

  function showError(message) {
    const banner = document.createElement('div');
    banner.className = 'err-banner';
    banner.textContent = message;
    timelineHost().appendChild(banner);
    setStatus('error', '出错');
  }

  /** 时间轴上的一条说明（不是错误：比如这次没有可交接的内容）。 */
  function showNote(message) {
    const banner = document.createElement('div');
    banner.className = 'note-banner';
    banner.textContent = message;
    timelineHost().appendChild(banner);
  }

  /**
   * 交接卡片：把上游 agent 的分析结果显示在下游时间轴的最上面。
   *
   * 这是页面自己渲染的，不走协议分片 —— 协议里没有"交接"这种类型，
   * 硬塞一个进去只会让 reducer 认不出。
   *
   * sourceName 是上游的名字：交接不再只有 A → B 这一条路，卡片上写死的
   * "AGENT A" 会把别的来源说错。
   */
  function renderHandoffCard(handoff, sourceName, host) {
    const card = document.createElement('div');
    card.className = 'handoff-card';

    const head = document.createElement('div');
    head.className = 'handoff-head';
    head.textContent = `← 已收到 ${sourceName} 的分析结果`;
    card.appendChild(head);

    const details = document.createElement('details');
    details.className = 'part-details';
    const summary = document.createElement('summary');
    summary.textContent = `展开 ${sourceName} 的原文`;
    const pre = document.createElement('pre');
    pre.textContent = handoff;
    details.append(summary, pre);
    card.appendChild(details);

    (host || timelineHost()).appendChild(card);
    return card;
  }

  // ---------- 轮次步骤 ----------

  /**
   * 把服务端给的默认步骤整理成面板能用的形状。
   *
   * 没有默认步骤是预期内的缺失（新建的 agent 本来就没有）。
   * 写坏了不是：形状不对就报出来，别把坏数据悄悄变成空步骤。
   */
  function normalizeStepList(list) {
    if (list === undefined || list === null) return [];
    if (!Array.isArray(list)) {
      showAlert('服务端给的默认步骤不是一个数组，这一栏按没有默认步骤开始。');
      return [];
    }
    const kept = [];
    list.forEach((entry, index) => {
      if (!entry || typeof entry !== 'object') {
        showAlert(`服务端给的默认步骤第 ${index} 条不是对象，已跳过这一条。`);
        return;
      }
      const title = typeof entry.title === 'string' ? entry.title : '';
      const instruction = typeof entry.instruction === 'string' ? entry.instruction : '';
      if (title.trim() === '' || instruction.trim() === '') {
        showAlert(`服务端给的默认步骤第 ${index} 条缺标题或缺指令，已跳过这一条。`);
        return;
      }
      kept.push({ title, instruction });
    });
    return kept;
  }

  /** 用户这一步的清单：存到服务端的形状（只有标题与指令）。 */
  function userStepsPayload() {
    return steps.map((step) => ({
      title: step.title,
      instruction: step.instruction,
    }));
  }

  /**
   * 把步骤配置存到服务端。
   *
   * 存的是用户这份（哪怕是空数组 —— 空数组是"用户清空了"，不是"没配过"）。
   * 「恢复默认步骤」存的是 null：抹掉用户这份改动，生效值回到代码里的默认。
   */
  function saveSteps(payload) {
    userEdited.steps = true;
    stepsConfigured = payload !== null;
    scheduleSave({ steps: payload });
    refreshStepsSummary();
  }

  function refreshStepsSummary() {
    if (!els.stepsSummary) return;
    if (steps.length === 0) {
      // 空着、但这个 agent 有默认步骤：多半是用户点过「清空」。
      // 把这件事写在摘要上 —— 不然只剩一个空步骤区，人只能猜。
      els.stepsSummary.textContent =
        defaultSteps.length > 0
          ? `轮次步骤（已清空 · 默认 ${defaultSteps.length} 步，点「恢复默认步骤」找回）`
          : '轮次步骤（未配置，一次跑完）';
    } else {
      els.stepsSummary.textContent = `轮次步骤（共 ${steps.length} 步）`;
    }
    if (els.stepsHint) {
      els.stepsHint.textContent =
        defaultSteps.length > 0
          ? `步骤按顺序跑：前一步的产出接到后一步。这个 agent 的默认步骤是 ${defaultSteps.length} 步。`
          : '步骤按顺序跑：前一步的产出接到后一步。';
    }
  }

  /** 重画步骤编辑区。 */
  function renderSteps() {
    if (!els.stepsList) return;
    els.stepsList.innerHTML = '';
    steps.forEach((step, index) => {
      const row = document.createElement('div');
      row.className = 'step-row';

      const head = document.createElement('div');
      head.className = 'step-row-head';

      const badge = document.createElement('span');
      badge.className = 'step-index';
      badge.textContent = String(index);

      const title = document.createElement('input');
      title.type = 'text';
      title.className = 'step-title';
      title.value = step.title;
      title.placeholder = '这一步做什么';
      title.addEventListener('input', () => {
        step.title = title.value;
        saveSteps(userStepsPayload());
      });

      const up = document.createElement('button');
      up.type = 'button';
      up.className = 'step-move';
      up.textContent = '↑';
      up.title = '上移这一步';
      up.addEventListener('click', () => moveStep(index, -1));

      const down = document.createElement('button');
      down.type = 'button';
      down.className = 'step-move';
      down.textContent = '↓';
      down.title = '下移这一步';
      down.addEventListener('click', () => moveStep(index, 1));

      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'step-move';
      del.textContent = '✕';
      del.title = '删掉这一步';
      del.addEventListener('click', () => removeStep(index));

      head.append(badge, title, up, down, del);

      const instruction = document.createElement('textarea');
      instruction.className = 'step-instruction';
      instruction.rows = 2;
      instruction.value = step.instruction;
      instruction.placeholder = '给这一步的具体指令';
      instruction.addEventListener('input', () => {
        step.instruction = instruction.value;
        saveSteps(userStepsPayload());
      });

      row.append(head, instruction);
      els.stepsList.appendChild(row);
    });
    refreshStepsSummary();
    // 步骤配好或改过之后，进度条上要跟着变：加一步就多一个 chip，
    // 删一步就少一个 —— 进度条显示的必须是"现在配好的那几步"。
    renderPlanChips();
  }

  function moveStep(index, delta) {
    const target = index + delta;
    if (target < 0 || target >= steps.length) return;
    const moved = steps.splice(index, 1)[0];
    steps.splice(target, 0, moved);
    saveSteps(userStepsPayload());
    renderSteps();
  }

  function removeStep(index) {
    steps.splice(index, 1);
    saveSteps(userStepsPayload());
    renderSteps();
  }

  function addStep() {
    steps.push({ title: '', instruction: '' });
    saveSteps(userStepsPayload());
    renderSteps();
    const inputs = els.stepsList.querySelectorAll('.step-title');
    const last = inputs[inputs.length - 1];
    if (last) last.focus();
  }

  // ---------- 权限 ----------

  /**
   * 把服务端给的权限目录整理成面板能用的形状。
   *
   * 没有目录是预期内的缺失（服务端还没回）。
   * 写坏了不是：形状不对就报出来，别把坏数据悄悄变成"没有权限" ——
   * 那会让 agent 变成什么也做不了，而页面上看不出哪儿出了问题。
   */
  function normalizePermissionCatalog(list) {
    if (list === undefined || list === null) return [];
    if (!Array.isArray(list)) {
      showAlert('服务端给的权限目录不是一个数组。');
      return [];
    }
    const kept = [];
    for (const entry of list) {
      if (!entry || typeof entry !== 'object') {
        showAlert('权限目录里有一项不是对象，已跳过。');
        continue;
      }
      const id = typeof entry.id === 'string' ? entry.id : '';
      if (id === '') {
        showAlert('权限目录里有一项缺 id，已跳过。');
        continue;
      }
      kept.push({
        id,
        label: typeof entry.label === 'string' ? entry.label : id,
        description: typeof entry.description === 'string' ? entry.description : '',
      });
    }
    return kept;
  }

  /**
   * 把权限配置存到服务端。
   *
   * 存的是用户勾的那几项（哪怕一项都不勾 —— 空数组是"用户全取消了"，
   * 不是"没配过"）。存过之后，这一轮真正生效的就是这份。
   */
  function savePermissions() {
    userEdited.permissions = true;
    scheduleSave({ permissions: Array.from(permissions) });
    renderPermissions();
  }

  function refreshPermissionsSummary() {
    if (!els.permissionsSummary) return;
    const count = permissions.size;
    if (!permissionCatalogLoaded) {
      els.permissionsSummary.textContent = '权限（还没从服务端取回来）';
    } else if (count === 0) {
      els.permissionsSummary.textContent = '权限（未授任何能力）';
    } else {
      els.permissionsSummary.textContent = `权限（已授 ${count} 项）`;
    }
    if (els.permissionsNote) {
      const names = permissionCatalog
        .filter((item) => permissions.has(item.id))
        .map((item) => item.label);
      els.permissionsNote.textContent =
        names.length === 0
          ? '没授的工具不会出现在这一轮的工具表里 —— 模型调不到它。'
          : `这一轮能用的能力：${names.join('、')}。没授的工具不会出现在工具表里 —— 模型调不到它。`;
    }
  }

  /** 重画权限区。 */
  function renderPermissions() {
    if (!els.permissionList) return;
    els.permissionList.innerHTML = '';
    if (permissionCatalog.length === 0) {
      const empty = document.createElement('span');
      empty.className = 'permission-empty';
      empty.textContent = permissionCatalogLoaded
        ? '服务端没有给权限目录。'
        : '权限目录还没从服务端取回来。';
      els.permissionList.appendChild(empty);
      refreshPermissionsSummary();
      return;
    }
    for (const item of permissionCatalog) {
      const label = document.createElement('label');
      label.className = 'permission-option';

      const box = document.createElement('input');
      box.type = 'checkbox';
      box.value = item.id;
      box.checked = permissions.has(item.id);
      box.addEventListener('change', () => {
        if (box.checked) permissions.add(item.id);
        else permissions.delete(item.id);
        savePermissions();
      });

      const text = document.createElement('span');
      text.className = 'permission-text';

      const title = document.createElement('span');
      title.className = 'permission-title';
      title.textContent = item.label;

      const desc = document.createElement('span');
      desc.className = 'permission-desc';
      desc.textContent = item.description;

      text.append(title, desc);
      label.append(box, text);
      els.permissionList.appendChild(label);
    }
    refreshPermissionsSummary();
  }

  /**
   * 装上服务端给的权限目录。
   *
   * 目录是全局的一份（所有 agent 共用一张权限表），每一栏勾哪几项是各的。
   * 目录里没有的 id 是过期配置 —— 服务端在给出生效值时已经把它们挑出去了
   * （见 profile.stalePermissions），这一层只负责把复选框铺出来。
   */
  function applyPermissionCatalog(list) {
    permissionCatalog = normalizePermissionCatalog(list);
    permissionCatalogLoaded = true;
    renderPermissions();
  }

  /**
   * 这一步要发出去的话：只有这一步自己的指令。
 *
 * 本轮输入由服务端拼（第一步那一次）：前端把它塞进 prompt 的话，
 * 同一个请求里会既有 prompt 里的那段、又有 roundPrompt 字段，白白重复一遍。
   *
   * 每一步仍然是一次独立请求（顺序由谁来调决定，不由模型自觉决定），
   * 但**不再把前面几步的产出重新贴一遍** —— 一轮的每一步共用同一个 pi 会话，
   * 会话本身带着前面的对话，重发只会让上下文按平方增长。
   *
   * 上游 agent 交来的结果不在这里拼：它跟着请求的 handoff 字段走，
   * 服务端会在最前面加上「【某某 的分析结果】」，与单步跑的交接一个格式。
   */
  function buildStepPrompt(step, index, total) {
    const title = step.title.trim() === '' ? '未命名' : step.title.trim();
    return (
      `【第 ${index} 步 / 共 ${total} 步 · ${title}】\n` +
      '只做这一步：做完就交出结论，后面的步骤由后续请求接着做（会话是连续的，前面做过的事不用重发）。\n' +
      step.instruction.trim()
    );
  }

  /** 时间轴上的步骤分隔卡：标出下面这一段是第几步。 */
  function renderRoundStepCard(index, step, total, host, roundId) {
    const card = document.createElement('div');
    card.className = 'round-step-card';
    // 这一步属于哪一轮：时间轴上会堆着好几轮，卡片得能认出自己的轮次。
    if (typeof roundId === 'string' && roundId !== '') card.dataset.round = roundId;

    const head = document.createElement('div');
    head.className = 'round-step-head';

    const name = document.createElement('span');
    name.className = 'round-step-name';
    const title = step.title.trim() === '' ? '未命名' : step.title.trim();
    name.textContent = `第 ${index} 步 / 共 ${total} 步 · ${title}`;

    const state = document.createElement('span');
    state.className = 'round-step-state';
    state.textContent = '进行中';

    head.append(name, state);
    card.appendChild(head);

    if (step.instruction.trim() !== '') {
      const body = document.createElement('div');
      body.className = 'round-step-body';
      body.textContent = step.instruction.trim();
      card.appendChild(body);
    }

    (host || timelineHost()).appendChild(card);
    return card;
  }

  /** 步骤卡收尾：写成"完成"或"失败"。 */
  function markRoundStepCard(card, label, failed) {
    const state = card.querySelector('.round-step-state');
    if (state) state.textContent = label;
    card.classList.add(failed ? 'failed' : 'done');
  }

  /** 一份 store 里现在有多少段文字：用来切出"这一步新产出"的那几段。 */
  function textPartCount(run) {
    if (!run) return 0;
    return run.store.steps.reduce(
      (sum, step) => sum + step.parts.filter((part) => part.kind === 'text').length,
      0,
    );
  }

  /** 一份 store 里最后一段文字回答；没有文字输出就返回空串。 */
  function finalAnswer(run) {
    if (!run) return '';
    const texts = run.store.steps.flatMap((step) => step.parts).filter((p) => p.kind === 'text');
    if (texts.length === 0) return '';
    return texts[texts.length - 1].text.trim();
  }

  function handleChunk(run, chunk) {
    applyChunk(run.store, chunk);
    renderRun(run);
    scrollToBottom();

    if (chunk.type === 'error') showError(chunk.errorText || 'unknown error');
    if (chunk.type === 'abort') showError(`已中止：${chunk.reason || 'aborted'}`);
    // 多步轮次里的一步跑完不算整轮跑完：状态与交接由整轮统一收尾。
    if (chunk.type === 'finish' && !currentPartOfRound) {
      setStatus('done', '完成');
      setRunning(false);
      if (finishHandler) finishHandler(finalAnswer(run));
    }
    if (run.store.unknownTypes.length > 0) {
      showAlert(
        `认不出的分片类型：${run.store.unknownTypes.join(', ')}。` +
          '这通常说明浏览器里跑的是旧版前端，请硬刷新（Ctrl+Shift+R）。',
      );
    }
  }

  /**
   * 把时间轴清空：回到空状态。
   *
   * 跑新一轮不再清时间轴了 —— 历史存在服务端，刷新页面还在。
   * 一次跑完就把前面的抹掉，等于自己把历史毁了。
   * 只有「清空历史」走这里。
   */
  function clearTimeline() {
    runs.length = 0;
    currentRun = null;
    els.timelineInner.innerHTML = '';
    els.timelineInner.classList.remove('has-content');
    els.timeline.scrollTop = 0;
    stickToBottom = true;
    if (els.emptyHint) {
      els.emptyHint.hidden = false;
      els.timelineInner.appendChild(els.emptyHint);
    }
  }

  /** 取消还没到点的下一轮：用户点了结束、或者手动又点了一次运行。 */
  function cancelLoop() {
    if (loopTimer !== null) {
      clearTimeout(loopTimer);
      loopTimer = null;
    }
    loopTarget = 0;
    loopDone = 0;
    refreshLoopState();
  }

  /**
   * 一轮跑完之后，按配置决定要不要再跑。
   *
   * 只有最新那一轮（mySeq 仍是当前序号）才配安排下一轮；用户点了结束、或者次数
   * 已用完，都不再排。第一轮由用户点击触发，loopContinuation 不再重置剩余轮数。
   */
  function scheduleLoopAfter(mySeq, options) {
    if (mySeq !== runSeq) return;
    if (abortedByUser) {
      cancelLoop();
      return;
    }
    const setting = loopSetting();
    if (!setting.enabled) {
      cancelLoop();
      return;
    }
    if (!options.loopContinuation) {
      loopTarget = setting.count;
      loopDone = 1;
    } else {
      loopDone += 1;
    }
    if (loopDone >= loopTarget) {
      const total = loopDone;
      loopTarget = 0;
      loopDone = 0;
      if (els.loopState) els.loopState.textContent = `循环完成：共 ${total} 轮。`;
      return;
    }
    const waitMs = Math.max(0, setting.intervalSeconds) * 1000;
    if (els.loopState) {
      els.loopState.textContent = waitMs > 0
        ? `第 ${loopDone} 轮已结束，${setting.intervalSeconds} 秒后开第 ${loopDone + 1} 轮。`
        : `第 ${loopDone} 轮已结束，马上开第 ${loopDone + 1} 轮。`;
    }
    loopTimer = setTimeout(() => {
      loopTimer = null;
      if (mySeq !== runSeq) return;
      if (abortedByUser) return;
      void run({ ...options, loopContinuation: true });
    }, waitMs);
  }

  /**
   * 中止当前这一轮。
   *
   * abort 掉 fetch 之后连接就断了：服务端收到 close 会把 agent 那一侧也 abort 掉，
   * 正在跑的那一轮是真的停下来，不只是前端不再往下显示。
   */
  function stop() {
    if (!running) {
      cancelLoop();
      setStatus('idle', '待运行');
      return;
    }
    abortedByUser = true;
    cancelLoop();
    controller.abort();
    running = false;
    setRunning(false);
    setStatus('stopped', '已中止');
    showNote('已中止本轮运行。');
  }

  /**
   * 面板的「运行」入口。
   *
   * 配了轮次步骤就按步骤一步步跑：每一步一次请求，顺序由这里定，
   * 前一步的产出接到后一步。一步都没配就一次跑完 —— 和以前一样。
   */
  async function run(options = {}) {
    // 手动点「运行」= 从头开始：取消还没到点的下一轮，并按次数重新计。
    if (!options.loopContinuation) cancelLoop();
    // 上一轮还在跑就把它停掉：新的一轮顶掉旧的一轮。
    if (controller) controller.abort();
    // 本轮的序号：收尾时只有最新的一轮才配改状态与按钮，
    // 上一轮被新的一轮顶掉时，它的收尾不许反过来覆盖新一轮。
    const mySeq = ++runSeq;
    // 配置改动先落到服务端：这一轮必须用新配置。
    // 存不上就不跑 —— 跑出来的那份不是用户配的那份，说清楚比跑错强。
    const saved = await flushConfigSave();
    if (!saved) {
      showError('配置没能存到服务端，这一轮没有跑。');
      return;
    }
    if (steps.length > 0) {
      await runRound(options, mySeq);
    } else {
      await runOne(options, mySeq);
    }
    // 单一收尾点：一轮结束之后，循环要不要继续在这里决定。
    scheduleLoopAfter(mySeq, options);
  }

  /**
   * 按配好的步骤跑一整轮。
   *
   * 每一步是一次独立的请求 —— 顺序天然成立，不用指望模型照着清单做事。
   * 某一步出错就停在那里：不接着往下跑，也不把半截结果交出去。
   */
  async function runRound(options, mySeq) {
    // 开跑前先抄一份步骤：跑的过程中用户改了配置，这一轮也不跟着变。
    const plan = steps.map((step) => ({
      title: step.title,
      instruction: step.instruction,
    }));
    // 交接是这一轮的最后一步：默认无条件交；配了交接条件，就由这一步的模型判定。
    // 一条步骤都没配的 agent（单次运行）不插这一步，交接保持原来的自动行为。
    const handoffTargets = api.targets();
    const hasHandoffStep = plan.length > 0 && handoffTargets.length > 0;
    if (hasHandoffStep) {
      plan.push({
        title: '交接结论',
        instruction: buildHandoffStepInstruction(handoffTargets, api.handoffCondition()),
        handoff: true,
      });
    }
    let handoffWanted = true;

    abortedByUser = false;
    running = true;
    setRunning(true);
    setStatus('running', '运行中');
    // 步骤进度条按这一轮的计划重画：每一步一个 chip，开跑前都还没开始。
    renderPlanChips();

    const handoff = typeof options.handoff === 'string' ? options.handoff : '';
    const handoffFrom =
      typeof options.handoffFrom === 'string' && options.handoffFrom.trim() !== ''
        ? options.handoffFrom.trim()
        : DEFAULT_HANDOFF_SOURCE;
    const roundInput = (
      options.instruction !== undefined ? options.instruction : els.prompt.value
    ).trim();

    // 预期内的缺失：没有输入、上游也没交来东西、步骤里也没写指令 —— 没活可干。
    const hasInstruction = plan.some((step) => step.instruction.trim() !== '');
    if (roundInput === '' && handoff === '' && !hasInstruction) {
      showError('没有可执行的内容：输入框是空的，步骤里也没写指令。');
      running = false;
      setRunning(false);
      return;
    }

    // 整轮的编号：这几步在服务端是同一个轮次里的几步。
    const roundId = `round-${Date.now()}-${++roundSeq}`;
    const outputs = [];
    for (let index = 0; index < plan.length; index += 1) {
      // 被新的一轮顶掉了：这一轮到此为止。
      if (mySeq !== runSeq) return;
      const step = plan[index];
      // 进度条上这一步标成"正在做"，后面几步还是"没轮到"。
      setPlanChipState(index, 'active');
      const prompt = buildStepPrompt(step, index, plan.length);
      // 上游的结果只接给第一步：后面每一步的上下文由同一个 pi 会话带着。
      const stepOptions = {
        ...options,
        instruction: prompt,
        partOfRound: true,
        handoff: index === 0 ? handoff : '',
        // 这一步在整轮里的位置：跟着请求走，服务端原样记进对话记录，
        // 刷新页面时才能把记录重新排成"第几步"。
        round: {
          roundId,
          stepIndex: index,
          stepTitle: step.title,
          stepInstruction: step.instruction,
          totalSteps: plan.length,
          roundPrompt: roundInput,
        },
      };
      const result = await runOne(stepOptions, mySeq);
      if (mySeq !== runSeq) return;
      if (!result.ok) {
        setPlanChipState(index, 'failed');
        if (result.card) markRoundStepCard(result.card, '失败', true);
        setStatus('error', `第 ${index} 步失败`);
        running = false;
        setRunning(false);
        return;
      }
      const stepText = result.text;
      setPlanChipState(index, 'filled');
      if (result.card) {
        markRoundStepCard(result.card, stepText === '' ? '完成（无产出）' : '完成', false);
      }

      // 这一步调了 endRound：本轮到此为止，后面的步骤不再发请求。
      // 判定依据是工具调用（结构化），不是模型在文字里写的"本轮结束"。
      if (result.ended) {
        const skipped = plan.length - index - 1;
        for (let rest = index + 1; rest < plan.length; rest += 1) {
          setPlanChipState(rest, 'skipped');
        }
        if (result.card) markRoundStepCard(result.card, '完成（本轮结束）', false);
        showNote(
          `第 ${index + 1} 步判定本轮到此结束` +
            (skipped > 0 ? `，后面 ${skipped} 步跳过。` : '。'),
        );
        outputs.push({
          index,
          title: step.title.trim() === '' ? '未命名' : step.title.trim(),
          text: stepText,
        });
        if (finishHandler) finishHandler(stepText);
        setStatus('done', '完成');
        running = false;
        setRunning(false);
        return;
      }

      if (step.handoff) handoffWanted = result.handoff === true;
      outputs.push({
        index,
        title: step.title.trim() === '' ? '未命名' : step.title.trim(),
        text: stepText,
        handoff: step.handoff === true,
      });
    }

    // 整轮跑完：交接步骤说要交，才把结果交出去。
    // 交出去的是"结论"那一步的产出 —— 交接步骤自己的回答只是"已交接"这句话，
    // 不该当下游的输入。
    const contentOutputs = outputs.filter((entry) => entry.handoff !== true);
    const finalText =
      contentOutputs.length > 0
        ? contentOutputs[contentOutputs.length - 1].text
        : outputs.length > 0
          ? outputs[outputs.length - 1].text
          : '';
    if (!handoffWanted) {
      showNote('交接步骤判定这次不交接：这一轮的结论留在这里。');
      setStatus('done', '完成');
      running = false;
      setRunning(false);
      return;
    }
    // 先把结果发出去，再把状态标成完成 —— 交接请求是在这一刻发起的，
    // 顺序反了的话，页面已经显示"完成"，下游却还没收到东西。
    if (finishHandler) finishHandler(finalText);
    setStatus('done', '完成');
    running = false;
    setRunning(false);
  }

  /**
   * 跑一次 agent（一个请求）。
   *
   * 一次请求 = 时间轴上的一个块。块里是这一轮自己的产出。
   *
   * options.handoff 有值时走 POST：上游的分析结果放请求体里（可能很长，
   * 塞 URL 会撞上长度限制）。没有交接就走 GET，和以前一样。
   *
   * options.partOfRound 为真时，这一次是多步轮次里的一步：
   * 不改面板状态、也不触发交接 —— 整轮的收尾由 runRound 管。
   */
  async function runOne(options, mySeq) {
    // 用户又跑了一轮：上一轮"还没到点"的延迟交接作废。
    clearPendingHandoffs(pendingHandoffTimers);
    controller = new AbortController();
    currentPartOfRound = options.partOfRound === true;

    const handoff = typeof options.handoff === 'string' ? options.handoff : '';
    // 谁把这轮的结果发过来的（agent 身份）。没上游就是空串。
    handoffSourceId = typeof options.handoffFromId === 'string' ? options.handoffFromId : '';
    // 上游的名字：没给就用固定的那个（直接调用接口时的老规矩）。
    const handoffFrom =
      typeof options.handoffFrom === 'string' && options.handoffFrom.trim() !== ''
        ? options.handoffFrom.trim()
        : DEFAULT_HANDOFF_SOURCE;
    const prompt = (
      options.instruction !== undefined ? options.instruction : els.prompt.value
    ).trim();

    // 这一轮（一次请求）在时间轴上占一个块。
    // 多步轮次的每一步各是一次请求，也就各有各的块。
    // 本轮输入只写在第一步的块上：后面几步的模型已经看过它了，
    // 每块都贴一遍只会让时间轴看起来像把同一个问题问了四遍。
    const showRoundPrompt =
      !options.round || (options.round.stepIndex === 0 ? options.round.roundPrompt : '');
    const run = beginRun({
      promptText: options.round ? showRoundPrompt : prompt,
      startedAt: Date.now(),
      roundId: options.round ? options.round.roundId : '',
    });
    let card = null;

    if (!currentPartOfRound) {
      abortedByUser = false;
      running = true;
      setRunning(true);
      setStatus('running', '运行中');

      // 预期内的缺失：没有要执行的话就不该启动 agent，直接说清楚，不去猜。
      // 有上游交接时不算"没有要执行的"：上游的分析结果就是要接着做的事。
      if (prompt === '' && handoff === '') {
        showError('没有可执行的内容：输入框是空的，也没有上游交接。');
        running = false;
        setRunning(false);
        finishRun(run, 'error');
        return { ok: false, text: '', card: null };
      }
    }

    // 交接卡片：上游的结果放在这一轮自己的块里。
    if (handoff !== '') renderHandoffCard(handoff, handoffFrom, run.body);

    // 多步轮次里的一步：块里写明这是第几步、这一步要做什么。
    if (options.round) {
      card = renderRoundStepCard(
        options.round.stepIndex,
        { title: options.round.stepTitle, instruction: options.round.stepInstruction },
        options.round.totalSteps,
        run.body,
        options.round.roundId,
      );
    }

    const init = { signal: controller.signal };
    let url;
    // 多步轮次一律走 POST：步骤的提示词与 roundId 走请求体，比塞 URL 稳，
    // 塞进 URL 会撞上服务端请求行的长度上限。
    if (handoff === '' && !currentPartOfRound) {
      // 简单的一轮走 GET：带上前端版本号与 agent 身份。
      // 版本号让服务端认出旧版前端，身份让服务端知道这一栏该跑哪个 agent。
      //
      // 提示词与权限不在请求里：它们存在服务端，服务端按自己存的那份来 ——
      // 页面上看到的就是真正生效的，不会出现"界面改了但这一轮没改"。
      url =
        `${API_BASE}/api/run?prompt=${encodeURIComponent(prompt)}` +
        `&v=${FRONTEND_VERSION}` +
        `&profile=${encodeURIComponent(root.dataset.profile)}`;
    } else {
      // 有上游结果、或者是多步轮次里的一步，就走 POST：
      // 提示词可能很长，塞 URL 里会被截断。
      url = `${API_BASE}/api/run`;
      init.method = 'POST';
      init.headers = { 'Content-Type': 'application/json' };
      const body = {
        prompt,
        v: FRONTEND_VERSION,
        profile: root.dataset.profile,
        handoff,
      };
      // 来源名字：下游看到的"这是谁的结果"得跟着实际发送方走。
      if (handoff !== '') body.handoffFrom = handoffFrom;
      // 这一步在整轮里的位置：服务端原样记进对话记录。
      if (options.round) {
        body.roundId = options.round.roundId;
        body.stepIndex = options.round.stepIndex;
        body.stepTitle = options.round.stepTitle;
        body.stepInstruction = options.round.stepInstruction;
        body.totalSteps = options.round.totalSteps;
        // 本轮输入只在第一步发：服务端会记住它，会话丢失时用来自动补上下文。
        if (options.round.stepIndex === 0) body.roundPrompt = options.round.roundPrompt;
      }
      init.body = JSON.stringify(body);
    }

    try {
      const res = await fetch(url, init);
      if (!res.ok || !res.body) {
        // 光甩一个 "HTTP 400" 给用户没有任何用 —— 服务端通常在响应体里
        // 写了原因（哪个字段不认、怎么修），把它读出来显示。
        let detail = '';
        try {
          const data = await res.json();
          if (data && typeof data.error === 'string') detail = data.error;
          if (data && typeof data.hint === 'string') {
            detail = detail ? `${detail} ${data.hint}` : data.hint;
          }
        } catch {
          // 响应体不是 JSON（网关错误之类），那就只报状态码。
        }
        throw new Error(detail || `HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let index;
        while ((index = buffer.indexOf('\n\n')) >= 0) {
          const raw = buffer.slice(0, index);
          buffer = buffer.slice(index + 2);
          for (const line of raw.split('\n')) {
            if (!line.startsWith('data:')) continue;
            const payload = line.slice(5).trim();
            if (payload === '') continue;
            if (payload === '[DONE]') continue;
            handleChunk(run, JSON.parse(payload));
          }
        }
      }

      if (!currentPartOfRound && run.store.error === null && run.store.aborted === null) {
        setStatus('done', '完成');
      }
      finishRun(run, runStatus(run));
    } catch (err) {
      if (err && err.name === 'AbortError') {
        // 用户点的「结束」：状态已经在 stop() 里设过了，不重复设。
        if (mySeq === runSeq && !abortedByUser) {
          setStatus('stopped', '已中止');
          showNote('已中止本轮运行。');
        }
        finishRun(run, 'aborted');
        return { ok: false, text: '', card };
      }
      if (mySeq !== runSeq) return { ok: false, text: '', card };
      showError((err && err.message) || String(err));
      finishRun(run, 'error');
      return { ok: false, text: '', card };
    } finally {
      if (mySeq === runSeq && !currentPartOfRound) {
        running = false;
        setRunning(false);
      }
    }
    return {
      ok: run.store.error === null && run.store.aborted === null,
      text: runText(run),
      card,
      // 这一步有没有要求"本轮到此结束"（endRound 工具调用）。
      ended: roundEnded(run.store),
      handoff: handoffRequested(run.store),
    };
  }

  /** 这一轮跑完是什么结果：出错、中止、还是完成。 */
  function runStatus(run) {
    if (run.store.error !== null) return 'error';
    if (run.store.aborted !== null) return 'aborted';
    return 'done';
  }

  /** 这一轮所有文字产出拼起来（一步的产出就是它自己那份）。 */
  function runText(run) {
    if (!run) return '';
    return run.store.steps
      .flatMap((step) => step.parts)
      .filter((part) => part.kind === 'text')
      .map((part) => part.text)
      .join('')
      .trim();
  }

  /**
   * 把服务端存的一条对话记录画回时间轴。
   *
   * 存下来的是原样的协议分片，回放时用同一份 reducer 还原 ——
   * 画出来的东西跟当时跑的时候一模一样。
   */
  function replayRecord(record) {
    if (!record || typeof record !== 'object') {
      throw new Error('对话记录里有一条不是一个对象');
    }
    const startedAt = typeof record.startedAt === 'number' ? record.startedAt : Date.now();
    // 这一轮问的是什么：多步轮次里记的是整轮的输入，其余记的是这一步的。
    const promptText =
      typeof record.roundPrompt === 'string' && record.roundPrompt !== ''
        ? record.roundPrompt
        : typeof record.prompt === 'string'
          ? record.prompt
          : '';
    const run = beginRun({
      id: typeof record.runId === 'string' && record.runId !== '' ? record.runId : undefined,
      promptText,
      startedAt,
      roundId: typeof record.roundId === 'string' ? record.roundId : '',
    });

    // 上游交来的结果：把那张交接卡片原样画回来。
    if (typeof record.handoff === 'string' && record.handoff !== '') {
      renderHandoffCard(
        record.handoff,
        typeof record.handoffFrom === 'string' && record.handoffFrom !== ''
          ? record.handoffFrom
          : DEFAULT_HANDOFF_SOURCE,
        run.body,
      );
    }

    // 多步轮次里的一步：把"第几步"那张卡也画回来。
    if (Number.isInteger(record.stepIndex)) {
      const card = renderRoundStepCard(
        record.stepIndex,
        {
          title: typeof record.stepTitle === 'string' ? record.stepTitle : '',
          instruction: typeof record.stepInstruction === 'string' ? record.stepInstruction : '',
        },
        Number.isInteger(record.totalSteps) ? record.totalSteps : record.stepIndex + 1,
        run.body,
        typeof record.roundId === 'string' ? record.roundId : '',
      );
      markRoundStepCard(card, '完成', false);
    }

    // 协议分片：跟当时跑的时候一模一样的一份。
    const chunks = Array.isArray(record.chunks) ? record.chunks : [];
    for (const chunk of chunks) {
      applyChunk(run.store, chunk);
    }
    renderRun(run);

    // 收尾：状态与耗时按当初记下来的写，不重新算。
    run.finishedAt = typeof record.finishedAt === 'number' ? record.finishedAt : startedAt;
    run.durationMs = typeof record.durationMs === 'number' ? record.durationMs : 0;
    const status =
      record.status === 'aborted' ? 'aborted' : record.status === 'error' ? 'error' : 'done';
    finishRun(run, status);
  }

  /**
   * 读出这一栏的对话记录，画回时间轴。
   *
   * 记录读不出来时报错给用户看：给它画一个空白的过去，等于骗人说
   * "这一栏从来没跑过"。
   */
  async function loadHistory() {
    let records;
    try {
      records = await getChats(root.dataset.profile);
    } catch (err) {
      showAlert(`读不出这一栏的对话记录：${(err && err.message) || String(err)}`);
      return;
    }
    clearTimeline();
    for (const record of records) {
      replayRecord(record);
    }
    // 回放完了没有"正在跑的那一轮"：往后的提示落在最外层，
    // 不落进最后一块历史里。
    currentRun = null;
  }

  /**
   * 清掉这一栏的对话记录：服务端那份删掉，页面上的时间轴也清空。
   *
   * 两边必须一起清 —— 只清一边的话，刷新页面它又回来了。
   */
  /**
   * 重置这一栏的会话：对话记录与模型上下文都清掉。
   *
   * 全局设置里也有同一个动作；这里是就近的一键版 —— 每一栏自己就能点。
   */
  async function resetThisSession() {
    try {
      const res = await fetch(`${API_BASE}/api/session/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agentId: root.dataset.profile }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        showAlert(`会话没能重置：${data.error || `HTTP ${res.status}`}`);
        return;
      }
      clearTimeline();
      showNote('这一栏的会话已重置：对话记录与模型上下文都清掉了。');
    } catch (err) {
      showAlert(`会话没能重置：${err && err.message ? err.message : String(err)}`);
    }
  }

  async function clearHistory() {
    try {
      await deleteChats(root.dataset.profile);
    } catch (err) {
      showAlert(`对话记录没能清掉：${(err && err.message) || String(err)}`);
      return;
    }
    clearTimeline();
  }

  els.form.addEventListener('submit', (event) => {
    event.preventDefault();
    run();
  });

  els.stopBtn.addEventListener('click', () => {
    stop();
  });

  if (els.instructions) {
    els.instructions.addEventListener('input', () => {
      // 改了就记成"用户改过"，并把新内容存到服务端。
      // 存在服务端而不是浏览器里：换个浏览器、换台机器、服务重启之后都在。
      userEdited.instructions = true;
      instructionsOverride = els.instructions.value;
      refreshInstructionsState();
      scheduleSave({ instructions: els.instructions.value });
    });
  }
  if (els.resetInstructions) {
    els.resetInstructions.addEventListener('click', resetInstructions);
  }
  /**
   * 保存名字。
   *
   * 名字是即时生效的（页头、侧栏、交接勾选项都要跟着变），
   * 所以存成功之后自己把这几个地方刷一遍 —— saveConfig 不会重画界面
   * （它怕把正在打字的输入冲掉）。
   */
  async function saveName(name) {
    const merged = await saveConfig({ name });
    if (!merged) return;
    applyNameToDom(merged.name);
    if (els.nameInput) els.nameInput.value = merged.name;
    agentNames.set(api.profileId, merged.name);
    nameCustomized = Boolean(merged.customized && merged.customized.name);
    refreshNameState();
    refreshAllTargets();
  }

  if (els.nameInput) {
    // 打字时即时改标题（看得见效果），但只在收手（change）时才发请求。
    els.nameInput.addEventListener('input', () => {
      userEdited.name = true;
      const trimmed = els.nameInput.value.trim();
      if (trimmed === '') {
        refreshNameState('名字不能为空（还没保存）');
        return;
      }
      applyNameToDom(trimmed);
      agentNames.set(api.profileId, trimmed);
      refreshAllTargets();
      nameCustomized = true;
      refreshNameState();
    });
    els.nameInput.addEventListener('change', () => {
      const trimmed = els.nameInput.value.trim();
      if (trimmed === '') {
        if (els.nameInput) els.nameInput.value = displayName;
        refreshNameState();
        return;
      }
      void saveName(trimmed);
    });
  }
  if (els.resetName) {
    els.resetName.addEventListener('click', async () => {
      const merged = await saveConfig({ name: null });
      if (!merged) return;
      userEdited.name = false;
      if (els.nameInput) els.nameInput.value = merged.name;
      applyNameToDom(merged.name);
      agentNames.set(api.profileId, merged.name);
      nameCustomized = false;
      refreshNameState();
      refreshAllTargets();
    });
  }

  const saveHandoffCondition = () => {
    refreshHandoffConditionState();
    void saveConfig({
      handoffCondition: els.handoffConditionText ? els.handoffConditionText.value.trim() : '',
    });
  };
  if (els.handoffConditionText) {
    els.handoffConditionText.addEventListener('change', saveHandoffCondition);
  }

  if (els.allowHandoffBack) {
    els.allowHandoffBack.addEventListener('change', () => {
      refreshAllowHandoffBackState();
      void saveConfig({ allowHandoffBack: Boolean(els.allowHandoffBack.checked) });
    });
  }

  if (els.resetSessionHandoff) {
    els.resetSessionHandoff.addEventListener('change', () => {
      userEdited.handoffReset = true;
      refreshResetSessionHandoffState();
      void saveConfig({ resetSessionOnHandoff: Boolean(els.resetSessionHandoff.checked) });
    });
  }

  if (els.handoffDelay) {
    els.handoffDelay.addEventListener('change', () => {
      const raw = Number(els.handoffDelay.value);
      if (!Number.isInteger(raw) || raw < 0 || raw > 3600) {
        showAlert('交接延迟必须是 0 到 3600 之间的整数秒。');
        els.handoffDelay.value = String(api.handoffDelaySeconds());
        refreshHandoffDelayState();
        return;
      }
      userEdited.handoffDelay = true;
      refreshHandoffDelayState();
      void saveConfig({ handoffDelaySeconds: raw });
    });
  }

  // 循环：勾选/改数值都存到服务端，改完立刻反映到折叠栏与状态行。
  for (const el of [els.loopEnabled, els.loopCount, els.loopInterval]) {
    if (!el) continue;
    el.addEventListener('input', () => refreshLoopState());
    el.addEventListener('change', () => {
      userEdited.loop = true;
      refreshLoopState();
      scheduleSave({ loop: loopSetting() });
    });
  }

  if (els.deleteBtn) {
    els.deleteBtn.addEventListener('click', armDelete);
  }
  if (els.clearHistoryBtn) {
    els.clearHistoryBtn.addEventListener('click', () => {
      void clearHistory();
    });
  }
  if (els.resetSessionBtn) {
    els.resetSessionBtn.addEventListener('click', () => {
      void resetThisSession();
    });
  }

  // 配置（提示词、步骤、权限、交接目标）都存在服务端。
  // 面板先空着：/api/health 回来之后，按每个 agent 自己那份配置铺上去。
  renderSteps();
  renderPermissions();
  if (els.addStepBtn) {
    els.addStepBtn.addEventListener('click', addStep);
  }
  if (els.clearStepsBtn) {
    els.clearStepsBtn.addEventListener('click', () => {
      steps = [];
      saveSteps([]);
      renderSteps();
    });
  }
  if (els.restoreStepsBtn) {
    els.restoreStepsBtn.addEventListener('click', () => {
      if (!defaultsLoaded) {
        showAlert('默认步骤还没从服务端取回来，等页面状态读完再点一次。');
        return;
      }
      if (defaultSteps.length === 0) {
        showAlert('这个 agent 没有默认步骤：新建的 agent 都从空白开始。');
        return;
      }
      steps = defaultSteps.map((step) => ({ ...step }));
      saveSteps(null);
      renderSteps();
    });
  }

  /**
   * 用户滚窗口时更新"贴底"状态。
   *
   * 往上是翻历史（stickToBottom 变假），往下回到底部又变真。
   * 程序自动滚到底部同样会触发这个事件 —— 那时它算出来仍是贴底，一致。
   */
  els.timeline.addEventListener('scroll', () => {
    stickToBottom = atBottom();
  });

  /** 名字落到页面上：页头、配置页标题、左侧栏。交接勾选项由 refreshAllTargets 刷新。 */
  function applyNameToDom(name) {
    displayName = name;
    els.name.textContent = name;
    if (els.configTitle) els.configTitle.textContent = `${name} · 配置`;
    syncSidebar();
  }

  const api = {
    root,
    profileId: root.dataset.profile,
    /** 面板的名字：交接配置的勾选项与交接卡片都用它。 */
    name() {
      return displayName;
    },
    /** 配置里写死的 agent 删不掉，删的只是页面上的面板。 */
    isBuiltin() {
      return root.dataset.builtin === 'true';
    },
    setName(name) {
      applyNameToDom(name);
    },
    /** 当前状态：状态点的样式与文字。侧边栏的圆点跟着它走。 */
    status() {
      return { kind: statusKind, label: statusLabel };
    },
    setPrompt(text) {
      els.prompt.value = text;
    },
    /**
     * 装上服务端给的这个 agent 的配置。
     *
     * 一份配置里有三样东西：生效的值、代码里的默认值、哪些字段是用户改过的。
     * 三者必须分开 —— 用户清空过的步骤与"用户还没配过步骤"不是一回事。
     */
    applyProfile(profile) {
      applyProfile(profile);
    },
    /**
     * 装上服务端给的权限目录。
     *
     * 目录是全局的（所有 agent 共用一份权限表），
     * 但每一栏的勾选状态是各的。
     */
    setPermissionCatalog(list) {
      applyPermissionCatalog(list);
    },
    /**
     * 读出这一栏的对话记录，画回时间轴。
     *
     * 历史存在服务端：刷新页面、服务重启之后，时间轴上都还是原来那些轮次。
     */
    loadHistory,
    /**
     * 清掉这一栏的对话记录：服务端那份删掉，页面上的时间轴也清空。
     */
    clearHistory,
    /**
     * 只清页面上的时间轴（服务端那份已经由别处清过了）。
     *
     * 「重置会话」是服务端先清记录、再由这里清时间轴 —— 两边一起清，
     * 刷新页面它才不会再回来。
     */
    clearTimelineOnly: clearTimeline,
    /** 输入框里现在写的（对下游 agent 来说就是"要求"）。 */
    getPrompt() {
      return els.prompt.value.trim();
    },
    /** 跑完之后的回调，收到本轮最后一段文字回答。 */
    setOnFinish(handler) {
      finishHandler = handler;
    },
    /** 交接目标：这一栏跑完后结果要发给谁（可以多选）。 */
    targets() {
      return Array.from(handoffTargets);
    },
    /** 交接条件：一句人话，交给轮次最后一步的模型判定。空串 = 无条件。 */
    handoffCondition() {
      const condition = profileState && profileState.handoffCondition;
      return typeof condition === 'string' ? condition.trim() : '';
    },
    /** 允许把结果回传给刚把它发给我的那一栏（审查—返修循环要打开）。 */
    allowHandoffBack() {
      return profileState ? Boolean(profileState.allowHandoffBack) : false;
    },
    /** 交接延迟（秒）：0 = 跑完立刻发。 */
    handoffDelaySeconds() {
      const value = profileState && profileState.handoffDelaySeconds;
      return typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : 0;
    },
    /** 延迟执行一次交接；定时器归这一栏管，跑新一轮或删面板时会被取消。 */
    scheduleHandoff(run, seconds) {
      const timer = setTimeout(() => {
        pendingHandoffTimers.delete(timer);
        run();
      }, seconds * 1000);
      pendingHandoffTimers.add(timer);
    },
    /** 设置交接目标（agent 身份列表）。 */
    setTargets(ids) {
      handoffTargets.clear();
      for (const id of ids) handoffTargets.add(id);
      renderHandoffTargets();
    },
    /** 重画交接配置的勾选项（页面上有 agent 新增或消失时）。 */
    refreshTargets: renderHandoffTargets,
    /**
     * 交接目标与服务端认识的 agent 对齐：已经不存在的目标摘掉并存回去。
     *
     * 页面上删掉一个 agent 时要调它 —— 别人勾了它就交不出去了。
     */
    syncTargets,
    /** 从页面上摘掉这个面板（正在跑的一轮同时中止）。 */
    destroy() {
      if (controller) controller.abort();
      clearPendingHandoffs(pendingHandoffTimers);
      root.remove();
    },
    /** 在时间轴上留一条说明。 */
    note(message) {
      showNote(message);
    },
    run,
  };

  // 「配置」进这个 agent 自己的配置页；「回到聊天」回聊天框。
  if (els.configBtn) {
    els.configBtn.addEventListener('click', () => setActiveAgent(api.profileId, 'config'));
  }
  if (els.backToChat) {
    els.backToChat.addEventListener('click', () => setActiveAgent(api.profileId, 'chat'));
  }

  // 跑完之后的收尾：把本轮结果交给交接配置里勾选的那些 agent。
  finishHandler = (answer) => deliverResult(api, answer, { skipProfileId: handoffSourceId });

  return api;
}

/**
 * 页面上所有面板。
 *
 * 运行时可以往里加 —— 「新建 Agent」造出来的面板也在这儿。
 * 首页模板里的那份不算：模板内容不在文档树里，querySelectorAll 看不到它。
 */
document.querySelectorAll('.panel').forEach((root) => panels.push(createPanel(root)));

/**
 * 全局设置：运行环境主机与工作目录。
 *
 * 所有 agent 共用这两项，服务端存着，改完立刻对下一次运行生效。
 * 侧栏底部一直显示当前生效的值：出问题时一眼能看出它在哪儿跑。
 */
let globalSettingsDefaults = null;

function renderGlobalSettings(data) {
  const settings = (data && data.settings) || {};
  if (settingsMetaEl) {
    const host = settings.agentHost || '?';
    const dir = settings.agentWorkdir || '?';
    settingsMetaEl.textContent = `${host} · ${dir}`;
  }
  if (settingsStateEl) {
    const problem = data && data.envProblem;
    settingsStateEl.textContent = problem
      ? `现在不能跑：${problem}`
      : `当前主机 ${(data && data.host) || '?'}，环境正常。`;
    settingsStateEl.classList.toggle('bad', Boolean(problem));
  }
}

async function loadGlobalSettings() {
  try {
    const res = await fetch(`${API_BASE}/api/settings`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    globalSettingsDefaults = data.defaults || null;
    if (settingsHostEl) settingsHostEl.value = (data.settings && data.settings.agentHost) || '';
    if (settingsWorkdirEl) {
      settingsWorkdirEl.value = (data.settings && data.settings.agentWorkdir) || '';
    }
    renderGlobalSettings(data);
    return data;
  } catch (err) {
    showAlert(`读全局设置失败：${err && err.message ? err.message : String(err)}`);
    return null;
  }
}

async function saveGlobalSettings() {
  const body = {
    agentHost: settingsHostEl ? settingsHostEl.value : '',
    agentWorkdir: settingsWorkdirEl ? settingsWorkdirEl.value : '',
  };
  try {
    const res = await fetch(`${API_BASE}/api/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      if (settingsStateEl) {
        settingsStateEl.textContent = `没存上：${data.error || `HTTP ${res.status}`}`;
        settingsStateEl.classList.add('bad');
      }
      return;
    }
    renderGlobalSettings(data);
  } catch (err) {
    showAlert(`存全局设置失败：${err && err.message ? err.message : String(err)}`);
  }
}

/**
 * 填「重置会话」那一栏的 agent 列表。
 *
 * 名单来自服务端给的 agent 清单（页面上的自定义 agent 也在里面）。
 * 第一项是"全部 agent"。
 */
function fillSettingsAgentList() {
  if (!settingsAgentEl) return;
  const names = [...agentNames.entries()].map(([id, name]) => ({ id, name }));
  names.sort((a, b) => a.name.localeCompare(b.name));
  settingsAgentEl.innerHTML = '';
  const all = document.createElement('option');
  all.value = '__all__';
  all.textContent = '全部 agent';
  settingsAgentEl.appendChild(all);
  for (const entry of names) {
    const option = document.createElement('option');
    option.value = entry.id;
    option.textContent = entry.name;
    settingsAgentEl.appendChild(option);
  }
}

/**
 * 重置 agent 会话：清掉它的对话记录与模型上下文。
 *
 * 服务端先清（记录 + 会话），页面再清时间轴 —— 只清一边的话，刷新页面它又回来了。
 */
async function resetAgentSession(agentId) {
  const body = agentId === '__all__' ? { all: true } : { agentId };
  const res = await fetch(`${API_BASE}/api/session/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  const affected = Array.isArray(data.agents) ? data.agents : [];
  for (const panel of panels) {
    if (affected.includes(panel.profileId)) panel.clearTimelineOnly();
  }
  return data;
}

/**
 * 组的管理：每个组一块（名字 + 成员），外加一块"新建组"。
 *
 * 成员勾选与保存都在这一块里做完，改完服务端存下来，页面重新读一次清单。
 */
function renderGroupEditors() {
  if (!settingsGroupsEl) return;
  settingsGroupsEl.innerHTML = '';
  for (const group of knownGroups.values()) {
    settingsGroupsEl.appendChild(groupEditor(group, false));
  }
  settingsGroupsEl.appendChild(groupEditor(null, true));
}

function groupEditor(group, isNew) {
  const block = document.createElement('div');
  block.className = 'settings-group';

  const name = document.createElement('input');
  name.type = 'text';
  name.value = group ? group.name : '';
  name.placeholder = '组的名字';
  name.dataset.role = 'groupName';

  const members = document.createElement('div');
  members.className = 'settings-group-members';
  const picked = new Set(group ? group.members : []);
  for (const [id, agentName] of agentNames) {
    const label = document.createElement('label');
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.value = id;
    box.checked = picked.has(id);
    box.addEventListener('change', () => {
      if (box.checked) picked.add(id);
      else picked.delete(id);
    });
    const text = document.createElement('span');
    text.textContent = agentName;
    label.append(box, text);
    members.appendChild(label);
  }

  const actions = document.createElement('div');
  actions.className = 'settings-group-actions';
  const save = document.createElement('button');
  save.type = 'button';
  save.dataset.role = 'groupSave';
  save.textContent = isNew ? '新建组' : '保存这个组';
  save.addEventListener('click', () => {
    const body = { name: name.value, members: Array.from(picked) };
    const url = isNew
      ? `${API_BASE}/api/groups`
      : `${API_BASE}/api/groups/${encodeURIComponent(group.id)}`;
    fetch(url, {
      method: isNew ? 'POST' : 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          showAlert(`组没存上：${data.error || `HTTP ${res.status}`}`);
          return;
        }
        await refreshGroups();
        renderGroupEditors();
        // 新建组用的那个对话框建完就收起来。
        if (newGroupDialogEl && newGroupDialogEl.open && typeof newGroupDialogEl.close === 'function') {
          newGroupDialogEl.close();
        }
      })
      .catch((err) => showAlert(`组没存上：${err && err.message ? err.message : String(err)}`));
  });
  actions.appendChild(save);

  if (!isNew) {
    const del = document.createElement('button');
    del.type = 'button';
    del.dataset.role = 'groupDelete';
    del.textContent = '删除这个组';
    del.addEventListener('click', () => {
      fetch(`${API_BASE}/api/groups/${encodeURIComponent(group.id)}`, { method: 'DELETE' })
        .then(async (res) => {
          const data = await res.json().catch(() => ({}));
          if (!res.ok) {
            showAlert(`组没删掉：${data.error || `HTTP ${res.status}`}`);
            return;
          }
          await refreshGroups();
          renderGroupEditors();
        })
        .catch((err) => showAlert(`组没删掉：${err && err.message ? err.message : String(err)}`));
    });
    actions.appendChild(del);
  }

  block.append(name, members, actions);
  return block;
}

/**
 * 「新建组」：打开对话框，填名字、勾成员，存下来。
 *
 * 建完的组立刻出现在各 agent 的交接目标里（那边刷新一下目标列表）。
 */
function openNewGroupDialog() {
  if (!newGroupDialogEl) return;
  if (newGroupBodyEl) {
    newGroupBodyEl.innerHTML = '';
    newGroupBodyEl.appendChild(groupEditor(null, true));
  }
  if (typeof newGroupDialogEl.showModal === 'function') newGroupDialogEl.showModal();
}

/** 重新读一次组清单（服务端是唯一的一份）。 */
async function refreshGroups() {
  try {
    const res = await fetch(`${API_BASE}/api/groups`);
    if (!res.ok) return;
    const data = await res.json();
    knownGroups.clear();
    for (const group of Array.isArray(data.groups) ? data.groups : []) {
      if (!group || typeof group.id !== 'string') continue;
      knownGroups.set(group.id, {
        id: group.id,
        name: typeof group.name === 'string' ? group.name : group.id,
        members: Array.isArray(group.members) ? group.members : [],
      });
    }
    for (const panel of panels) panel.refreshTargets();
    renderGroupList();
  } catch {
    // 读不到就先用手上这份：不影响别的功能。
  }
}

function resetGlobalSettingsForm() {
  const defaults = globalSettingsDefaults || {};
  if (settingsHostEl) settingsHostEl.value = defaults.agentHost || '';
  if (settingsWorkdirEl) settingsWorkdirEl.value = defaults.agentWorkdir || '';
}

/**
 * 组名就地改：点一下名字变成输入框，回车或失焦存下，Esc 放弃。
 */
function makeGroupNameEditable(group, nameEl) {
  nameEl.addEventListener('click', (event) => {
    // 别让 details 折叠掉 —— 这次点击是去改名字的。
    event.preventDefault();
    event.stopPropagation();
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'group-name-input';
    input.value = group.name;
    nameEl.replaceWith(input);
    input.focus();
    input.select();
    let done = false;
    const commit = async (save) => {
      if (done) return;
      done = true;
      const value = input.value.trim();
      if (!save || value === '' || value === group.name) {
        input.replaceWith(nameEl);
        return;
      }
      try {
        const res = await fetch(`${API_BASE}/api/groups/${encodeURIComponent(group.id)}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: value }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          showAlert(`组没改名：${data.error || `HTTP ${res.status}`}`);
          input.replaceWith(nameEl);
          return;
        }
        await refreshGroups();
      } catch (err) {
        showAlert(`组没改名：${err && err.message ? err.message : String(err)}`);
        input.replaceWith(nameEl);
      }
    };
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        void commit(true);
      }
      if (event.key === 'Escape') {
        event.preventDefault();
        void commit(false);
      }
    });
    input.addEventListener('blur', () => {
      void commit(true);
    });
  });
}

/**
 * 左侧栏的「组」：每个组一行，展开看组里有哪些 agent。
 *
 * 组排在 agent 列表上面：先看分组，再看全部 agent。组里的成员可以点，
 * 直接跳到那个 agent 的聊天。
 */
function renderGroupList() {
  if (!groupListEl) return;
  groupListEl.innerHTML = '';
  if (knownGroups.size === 0) {
    const empty = document.createElement('li');
    empty.className = 'group-empty';
    empty.textContent = '还没有组。点上面的「＋ 新建组」把几个 agent 放在一起。';
    groupListEl.appendChild(empty);
    return;
  }
  for (const group of knownGroups.values()) {
    const item = document.createElement('li');
    item.className = 'group-item';
    item.dataset.groupId = group.id;

    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.className = 'group-summary';
    const nameSpan = document.createElement('span');
    nameSpan.className = 'group-name';
    nameSpan.textContent = group.name;
    makeGroupNameEditable(group, nameSpan);
    summary.append(nameSpan);
    const count = document.createElement('span');
    count.className = 'group-count';
    count.textContent = `（${group.members.length} 人）`;
    summary.append(count);
    details.appendChild(summary);

    const members = document.createElement('ul');
    members.className = 'group-members';
    for (const member of group.members) {
      const row = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'group-member';
      button.textContent = agentNames.get(member) || member;
      button.addEventListener('click', () => setActiveAgent(member, 'chat'));
      row.appendChild(button);
      members.appendChild(row);
    }
    details.appendChild(members);
    item.appendChild(details);
    groupListEl.appendChild(item);
  }
}

/**
 * 左侧栏：把页面上所有 agent 列出来。
 *
 * 一行一个：状态点 + 名字（点它，右侧换成它的聊天框）+ 配置按钮
 * （点它，进它自己的配置页）。
 */
function renderSidebar() {
  if (!agentListEl) return;
  agentListEl.innerHTML = '';
  sidebarRows.clear();
  for (const panel of panels) {
    const li = document.createElement('li');
    li.className = 'agent-item';
    li.dataset.agentId = panel.profileId;

    const dot = document.createElement('span');
    dot.className = 'dot';
    li.appendChild(dot);

    const pick = document.createElement('button');
    pick.type = 'button';
    pick.className = 'agent-pick';
    pick.setAttribute('data-role', 'pickAgent');
    pick.textContent = panel.name();
    pick.addEventListener('click', () => setActiveAgent(panel.profileId, 'chat'));
    li.appendChild(pick);

    const config = document.createElement('button');
    config.type = 'button';
    config.className = 'agent-config';
    config.setAttribute('data-role', 'sideConfig');
    config.textContent = '配置';
    config.title = `配置 ${panel.name()}`;
    config.addEventListener('click', () => setActiveAgent(panel.profileId, 'config'));
    li.appendChild(config);

    agentListEl.appendChild(li);
    sidebarRows.set(panel.profileId, { li, dot, pick, config });
  }
  syncSidebar();
}

/** 侧边栏跟着当前状态走：谁被选中、谁在跑。 */
function syncSidebar() {
  for (const panel of panels) {
    const row = sidebarRows.get(panel.profileId);
    if (!row) continue;
    row.li.dataset.active = panel.profileId === activeAgentId ? 'true' : 'false';
    const status = panel.status();
    row.dot.className = 'dot' + (status.kind ? ' ' + status.kind : '');
    row.pick.textContent = panel.name();
    row.config.title = `配置 ${panel.name()}`;
  }
}

/**
 * 把当前状态落到 DOM 上：右侧只显示被选中的那一个 agent。
 */
function applyView() {
  for (const panel of panels) {
    const isActive = panel.profileId === activeAgentId;
    panel.root.hidden = !isActive;
    panel.root.dataset.view = isActive ? activeView : 'chat';
  }
  syncSidebar();
}

/**
 * 切到某个 agent（可选切到它自己的配置页）。
 */
function setActiveAgent(profileId, view = 'chat') {
  const panel = panels.find((item) => item.profileId === profileId);
  if (!panel) {
    throw new Error(`页面上没有 agent「${profileId}」，切不过去。`);
  }
  activeAgentId = profileId;
  activeView = view;
  applyView();
  writeRoute();
}

/**
 * 地址栏里记着当前看的是哪个 agent、哪种视图。
 *
 * 刷新页面回到同一个地方；也能把链接直接发给另一个人。
 */
function writeRoute() {
  if (activeAgentId === null) return;
  const next = `#/agent/${encodeURIComponent(activeAgentId)}/${activeView}`;
  if (location.hash !== next) location.hash = next;
}

/** 从地址栏读出该显示谁。地址栏没写就没有路由信息。 */
function readRoute() {
  const matched = /^#\/agent\/([^/]+)\/(chat|config)$/.exec(location.hash);
  if (!matched) return null;
  return { profileId: decodeURIComponent(matched[1]), view: matched[2] };
}

/** 按地址栏显示对应的 agent 与视图。 */
function applyRoute() {
  const route = readRoute();
  if (route === null) {
    // 地址栏没指定：显示第一个 agent 的聊天。
    if (activeAgentId === null && panels.length > 0) {
      setActiveAgent(panels[0].profileId, 'chat');
    }
    return;
  }
  const panel = panels.find((item) => item.profileId === route.profileId);
  if (!panel) {
    // 配置里的角色（例如 AGENT MATH）是从模板挂的，要等 /api/health 回来才挂上：
    // 这时候地址栏指的 agent 不是"不存在"，只是还没挂出来 —— 先记下，面板挂好再按
    // 地址栏显示。只有服务端压根不认这个身份，才是真错了。
    if (!healthLoaded || knownAgentIds.has(route.profileId)) {
      pendingRoute = route;
      return;
    }
    showAlert(`地址里指定的 agent「${route.profileId}」页面上不存在。`);
    return;
  }
  activeAgentId = route.profileId;
  activeView = route.view;
  applyView();
}

window.addEventListener('hashchange', applyRoute);

/**
 * 服务端给的权限目录（全局一份）。
 *
 * 「新建 Agent」造出来的面板也要铺同一份权限目录 —— 它跟 AGENT A / AGENT B
 * 用的是同一张权限表，只是默认勾哪几项不同。
 */
let permissionCatalogSnapshot = [];

/**
 * 把一个新 agent 装成页面上的一个面板。
 *
 * 面板结构来自首页里的模板 —— 与 AGENT A / AGENT B 完全同构，
 * 区别只是系统提示词默认是空的、输入框也是空的，等用户自己填。
 */
function mountPanel(profile, options = {}) {
  const tpl = document.getElementById('panelTemplate');
  if (!tpl) {
    throw new Error('首页里缺少面板模板，新建不了 agent。');
  }
  if (!layoutEl) {
    throw new Error('页面上找不到面板容器，新建不了 agent。');
  }

  const root = tpl.content.firstElementChild.cloneNode(true);
  root.id = `panel-${profile.id}`;
  root.dataset.profile = profile.id;

  // 这个身份服务端认得：交接目标的候选与名字要把它列上 ——
  // 不列的话，别的 agent 没法把结果交给它。
  knownAgentIds.add(profile.id);
  agentNames.set(profile.id, profile.name);

  // 配置里写死的身份：面板能删，但服务端的身份删不掉（与 AGENT A/B 同规矩）。
  if (profile.builtin) root.dataset.builtin = 'true';
  const nameEl = root.querySelector('[data-role=name]');
  if (nameEl) nameEl.textContent = profile.name;
  const promptEl = root.querySelector('[data-role=prompt]');
  if (promptEl) promptEl.value = typeof profile.defaultPrompt === 'string' ? profile.defaultPrompt : '';
  const hintEl = root.querySelector('.hint');
  if (hintEl) {
    const instrLen =
      typeof profile.instructions === 'string' ? profile.instructions.trim().length : 0;
    const prefix = profile.builtin ? '配置里的固定角色' : '新建的 agent';
    hintEl.textContent =
      instrLen > 0
        ? `${prefix} · 系统提示词已配置 ${instrLen} 个字符（「配置」里可改）。`
        : `${prefix} · 系统提示词默认是空的，自己写。`;
  }

  layoutEl.appendChild(root);

  const panel = createPanel(root);
  panel.setName(profile.name);
  // 权限目录用全局那份；这一栏的生效配置用服务端给的这份。
  panel.setPermissionCatalog(permissionCatalogSnapshot);
  panel.applyProfile(profile);
  panels.push(panel);
  // 页面上多了一个 agent：左侧栏要列出它。
  renderSidebar();
  // 新建的 agent 切过去看；刷新页面时补挂回来的 agent 不抢当前的视线。
  if (options.activate !== false) setActiveAgent(panel.profileId, 'chat');
  // 别的面板的交接目标列表也要把它列出来。
  refreshAllTargets();
  return panel;
}

/**
 * 新建一个 agent。
 *
 * 先让服务端登记这个身份（id、名字、默认系统提示词），再把面板插到页面上 ——
 * 服务端认得这个身份，它才能被运行。
 */
async function createNewAgent() {
  if (!newAgentBtnEl) return;
  newAgentBtnEl.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/api/agents`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    if (!res.ok) {
      // 服务端通常在响应体里写了原因，把它读出来显示，别只甩一个状态码。
      let detail = '';
      try {
        const data = await res.json();
        if (data && typeof data.error === 'string') detail = data.error;
      } catch {
        // 响应体不是 JSON（网关错误之类），那就只报状态码。
      }
      throw new Error(detail || `HTTP ${res.status}`);
    }
    const profile = await res.json();
    mountPanel(profile);
  } catch (err) {
    showAlert(`新建 agent 失败：${(err && err.message) || String(err)}`);
  } finally {
    newAgentBtnEl.disabled = false;
  }
}

if (newAgentBtnEl) {
  newAgentBtnEl.addEventListener('click', () => {
    createNewAgent();
  });
}

// 全局设置：打开时读一遍当前生效的值，保存后立刻刷新侧栏底部那行字。
if (newGroupBtnEl) {
  newGroupBtnEl.addEventListener('click', () => {
    openNewGroupDialog();
  });
}
if (newGroupCloseEl) {
  newGroupCloseEl.addEventListener('click', () => {
    if (newGroupDialogEl && typeof newGroupDialogEl.close === 'function') newGroupDialogEl.close();
  });
}

if (settingsBtnEl) {
  settingsBtnEl.addEventListener('click', () => {
    loadGlobalSettings().then(() => {
      fillSettingsAgentList();
      renderGroupEditors();
      if (settingsDialogEl && typeof settingsDialogEl.showModal === 'function') {
        settingsDialogEl.showModal();
      }
    });
  });
}
if (settingsSaveEl) {
  settingsSaveEl.addEventListener('click', () => {
    saveGlobalSettings();
  });
}
if (settingsResetEl) {
  settingsResetEl.addEventListener('click', () => {
    resetGlobalSettingsForm();
  });
}
if (settingsCloseEl) {
  settingsCloseEl.addEventListener('click', () => {
    if (settingsDialogEl && typeof settingsDialogEl.close === 'function') settingsDialogEl.close();
  });
}
if (settingsResetSessionEl) {
  // 两步确认：第一次点只上膛（按钮变成"确认重置"），第二次才真重置。
  // 不用 window.confirm —— 弹窗在自动化里会被直接按掉，按钮上膛更可靠。
  let resetArmed = false;
  settingsResetSessionEl.addEventListener('click', () => {
    const agentId = settingsAgentEl ? settingsAgentEl.value : '';
    if (agentId === '') return;
    const label = agentId === '__all__' ? '全部 agent' : (agentNames.get(agentId) || agentId);
    if (!resetArmed) {
      resetArmed = true;
      settingsResetSessionEl.textContent = '确认重置';
      settingsResetSessionEl.classList.add('armed');
      if (settingsStateEl) {
        settingsStateEl.textContent = `再点一次就重置 ${label} 的会话（对话记录与模型上下文都清掉）。`;
        settingsStateEl.classList.remove('bad');
      }
      return;
    }
    resetArmed = false;
    settingsResetSessionEl.textContent = '重置会话';
    settingsResetSessionEl.classList.remove('armed');
    settingsResetSessionEl.disabled = true;
    resetAgentSession(agentId)
      .then((data) => {
        const count = Array.isArray(data.agents) ? data.agents.length : 0;
        if (settingsStateEl) {
          settingsStateEl.textContent = `已重置 ${count} 个 agent 的会话。`;
          settingsStateEl.classList.remove('bad');
        }
        fillSettingsAgentList();
      })
      .catch((err) => {
        if (settingsStateEl) {
          settingsStateEl.textContent = `重置失败：${err && err.message ? err.message : String(err)}`;
          settingsStateEl.classList.add('bad');
        }
      })
      .finally(() => {
        settingsResetSessionEl.disabled = false;
      });
  });
}

/**
 * 重画所有面板的交接配置。
 *
 * 页面上新增或删掉 agent 之后要调用：勾选项里的名字与被删掉的目标都得跟上。
 *
 * 用的是 syncTargets 而不是单纯重画：被删掉的那个 agent 不能继续留在
 * 别人的交接目标里 —— 服务端已经没这个身份了，交过去会失败。
 */
function refreshAllTargets() {
  for (const panel of panels) panel.syncTargets();
}

/**
 * 交接：这一栏跑完后，把结果发给配置里勾选的那些 agent。
 *
 * 目标可以多选：每一个被勾上的 agent 都拿到这份结果，以及它自己输入框里
 * 写的要求（要求是空的就不发起交接 —— 没有要执行的事）。
 */
/** 取消这一栏所有"还没到点"的延迟交接。 */
function clearPendingHandoffs(pending) {
  for (const timer of pending) clearTimeout(timer);
  pending.clear();
}

/**
 * 交接步骤的指令：把"交给谁"和"什么条件下才交"写清楚，判定权在模型手里。
 *
 * 不写关键词规则：条件是一句人话，由这一步的模型对照这一轮的产出判断。
 */
function buildHandoffStepInstruction(targets, condition) {
  const names = targets
    .map((id) =>
      id.startsWith('group:')
        ? `组「${(knownGroups.get(id.slice('group:'.length)) || {}).name || id}」`
        : agentNames.get(id) || id,
    )
    .join('、');
  const head = `【交接步骤】这一轮已经跑完。按这一栏的交接配置，结论要交给：${names}。`;
  if (condition === '') {
    return (
      `${head}\n` +
      '这一栏没有设置交接条件：无条件交接。' +
      '调用 requestHandoff（summary 写一句话说明交的是什么），然后结束这一步。'
    );
  }
  return (
    `${head}\n` +
    `交接条件：${condition}\n` +
    '先判断这个条件在这一轮是否满足：满足就调用 requestHandoff（summary 写一句话）；' +
    '不满足就不要调用它，并在回答里说清为什么这次不交接。' +
    '判定要依据这一轮的产出（以及你读到的文件），不要猜。'
  );
}

/**
 * 交接循环的两道闸。
 *
 * - 这一栏自己配的交接条件（answer 里含不含某段文本）→ 该跳就跳、该发才发；
 * - 连续交接超过 HANDOFF_MAX_HOPS 跳 → 停手并说清楚（防失控循环）。
 */
const HANDOFF_MAX_HOPS = 8;

function deliverResult(source, answer, options = {}) {
  if (answer === '') {
    source.note('这一栏这轮没有产出文字结果，没有可交接的内容。');
    return;
  }

  const depth = Number.isInteger(options.handoffDepth) ? options.handoffDepth : 0;
  if (depth + 1 > HANDOFF_MAX_HOPS) {
    source.note(
      `这一条链已经连续交接 ${depth} 跳，到上限 ${HANDOFF_MAX_HOPS} 了，本轮不再往下交。`,
    );
    return;
  }
  const skipProfileId =
    typeof options.skipProfileId === 'string' ? options.skipProfileId : '';
  // 默认不把结果交回给刚把它发过来的那一栏（防来回触发）；
  // 审查—返修这种有意的循环，靠 agent 自己的 allowHandoffBack 打开。
  const backAllowed = source.allowHandoffBack();
  const targets = source.targets().filter((id) => backAllowed || id !== skipProfileId);
  const skipped = source.targets().length - targets.length;
  if (targets.length === 0) {
    source.note(
      skipped > 0
        ? '本轮结果来自上游，不再交回给它（避免来回触发），没有其他交接目标。'
        : '没有勾选交接目标，本轮结果没有发出去。',
    );
    return;
  }
  // 延迟是"这一栏自己的配置"：跑完隔多少秒才发。0 = 立刻。
  const delaySeconds = source.handoffDelaySeconds();
  // 目标里可能有组：组里每个人都收到同一条消息，并且各自都要回应。
  // 同一个人既是直接目标又是组成员时，只发一次。
  const recipients = [];
  for (const target of targets) {
    if (target.startsWith('group:')) {
      const group = knownGroups.get(target.slice('group:'.length));
      if (!group) continue;
      for (const member of group.members) {
        if (!recipients.includes(member)) recipients.push(member);
      }
      continue;
    }
    if (!recipients.includes(target)) recipients.push(target);
  }
  for (const targetId of recipients) {
    const target = panels.find((item) => item.profileId === targetId);
    if (!target) continue;
    const requirement = target.getPrompt();
    // 下游的输入框是空的，不等于没活干：
    // 上游的结果本身就是它要接着做的事，用户配了交接就该交出去。
    // （要求只是"额外要它做什么"，不是交接触发的必要条件。）
    if (requirement === '') {
      target.note(
        `${source.name()} 的结果已经交给它了（它没有写要求，按上游结果继续）。`,
      );
    }
    // 不自动把右侧切过去：正在看的那一栏被换掉会让人措手不及。
    // 下游开始跑了 —— 左侧栏它那一行的圆点会亮起来，点它就能过去看。
    const payload = {
      handoff: answer,
      instruction: requirement,
      handoffFrom: source.name(),
      // 发送方的身份：服务端据此判断"发送方要不要先重置自己的会话"。
      handoffFromId: source.profileId,
      // 这一跳是第几跳：审查—返修那种循环要有个上限。
      handoffDepth: depth + 1,
    };
    if (delaySeconds > 0) {
      // 延迟交接：定时器归来源这一栏管 —— 它重新跑一轮或删面板时会被取消，
      // 不会让一条旧结果在半分钟后突然打到下游。
      source.note(`${target.name()} 将在 ${delaySeconds} 秒后收到本轮结果。`);
      source.scheduleHandoff(() => target.run(payload), delaySeconds);
    } else {
      target.run(payload);
    }
  }
}

/**
 * 删掉一个面板。
 *
 * 新建的 agent 还要从服务端的身份表里删掉 —— 留着的话刷新页面它又会冒出来。
 * 配置里写死的（AGENT A / AGENT B / AGENT MATH）删不掉，删的只是页面上的面板。
 *
 * 服务端删不掉时面板也不动：面板还在、身份没了是半截状态。
 */
async function removePanel(panel) {
  const profileId = panel.profileId;
  // 服务端的身份还在不在：配置里写死的不给删，页面上的面板只是本页的挂载。
  let removedFromServer = false;
  if (!panel.isBuiltin()) {
    const res = await fetch(`${API_BASE}/api/agents/${encodeURIComponent(profileId)}`, {
      method: 'DELETE',
    });
    if (!res.ok) {
      let detail = '';
      try {
        const data = await res.json();
        if (data && typeof data.error === 'string') detail = data.error;
      } catch {
        // 响应体不是 JSON（网关错误之类），那就只报状态码。
      }
      showAlert(`删除 agent 失败：${detail || `HTTP ${res.status}`}`);
      return;
    }
    removedFromServer = true;
  }

  // 服务端的身份已经没了：本地这份清单也得跟着摘 ——
  // 不摘的话，它还会出现在别的面板的交接目标里，交过去必然失败。
  //
  // 配置里写死的不在这里摘：它们在服务端还在（只是页面上不挂了），
  // 摘了会让别人交不出去的目标被无声地抹掉。
  if (removedFromServer) {
    knownAgentIds.delete(profileId);
    agentNames.delete(profileId);
  }

  panel.destroy();
  const index = panels.indexOf(panel);
  if (index >= 0) panels.splice(index, 1);
  // 左侧栏少了一行。
  renderSidebar();
  // 删掉的正好是当前在看的那一个：换到还剩的第一个去。
  if (activeAgentId === profileId) {
    activeAgentId = null;
    activeView = 'chat';
    if (panels.length > 0) setActiveAgent(panels[0].profileId, 'chat');
  }
  applyView();
  refreshAllTargets();
}

const panelA = panels.find((panel) => panel.profileId === 'agent-a');
const panelB = panels.find((panel) => panel.profileId === 'agent-b');

// 默认的闭环（AGENT A 跑完把实现分析交给 AGENT B 审查）写在服务端的配置里。
// 页面不再自己设一遍 —— 用户改过的目标存在服务端，页面上的这份只是副本。
if (!panelA || !panelB) {
  showAlert('页面上找不到 AGENT A / AGENT B 两个面板，默认的交接没有接通。');
}

refreshAllTargets();

// 左侧栏先摆出来，右侧先显示第一个 agent 的聊天框。
renderSidebar();
applyRoute();

/** 启动时读一次 /api/health，把运行模式与模型显示在顶栏，并核对前端版本与 agent 身份。 */
async function loadHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(`/api/health 返回 ${res.status}`);
  const info = await res.json();

  if (globalMetaEl) {
    globalMetaEl.textContent = `${info.mode} · ${info.model}`;
  }
  if (buildTagEl) buildTagEl.textContent = `前端 ${FRONTEND_VERSION}`;

  // 全局设置（运行环境主机与工作目录）：侧栏底部一直显示当前生效的值。
  globalSettingsDefaults = info.defaultSettings || null;
  renderGlobalSettings({
    settings: info.settings,
    envProblem: info.envProblem,
    host: info.host,
  });

  if (info.frontendVersion && info.frontendVersion !== FRONTEND_VERSION) {
    showAlert(
      `前端版本不一致：本页是 ${FRONTEND_VERSION}，服务端是 ${info.frontendVersion}。` +
        '请硬刷新（Ctrl+Shift+R）后重试。',
    );
  }

  if (!Array.isArray(info.agents)) {
    showAlert('服务端没有返回 agent 列表，页面不知道左右两栏各跑哪个 agent。');
    return;
  }
  // 不比数量：页面上的面板可以比服务端多 ——「新建 Agent」造出来的新身份
  // 是在页面上建的，服务端的列表里不一定有。真正要查的是下面那条：
  // 每个面板的身份服务端认不认得。

  const agents = new Map(info.agents.map((agent) => [agent.id, agent]));

  // 服务端认识的 agent 身份：交接目标的候选、已删 agent 的摘除，都按这份清单。
  knownAgentIds.clear();
  agentNames.clear();
  for (const agent of info.agents) {
    knownAgentIds.add(agent.id);
    agentNames.set(agent.id, agent.name);
  }

  // 组：交接目标可以指向一整组。
  knownGroups.clear();
  for (const group of Array.isArray(info.groups) ? info.groups : []) {
    if (!group || typeof group.id !== 'string') continue;
    knownGroups.set(group.id, {
      id: group.id,
      name: typeof group.name === 'string' ? group.name : group.id,
      members: Array.isArray(group.members) ? group.members : [],
    });
  }
  renderGroupList();

  // 权限目录是全局的一份：所有面板共用同一张权限表，
  // 每一栏勾哪几项是各的（存在服务端的那份配置里）。
  if (Array.isArray(info.permissionCatalog)) {
    permissionCatalogSnapshot = info.permissionCatalog;
    for (const panel of panels) panel.setPermissionCatalog(info.permissionCatalog);
  }

  // 服务端有、页面上还没有的 agent：把它挂上来。
  // 运行时新建的存在服务端（agents.json）、刷新不该弄丢；配置里新加的内置角色
  // （例如 AGENT MATH）页面上没有静态面板，也从模板挂一份。
  for (const agent of info.agents) {
    if (panels.some((panel) => panel.profileId === agent.id)) continue;
    mountPanel(agent, { activate: false });
  }

  for (const panel of panels) {
    const agent = agents.get(panel.profileId);
    if (!agent) {
      showAlert(`页面上的 agent 身份「${panel.profileId}」服务端不认识。`);
      continue;
    }
    panel.setName(agent.name);
    // 生效配置、代码里的默认值、用户改过没有 —— 一次都给齐。
    panel.applyProfile(agent);
  }
  // 名字可能和服务端给的不同（用户改过名字的 agent）：勾选项里的名字要跟上。
  refreshAllTargets();
  // 左侧栏里的名字也用服务端的这份。
  renderSidebar();

  // 清单到了，面板也挂齐了：这时才按地址栏显示。
  // 配置里的角色（AGENT MATH 等）是这一步才挂上来的 —— 地址栏指着它们时，
  // 不能在它们挂出来之前就报"不存在"。
  healthLoaded = true;
  if (pendingRoute) {
    pendingRoute = null;
  }
  applyRoute();

  // 历史：每一栏跑过的那几轮都存在服务端，刷新页面后照样在。
  for (const panel of panels) {
    await panel.loadHistory();
  }
}

// ---------------------------------------------------------------------------
// IDEA 文本：全局一份，随每次运行发给所有 agent（优先于仓库里的 idea.txt）。
// ---------------------------------------------------------------------------

const ideaUi = {
  btn: document.querySelector('[data-role=ideaBtn]'),
  modal: document.querySelector('[data-role=ideaModal]'),
  backdrop: document.querySelector('[data-role=ideaBackdrop]'),
  close: document.querySelector('[data-role=ideaClose]'),
  input: document.querySelector('[data-role=ideaInput]'),
  clear: document.querySelector('[data-role=ideaClear]'),
  state: document.querySelector('[data-role=ideaState]'),
  chars: document.querySelector('[data-role=ideaChars]'),
  kv: document.querySelector('[data-role=ideaKv]'),
  count: document.querySelector('[data-role=ideaCount]'),
  struct: document.querySelector('[data-role=ideaStruct]'),
  mode: document.querySelector('[data-role=ideaMode]'),
};

// 机制文件回答两件事：目标与方法（方法里写明要新建哪些文件）。
// 假设、起点、开关、预注册、检查、记录、边界由 verification.json 与仓库规则承担，
// 所以完整性门槛按这两段判定，而不是按固定的十段。
const IDEA_REQUIRED_SECTIONS = ['1_goal', '4_method'];
const IDEA_REQUIRED_COUNT = IDEA_REQUIRED_SECTIONS.length;

let ideaSaveTimer = null;
let ideaDoc = null;
let ideaMode = 'kv';
let ideaCountSections = 0;

function sectionLabel(key) {
  return key.replace(/^(\d+)_/, '$1. ').replace(/_/g, ' ');
}

function fmtValue(value) {
  if (value === null || value === undefined || value === '') {
    return { text: '缺失', cls: 'kv-missing' };
  }
  if (typeof value === 'number') return { text: String(value), cls: 'kv-num' };
  if (typeof value === 'boolean') return { text: value ? 'true' : 'false', cls: 'kv-bool' };
  if (Array.isArray(value)) return { text: value.map((v) => String(v)).join('、'), cls: 'kv-arr' };
  if (typeof value === 'object') return { text: JSON.stringify(value), cls: 'kv-obj' };
  return { text: String(value), cls: 'kv-str' };
}

/** 把任意 JSON 展平成 [路径, 值] 行；对象按路径展开，数组逐项。 */
function flattenRows(value, path, out) {
  if (Array.isArray(value)) {
    value.forEach((item, i) => flattenRows(item, `${path}[${i}]`, out));
    return out;
  }
  if (value && typeof value === 'object') {
    for (const [k, v] of Object.entries(value)) { flattenRows(v, path === '' ? k : `${path}.${k}`, out); }
    return out;
  }
  out.push([path, value]);
  return out;
}

function kvSection(title, entries) {
  const box = document.createElement('section');
  box.className = 'kv-section';
  const h = document.createElement('h3');
  h.textContent = title;
  box.appendChild(h);
  for (const [path, value] of entries) {
    const row = document.createElement('div');
    row.className = 'kv-row';
    row.dataset.role = 'kv-row';
    const key = document.createElement('span');
    key.className = 'kv-key';
    key.textContent = path;
    const val = document.createElement('span');
    const f = fmtValue(value);
    val.className = `kv-value ${f.cls}`;
    val.textContent = f.text;
    row.appendChild(key);
    row.appendChild(val);
    box.appendChild(row);
  }
  return box;
}

function countSections(doc) {
  if (!doc || !doc.sections) return 0;
  return IDEA_REQUIRED_SECTIONS.filter((k) => {
    const v = doc.sections[k];
    if (v === undefined || v === null) return false;
    const rows = flattenRows(v, '', []);
    return rows.length > 0 && rows.every(([, x]) => String(x ?? '').trim() !== '');
  }).length;
}

function renderIdeaDoc(doc, error, extras = {}) {
  const { mainLine = null, baseline = null, verification = null, math = null } = extras || {};
  if (!ideaUi.kv) return;
  ideaUi.kv.innerHTML = '';
  ideaDoc = doc || null;
  ideaCountSections = countSections(doc);
  if (!doc) {
    const warn = document.createElement('div');
    warn.className = 'kv-invalid';
    warn.textContent = error || '结构不合格：读不到 idea.json';
    ideaUi.kv.appendChild(warn);
  } else {
    const meta = ['id', 'title', 'type', 'status', 'issue', 'date', 'supersedes']
      .filter((k) => doc[k] !== undefined);
    if (meta.length) ideaUi.kv.appendChild(kvSection('元信息', meta.map((k) => [k, doc[k]])));
    // 主线单独一块、排在正文之前：它来自 ideas/main_line.json，机制文件不再自带副本。
    const line = mainLine || doc.main_line;
    if (line !== undefined && line !== null) {
      ideaUi.kv.appendChild(kvSection('主线', flattenRows(line, '', [])));
    }
    // 基线事实每个候选都一样，单独一块，机制文件不再抄。
    if (baseline && typeof baseline === 'object') {
      ideaUi.kv.appendChild(kvSection('基线', flattenRows(baseline, '', [])));
    }
    // 用什么算式在方向目录的 math.json：先摆公式，再摆机制正文。
    if (math && typeof math === 'object') {
      ideaUi.kv.appendChild(kvSection('公式', flattenRows(math, '', [])));
    }
    // 要验证什么（假设/开关/预注册/检查/期望）在方向目录的 verification.json。
    if (verification && typeof verification === 'object') {
      ideaUi.kv.appendChild(kvSection('验证计划', flattenRows(verification, '', [])));
    }
    if (doc.sections) {
      for (const key of Object.keys(doc.sections)) {
        ideaUi.kv.appendChild(kvSection(sectionLabel(key), flattenRows(doc.sections[key], '', [])));
      }
    }
    for (const key of ['references', 'result', 'web_contract', 'self_check']) {
      if (doc[key] !== undefined) ideaUi.kv.appendChild(kvSection(key, flattenRows(doc[key], '', [])));
    }
  }
  if (ideaUi.count) ideaUi.count.textContent = `${ideaCountSections}/${IDEA_REQUIRED_COUNT} 段`;
  if (ideaUi.struct) {
    ideaUi.struct.textContent = ideaCountSections === IDEA_REQUIRED_COUNT ? '结构合格' : '结构不合格（缺段或空段）';
    ideaUi.struct.dataset.ok = ideaCountSections === IDEA_REQUIRED_COUNT ? '1' : '0';
  }
  if (ideaUi.input && doc) ideaUi.input.value = JSON.stringify(doc, null, 2);
  refreshIdeaUi();
}

function setIdeaMode(mode) {
  ideaMode = mode === 'src' ? 'src' : 'kv';
  if (ideaUi.kv) ideaUi.kv.hidden = ideaMode !== 'kv';
  if (ideaUi.input) ideaUi.input.hidden = ideaMode !== 'src';
  if (ideaUi.mode) ideaUi.mode.textContent = ideaMode === 'kv' ? '源码（JSON）' : '键值视图';
  if (ideaMode === 'src' && ideaUi.input) ideaUi.input.focus();
}

/** 把输入长度、段数与按钮摘要刷一遍。 */
function refreshIdeaUi() {
  const text = ideaUi.input ? ideaUi.input.value : '';
  if (ideaUi.chars) ideaUi.chars.textContent = `${text.length} 字符`;
  if (ideaUi.btn) {
    ideaUi.btn.textContent = text.trim() === '' ? 'IDEA（空）' : `IDEA · ${ideaCountSections}/${IDEA_REQUIRED_COUNT} 段`;
  }
}

/** 从服务端读出结构化 idea（真源 idea.json）。 */
async function loadIdea() {
  if (!ideaUi.input) return;
  try {
    const res = await fetch(`${API_BASE}/api/idea`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const doc = data.doc && typeof data.doc === 'object' ? data.doc : null;
    if (doc) {
      const where = typeof data.ideaPath === 'string' && data.ideaPath
        ? data.ideaPath : 'ideas/*_idea.json';
      if (ideaUi.state) ideaUi.state.textContent = `来自仓库 ${where}（结构化真源）`;
      renderIdeaDoc(doc, null, {
        mainLine: data.mainLine || null,
        baseline: data.baseline || null,
        verification: data.verification || null,
        math: data.math || null,
      });
      return;
    }
    const storedText = typeof data.text === 'string' ? data.text : '';
    const fileText = typeof data.fileText === 'string' ? data.fileText : '';
    const raw = storedText.trim() !== '' ? storedText : fileText;
    ideaUi.input.value = raw;
    let parsed = null;
    try { parsed = raw.trim() === '' ? null : JSON.parse(raw); } catch { parsed = null; }
    if (ideaUi.state) {
      ideaUi.state.textContent = parsed
        ? '已从服务端载入'
        : (data.docError || '结构不合格：ideas/ 下没有可解析的 *_idea.json，当前只有纯文本');
    }
    renderIdeaDoc(parsed, parsed ? null : (data.docError || 'ideas/ 下没有可解析的 *_idea.json'));
  } catch (err) {
    showAlert(`读不到 IDEA：${err && err.message ? err.message : String(err)}`);
  }
}

/** 中文输入法下每敲一下存一次太吵：停手 600ms 再存。 */
function scheduleIdeaSave() {
  if (ideaSaveTimer !== null) clearTimeout(ideaSaveTimer);
  ideaSaveTimer = setTimeout(() => {
    ideaSaveTimer = null;
    void saveIdeaNow();
  }, 600);
}

async function saveIdeaNow() {
  const text = ideaUi.input ? ideaUi.input.value : '';
  let doc = null;
  try { doc = text.trim() === '' ? null : JSON.parse(text); } catch (err) {
    if (ideaUi.state) ideaUi.state.textContent = `结构不合格：${err.message}`;
    return;
  }
  if (doc && countSections(doc) < IDEA_REQUIRED_COUNT) {
    if (ideaUi.state) ideaUi.state.textContent = `结构不合格：只有 ${countSections(doc)}/${IDEA_REQUIRED_COUNT} 段，未保存`;
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/api/idea`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      throw new Error((data && typeof data.error === 'string' && data.error) || `HTTP ${res.status}`);
    }
    if (doc) renderIdeaDoc(doc, null);
    if (ideaUi.state) ideaUi.state.textContent = '已保存（随每次运行发送）';
  } catch (err) {
    showAlert(`IDEA 没能存到服务端：${err && err.message ? err.message : String(err)}`);
  }
}

function flushIdeaSave() {
  if (ideaSaveTimer !== null) {
    clearTimeout(ideaSaveTimer);
    ideaSaveTimer = null;
    void saveIdeaNow();
  }
}

function setIdeaModal(open) {
  if (!ideaUi.modal) return;
  if (!open) flushIdeaSave();
  ideaUi.modal.hidden = !open;
  if (open && ideaMode === 'src' && ideaUi.input) ideaUi.input.focus();
}

if (ideaUi.btn) {
  ideaUi.btn.addEventListener('click', () => setIdeaModal(true));
}
if (ideaUi.close) {
  ideaUi.close.addEventListener('click', () => setIdeaModal(false));
}
if (ideaUi.backdrop) {
  ideaUi.backdrop.addEventListener('click', () => setIdeaModal(false));
}
if (ideaUi.input) {
  ideaUi.input.addEventListener('input', () => {
    refreshIdeaUi();
    scheduleIdeaSave();
  });
  ideaUi.input.addEventListener('blur', flushIdeaSave);
  ideaUi.input.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') setIdeaModal(false);
  });
}
if (ideaUi.mode) {
  ideaUi.mode.addEventListener('click', () => setIdeaMode(ideaMode === 'kv' ? 'src' : 'kv'));
}
if (ideaUi.clear) {
  ideaUi.clear.addEventListener('click', () => {
    if (!ideaUi.input) return;
    ideaUi.input.value = '';
    renderIdeaDoc(null, '结构不合格：已清空');
    if (ideaUi.state) ideaUi.state.textContent = '已清空';
    void saveIdeaNow();
  });
}

loadIdea().catch((err) => {
  showAlert(`读不到 IDEA 文本：${err && err.message ? err.message : String(err)}`);
});

loadHealth().catch((err) => {
  showAlert(`读不到服务端状态：${err && err.message ? err.message : String(err)}`);
});

/**
 * 把服务端新登记的 agent 身份补挂到页面上。
 *
 * 只在页面加载时挂一次的话，别处（另一个标签页、脚本、API）新建的 agent
 * 必须手动刷新才出现 —— 用户看到的就是「网页上没有这个 agent」。
 * 这里定期问一次服务端：新身份直接挂上来，改过的名字跟着更新。
 * 不删面板：页面上的面板可以比服务端多，删身份是用户自己的操作。
 */
async function syncAgentPanels() {
  if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return;
  let info;
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) return;
    info = await res.json();
  } catch {
    return;
  }
  if (!info || !Array.isArray(info.agents)) return;

  let mounted = false;
  for (const agent of info.agents) {
    if (panels.some((panel) => panel.profileId === agent.id)) continue;
    try {
      const panel = mountPanel(agent, { activate: false });
      mounted = true;
      panel.loadHistory().catch(() => {});
    } catch (err) {
      showAlert(
        `挂载 agent「${agent.name || agent.id}」失败：${err && err.message ? err.message : String(err)}`,
      );
    }
  }

  // 名字可能被改过（例如改名成 AGENT IDEA）：侧栏与面板头要跟上。
  // 只同步名字，不重放整份配置 —— 那会把用户正在编辑的输入框冲掉。
  for (const agent of info.agents) {
    const panel = panels.find((entry) => entry.profileId === agent.id);
    if (panel) panel.setName(agent.name);
  }
  if (mounted) renderSidebar();
  refreshAllTargets();
}

setInterval(() => {
  void syncAgentPanels();
}, 10000);

// `?autostart=1` 自动跑一次所有面板；`&prompt=...` 预填问题。
const params = new URLSearchParams(window.location.search);
const seed = params.get('prompt');
if (seed) {
  for (const panel of panels) panel.setPrompt(seed);
}
if (params.has('autostart')) {
  for (const panel of panels) panel.run();
}
