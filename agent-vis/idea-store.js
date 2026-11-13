/**
 * 页面上的 IDEA 文本：存盘、读回。
 *
 * 它是"这一轮工作的机制定义"，随每次运行发给 agent（优先于仓库里的 idea.txt）。
 * 存在数据目录下（默认 .data/idea.json），与 agent 配置、对话记录放一起 ——
 * 刷新页面、重启服务之后都还在。写盘先写临时文件再改名，不留半截 JSON。
 */
import fs from 'node:fs';
import path from 'node:path';

import { currentWorkdir, dataDir, IDEAS_ARCHIVE_NAME } from './config.js';

/** 一份 idea 的长度上限：够放一份机制定义，又不至于把每次请求撑爆。 */
export const MAX_IDEA_CHARS = 200_000;

function ideaFile() {
  return path.join(dataDir(), 'idea.json');
}

/**
 * 仓库里的机制定义：`ideas/<机制>_idea.json`，不叫固定名字。
 *
 * 一个目录可以放多份机制方向（AF3 曲率、行为图、容量分配……），文件名就是机制名。
 * 页面读**最新修改**的那一份；目录里恰好只有一份时就是它。这样新增一个方向只需要
 * 往 ideas/ 里放一个文件，不用改代码，也不会让旧方向被静默覆盖。
 */
/**
 * 归档目录里已经有的方向名。
 *
 * 归档过的方向不是待做的方向：新的 idea 不许跟归档的重复，
 * 所以扫描待做方向时要把这些名字排掉。
 */
export function archivedDirectionNames(folder) {
  const archive = path.join(folder, IDEAS_ARCHIVE_NAME);
  const names = new Set();
  let entries = [];
  try {
    entries = fs.readdirSync(archive, { withFileTypes: true });
  } catch {
    return names;
  }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    names.add(entry.name);
    // 扁平存放的老写法：归档目录里直接躺着 <方向>_idea.json。
    const flat = path.join(archive, `${entry.name}_idea.json`);
    if (fs.existsSync(flat)) names.add(entry.name);
  }
  return names;
}

export function scanIdeaFiles(folder) {
  const found = [];
  const archived = archivedDirectionNames(folder);
  // 方向是目录：ideas/<方向>/<方向>_idea.json；顶层直接放 *_idea.json 的老写法也认，
  // 于是一个方向一个目录与扁平存放并存，页面读最新的那一份。
  const walk = (dir) => {
    let entries = [];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        // 归档目录跳过：归档过的方向不是待做的方向，页面不该再看到它们。
        if (entry.name === IDEAS_ARCHIVE_NAME) continue;
        walk(full);
      } else if (entry.name.endsWith('_idea.json')) {
        // 方向名跟归档里的重了 = 新 idea 跟归档的重复：不当作待做的方向。
        const direction = entry.name.replace(/_idea\.json$/, '');
        if (archived.has(direction)) continue;
        let mtime = 0;
        try {
          mtime = fs.statSync(full).mtimeMs;
        } catch {
          mtime = 0;
        }
        found.push({ full, mtime });
      }
    }
  };
  walk(folder);
  return found;
}

function ideaDocPath() {
  const folder = path.join(currentWorkdir(), 'ideas');
  const found = scanIdeaFiles(folder);
  if (found.length === 0) return null;
  found.sort((a, b) => b.mtime - a.mtime || a.full.localeCompare(b.full));
  return found[0].full;
}

/**
 * 仓库里的机制定义文本（只读）。
 *
 * 页面上的窗口第一次打开时用它当**草稿**：用户看到的是真正在用的那份机制定义，
 * 而不是一片空白。它不会被自动保存 —— 保存是用户显式动作，保存之后才随运行发送。
 */
export function readIdeaFile() {
  const file = ideaDocPath();
  if (!file) return '';
  try {
    return fs.readFileSync(file, 'utf8');
  } catch {
    return '';
  }
}

/**
 * 逻辑与渲染都基于这份 JSON（只读、结构化真源）。
 *
 * 页面按它的键值渲染成界面；解析失败返回 `{doc:null, error}`，
 * 由页面显示「结构不合格」，而不是悄悄退回纯文本。
 */
export function readIdeaDoc() {
  const file = ideaDocPath();
  if (!file) {
    return { doc: null, error: 'ideas/ 下没有 *_idea.json', path: null };
  }
  const name = path.relative(currentWorkdir(), file);
  let text = '';
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch {
    return { doc: null, error: `${name} 读不出来`, path: name };
  }
  try {
    return { doc: JSON.parse(text), error: null, path: name };
  } catch (err) {
    return { doc: null, path: name,
             error: `${name} 解析失败：${err && err.message ? err.message : String(err)}` };
  }
}

/**
 * 主线：`ideas/main_line.json`，整份方向共用一份。
 *
 * 机制文件只写"这一轮怎么变"，主线写在单独的文件里：换机制时它保持原样，页面
 * 把两块并排显示，读者一眼能看出某个机制是否还在这条线上。
 */
export function readMainLine() {
  const file = path.join(currentWorkdir(), 'ideas', 'main_line.json');
  let text = '';
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch {
    return { doc: null, error: 'ideas/main_line.json 不存在' };
  }
  try {
    return { doc: JSON.parse(text), error: null };
  } catch (err) {
    return { doc: null,
             error: `ideas/main_line.json 解析失败：${err && err.message ? err.message : String(err)}` };
  }
}

/**
 * 基线事实：`ideas/baseline.json`，每个候选共用一份。
 *
 * 复制起点、基线 digest、当前 SID 与读数对所有机制都一样，写在机制文件里就是
 * 每一份都抄一遍。页面把它单独列一块，机制文件只写自己多出来的部分。
 */
export function readBaseline() {
  const file = path.join(currentWorkdir(), 'ideas', 'baseline.json');
  let text = '';
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch {
    return { doc: null, error: 'ideas/baseline.json 不存在' };
  }
  try {
    return { doc: JSON.parse(text), error: null };
  } catch (err) {
    return { doc: null,
             error: `ideas/baseline.json 解析失败：${err && err.message ? err.message : String(err)}` };
  }
}

/**
 * 验证计划：方向目录里的 `verification.json`。
 *
 * "要验证什么"（假设、开关、预注册、检查清单、期望）与"怎么实现"分开存：换机制
 * 时这份文件可以整份复用或整份替换，页面把它单独列一块。
 */
export function readVerification() {
  const ideaFile = ideaDocPath();
  if (!ideaFile) return { doc: null, error: 'ideas/ 下没有 *_idea.json，不知道读哪个方向的验证计划' };
  const file = path.join(path.dirname(ideaFile), 'verification.json');
  let text = '';
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch {
    return { doc: null, error: `${path.relative(currentWorkdir(), file)} 不存在` };
  }
  try {
    return { doc: JSON.parse(text), error: null, path: path.relative(currentWorkdir(), file) };
  } catch (err) {
    return { doc: null, path: path.relative(currentWorkdir(), file),
             error: `${path.relative(currentWorkdir(), file)} 解析失败：${err && err.message ? err.message : String(err)}` };
  }
}

/**
 * 公式：方向目录里的 `math.json`。
 *
 * "用什么算式"单独存一份，机制文件只写目标与方法；页面把它列成一块，
 * 缺这份文件时整块不显示（不是错误）。
 */
export function readMath() {
  const ideaFile = ideaDocPath();
  if (!ideaFile) return { doc: null, error: 'ideas/ 下没有 *_idea.json，不知道读哪个方向的公式' };
  const file = path.join(path.dirname(ideaFile), 'math.json');
  let text = '';
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch {
    return { doc: null, error: `${path.relative(currentWorkdir(), file)} 不存在` };
  }
  try {
    return { doc: JSON.parse(text), error: null, path: path.relative(currentWorkdir(), file) };
  } catch (err) {
    return { doc: null, path: path.relative(currentWorkdir(), file),
             error: `${path.relative(currentWorkdir(), file)} 解析失败：${err && err.message ? err.message : String(err)}` };
  }
}

/**
 * 读出当前的 IDEA 文本。
 *
 * 没有文件是预期内的（还没填过）；文件坏了要报错，不能悄悄当成空 ——
 * 那会让用户以为自己的文本丢了。
 */
export function readIdea() {
  const file = ideaFile();
  let raw;
  try {
    raw = fs.readFileSync(file, 'utf8');
  } catch (err) {
    if (err && err.code === 'ENOENT') return { text: '', updatedAt: null };
    throw err;
  }
  const parsed = JSON.parse(raw);
  return {
    text: typeof parsed.text === 'string' ? parsed.text : '',
    updatedAt: typeof parsed.updatedAt === 'number' ? parsed.updatedAt : null,
  };
}

/** 写入 IDEA 文本。空串表示清空（文件还在，只是没有内容）。 */
export function writeIdea(text) {
  if (typeof text !== 'string') throw new Error('idea 必须是字符串');
  if (text.length > MAX_IDEA_CHARS) {
    throw new Error(`idea 太长（${text.length} 字符，上限 ${MAX_IDEA_CHARS}）`);
  }
  const dir = dataDir();
  fs.mkdirSync(dir, { recursive: true });
  const file = ideaFile();
  const tmp = `${file}.tmp-${process.pid}`;
  const payload = { text, updatedAt: Date.now() };
  fs.writeFileSync(tmp, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  fs.renameSync(tmp, file);
  return payload;
}
