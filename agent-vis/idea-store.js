/**
 * 页面上的 IDEA 文本：存盘、读回。
 *
 * 它是"这一轮工作的机制定义"，随每次运行发给 agent（优先于仓库里的 idea.txt）。
 * 存在数据目录下（默认 .data/idea.json），与 agent 配置、对话记录放一起 ——
 * 刷新页面、重启服务之后都还在。写盘先写临时文件再改名，不留半截 JSON。
 */
import fs from 'node:fs';
import path from 'node:path';

import { SANDBOX_ROOT, dataDir } from './config.js';

/** 一份 idea 的长度上限：够放一份机制定义，又不至于把每次请求撑爆。 */
export const MAX_IDEA_CHARS = 200_000;

function ideaFile() {
  return path.join(dataDir(), 'idea.json');
}

/**
 * 仓库根里的 idea.txt（只读）。
 *
 * 页面上的窗口第一次打开时用它当**草稿**：用户看到的是真正在用的那份机制定义，
 * 而不是一片空白。它不会被自动保存 —— 保存是用户显式动作，保存之后才随运行发送。
 */
export function readIdeaFile() {
  try {
    return fs.readFileSync(path.join(SANDBOX_ROOT, 'idea.txt'), 'utf8');
  } catch {
    return '';
  }
}

/**
 * 仓库根里的 idea.json（只读、结构化真源）。
 *
 * 页面按这份 JSON 的键值渲染成界面；解析失败返回 `{doc:null, error}`，
 * 由页面显示「结构不合格」，而不是悄悄退回纯文本。
 */
export function readIdeaDoc() {
  let text = '';
  try {
    text = fs.readFileSync(path.join(SANDBOX_ROOT, 'idea.json'), 'utf8');
  } catch {
    return { doc: null, error: 'idea.json 不存在' };
  }
  try {
    return { doc: JSON.parse(text), error: null };
  } catch (err) {
    return { doc: null, error: `idea.json 解析失败：${err && err.message ? err.message : String(err)}` };
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
