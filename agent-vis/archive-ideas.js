#!/usr/bin/env node
/**
 * 归档已经跑过的机制方向。
 *
 * `ideas/` 只放还没执行的机制（`result.status === "not_run"`）。一旦某份 idea 真的
 * 跑过（`running` / `completed` / `failed`），它的数值就固定了，继续留在 `ideas/`
 * 会让页面把"历史结论"当成"待做的方向"读。这个脚本把它们移到
 * `ideas/_archive/`，文件名保持不变；页面扫 `ideas/` 时跳过归档目录，
 * 于是自然只看到待做的方向。
 *
 * 判定规则写在一处：结果对象的 `status` 只要不是 `not_run` 就归档；缺 status 或
 * 解析失败的文件原地不动并被报告出来 —— 判不准的东西不搬。
 *
 *   node agent-vis/archive-ideas.js            # 归档
 *   node agent-vis/archive-ideas.js --dry-run  # 只报告要搬什么
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { currentWorkdir, ideasArchiveDir, IDEAS_ARCHIVE_NAME } from './config.js';

/** 正在执行的机制留在 ideas/ 的那个状态。 */
export const ACTIVE_STATUS = 'not_run';

/** 归档规则，出现在输出与文档里，跑的时候只用这一份。 */
export const ARCHIVE_RULE = `result.status 不是 ${ACTIVE_STATUS} 就归档到 ideas/${IDEAS_ARCHIVE_NAME}/`;

/**
 * 扫描一份目录并按规则搬文件。
 *
 * 返回 `{moved, kept, broken, dryRun}`：搬走的、留着的、看不懂的，各带原因。
 * `dryRun` 只报告，不改磁盘。
 */
export function archiveReady({ ideasDir, archiveDir, dryRun = false } = {}) {
  const source = ideasDir || path.join(currentWorkdir(), 'ideas');
  // 归档目录在 ideas/ 下面：待做与归档同根，但互不串味。
  const target = archiveDir || ideasArchiveDir();
  const moved = [];
  const kept = [];
  const broken = [];

  // 方向是目录（ideas/<方向>/<方向>_idea.json），扁平存放也认：递归找 *_idea.json，
  // 归档时保留相对路径，方向目录搬空后顺手删掉。
  const files = [];
  const walk = (dir) => {
    let entries = [];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch (err) {
      if (err && err.code === 'ENOENT') return;
      throw err;
    }
    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue;
      const full = path.join(dir, entry.name);
      // 归档目录本身跳过：归档过的东西不再归档一次（否则会一层层套进自己）。
      if (entry.isDirectory()) {
        if (entry.name === IDEAS_ARCHIVE_NAME) continue;
        walk(full);
      } else if (entry.name.endsWith('_idea.json')) {
        files.push(full);
      }
    }
  };
  walk(source);
  files.sort();

  for (const from of files) {
    const relative = path.relative(source, from);
    const name = relative;
    let doc;
    try {
      doc = JSON.parse(fs.readFileSync(from, 'utf8'));
    } catch (err) {
      broken.push({ name, error: err && err.message ? err.message : String(err) });
      continue;
    }
    const status =
      doc && doc.result && typeof doc.result.status === 'string' ? doc.result.status : null;
    if (!status || status === ACTIVE_STATUS) {
      kept.push({ name, status });
      continue;
    }
    if (!dryRun) {
      const to = path.join(target, relative);
      fs.mkdirSync(path.dirname(to), { recursive: true });
      fs.renameSync(from, to);
      // 方向目录空了就删掉，ideas/ 里不留空壳。
      let parent = path.dirname(from);
      while (parent !== source && parent.startsWith(source)) {
        try {
          if (fs.readdirSync(parent).length > 0) break;
          fs.rmdirSync(parent);
        } catch {
          break;
        }
        parent = path.dirname(parent);
      }
    }
    moved.push({ name, status });
  }
  return { moved, kept, broken, dryRun };
}

function main(argv) {
  const dryRun = argv.includes('--dry-run');
  const result = archiveReady({ dryRun });
  const summary = {
    rule: ARCHIVE_RULE,
    moved: result.moved,
    kept: result.kept.length,
    broken: result.broken,
    dryRun,
  };
  console.log(JSON.stringify(summary, null, 2));
  return result.broken.length > 0 ? 1 : 0;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  process.exit(main(process.argv.slice(2)));
}
