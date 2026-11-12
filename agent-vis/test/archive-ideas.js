/**
 * 归档规则的测试：跑过的搬走，待做的留下，看不懂的原地不动。
 *
 * 临时目录放在 agent-vis/.out 下（已被 .gitignore 忽略），用完即删 —— 归档脚本
 * 会真的搬文件，测试必须跑在沙箱里，不能碰到仓库里的 ideas/。
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import { ACTIVE_STATUS, archiveReady } from '../archive-ideas.js';
import { scanIdeaFiles } from '../idea-store.js';
import { IDEAS_ARCHIVE_NAME } from '../config.js';

const here = path.dirname(new URL(import.meta.url).pathname);
const scratch = path.join(here, '..', '.out', `archive-ideas-${process.pid}`);

function writeIdea(dir, name, status) {
  fs.mkdirSync(dir, { recursive: true });
  const doc = { schema: 'genrec.idea.v1', id: name, result: {} };
  if (status !== null) doc.result.status = status;
  fs.writeFileSync(path.join(dir, name), `${JSON.stringify(doc, null, 2)}\n`);
}

/** 列 ideas/ 下的东西：归档目录本身不算"待做的方向"。 */
function listIdeas(ideas) {
  return fs
    .readdirSync(ideas)
    .filter((name) => name !== IDEAS_ARCHIVE_NAME)
    .sort();
}

function fresh() {
  fs.rmSync(scratch, { recursive: true, force: true });
  const ideas = path.join(scratch, 'ideas');
  // 归档目录在 ideas/ 下面（与待做同根），两边互不串味。
  const archive = path.join(ideas, IDEAS_ARCHIVE_NAME);
  fs.mkdirSync(ideas, { recursive: true });
  return { ideas, archive };
}

function run() {
  fs.rmSync(scratch, { recursive: true, force: true });
  let failures = 0;
  let checks = 0;
  const test = (name, fn) => {
    checks += 1;
    try {
      fn();
      console.log(`  ✓ ${name}`);
    } catch (err) {
      failures += 1;
      console.log(`  ✗ ${name}: ${err && err.message ? err.message : err}`);
    }
  };

  test('待做的留下，跑过的搬走', () => {
    const { ideas, archive } = fresh();
    writeIdea(ideas, 'A_idea.json', ACTIVE_STATUS);
    writeIdea(ideas, 'B_idea.json', 'completed');
    writeIdea(ideas, 'C_idea.json', 'running');
    const out = archiveReady({ ideasDir: ideas, archiveDir: archive });
    assert.deepEqual(out.moved.map((m) => m.name).sort(), ['B_idea.json', 'C_idea.json']);
    assert.deepEqual(out.kept.map((k) => k.name), ['A_idea.json']);
    assert.deepEqual(listIdeas(ideas), ['A_idea.json']);
    assert.deepEqual(fs.readdirSync(archive).sort(), ['B_idea.json', 'C_idea.json']);
  });

  test('归档目录就在 ideas/ 下面，两边互不串味', () => {
    const { ideas, archive } = fresh();
    writeIdea(ideas, 'E_idea.json', 'completed');
    const first = archiveReady({ ideasDir: ideas, archiveDir: archive });
    assert.deepEqual(first.moved.map((m) => m.name), ['E_idea.json']);
    assert.equal(fs.existsSync(path.join(archive, 'E_idea.json')), true, '没搬进归档目录');
    // 再跑一次：归档过的东西不会被再归档一次（不会套成 ideas/_archive/_archive/…）。
    const second = archiveReady({ ideasDir: ideas, archiveDir: archive });
    assert.deepEqual(second.moved.map((m) => m.name), [], '归档过的东西不该再被归档');
    assert.deepEqual(second.kept.map((k) => k.name), [], '归档目录里的文件不该被当成待做的');
    assert.equal(
      fs.existsSync(path.join(archive, IDEAS_ARCHIVE_NAME, 'E_idea.json')),
      false,
      '不该套出第二层归档目录',
    );
  });

  test('页面扫 ideas/ 时跳过归档目录', () => {
    const { ideas, archive } = fresh();
    writeIdea(ideas, 'F_idea.json', ACTIVE_STATUS);
    writeIdea(archive, 'G_idea.json', 'completed');
    const found = scanIdeaFiles(ideas).map((f) => path.basename(f.full)).sort();
    assert.deepEqual(found, ['F_idea.json'], '归档目录里的方向不该被页面读到');
  });

  test('文件名原样保留，内容一字不动', () => {
    const { ideas, archive } = fresh();
    writeIdea(ideas, 'D_idea.json', 'failed');
    const before = fs.readFileSync(path.join(ideas, 'D_idea.json'), 'utf8');
    archiveReady({ ideasDir: ideas, archiveDir: archive });
    assert.equal(fs.readFileSync(path.join(archive, 'D_idea.json'), 'utf8'), before);
  });

  test('dry-run 只报告，磁盘不动', () => {
    const { ideas, archive } = fresh();
    writeIdea(ideas, 'E_idea.json', 'completed');
    const out = archiveReady({ ideasDir: ideas, archiveDir: archive, dryRun: true });
    assert.deepEqual(out.moved.map((m) => m.name), ['E_idea.json']);
    assert.deepEqual(listIdeas(ideas), ['E_idea.json']);
    assert.equal(fs.existsSync(archive), false);
  });

  test('解析失败与缺 status 的文件原地不动并被报告', () => {
    const { ideas, archive } = fresh();
    fs.writeFileSync(path.join(ideas, 'F_idea.json'), '{ not json');
    writeIdea(ideas, 'G_idea.json', null);
    const out = archiveReady({ ideasDir: ideas, archiveDir: archive });
    assert.equal(out.moved.length, 0);
    assert.deepEqual(out.kept.map((k) => k.name), ['G_idea.json']);
    assert.deepEqual(out.broken.map((b) => b.name), ['F_idea.json']);
    assert.deepEqual(listIdeas(ideas), ['F_idea.json', 'G_idea.json']);
  });

  test('只认 *_idea.json', () => {
    const { ideas, archive } = fresh();
    writeIdea(ideas, 'H_idea.json', 'completed');
    fs.writeFileSync(path.join(ideas, 'notes.json'), '{"result":{"status":"completed"}}');
    const out = archiveReady({ ideasDir: ideas, archiveDir: archive });
    assert.deepEqual(out.moved.map((m) => m.name), ['H_idea.json']);
    assert.equal(fs.existsSync(path.join(ideas, 'notes.json')), true);
  });

  test('方向目录：保留相对路径归档，空了就删目录', () => {
    const { ideas, archive } = fresh();
    const direction = path.join(ideas, 'D1');
    writeIdea(direction, 'D1_idea.json', 'completed');
    const out = archiveReady({ ideasDir: ideas, archiveDir: archive });
    assert.deepEqual(out.moved.map((m) => m.name), [path.join('D1', 'D1_idea.json')]);
    assert.equal(fs.existsSync(path.join(archive, 'D1', 'D1_idea.json')), true);
    assert.equal(fs.existsSync(direction), false, '搬空的方向目录应被删掉');
  });

  test('方向目录里待做的留下、同目录的共享文件不动', () => {
    const { ideas, archive } = fresh();
    const direction = path.join(ideas, 'D2');
    writeIdea(direction, 'D2_idea.json', ACTIVE_STATUS);
    fs.writeFileSync(path.join(direction, 'main_line.json'), '{}');
    const out = archiveReady({ ideasDir: ideas, archiveDir: archive });
    assert.equal(out.moved.length, 0);
    assert.deepEqual(fs.readdirSync(direction).sort(), ['D2_idea.json', 'main_line.json']);
  });

  test('目录不存在时安静返回', () => {
    const out = archiveReady({ ideasDir: path.join(scratch, 'nope'),
                              archiveDir: path.join(scratch, 'nope_archive') });
    assert.deepEqual(out, { moved: [], kept: [], broken: [], dryRun: false });
  });

  fs.rmSync(scratch, { recursive: true, force: true });
  console.log(failures === 0
    ? `ARCHIVE_IDEAS_PASS ${checks} checks`
    : `ARCHIVE_IDEAS_FAIL ${failures} of ${checks} check(s)`);
  return failures === 0 ? 0 : 1;
}

process.exit(run());
