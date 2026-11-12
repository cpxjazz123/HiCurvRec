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

const here = path.dirname(new URL(import.meta.url).pathname);
const scratch = path.join(here, '..', '.out', `archive-ideas-${process.pid}`);

function writeIdea(dir, name, status) {
  fs.mkdirSync(dir, { recursive: true });
  const doc = { schema: 'genrec.idea.v1', id: name, result: {} };
  if (status !== null) doc.result.status = status;
  fs.writeFileSync(path.join(dir, name), `${JSON.stringify(doc, null, 2)}\n`);
}

function fresh() {
  fs.rmSync(scratch, { recursive: true, force: true });
  const ideas = path.join(scratch, 'ideas');
  const archive = path.join(scratch, 'ideas_archive');
  fs.mkdirSync(ideas, { recursive: true });
  return { ideas, archive };
}

function run() {
  fs.rmSync(scratch, { recursive: true, force: true });
  let failures = 0;
  const test = (name, fn) => {
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
    assert.deepEqual(fs.readdirSync(ideas), ['A_idea.json']);
    assert.deepEqual(fs.readdirSync(archive).sort(), ['B_idea.json', 'C_idea.json']);
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
    assert.deepEqual(fs.readdirSync(ideas), ['E_idea.json']);
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
    assert.deepEqual(fs.readdirSync(ideas).sort(), ['F_idea.json', 'G_idea.json']);
  });

  test('只认 *_idea.json', () => {
    const { ideas, archive } = fresh();
    writeIdea(ideas, 'H_idea.json', 'completed');
    fs.writeFileSync(path.join(ideas, 'notes.json'), '{"result":{"status":"completed"}}');
    const out = archiveReady({ ideasDir: ideas, archiveDir: archive });
    assert.deepEqual(out.moved.map((m) => m.name), ['H_idea.json']);
    assert.equal(fs.existsSync(path.join(ideas, 'notes.json')), true);
  });

  test('目录不存在时安静返回', () => {
    const out = archiveReady({ ideasDir: path.join(scratch, 'nope'),
                              archiveDir: path.join(scratch, 'nope_archive') });
    assert.deepEqual(out, { moved: [], kept: [], broken: [], dryRun: false });
  });

  fs.rmSync(scratch, { recursive: true, force: true });
  console.log(failures === 0
    ? 'ARCHIVE_IDEAS_PASS 6 checks'
    : `ARCHIVE_IDEAS_FAIL ${failures} check(s)`);
  return failures === 0 ? 0 : 1;
}

process.exit(run());
