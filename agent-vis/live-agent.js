/**
 * 真实 agent：跑的是 pi（@earendil-works/pi-coding-agent）自己的 agent 会话。
 *
 * 一个 agent = 一个 pi AgentSession：模型调用、工具循环、事件流、压缩、abort 全部由 pi 负责，
 * 我们只做三件它不管的事：
 *
 *   1. 工具：pi 的内置工具（bash / edit / write …）没有仓库边界，所以这里关掉它们，
 *      把 agent-vis 自己的 8 个沙箱工具注册成 pi 的自定义工具（权限怎么裁表见下）；
 *   2. 提示词：人设提示词与权限说明追加在 pi 的默认 harness 提示词之后，不整体替换；
 *   3. 协议：把 pi 的事件流翻成页面认识的 UI message stream 分片。
 *
 * 给模型哪些工具由这一轮的权限决定（read / write / delete / execute / network 五项基础权限，
 * 每项绑一组真实工具）；没授的工具压根不进 pi 的工具表。所有会动仓库的工具都限制在仓库根
 * 下面；越界或读不到就直接报错，错误会变成协议里的 tool-output-error，正好在时间轴上显示出来。
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFile, spawn } from 'node:child_process';
import { promisify } from 'node:util';

import {
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
  createAgentSession,
  createEditToolDefinition,
  createFindToolDefinition,
  createGrepToolDefinition,
  createLsToolDefinition,
  createReadToolDefinition,
  createWriteToolDefinition,
  defineTool,
} from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';

import { IDEA_SCHEMA_VERSION, validateIdeaDoc } from './idea-schema.js';

import {
  API_KEY_FILE,
  BASE_URL,
  CONTEXT_WINDOW,
  FROZEN_BASELINE_TREE,
  IDEA_RULE,
  LANGUAGE_RULE,
  MODEL_ID,
  PERMISSION_CATALOG,
  PROVIDER_NAME,
  ROUND_END_RULE,
  STAGE2_PYTHON,
  ROUND_END_TOOL,
  ROUND_HANDOFF_TOOL,
  ROUND_SESSION_TTL_MS,
  currentWorkdir,
  buildHandoffPrompt,
  buildPermissionPrompt,
  dataDir,
  normalizePermissions,
  resolvePermissions,
  resolveProfile,
} from './config.js';

/** 目录遍历时跳过的重目录：太大且与 agent 的问题无关。 */
const SKIP_DIRS = new Set([
  'node_modules',
  '.git',
  'results',
  'logs',
  '__pycache__',
  '.pytest_cache',
  '.mypy_cache',
  '.ruff_cache',
  '.venv',
  'venv',
]);

/** 二进制/大文件不参与内容搜索。 */
const SKIP_FILE_RE = /\.(pth|pt|npy|npz|parquet|zip|tar|gz|bz2|xz|pickle|pkl|so|dylib|dll|png|jpg|jpeg|gif|webp|mp4|mp3)$/i;
const MAX_SEARCH_FILE_BYTES = 512 * 1024;
const MAX_WRITE_BYTES = 1024 * 1024;

/**
 * API key 只在运行时从 OpenCode 的密钥文件读；任何日志、错误信息都不含它。
 */
export function loadApiKey() {
  const raw = fs.readFileSync(API_KEY_FILE, 'utf8');
  const key = raw.trim();
  if (key.length === 0) {
    throw new Error(`API key 文件为空: ${API_KEY_FILE}`);
  }
  return key;
}

/**
 * 当前的工作目录（面板上「全局设置」可改，改完立刻生效）。
 *
 * 会话的 cwd、只读命令的 cwd、文件工具的边界都取它 —— 三者必须是同一个目录，
 * 否则 agent 读到的路径和它能写的路径会不是一棵树。
 */
function sandboxRoot() {
  return currentWorkdir();
}

/** 把 agent 给的路径解析到沙箱根内；越界、缺文件都直接抛错。 */
function resolveInSandbox(relPath) {
  const root = sandboxRoot();
  const target = path.resolve(root, relPath);
  if (target !== root && !target.startsWith(root + path.sep)) {
    throw new Error(`路径越界（只允许仓库根 ${root} 之内）: ${relPath}`);
  }
  if (!fs.existsSync(target)) {
    throw new Error(`路径不存在: ${relPath}`);
  }
  return target;
}

/**
 * 把相对路径解析成仓库内一个**可以改动**的绝对路径。
 *
 * 写与删共用这一套边界：越界、.git 一律当场报错，不做任何回退。
 */
/** 一个 shell 风格的通配（`*` 单段、`**` 任意层）转正则。 */
function globToRegExp(pattern) {
  let out = '';
  for (let i = 0; i < pattern.length; i += 1) {
    const ch = pattern[i];
    if (ch === '*') {
      if (pattern[i + 1] === '*') {
        out += '.*';
        i += 1;
      } else {
        out += '[^/]*';
      }
    } else if ('\\^$.|?+()[]{}'.includes(ch)) {
      out += `\\${ch}`;
    } else {
      out += ch;
    }
  }
  return new RegExp(`^${out}$`);
}

/** 通配里第一个元字符之前的部分：用来判断"这个目录是不是某个允许路径的祖先"。 */
function literalPrefix(pattern) {
  const cut = pattern.search(/[*?[]/);
  return cut < 0 ? pattern : pattern.slice(0, cut);
}

/**
 * 写作用域：非空时，只有命中其中一条通配的路径才可写。
 *
 * 这是并行写同一棵树的硬前提：几个 agent 分文件写，谁都不能越界改别人的文件。
 */
export function scopeAllows(scope, relPosix, { isDir = false } = {}) {
  if (!Array.isArray(scope) || scope.length === 0) return true;
  if (scope.some((pattern) => globToRegExp(pattern).test(relPosix))) return true;
  if (isDir) {
    const dir = relPosix.replace(/\/+$/, '');
    return scope.some((pattern) => {
      const prefix = literalPrefix(pattern).replace(/\/+$/, '');
      return prefix === dir || prefix.startsWith(`${dir}/`);
    });
  }
  return false;
}

function resolveMutableInSandbox(relPath, scope, { isDir = false } = {}) {
  const root = sandboxRoot();
  const target = path.resolve(root, relPath);
  if (target !== root && !target.startsWith(root + path.sep)) {
    throw new Error(`路径越界（只允许仓库根 ${root} 之内）: ${relPath}`);
  }
  const rel = path.relative(root, target);
  if (rel.split(path.sep)[0] === '.git') {
    throw new Error(`不许动 .git 目录下的东西: ${relPath}`);
  }
  const relPosix = rel.split(path.sep).join('/');
  if (!scopeAllows(scope, relPosix, { isDir })) {
    throw new Error(
      `写入越界：${relPosix} 不在这个 agent 的文件范围里（允许：${(scope || []).join('、')}）`,
    );
  }
  return target;
}

function relativeToRoot(target) {
  const rel = path.relative(sandboxRoot(), target);
  return rel === '' ? '.' : rel;
}

/**
 * 跑一条只读命令。命令与参数都是代码里写死的，不接受模型拼出来的任意命令行。
 */
const execFileAsync = promisify(execFile);

const READ_ONLY_TIMEOUT_MS = 30000;

async function runReadOnly(command, args) {
  const { stdout } = await execFileAsync(command, args, {
    cwd: sandboxRoot(),
    timeout: READ_ONLY_TIMEOUT_MS,
    maxBuffer: 4 * 1024 * 1024,
  });
  return stdout;
}

/**
 * 工具的实际实现。
 *
 * 与 pi 无关，纯函数式地做一件事并返回结构化结果：这里能被单测直接调用，
 * pi 那边只是把同一份实现包成工具定义。
 */
export const TOOL_IMPLS = {
  async listFiles({ dir }) {
    const target = resolveInSandbox(dir);
    const entries = await fs.promises.readdir(target, { withFileTypes: true });
    const items = [];
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (SKIP_DIRS.has(entry.name)) continue;
      const full = path.join(target, entry.name);
      let size = null;
      if (entry.isFile()) size = (await fs.promises.stat(full)).size;
      items.push({
        name: entry.name,
        type: entry.isDirectory() ? 'dir' : entry.isFile() ? 'file' : 'other',
        size,
      });
    }
    return { dir: relativeToRoot(target), count: items.length, items };
  },

  async readFile({ path: relPath, maxLines = 120 }) {
    const target = resolveInSandbox(relPath);
    const stat = await fs.promises.stat(target);
    if (stat.isDirectory()) {
      throw new Error(`这是目录不是文件，请改用 listFiles: ${relPath}`);
    }
    const content = await fs.promises.readFile(target, 'utf8');
    const lines = content.split('\n');
    return {
      path: relativeToRoot(target),
      totalLines: lines.length,
      truncated: lines.length > maxLines,
      content: lines.slice(0, maxLines).join('\n'),
    };
  },

  /**
   * 从冻结基线复制一份候选树。
   *
   * 每一轮的新机制都从 stage2_RQ-VAE/TIGER_RQ-VAE 起步（仓库规则 19b）：
   * 复制一份、只改 ARM_NAME 这一个常量、其余逐文件一致。已经存在的目录不覆盖 ——
   * 候选树要是旧的，机制就说不清是哪一版的。
   */
  async copyBaselineTree({ arm, from = 'stage2_RQ-VAE/TIGER_RQ-VAE', renameArm = true }) {
    const name = typeof arm === 'string' ? arm.trim() : '';
    if (!/^TIGER_[A-Za-z0-9_]+$/.test(name)) {
      throw new Error(`arm 名字不对（要像 TIGER_af3_curvature_drive_rq）: ${arm}`);
    }
    const source = resolveInSandbox(from);
    const stageRoot = path.dirname(source);
    const target = path.join(stageRoot, name);
    if (path.dirname(target) !== stageRoot) {
      throw new Error(`候选树必须建在 stage2_RQ-VAE 下面: ${name}`);
    }
    // 独占建目录就是锁：两个 writer 同时建同一棵树时，只有一个能拿到 EEXIST 之外的结果。
    try {
      fs.mkdirSync(target);
    } catch (err) {
      if (err && err.code === 'EEXIST') {
        throw new Error(`候选树已经存在，不覆盖: ${name}`);
      }
      throw err;
    }
    try {
      await fs.promises.cp(source, target, {
        recursive: true,
        filter: (src) => {
          const base = path.basename(src);
          return base !== '__pycache__' && base !== 'logs' && !base.endsWith('.pyc');
        },
      });
    } catch (err) {
      // 复制失败就把占位目录删掉，别留下半棵树挡住下一次。
      try { fs.rmSync(target, { recursive: true, force: true }); } catch { /* 忽略 */ }
      throw err;
    }
    // 默认把 ARM_NAME 改成新树名（产物目录由它推出来）；renameArm=false 时整棵树
    // 与基线逐字节相同，只换了目录名 —— 需要 pristine 副本时用这个。
    const oldArm = path.basename(source);
    let rewritten = 0;
    if (renameArm === false) {
      return { path: relativeToRoot(target), arm: name, files: null, renamed: 0, pristine: true };
    }
    const walk = async (dir) => {
      const entries = await fs.promises.readdir(dir, { withFileTypes: true });
      for (const entry of entries) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          await walk(full);
          continue;
        }
        if (!/\.(py|json|md|txt)$/.test(entry.name)) continue;
        const text = await fs.promises.readFile(full, 'utf8');
        const next = text.replace(
          /(ARM_NAME\s*=\s*)"[^"]*"/,
          `$1"${name}"`,
        );
        if (next !== text) {
          await fs.promises.writeFile(full, next, 'utf8');
          rewritten += 1;
        }
      }
    };
    await walk(target);
    const files = [];
    const count = async (dir) => {
      const entries = await fs.promises.readdir(dir, { withFileTypes: true });
      for (const entry of entries) {
        if (entry.isDirectory()) await count(path.join(dir, entry.name));
        else files.push(path.join(dir, entry.name));
      }
    };
    await count(target);
    return {
      path: relativeToRoot(target),
      arm: name,
      from: relativeToRoot(source),
      files: files.length,
      rewritten,
    };
  },

  async searchFiles({ pattern, dir = '.', maxMatches = 30 }) {
    const root = resolveInSandbox(dir);
    const re = new RegExp(pattern);
    const matches = [];

    const walk = (current, depth) => {
      if (depth > 5 || matches.length >= maxMatches) return;
      let entries;
      try {
        entries = fs.readdirSync(current, { withFileTypes: true });
      } catch {
        return;
      }
      for (const entry of entries) {
        if (matches.length >= maxMatches) return;
        if (SKIP_DIRS.has(entry.name)) continue;
        const full = path.join(current, entry.name);
        if (entry.isDirectory()) {
          walk(full, depth + 1);
          continue;
        }
        if (!entry.isFile()) continue;
        if (SKIP_FILE_RE.test(entry.name)) continue;
        let stat;
        try {
          stat = fs.statSync(full);
        } catch {
          continue;
        }
        if (stat.size > MAX_SEARCH_FILE_BYTES) continue;
        let text;
        try {
          text = fs.readFileSync(full, 'utf8');
        } catch {
          continue;
        }
        const lines = text.split('\n');
        for (let i = 0; i < lines.length && matches.length < maxMatches; i += 1) {
          if (re.test(lines[i])) {
            matches.push({
              file: relativeToRoot(full),
              line: i + 1,
              text: lines[i].slice(0, 200),
            });
          }
        }
      }
    };

    walk(root, 0);
    return {
      pattern,
      dir: relativeToRoot(root),
      matchCount: matches.length,
      truncated: matches.length >= maxMatches,
      matches,
    };
  },

  async listIssues({ limit = 10 } = {}) {
    const stdout = await runReadOnly('gh', [
      'issue',
      'list',
      '--state',
      'open',
      '--limit',
      String(limit),
      '--json',
      'number,title,labels,state,updatedAt,url',
    ]);
    return JSON.parse(stdout);
  },

  async readIssue({ number }) {
    const stdout = await runReadOnly('gh', [
      'issue',
      'view',
      String(number),
      '--json',
      'number,title,body,labels,state,createdAt,updatedAt,url',
    ]);
    const issue = JSON.parse(stdout);
    // 评论单独拉，正文可能很长，整体返回会撑爆上下文。
    const comments = JSON.parse(
      await runReadOnly('gh', ['issue', 'view', String(number), '--json', 'comments']),
    ).comments || [];
    return {
      number: issue.number,
      title: issue.title,
      state: issue.state,
      labels: (issue.labels || []).map((label) => label.name),
      body: issue.body,
      comments: comments.map((comment) => ({
        author: comment.author && comment.author.login,
        createdAt: comment.createdAt,
        body: comment.body,
      })),
    };
  },

  async gitLog({ count = 3 } = {}) {
    // %x1e 分隔 commit，%x1f 分隔字段，%x00 分隔文件名。
    const stdout = await runReadOnly('git', [
      'log',
      `-${count}`,
      '--name-only',
      '--pretty=format:%x1e%H%x1f%an%x1f%aI%x1f%s%x1f%b',
    ]);
    const commits = stdout
      .split('\x1e')
      .map((block) => block.trim())
      .filter((block) => block !== '')
      .map((block) => {
        const [head, ...rest] = block.split('\n');
        const [sha, author, date, subject, body = ''] = head.split('\x1f');
        return {
          sha,
          author,
          date,
          subject,
          body: body.trim(),
          files: rest.map((line) => line.trim()).filter(Boolean),
        };
      });
    return { commits };
  },

  async writeFile({ path: relPath, content }) {
    const target = resolveMutableInSandbox(relPath);
    const bytes = Buffer.byteLength(content, 'utf8');
    if (bytes > MAX_WRITE_BYTES) {
      throw new Error(
        `要写的内容太大（${bytes} 字节，上限 ${MAX_WRITE_BYTES} 字节）: ${relPath}`,
      );
    }
    const existed = fs.existsSync(target);
    await fs.promises.mkdir(path.dirname(target), { recursive: true });
    await fs.promises.writeFile(target, content, 'utf8');
    return { path: relativeToRoot(target), bytes, created: !existed };
  },

  async deleteFile({ path: relPath }, options = {}) {
    const target = resolveMutableInSandbox(relPath, options.writeScope);
    if (target === sandboxRoot()) {
      throw new Error(`不许删仓库根目录: ${relPath}`);
    }
    if (!fs.existsSync(target)) {
      throw new Error(`路径不存在，删不了: ${relPath}`);
    }
    const isDir = fs.statSync(target).isDirectory();
    await fs.promises.rm(target, { recursive: isDir, force: false });
    return { path: relativeToRoot(target), deleted: true, type: isDir ? 'dir' : 'file' };
  },
};

/**
 * 把一份工具实现包成 pi 的工具定义。
 *
 * pi 的工具结果要分两半：content 是给模型读的文本，details 是给界面画的结构化结果。
 * 页面要的是后者 —— 与切到 pi 之前完全一样的那个对象。
 */
function piTool({ name, label, description, parameters, impl, callArgs = (params) => params }) {
  return defineTool({
    name,
    label,
    description,
    parameters,
    execute: async (toolCallId, params, signal, onUpdate, ctx) => {
      const details = await impl(callArgs(params, ctx));
      return {
        content: [{ type: 'text', text: JSON.stringify(details) }],
        details,
      };
    },
  });
}

const runTestsTool = piTool({
  name: 'runTests',
  label: '跑候选树的单元测试',
  description:
    'Run the unit tests in a candidate tree (<tree>/tests) on CPU with the Stage2 interpreter. ' +
    'Only for you to check your own work: it is not admission evidence (the framework runs the ' +
    'four verification suites itself). Returns exit code and the tail of the output.',
  parameters: Type.Object({
    tree: Type.String({ description: '候选树路径，例如 stage2_RQ-VAE/TIGER_xxx_rq' }),
    timeoutSeconds: Type.Optional(
      Type.Integer({ minimum: 1, maximum: 1800, description: '超时秒数，默认 600' }),
    ),
  }),
  impl: runTests,
});

const treeDiffTool = piTool({
  name: 'treeDiff',
  label: '比两棵树',
  description:
    'Compare a candidate tree with the frozen baseline: byte-identical files, changed files with the ' +
    'first lines of each unified diff, and files that exist on one side only. Read-only. ' +
    'Use it instead of walking both trees by hand - the review only has to look at the delta.',
  parameters: Type.Object({
    baseline: Type.String({
      description: '冻结基线目录（仓库内相对路径），例如 stage2_RQ-VAE/TIGER_RQ-VAE',
    }),
    tree: Type.String({ description: '候选树目录，例如 stage2_RQ-VAE/TIGER_xxx_rq' }),
    maxHunkLines: Type.Optional(
      Type.Integer({
        minimum: 1,
        maximum: 200,
        description: '每个改动文件最多回多少行 diff，默认 40',
      }),
    ),
  }),
  impl: treeDiff,
});

const readDiffTool = piTool({
  name: 'readDiff',
  label: '读 diff',
  description:
    'Unified diff of one file (or of every changed file) between the frozen baseline and the candidate tree. ' +
    'Use it instead of reading a whole file: the review surface is the changed lines plus their context.',
  parameters: Type.Object({
    tree: Type.String({ description: '候选树目录，例如 stage2_RQ-VAE/TIGER_xxx_rq' }),
    file: Type.Optional(
      Type.String({ description: '相对树根的单个文件，例如 train_rqvae.py；不给就列出全部改动文件' }),
    ),
    baseline: Type.Optional(Type.String({ description: '冻结基线目录，默认 stage2_RQ-VAE/TIGER_RQ-VAE' })),
    maxLines: Type.Optional(
      Type.Integer({ minimum: 1, maximum: 2000, description: '每个文件最多回多少行 diff，默认 200' }),
    ),
  }),
  impl: readDiff,
});

const specCheckTool = piTool({
  name: 'specCheck',
  label: '机械预检',
  description:
    'Mechanical pre-checks for a candidate tree: declared files present, absolute paths / Path(__file__).parents[N] / ' +
    'ARM_NAME values, frozen hyper-parameters against the baseline (JSON + Python constants), spec metrics that never ' +
    'landed, and tests with no or vacuous assertions. Read-only; it does not judge formula semantics or switch ' +
    'degeneration - those still need reading the code.',
  parameters: Type.Object({
    tree: Type.String({ description: '候选树目录，例如 stage2_RQ-VAE/TIGER_xxx_rq' }),
    idea: Type.Optional(
      Type.String({ description: 'idea 目录（含 *_idea.json 与 math.json）；不给就按 arm 名在 ideas/ 里找' }),
    ),
    baseline: Type.Optional(
      Type.String({ description: '冻结基线目录，默认 stage2_RQ-VAE/TIGER_RQ-VAE' }),
    ),
  }),
  impl: specCheck,
});

const stage2StartTool = piTool({
  name: 'stage2Start',
  label: '启动 Stage2 训练',
  description:
    'Start the Stage2 training for one candidate tree (stage2_RQ-VAE/TIGER_<ARM>/) through the ' +
    'repository MCP server. It archives old artifacts, runs the one-epoch smoke, then trains.',
  parameters: Type.Object({
    tree: Type.String({ description: '候选树路径，例如 stage2_RQ-VAE/TIGER_af3_curvature_drive_rq' }),
  }),
  impl: async ({ tree }) => {
    // 绝对路径：MCP 那边的 source_dir 要么是树名，要么是能解析到那棵树的绝对路径。
    return callGenrecMcp('stage2_start', { source_dir: stage2TreePath(tree) });
  },
});

const stage2StatusTool = piTool({
  name: 'stage2Status',
  label: '看 Stage2 训练状态',
  description:
    'Read where the Stage2 run of one candidate tree stands (status, worker, latest epoch, ' +
    'log tails, exported artifacts). Calls for the same tree are spaced one minute apart, so ' +
    'calling it repeatedly is how you watch the training.',
  parameters: Type.Object({
    tree: Type.String({ description: '候选树路径' }),
  }),
  impl: async ({ tree }) => {
    stage2TreePath(tree);
    const key = path.resolve(currentWorkdir(), tree);
    const previous = lastStage2StatusAt.get(key) || 0;
    const wait = Math.max(0, STAGE2_STATUS_INTERVAL_MS - (Date.now() - previous));
    if (wait > 0) await sleepMs(wait);
    lastStage2StatusAt.set(key, Date.now());
    const status = await callGenrecMcp('stage2_status', { source_dir: stage2TreePath(tree) });
    return { waited_seconds: Math.round(wait / 1000), ...status };
  },
});

const stage2StopTool = piTool({
  name: 'stage2Stop',
  label: '暂停 Stage2 训练',
  description:
    'Stop the Stage2 run of one candidate tree: the training and its smoke worker are ' +
    'terminated, logs and partial artifacts are kept.',
  parameters: Type.Object({
    tree: Type.String({ description: '候选树路径' }),
  }),
  impl: async ({ tree }) => {
    return callGenrecMcp('stage2_stop', { source_dir: stage2TreePath(tree) });
  },
});

const webSearchTool = piTool({
  name: 'webSearch',
  label: '检索文献',
  description:
    'Search published literature (OpenAlex). Returns titles, year, venue, DOI and citation ' +
    'counts for the matches; it does not summarise them for you.',
  parameters: Type.Object({
    query: Type.String({ description: '检索词，例如 "hyperbolic residual quantization recommendation"' }),
    limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 25, description: '最多返回几条，默认 8' })),
  }),
  impl: searchLiterature,
});

const checkJsonTool = piTool({
  name: 'checkJson',
  label: '校验 JSON',
  description:
    'Parse one JSON file inside the repo and report whether it is valid; on failure you get ' +
    'the parser message with line and column, so you can fix exactly that spot.',
  parameters: Type.Object({
    path: Type.String({ description: '仓库相对路径，例如 ideas/xxx/xxx_idea.json' }),
  }),
  impl: checkJson,
});

const checkIdeaTool = piTool({
  name: 'checkIdea',
  label: '校验 idea 字段',
  description:
    'Check one idea file against the fixed idea schema (genrec.idea.v2): required fields, closed ' +
    'value sets, and the fields that must not exist (run plans, self-made thresholds, status). ' +
    'Returns ok plus a per-field error list, so you can fix exactly what is wrong.',
  parameters: Type.Object({
    path: Type.String({ description: '仓库相对路径，例如 ideas/xxx/xxx_idea.json' }),
  }),
  impl: checkIdea,
});

const copyBaselineTreeTool = piTool({
  name: 'copyBaselineTree',
  label: '复制基线候选树',
  description:
    'Copy the frozen baseline tree into a new candidate tree (stage2_RQ-VAE/TIGER_<ARM>/). ' +
    'By default only ARM_NAME is rewritten; with renameArm:false every file stays byte-identical and ' +
    'only the directory name differs. An existing target is never overwritten.',
  parameters: Type.Object({
    arm: Type.String({ description: '新候选树的名字，例如 TIGER_af3_curvature_drive_rq' }),
    from: Type.Optional(
      Type.String({ description: '基线树路径，默认 stage2_RQ-VAE/TIGER_RQ-VAE' }),
    ),
    renameArm: Type.Optional(
      Type.Boolean({
        description: '是否把 ARM_NAME 改成新树名；false = 除目录名外与基线逐字节相同，默认 true',
      }),
    ),
  }),
  impl: TOOL_IMPLS.copyBaselineTree,
});

const listIssuesTool = piTool({
  name: 'listIssues',
  label: '列 Issue',
  description:
    'List the open GitHub issues of this repository (read-only). Returns number, title, labels and updated time.',
  parameters: Type.Object({
    limit: Type.Optional(
      Type.Integer({ minimum: 1, maximum: 50, description: '最多返回多少个，默认 10' }),
    ),
  }),
  impl: TOOL_IMPLS.listIssues,
});

const readIssueTool = piTool({
  name: 'readIssue',
  label: '读 Issue',
  description: 'Read one GitHub issue in full: its body, labels, state and comments (read-only).',
  parameters: Type.Object({
    number: Type.Integer({ minimum: 1, description: 'Issue 编号' }),
  }),
  impl: TOOL_IMPLS.readIssue,
});

const gitLogTool = piTool({
  name: 'gitLog',
  label: '读 commit',
  description:
    'Show the most recent commits of this repository together with the files each one changed (read-only).',
  parameters: Type.Object({
    count: Type.Optional(
      Type.Integer({ minimum: 1, maximum: 20, description: '读几条 commit，默认 3' }),
    ),
  }),
  impl: TOOL_IMPLS.gitLog,
});

function makeDeleteTool(scope) {
  return piTool({
    name: 'deleteFile',
    label: '删文件',
    description: 'Delete a file or a directory inside the repo. Path is relative to the repo root.',
    parameters: Type.Object({
      path: Type.String({ description: '仓库相对路径' }),
    }),
    impl: (args) => TOOL_IMPLS.deleteFile(args, { writeScope: scope }),
  });
}

const deleteFileTool = makeDeleteTool(undefined);

/**
 * 轮次控制工具：本步判定"本轮不用继续"时调用它。
 *
 * 它不属于 read / write / delete / execute / network 任何一项权限 —— 它不是仓库能力，
 * 是轮次机制自己的控件。客户端只能看见工具调用这种结构化信号，
 * 看不见模型在文字里写的"本轮结束"，所以这个信号必须是工具。
 */
const endRoundTool = defineTool({
  name: ROUND_END_TOOL,
  label: '结束本轮',
  description:
    'Call this when this step decides the round should stop here (for example: no open issue at all, or an open issue means the controller must do nothing). The remaining steps will be skipped.',
  parameters: Type.Object({
    reason: Type.String({ description: '为什么本轮到此为止' }),
  }),
  execute: async (toolCallId, { reason }) => ({
    content: [{ type: 'text', text: `本轮到此结束：${reason}` }],
    details: { reason },
  }),
});

/**
 * 交接工具：轮次的最后一步靠它决定"要不要交"。
 *
 * 和 endRound 一样，它不是仓库能力，是轮次机制自己的控件 —— 客户端只能看见
 * 工具调用这种结构化信号，看不见模型在文字里写的"该交/不该交"。
 * 交接条件（人写的一句话）由最后一步的模型判定：判定该交才调用它。
 */
const requestHandoffTool = defineTool({
  name: ROUND_HANDOFF_TOOL,
  label: '交接结论',
  description:
    'Call this on the handoff step when this round result should be sent to the configured targets. ' +
    'If the handoff condition says it should not be sent, do not call it, and say why in your answer.',
  parameters: Type.Object({
    summary: Type.String({ description: '一句话说明这次交的是什么' }),
  }),
  execute: async (toolCallId, { summary }) => ({
    content: [{ type: 'text', text: `已交接：${summary}` }],
    details: { summary },
  }),
});

/**
 * 所有工具，按名字索引。
 *
 * 权限目录里写的工具名必须都在这张表里 —— 对不上就是配置坏了。
 */
/**
 * 调用仓库里那个 MCP 服务（tools/genrec-mcp/server.py）的一个工具。
 *
 * 训练只有这一条入口（CLAUDE.md §39：只使用现有 MCP 训练接口），所以这里
 * 不自己实现训练、不自己调度进程 —— 只是把一次 tools/call 转过去。
 */
async function callGenrecMcp(toolName, args, { timeoutMs = 180000 } = {}) {
  const root = currentWorkdir();
  const server = path.join(root, 'tools', 'genrec-mcp', 'server.py');
  if (!fs.existsSync(server)) {
    throw new Error(`找不到 MCP 服务：${server}`);
  }
  const request = {
    jsonrpc: '2.0',
    id: 1,
    method: 'tools/call',
    params: { name: toolName, arguments: args || {} },
  };
  return new Promise((resolve, reject) => {
    const child = spawn('python3', [server], {
      cwd: root,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let out = '';
    let err = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new Error(`MCP 工具 ${toolName} 超时（${timeoutMs} ms）`));
    }, timeoutMs);
    child.stdout.on('data', (chunk) => {
      out += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      err += chunk.toString();
    });
    child.on('error', (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on('close', () => {
      clearTimeout(timer);
      for (const line of out.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('{')) continue;
        let message;
        try {
          message = JSON.parse(trimmed);
        } catch {
          continue;
        }
        if (message.id !== 1) continue;
        const content = message.result && message.result.content;
        const text = content && content[0] ? content[0].text : '';
        if (message.error) {
          reject(new Error(message.error.message || 'MCP 调用失败'));
          return;
        }
        if (message.result && message.result.isError) {
          reject(new Error(String(text).slice(0, 400)));
          return;
        }
        try {
          resolve(JSON.parse(text));
        } catch {
          resolve({ text: String(text) });
        }
        return;
      }
      reject(new Error(`MCP 工具 ${toolName} 没有回话${err ? `：${err.slice(0, 300)}` : ''}`));
    });
    child.stdin.write(`${JSON.stringify(request)}\n`);
    child.stdin.end();
  });
}

/**
 * 状态检查的节奏：同一棵树两次检查之间至少隔一分钟。
 *
 * 需求是"每 1 分钟检查一次训练状态"，所以这个工具自己把关节奏：
 * 离上次检查不到一分钟时，它等到满一分钟再回话。返回值带 waited_seconds，
 * 模型能看见自己等了多久。
 */
const STAGE2_STATUS_INTERVAL_MS = 60000;
const lastStage2StatusAt = new Map();

function sleepMs(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function stage2TreePath(relPath) {
  const target = path.resolve(currentWorkdir(), relPath);
  const root = currentWorkdir();
  if (target !== root && !target.startsWith(root + path.sep)) {
    throw new Error(`路径越界（只允许仓库根 ${root} 之内）: ${relPath}`);
  }
  return target;
}

/**
 * 检索文献（OpenAlex：公开接口，不需要 key）。
 *
 * 返回的是"这个方向有没有现成的工作、他们做了什么"这种读数，
 * 不是结论 —— 结论由 agent 自己从这些条目里读出来。
 */
async function searchLiterature({ query, limit = 8 }) {
  const text = typeof query === 'string' ? query.trim() : '';
  if (text === '') throw new Error('检索词不能为空');
  const count = Math.min(Math.max(Number(limit) || 8, 1), 25);
  const url = new URL('https://api.openalex.org/works');
  url.searchParams.set('search', text);
  url.searchParams.set('per-page', String(count));
  const res = await fetch(url, { headers: { 'User-Agent': 'agent-vis/1.0' } });
  if (!res.ok) throw new Error(`文献检索失败：HTTP ${res.status}`);
  const data = await res.json();
  const works = Array.isArray(data.results) ? data.results : [];
  return {
    query: text,
    count: works.length,
    works: works.map((work) => ({
      title: work.title || '',
      year: work.publication_year || null,
      venue:
        (work.primary_location &&
          work.primary_location.source &&
          work.primary_location.source.display_name) ||
        '',
      doi: work.doi || '',
      cited_by: work.cited_by_count || 0,
      authors: (work.authorships || [])
        .slice(0, 4)
        .map((entry) => (entry.author && entry.author.display_name) || '')
        .filter(Boolean),
    })),
  };
}

/**
 * 校验一个仓库里的 JSON 文件能不能解析。
 *
 * 模型写字容易漏逗号、丢引号，而它自己没有执行工具、没法当场验证 ——
 * 这个工具就是它的"解析器"：返回 ok 或者具体的行列与错误信息。
 */
async function checkJson({ path: relPath }) {
  const target = resolveInSandbox(relPath);
  let text = '';
  try {
    text = await fs.promises.readFile(target, 'utf8');
  } catch (err) {
    throw new Error(`读不了这个文件: ${relPath}`);
  }
  try {
    const value = JSON.parse(text);
    return { ok: true, path: relativeToRoot(target), bytes: text.length, kind: Array.isArray(value) ? 'array' : typeof value, keys: Array.isArray(value) ? null : Object.keys(value) };
  } catch (err) {
    const message = err && err.message ? err.message : String(err);
    // JSON.parse 的错误里带 "line X column Y" —— 原样给出去，模型才改得准。
    const matched = /line (\d+) column (\d+)/.exec(message);
    return {
      ok: false,
      path: relativeToRoot(target),
      bytes: text.length,
      error: message,
      line: matched ? Number(matched[1]) : null,
      column: matched ? Number(matched[2]) : null,
    };
  }
}

/**
 * 校验一份 idea 文件的字段形状。
 *
 * checkJson 只管"能不能解析"；这个工具管"字段对不对"：
 * 必填字段、闭合取值、以及那些**不该存在**的字段（运行计划、自定门槛、状态）。
 * 逐条报错，模型照着改就能过。
 */
async function checkIdea({ path: relPath }) {
  const target = resolveInSandbox(relPath);
  let text = '';
  try {
    text = await fs.promises.readFile(target, 'utf8');
  } catch (err) {
    throw new Error(`读不了这个文件: ${relPath}`);
  }
  let value;
  try {
    value = JSON.parse(text);
  } catch (err) {
    const message = err && err.message ? err.message : String(err);
    const matched = /line (\d+) column (\d+)/.exec(message);
    return {
      ok: false,
      path: relativeToRoot(target),
      schema: IDEA_SCHEMA_VERSION,
      error: message,
      line: matched ? Number(matched[1]) : null,
      column: matched ? Number(matched[2]) : null,
    };
  }
  // 基线树给进去：这样"新文件"与"改动文件"分得开 ——
  // 基线里已有的文件写进 new_files 会被指出来。
  const verdict = validateIdeaDoc(value, {
    baselineRoot: path.join(sandboxRoot(), FROZEN_BASELINE_TREE),
  });
  return {
    ok: verdict.ok,
    path: relativeToRoot(target),
    schema: verdict.version,
    errors: verdict.errors,
  };
}

/**
 * 问一句、判断一句 —— 循环闸用。
 *
 * 不带任何工具权限、不记轮次：跑一次，把文字收回来就够了。
 */
export async function askOnce(prompt, { profileId, instructions } = {}) {
  let text = '';
  for await (const chunk of runLiveChunks(prompt, {
    profileId,
    instructions,
    permissions: [],
  })) {
    if (chunk && chunk.type === 'text-delta' && typeof chunk.delta === 'string') {
      text += chunk.delta;
    }
  }
  return text.trim();
}

/**
 * pi 自己的文件工具：read / write / edit / ls / grep / find。
 *
 * 引擎全是 pi 的 —— 不重复实现它已经有的能力。
 * 但"能碰哪些文件"由我们说了算：这些工具的文件操作被换成我们的实现，
 * 于是仓库根之外、.git 之内这些仍然被挡着（pi 的工具不认识仓库的规矩）。
 *
 * 只读的那几个（read/ls/grep/find）用 pi 的默认实现即可：它们只读，
 * 而会话的 cwd 就是仓库根，边界天然是仓库根。
 */
const sandboxReadOperations = {
  readFile: async (absolutePath) => {
    const target = resolveInSandbox(absolutePath);
    return fs.promises.readFile(target);
  },
  access: async (absolutePath) => {
    const target = resolveInSandbox(absolutePath);
    await fs.promises.access(target, fs.constants.R_OK);
  },
};

/** 作用域所属的树根：`stage2_RQ-VAE/<arm>`（作用域第一条通配的前两段）。 */
function scopeTreeRoot(scope) {
  if (!Array.isArray(scope) || scope.length === 0) return '';
  const pattern = String(scope[0] || '');
  if (!pattern.startsWith('stage2_RQ-VAE/')) return '';
  return pattern.split('/').slice(0, 2).join('/');
}

/**
 * 作用域内写入前先确认树根存在。
 *
 * 不确认的话，write 会自动 mkdir 出一条只有它自己文件的"树" —— 没有基线、
 * 没有接口文件，后面所有审计都无从谈起。建树是 copyBaselineTree 的事。
 */
function requireTreeRoot(scope) {
  const treeRoot = scopeTreeRoot(scope);
  if (treeRoot === '') return;
  if (!fs.existsSync(path.join(sandboxRoot(), treeRoot))) {
    throw new Error(
      `候选树还不存在：${treeRoot}。它由 AGENT IDEA 在建 idea 时复制好（只换目录名）；` +
        '你这边不建树 —— 停下，并在结论里写清"树缺失"',
    );
  }
}

/** 写操作按作用域生成：同一个进程里并行跑多个 agent 时，作用域不能是全局状态。 */
function makeWriteOperations(scope) {
  return {
    writeFile: async (absolutePath, content) => {
      requireTreeRoot(scope);
      const target = resolveMutableInSandbox(absolutePath, scope);
      const buf = Buffer.from(content);
      if (buf.byteLength > MAX_WRITE_BYTES) {
        throw new Error(
          `要写的内容太大（${buf.byteLength} 字节，上限 ${MAX_WRITE_BYTES} 字节）: ${absolutePath}`,
        );
      }
      await fs.promises.writeFile(target, buf);
    },
    mkdir: async (dir) => {
      requireTreeRoot(scope);
      const root = sandboxRoot();
      const abs = path.resolve(root, dir);
      const relPosix = path.relative(root, abs).split(path.sep).join('/');
      if (!scopeAllows(scope, relPosix, { isDir: true })) {
        throw new Error(`建目录越界：${relPosix} 不在这个 agent 的文件范围里`);
      }
      const target = resolveMutableInSandbox(dir, scope, { isDir: true });
      await fs.promises.mkdir(target, { recursive: true });
    },
    access: async (absolutePath) => {
      requireTreeRoot(scope);
      const target = resolveMutableInSandbox(absolutePath, scope);
      await fs.promises.access(target, fs.constants.R_OK | fs.constants.W_OK);
    },
  };
}

const sandboxWriteOperations = makeWriteOperations(undefined);

const piReadTool = createReadToolDefinition(sandboxRoot(), {
  operations: sandboxReadOperations,
});

/** 带作用域的写工具：多个 agent 并行写同一棵树时，各写各的文件。 */
function makeWriteTool(scope) {
  return createWriteToolDefinition(sandboxRoot(), {
    operations: makeWriteOperations(scope),
  });
}

const piWriteTool = makeWriteTool(undefined);

const piLsTool = createLsToolDefinition(sandboxRoot());
const piGrepTool = createGrepToolDefinition(sandboxRoot());
const piFindTool = createFindToolDefinition(sandboxRoot());

/**
 * pi 自带的 edit 工具：按原文精确替换（一处找不到会试模糊匹配），
 * 一次调用可以改多处。
 */
function makeEditTool(scope) {
  const operations = makeWriteOperations(scope);
  return createEditToolDefinition(sandboxRoot(), {
    operations: {
      readFile: async (absolutePath) => {
        const target = resolveMutableInSandbox(absolutePath, scope);
        const buf = await fs.promises.readFile(target);
        if (buf.byteLength > MAX_WRITE_BYTES) {
          throw new Error(
            `文件太大（${buf.byteLength} 字节，上限 ${MAX_WRITE_BYTES} 字节），改不动: ${absolutePath}`,
          );
        }
        return buf;
      },
      writeFile: operations.writeFile,
      access: operations.access,
    },
  });
}

const piEditTool = makeEditTool(undefined);

/**
 * 在 CPU 上跑候选树 tests/ 里的单元测试。
 *
 * 只给 agent 自测用 —— 准入由框架用四套独立验证套件判，这个结果不算证据。
 * 命令行是写死的：解释器、`-m pytest`、测试目录都由代码决定，模型只能指定跑哪棵树。
 */
async function runTests({ tree, timeoutSeconds = 600 }) {
  // stage2TreePath 已经给出绝对路径，别再 join 一次（会把绝对路径拼成两级）。
  const treeDir = stage2TreePath(tree);
  const target = relativeToRoot(treeDir);
  const testsDir = path.join(treeDir, 'tests');
  if (!fs.existsSync(testsDir)) {
    return {
      ok: false,
      tree: target,
      error: `这棵树里没有 tests/ 目录：${target}/tests`,
    };
  }
  const started = Date.now();
  try {
    const { stdout, stderr } = await execFileAsync(
      STAGE2_PYTHON,
      ['-m', 'pytest', '-q', testsDir],
      {
        cwd: treeDir,
        timeout: Math.min(1800, Math.max(1, timeoutSeconds)) * 1000,
        maxBuffer: 4 * 1024 * 1024,
        env: { ...process.env, CUDA_VISIBLE_DEVICES: '', PYTHONHASHSEED: '0' },
      },
    );
    return {
      ok: true,
      tree: target,
      exitCode: 0,
      durationMs: Date.now() - started,
      output: tailLines(stdout, 40),
      stderr: stderr.trim() === '' ? '' : tailLines(stderr, 10),
    };
  } catch (err) {
    const stdout = (err && err.stdout) || '';
    const stderr = (err && err.stderr) || '';
    return {
      ok: false,
      tree: target,
      exitCode: typeof err.code === 'number' ? err.code : null,
      durationMs: Date.now() - started,
      error: (err && err.message) || String(err),
      output: tailLines(stdout, 40),
      stderr: tailLines(stderr, 10),
    };
  }
}

/** 单个文件的 unified diff（缺口时返回空串，不抛）。 */
async function unifiedDiff(baseFile, treeFile) {
  try {
    const { stdout } = await execFileAsync('diff', ['-u', baseFile, treeFile], {
      maxBuffer: 4 * 1024 * 1024,
    });
    return stdout;
  } catch (err) {
    if (err && err.code === 1) return err.stdout || '';
    throw err;
  }
}

/**
 * 两棵树里某个文件（或全部改动文件）的 diff。
 *
 * 审"改了哪里"不需要 71 KB 的整文件：+860/−10 的 diff 就是审查面。
 * 只回 diff 行，正文由模型按需再 read。
 */
async function readDiff({ baseline = 'stage2_RQ-VAE/TIGER_RQ-VAE', tree, file, maxLines = 200 }) {
  const baseDir = stage2TreePath(baseline);
  const treeDir = stage2TreePath(tree);
  if (!fs.existsSync(baseDir)) return { ok: false, error: `基线不存在: ${baseline}` };
  if (!fs.existsSync(treeDir)) return { ok: false, error: `候选树不存在: ${tree}` };
  const limit = Math.max(1, Math.min(2000, maxLines));
  const rels = [];
  if (file !== undefined && file !== '') {
    rels.push(file);
  } else {
    const baseFiles = listTextFiles(baseDir);
    const treeFiles = listTextFiles(treeDir);
    for (const rel of [...new Set([...baseFiles.keys(), ...treeFiles.keys()])].sort()) {
      const a = baseFiles.get(rel);
      const b = treeFiles.get(rel);
      if (!a || !b) {
        rels.push(rel);
        continue;
      }
      if (!fs.readFileSync(a).equals(fs.readFileSync(b))) rels.push(rel);
    }
  }
  const out = [];
  for (const rel of rels.slice(0, 20)) {
    const a = path.join(baseDir, rel);
    const b = path.join(treeDir, rel);
    const missing = [];
    if (!fs.existsSync(a)) missing.push('baseline');
    if (!fs.existsSync(b)) missing.push('tree');
    if (missing.length > 0) {
      out.push({ file: rel, onlyIn: missing.length === 2 ? 'neither' : missing[0] === 'baseline' ? 'tree' : 'baseline' });
      continue;
    }
    const unified = await unifiedDiff(a, b);
    if (unified === '') continue;
    const lines = unified.split('\n');
    const { added, removed } = countDiffLines(unified);
    out.push({
      file: rel,
      added,
      removed,
      truncated: lines.length > limit,
      diff: lines.slice(0, limit).join('\n'),
    });
  }
  return {
    ok: true,
    baseline: relativeToRoot(baseDir),
    tree: relativeToRoot(treeDir),
    files: out,
    hint: 'diff 的 `-` 是基线的写法、`+` 是当前的写法；需要上下文再 read 该文件的相关区间。',
  };
}

/**
 * 一棵树里的文本文件（相对路径 → 绝对路径）。
 *
 * 和 grep/find 用同一套跳过规矩：缓存目录与二进制不进表，超过上限的文件也不进 ——
 * 比对的对象是源码，不是训练产物。
 */
function listTextFiles(rootDir) {
  const files = new Map();
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        if (SKIP_DIRS.has(entry.name)) continue;
        walk(path.join(dir, entry.name));
        continue;
      }
      if (!entry.isFile() || SKIP_FILE_RE.test(entry.name)) continue;
      const abs = path.join(dir, entry.name);
      let stat;
      try {
        stat = fs.statSync(abs);
      } catch {
        continue;
      }
      if (stat.size > MAX_WRITE_BYTES) continue;
      files.set(path.relative(rootDir, abs), abs);
    }
  };
  walk(rootDir);
  return files;
}

/** 一份 unified diff 里加了多少行、减了多少行（不算 ---/+++ 头）。 */
function countDiffLines(unified) {
  let added = 0;
  let removed = 0;
  for (const line of unified.split('\n')) {
    if (line.startsWith('+++') || line.startsWith('---')) continue;
    if (line.startsWith('+')) added += 1;
    else if (line.startsWith('-')) removed += 1;
  }
  return { added, removed };
}

/**
 * 两棵树的差异：逐字节相同的、改了的（带 diff 头几行）、只在一边的。
 *
 * 候选树是冻结基线的一份副本加上机制增量，所以"改了什么"本身就是核对项：
 * 判定只需要落在增量上，没变的文件不必重读。只读，两棵树都必须在仓库根之内。
 */
async function treeDiff({ baseline = 'stage2_RQ-VAE/TIGER_RQ-VAE', tree, maxHunkLines = 40, maxChangedFiles = 40 }) {
  const baseDir = stage2TreePath(baseline);
  const treeDir = stage2TreePath(tree);
  for (const [label, dir] of [['baseline', baseDir], ['tree', treeDir]]) {
    let stat;
    try {
      stat = fs.statSync(dir);
    } catch {
      return { ok: false, error: `${label} 路径不存在: ${dir}` };
    }
    if (!stat.isDirectory()) return { ok: false, error: `${label} 不是目录: ${dir}` };
  }
  const baseFiles = listTextFiles(baseDir);
  const treeFiles = listTextFiles(treeDir);
  const names = [...new Set([...baseFiles.keys(), ...treeFiles.keys()])].sort();
  const identical = [];
  const onlyInBaseline = [];
  const onlyInTree = [];
  const changed = [];
  for (const rel of names) {
    const a = baseFiles.get(rel);
    const b = treeFiles.get(rel);
    if (a && !b) {
      onlyInBaseline.push(rel);
      continue;
    }
    if (b && !a) {
      onlyInTree.push(rel);
      continue;
    }
    if (fs.readFileSync(a).equals(fs.readFileSync(b))) {
      identical.push(rel);
      continue;
    }
    let unified = '';
    try {
      const { stdout } = await execFileAsync('diff', ['-u', a, b], {
        maxBuffer: 4 * 1024 * 1024,
      });
      unified = stdout;
    } catch (err) {
      // diff 的退出码 1 就是"有差异"，不是失败。
      if (err && err.code === 1) unified = err.stdout || '';
      else throw err;
    }
    const { added, removed } = countDiffLines(unified);
    changed.push({
      file: rel,
      added,
      removed,
      hunks: unified.split('\n').slice(2, 2 + Math.max(1, maxHunkLines)).join('\n'),
    });
  }
  return {
    ok: true,
    baseline: relativeToRoot(baseDir),
    tree: relativeToRoot(treeDir),
    summary: {
      identical: identical.length,
      changed: changed.length,
      onlyInTree: onlyInTree.length,
      onlyInBaseline: onlyInBaseline.length,
    },
    identical,
    changed: changed.slice(0, Math.max(1, maxChangedFiles)),
    changedTruncated: changed.length > maxChangedFiles,
    onlyInTree,
    onlyInBaseline,
  };
}

// ---------------------------------------------------------------------------
// specCheck：能机器判的项，不让模型用眼睛判
// ---------------------------------------------------------------------------

/** 读 idea：按 arm 名在 ideas 下逐个目录找，取 4_method 的 new_files / modified_files。 */
function findIdeaForArm(arm) {
  const ideasDir = path.join(sandboxRoot(), 'ideas');
  if (!fs.existsSync(ideasDir) || arm === '') return null;
  for (const dir of fs.readdirSync(ideasDir, { withFileTypes: true })) {
    if (!dir.isDirectory()) continue;
    const dirPath = path.join(ideasDir, dir.name);
    for (const entry of fs.readdirSync(dirPath, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.endsWith('.json') || entry.name === 'math.json') continue;
      const file = path.join(dirPath, entry.name);
      let parsed;
      try {
        parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
      } catch {
        continue;
      }
      if (parsed && parsed.arm === arm) {
        return {
          file: relativeToRoot(file),
          dir: relativeToRoot(dirPath),
          idea: parsed,
          math: readJsonIfExists(path.join(dirPath, 'math.json')),
        };
      }
    }
  }
  return null;
}

function readJsonIfExists(file) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return null;
  }
}

/** 把嵌套 JSON 拍平成 `a.b.c` → 值，用来逐项比冻结配置。 */
function flattenJson(value, prefix = '', out = new Map()) {
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    for (const [key, item] of Object.entries(value)) {
      flattenJson(item, prefix === '' ? key : `${prefix}.${key}`, out);
    }
    return out;
  }
  out.set(prefix, Array.isArray(value) ? JSON.stringify(value) : String(value));
  return out;
}

/** 模块级 `NAME = 值` 常量（只认全大写名字），用来比两棵树里的冻结超参。 */
function pyConstants(text) {
  const out = new Map();
  for (const line of text.split('\n')) {
    const match = line.match(/^([A-Z][A-Z0-9_]{2,})\s*=\s*(.+?)\s*(#.*)?$/);
    if (match) out.set(match[1], match[2].trim());
  }
  return out;
}

/** `文件:行` 形式的命中清单，带条数上限。 */
function grepFiles(files, pattern, limit = 20) {
  const hits = [];
  let total = 0;
  for (const [rel, abs] of files) {
    const text = fs.readFileSync(abs, 'utf8');
    const lines = text.split('\n');
    for (let i = 0; i < lines.length; i += 1) {
      if (!pattern.test(lines[i])) continue;
      total += 1;
      if (hits.length < limit) {
        hits.push({ where: `${rel}:${i + 1}`, text: lines[i].trim().slice(0, 160) });
      }
    }
  }
  return { total, hits };
}

/**
 * 机械预检：只算能确定的事，不做判断。
 *
 * 文件的齐缺、路径违规、冻结超参有没有动、规格里的 metrics 名字有没有落地、
 * 测试有没有空断言 —— 这些不需要模型，工具一次算完；模型只判公式语义与开关退化。
 */
async function specCheck({ tree, idea, baseline = 'stage2_RQ-VAE/TIGER_RQ-VAE' }) {
  const treeDir = stage2TreePath(tree);
  const baseDir = stage2TreePath(baseline);
  if (!fs.existsSync(treeDir) || !fs.statSync(treeDir).isDirectory()) {
    return { ok: false, error: `候选树不存在: ${tree}` };
  }
  const arm = path.basename(treeDir);
  const ideaInfo = idea !== undefined && idea !== ''
    ? (() => {
        const dir = stage2TreePath(idea);
        const ideaFile = fs.readdirSync(dir)
          .filter((name) => name.endsWith('.json') && name !== 'math.json')
          .map((name) => path.join(dir, name))[0];
        return {
          file: relativeToRoot(ideaFile),
          dir: relativeToRoot(dir),
          idea: readJsonIfExists(ideaFile),
          math: readJsonIfExists(path.join(dir, 'math.json')),
        };
      })()
    : findIdeaForArm(arm);
  const method = (ideaInfo && ideaInfo.idea && ideaInfo.idea.sections && ideaInfo.idea.sections['4_method']) || {};
  const treeRel = relativeToRoot(treeDir);
  const treeFiles = listTextFiles(treeDir);
  const baseFiles = fs.existsSync(baseDir) ? listTextFiles(baseDir) : new Map();

  // a) 声明的文件齐不齐；声明的"改过"是不是真的改了。
  const declaredNew = (method.new_files || []).map((entry) => String(entry.path || ''));
  const declaredModified = (method.modified_files || []).map((entry) => String(entry.path || ''));
  const relOf = (declared) => {
    const norm = declared.replace(/^\.\//, '');
    const marker = `${treeRel}/`;
    const cut = norm.indexOf(marker);
    return cut >= 0 ? norm.slice(cut + marker.length) : norm;
  };
  const newFiles = declaredNew.map((declared) => {
    const rel = relOf(declared);
    const present = treeFiles.has(rel);
    return { file: rel, present, inTree: declared.startsWith(treeRel) };
  });
  const modifiedFiles = declaredModified.map((declared) => {
    const rel = relOf(declared);
    const inTree = treeFiles.get(rel);
    const inBase = baseFiles.get(rel);
    let changed = null;
    if (inTree && inBase) changed = !fs.readFileSync(inTree).equals(fs.readFileSync(inBase));
    return { file: rel, present: !!inTree, differsFromBaseline: changed, inTree: declared.startsWith(treeRel) };
  });

  // e) 路径与身份：绝对路径、parents[N]、ARM_NAME 的取值、跨臂引用。
  // 候选树必须自包含（只依赖冻结基线 + 本树）：引用别的臂就是外部依赖，树不完整。
  const ownArm = (ideaInfo && ideaInfo.idea && typeof ideaInfo.idea.arm === 'string'
    && ideaInfo.idea.arm.trim() !== '')
    ? ideaInfo.idea.arm.trim()
    : path.basename(treeDir);
  const crossArmHits = [];
  for (const [rel, abs] of treeFiles) {
    const lines = fs.readFileSync(abs, 'utf8').split('\n');
    for (let i = 0; i < lines.length; i += 1) {
      for (const match of lines[i].matchAll(/stage2_RQ-VAE\/(TIGER_[A-Za-z0-9_-]+)\//g)) {
        const arm = match[1];
        if (arm === ownArm || arm === 'TIGER_RQ-VAE') continue;
        if (crossArmHits.length < 20) {
          crossArmHits.push({ where: `${rel}:${i + 1}`, arm, text: lines[i].trim().slice(0, 160) });
        }
      }
    }
  }
  const absolute = grepFiles(treeFiles, /(^|[^A-Za-z0-9_.])\/(home|fs[0-9]+|tmp|Users|scratch|var\/tmp)\//);
  const parents = grepFiles(treeFiles, /parents\[\s*\d+\s*\]/);
  const armNames = [];
  const armPattern = /^\s*ARM_NAME\s*=\s*(['"])(.*?)\1/gm;
  for (const [rel, abs] of treeFiles) {
    const text = fs.readFileSync(abs, 'utf8');
    for (const match of text.matchAll(armPattern)) armNames.push({ where: rel, value: match[2] });
  }

  // f) 冻结超参：JSON 逐项比，Python 常量逐名字比。
  const frozen = { json: null, constants: null };
  const baseConfigJson = path.join(baseDir, 'config', 'train_config.json');
  const treeConfigJson = path.join(treeDir, 'config', 'train_config.json');
  if (fs.existsSync(baseConfigJson) && fs.existsSync(treeConfigJson)) {
    const a = flattenJson(readJsonIfExists(baseConfigJson) || {});
    const b = flattenJson(readJsonIfExists(treeConfigJson) || {});
    const changed = [];
    for (const [key, value] of a) {
      if (!b.has(key)) changed.push({ key, was: value, now: '(missing)' });
      else if (b.get(key) !== value) changed.push({ key, was: value, now: b.get(key) });
    }
    for (const [key, value] of b) if (!a.has(key)) changed.push({ key, was: '(missing)', now: value });
    frozen.json = { file: 'config/train_config.json', checked: a.size, changed };
  }
  const basePy = path.join(baseDir, 'train_config.py');
  const treePy = path.join(treeDir, 'train_config.py');
  if (fs.existsSync(basePy) && fs.existsSync(treePy)) {
    const a = pyConstants(fs.readFileSync(basePy, 'utf8'));
    const b = pyConstants(fs.readFileSync(treePy, 'utf8'));
    const changed = [];
    for (const [key, value] of a) {
      if (!b.has(key)) changed.push({ key, was: value, now: '(missing)' });
      else if (b.get(key) !== value) changed.push({ key, was: value, now: b.get(key) });
    }
    // 只在候选里出现的常量单独列：多半是机制自己的常量，要看是不是该冻结的。
    const extra = [...b.keys()].filter((key) => !a.has(key));
    frozen.constants = { file: 'train_config.py', checked: a.size, changed, added: extra };
  }

  // d) 分配规则：只给调用点，判读仍归模型。
  const sinkhorn = grepFiles(treeFiles, /sinkhorn\s*\(/);
  const argmin = grepFiles(treeFiles, /argmin\s*\(|\.argmin\s*\(|torch\.argmin/);

  // g) 测试：有没有空断言、有没有断言数。
  const tests = [];
  for (const [rel, abs] of treeFiles) {
    if (!/(^|\/)tests?\/.*\.py$/.test(rel)) continue;
    const text = fs.readFileSync(abs, 'utf8');
    const fn = (text.match(/^\s*def test_/gm) || []).length;
    const asserts = (text.match(/^\s*assert\b/gm) || []).length;
    const vacuous = (text.match(/^\s*assert\s+True\s*$/gm) || []).length;
    if (fn === 0) continue;
    tests.push({ file: rel, tests: fn, asserts, vacuous, suspicious: asserts === 0 || vacuous > 0 });
  }

  // b') 规格里的 metrics 名字有没有在树里落地。
  const metricNames = ideaInfo && ideaInfo.math && ideaInfo.math.metrics
    ? Object.keys(ideaInfo.math.metrics)
    : [];
  const missingMetrics = [];
  const escapeRe = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  for (const name of metricNames) {
    if (grepFiles(treeFiles, new RegExp(`\\b${escapeRe(name)}\\b`), 1).total === 0) missingMetrics.push(name);
  }

  // 规格声明的常量与公式名，落在哪个文件哪一行 —— 省掉 step 1 的逐名 grep。
  const declareLocations = (entries) => entries.map((entry) => {
    const name = String(entry && entry.name ? entry.name : '').trim();
    if (name === '') return null;
    const found = grepFiles(treeFiles, new RegExp(`\\b${escapeRe(name)}\\b`), 2);
    return {
      name,
      hits: found.total,
      where: found.hits[0] ? found.hits[0].where : null,
      text: found.hits[0] ? found.hits[0].text : null,
    };
  }).filter(Boolean);
  const math = (ideaInfo && ideaInfo.math) || {};
  const declarations = {
    constants: declareLocations(Array.isArray(math.constants) ? math.constants : []),
    formulas: declareLocations(Array.isArray(math.formulas) ? math.formulas : []),
  };

  return {
    ok: true,
    arm,
    tree: treeRel,
    baseline: relativeToRoot(baseDir),
    idea: ideaInfo ? { file: ideaInfo.file, dir: ideaInfo.dir } : null,
    files: {
      newFiles: newFiles.filter((entry) => !entry.present || !entry.inTree),
      newFilesChecked: newFiles.length,
      modifiedFiles: modifiedFiles.filter((entry) => !entry.present || !entry.inTree || entry.differsFromBaseline === false),
      modifiedFilesChecked: modifiedFiles.length,
    },
    paths: {
      absolute: { total: absolute.total, hits: absolute.hits },
      parentsN: { total: parents.total, hits: parents.hits },
      armNames,
      crossArm: { total: crossArmHits.length, hits: crossArmHits },
    },
    frozen,
    assignment: { sinkhornCallsites: sinkhorn.total, argminCallsites: argmin.total, argminHits: argmin.hits },
    metrics: { declared: metricNames.length, missing: missingMetrics },
    declarations,
    tests: { files: tests.length, suspicious: tests.filter((entry) => entry.suspicious) },
    hints: [
      'files.* 里剩下的条目就是问题：missing 的没写，differsFromBaseline=false 的声明改了却没改。',
      'paths.absolute / paths.parentsN 的 total 必须是 0；armNames 只允许一个值且等于候选树目录名。',
      'paths.crossArm 必须为空：候选树只能依赖冻结基线与本树，引用别的臂就是外部依赖。',
      'frozen.*.changed 必须为空；frozen.constants.added 是机制自己的常量，属正常。',
      'metrics.missing 是规格要求但树里没有的读数名。',
      'declarations.constants/formulas 是规格声明的名字在树里的位置（hits=0 表示没找到，需要你判断是命名不同还是真缺）。',
      'tests.suspicious 是空断言或无断言的测试文件。',
      '公式语义与开关是否真退化，工具判不了 —— 这两类仍要你读代码。',
    ],
  };
}

/** 取最后 n 行：测试输出前面一大截是噪声，结论在尾巴上。 */
function tailLines(text, n) {
  const lines = String(text || '').trimEnd().split('\n');
  return lines.slice(Math.max(0, lines.length - n)).join('\n');
}

export const PI_TOOLS = {
  read: piReadTool,
  write: piWriteTool,
  edit: piEditTool,
  ls: piLsTool,
  grep: piGrepTool,
  find: piFindTool,
  checkJson: checkJsonTool,
  checkIdea: checkIdeaTool,
  webSearch: webSearchTool,
  stage2Start: stage2StartTool,
  stage2Status: stage2StatusTool,
  stage2Stop: stage2StopTool,
  listIssues: listIssuesTool,
  readIssue: readIssueTool,
  gitLog: gitLogTool,
  runTests: runTestsTool,
  treeDiff: treeDiffTool,
  specCheck: specCheckTool,
  readDiff: readDiffTool,
  copyBaselineTree: copyBaselineTreeTool,
  deleteFile: deleteFileTool,
};

/**
 * 按权限裁出这一轮真正能用的工具。
 *
 * 没授的工具压根不进 pi 的工具表 —— 模型看不见它，也调不到它。
 * 这是权限的硬约束：提示词里写"你没写权限"只是补充说明，真正管用的是这张表。
 */
/** 建树工具也受作用域约束：目标树必须落在允许的路径里。 */
function makeCopyBaselineTool(scope) {
  return piTool({
    name: 'copyBaselineTree',
    label: '建候选树',
    description: 'Copy one frozen baseline tree into a new candidate tree under stage2_RQ-VAE (never overwrites).',
    parameters: Type.Object({
      arm: Type.String({ description: '新候选树的名字，例如 TIGER_xxx_rq' }),
      from: Type.Optional(Type.String({ description: '冻结基线路径，默认 stage2_RQ-VAE/TIGER_RQ-VAE' })),
    }),
    impl: (args) => {
      const arm = typeof args.arm === 'string' ? args.arm.trim() : '';
      const target = `stage2_RQ-VAE/${arm}`;
      if (!scopeAllows(scope, target, { isDir: true })) {
        throw new Error(`建树越界：${target} 不在这个 agent 的文件范围里`);
      }
      return TOOL_IMPLS.copyBaselineTree(args);
    },
  });
}

const SCOPED_TOOL_NAMES = new Set(['write', 'edit', 'deleteFile', 'copyBaselineTree']);
const scopedToolCache = new Map();

/** 某个写作用域下的写工具集（read 工具不需要作用域，它们本来就只读）。 */
function scopedTools(scope) {
  const key = Array.isArray(scope) && scope.length > 0 ? scope.join('|') : '';
  if (key === '') return null;
  if (!scopedToolCache.has(key)) {
    scopedToolCache.set(key, {
      write: makeWriteTool(scope),
      edit: makeEditTool(scope),
      deleteFile: makeDeleteTool(scope),
      copyBaselineTree: makeCopyBaselineTool(scope),
    });
  }
  return scopedToolCache.get(key);
}

export function toolsForPermissions(permissionIds, options = {}) {
  const ids = normalizePermissions(permissionIds);
  const scoped = scopedTools(options.writeScope);
  const denied = new Set(Array.isArray(options.denyTools) ? options.denyTools : []);
  const picked = new Map();
  for (const id of ids) {
    const item = PERMISSION_CATALOG.find((entry) => entry.id === id);
    for (const name of item.tools) {
      if (denied.has(name)) continue;
      const tool = (scoped && SCOPED_TOOL_NAMES.has(name) ? scoped[name] : PI_TOOLS[name]);
      if (!tool) {
        throw new Error(`权限 ${id} 指向了不存在的工具 ${name}`);
      }
      picked.set(name, tool);
    }
  }
  return [...picked.values()];
}

// ---------------------------------------------------------------------------
// pi 运行时：模型目录 + 凭证 + 会话
// ---------------------------------------------------------------------------

/** pi 的派生配置挂在数据目录下，测试换目录时跟着走，不碰用户自己的 pi 配置。 */
function piStateDir() {
  return path.join(dataDir(), 'pi');
}

/**
 * 把 config.js 里的模型常量写成 pi 认识的 models.json。
 *
 * 内容变了才写，避免每次运行都改文件时间戳。这个文件是派生物，不是配置源 ——
 * 要换模型改 config.js，不要改它。
 */
export function ensurePiModelsFile() {
  const dir = piStateDir();
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, 'models.json');
  const config = {
    providers: {
      [PROVIDER_NAME]: {
        baseUrl: BASE_URL,
        api: 'openai-completions',
        models: [
          {
            id: MODEL_ID,
            name: `${MODEL_ID}（${PROVIDER_NAME}）`,
            reasoning: false,
            input: ['text'],
            contextWindow: CONTEXT_WINDOW,
            // 不写 maxTokens：这个字段会作为 max_tokens 发给 provider，由它按模型
            // 真实上限截断。我们以前手写 8192，结果响应正好在 8192 处被切
            // （finishReason=length），把"我们自己设的栏杆"当成了模型的限制。
            // 不设它，就只剩模型自己的上限与上下文窗口。
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          },
        ],
      },
    },
  };
  const text = `${JSON.stringify(config, null, 2)}\n`;
  let existing = null;
  try {
    existing = fs.readFileSync(file, 'utf8');
  } catch {
    existing = null;
  }
  if (existing !== text) fs.writeFileSync(file, text, 'utf8');
  return file;
}

let cachedRuntime = null;
let cachedRuntimeDir = null;

/**
 * 建 pi 的模型/凭证运行时。
 *
 * API key 只留在内存里（setRuntimeApiKey 不落盘）：凭证文件那个位置只是给 pi
 * 一个合法的读路径，里面没有我们的 key。
 */
export async function createPiRuntime() {
  const modelsPath = ensurePiModelsFile();
  const authPath = path.join(piStateDir(), 'auth.json');
  const modelRuntime = await ModelRuntime.create({
    authPath,
    modelsPath,
    allowModelNetwork: false,
  });
  await modelRuntime.setRuntimeApiKey(PROVIDER_NAME, loadApiKey());
  const model = modelRuntime.getModel(PROVIDER_NAME, MODEL_ID);
  if (!model) {
    throw new Error(`pi 的模型目录里没有 ${PROVIDER_NAME}/${MODEL_ID}（看 ${modelsPath}）`);
  }
  return { modelRuntime, model, modelsPath, authPath };
}

/** 同一个进程里复用运行时；换了数据目录就重建。 */
async function piRuntime() {
  const dir = piStateDir();
  if (cachedRuntime && cachedRuntimeDir === dir) return cachedRuntime;
  cachedRuntime = await createPiRuntime();
  cachedRuntimeDir = dir;
  return cachedRuntime;
}

/**
 * 这次运行实际要用人设提示词。
 *
 * instructions === undefined：请求里没这个字段 → 用这个 agent 自己的默认提示词。
 * instructions 是字符串：请求里带了 → 就用这一份，空串表示"明确要没有系统提示词"。
 */
export function resolveInstructions(profile, instructions) {
  if (instructions === undefined) return profile.instructions;
  return instructions.trim();
}

/**
 * 这一轮追加在 pi 默认提示词后面的内容：人设提示词 + 权限说明。
 *
 * 权限说明总是加 —— 用户把提示词框清空，是"不要人设"，不是"连运行时约束也不要了"。
 * 工具裁剪本来就是硬约束，提示词里那几句只是配套说明。
 */
export function buildSystemPromptParts(profile, instructions, permissionIds) {
  const systemPrompt = resolveInstructions(profile, instructions).trim();
  const permissionText = buildPermissionPrompt(permissionIds);
  const parts = systemPrompt === '' ? [] : [systemPrompt];
  // 语言约束与权限说明一样，每次运行都挂上：用户清空提示词是"不要人设"，
  // 不是"用英文思考"。
  return [...parts, permissionText, LANGUAGE_RULE, IDEA_RULE, ROUND_END_RULE];
}

/**
 * 按「身份 + 提示词 + 权限」建一个 pi 会话。
 *
 * 内置工具全关（noTools: 'builtin'）：pi 的 bash / edit / write 没有仓库边界，
 * 挂到公网入口上等于给任何拿到链接的人一个 shell。这一轮的工具有哪些，只看权限表。
 */
export async function createPiSession({ profile, instructions, permissionIds }) {
  const { modelRuntime, model } = await piRuntime();
  // 轮次控制工具始终可用：没有它，"本步判定本轮结束"就只能写在文字里，
  // 客户端读不到，后面的步骤照样会发出去。
  const tools = [
    ...toolsForPermissions(permissionIds, {
      writeScope: profile && profile.writeScope,
      denyTools: profile && profile.denyTools,
    }),
    endRoundTool,
    requestHandoffTool,
  ];
  const loader = new DefaultResourceLoader({
    cwd: sandboxRoot(),
    agentDir: path.join(piStateDir(), 'agent'),
    noExtensions: true,
    noSkills: true,
    noPromptTemplates: true,
    noThemes: true,
    noContextFiles: true,
    appendSystemPrompt: buildSystemPromptParts(profile, instructions, permissionIds),
  });
  await loader.reload();

  const { session } = await createAgentSession({
    cwd: sandboxRoot(),
    modelRuntime,
    model,
    noTools: 'builtin',
    tools: tools.map((tool) => tool.name),
    customTools: tools,
    resourceLoader: loader,
    sessionManager: SessionManager.inMemory(sandboxRoot()),
    thinkingLevel: 'off',
  });
  return session;
}

// ---------------------------------------------------------------------------
// pi 事件流 → UI message stream 分片
// ---------------------------------------------------------------------------

/** pi 的 stopReason → 页面认识的 finishReason。 */
const FINISH_REASONS = {
  stop: 'stop',
  toolUse: 'tool-calls',
  length: 'length',
  error: 'error',
  aborted: 'aborted',
};

function textOf(result) {
  const content = result && Array.isArray(result.content) ? result.content : [];
  return content
    .filter((part) => part && part.type === 'text' && typeof part.text === 'string')
    .map((part) => part.text)
    .join('\n');
}

/**
 * 跑一个 pi 会话，把事件流翻成页面认识的 UI message stream 分片。
 *
 * session 只要求实现 pi AgentSession 的那几个方法（subscribe / prompt / abort / dispose），
 * 测试因此可以拿一个假会话把整条翻译路径跑一遍，不用打真实模型。
 *
 * 收尾契约：正常结束是 finish + data-usage；被 abort 是 abort；出错是 error。
 * 三者都只出一个，不会既有 error 又有 finish 让人分不清这次到底成没成。
 */
export async function* translateSession(
  session,
  { prompt, signal, messageId, dispose = true },
) {
  const queue = [];
  let wake = null;
  let settled = false;
  let runError = null;

  const notify = () => {
    if (wake) {
      const resume = wake;
      wake = null;
      resume();
    }
  };
  const unsubscribe = session.subscribe((event) => {
    if (event && typeof event.type === 'string') {
      queue.push(event);
      notify();
    }
  });

  const run = session.prompt(prompt).then(
    () => {
      settled = true;
      notify();
    },
    (err) => {
      runError = err;
      settled = true;
      notify();
    },
  );

  const onAbort = () => {
    void Promise.resolve(session.abort()).catch(() => {});
  };
  if (signal) signal.addEventListener('abort', onAbort, { once: true });

  const usage = { inputTokens: 0, outputTokens: 0, reasoningTokens: 0, totalTokens: 0 };
  let turns = 0;
  let messageSeq = 0;
  let finishReason = 'stop';
  const toolCallsByContent = new Map();
  // 这一步有没有调用 endRound（本步判定"本轮到此结束"）。
  let roundEnded = false;

  try {
    yield { type: 'start', messageId };

    for (let index = 0; ; index += 1) {
      if (index >= queue.length) {
        if (settled && index >= queue.length) break;
        await new Promise((resolve) => {
          wake = resolve;
        });
        index -= 1;
        continue;
      }
      const event = queue[index];

      if (event.type === 'turn_start') {
        yield { type: 'start-step' };
        continue;
      }

      if (event.type === 'turn_end') {
        yield { type: 'finish-step' };
        continue;
      }

      if (event.type === 'message_start') {
        if (event.message && event.message.role === 'assistant') messageSeq += 1;
        continue;
      }

      if (event.type === 'message_end') {
        const message = event.message || {};
        if (message.role !== 'assistant') continue;
        turns += 1;
        if (message.usage) {
          usage.inputTokens += message.usage.input ?? 0;
          usage.outputTokens += message.usage.output ?? 0;
          usage.reasoningTokens += message.usage.reasoning ?? 0;
          usage.totalTokens += message.usage.totalTokens ?? 0;
        }
        finishReason = FINISH_REASONS[message.stopReason] ?? 'stop';
        continue;
      }

      if (event.type === 'message_update') {
        const delta = event.assistantMessageEvent;
        if (!delta) continue;
        const contentIndex = delta.contentIndex ?? 0;
        const textId = `${messageSeq}:text:${contentIndex}`;
        const reasoningId = `${messageSeq}:reasoning:${contentIndex}`;
        const contentKey = `${messageSeq}:${contentIndex}`;
        const toolCallId = toolCallsByContent.get(contentKey);

        switch (delta.type) {
          case 'text_start':
            yield { type: 'text-start', id: textId };
            break;
          case 'text_delta':
            yield { type: 'text-delta', id: textId, delta: delta.delta ?? '' };
            break;
          case 'text_end':
            yield { type: 'text-end', id: textId };
            break;
          case 'thinking_start':
            yield { type: 'reasoning-start', id: reasoningId };
            break;
          case 'thinking_delta':
            yield { type: 'reasoning-delta', id: reasoningId, delta: delta.delta ?? '' };
            break;
          case 'thinking_end':
            yield { type: 'reasoning-end', id: reasoningId };
            break;
          case 'toolcall_start': {
            const content =
              delta.partial && Array.isArray(delta.partial.content)
                ? delta.partial.content[contentIndex]
                : null;
            if (content) {
              toolCallsByContent.set(contentKey, content.id);
              yield {
                type: 'tool-input-start',
                toolCallId: content.id,
                toolName: content.name,
              };
            }
            break;
          }
          case 'toolcall_delta':
            yield {
              type: 'tool-input-delta',
              toolCallId: toolCallId ?? contentKey,
              inputTextDelta: delta.delta ?? '',
            };
            break;
          case 'toolcall_end': {
            const call = delta.toolCall || {};
            toolCallsByContent.set(contentKey, call.id);
            yield {
              type: 'tool-input-available',
              toolCallId: call.id ?? toolCallId,
              toolName: call.name,
              input: call.arguments ?? {},
            };
            break;
          }
          default:
            break;
        }
        continue;
      }

      if (event.type === 'tool_execution_end') {
        // 结束本轮是轮次控件，不是仓库能力：调用成功就把"本轮结束"记下来，
        // 让外层释放这一轮的会话（后面的步骤不会再来了）。
        if (event.toolName === ROUND_END_TOOL && !event.isError) roundEnded = true;
        if (event.isError) {
          yield {
            type: 'tool-output-error',
            toolCallId: event.toolCallId,
            errorText: textOf(event.result) || '工具执行失败（没有给出原因）',
          };
        } else {
          const details = event.result ? event.result.details : undefined;
          yield {
            type: 'tool-output-available',
            toolCallId: event.toolCallId,
            output: details === undefined ? textOf(event.result) : details,
          };
        }
      }
    }

    // 客户端已经断开时不要再报"这次运行出错"：断线是这轮结束的原因，
    // 不是 agent 的失败。abort 分片优先于 abort 过程中 prompt 抛出的错误。
    if (signal && signal.aborted) {
      yield { type: 'abort', reason: 'aborted' };
      return { status: 'aborted' };
    }
    if (runError) {
      yield {
        type: 'error',
        errorText: runError && runError.message ? runError.message : String(runError),
      };
      return 'error';
    }

    yield { type: 'finish', finishReason };
    yield {
      type: 'data-usage',
      data: {
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        reasoningTokens: usage.reasoningTokens,
        totalTokens: usage.totalTokens,
        steps: turns,
      },
    };
    return { status: 'done', roundEnded };
  } finally {
    if (signal) signal.removeEventListener('abort', onAbort);
    unsubscribe();
    if (dispose) {
      try {
        session.dispose();
      } catch {
        // 释放失败不该盖住这一轮的真实结论。
      }
    }
    await run.catch(() => {});
  }
}

// ---------------------------------------------------------------------------
// 多步轮次的 pi 会话：一轮 = 一个会话，每一步只发自己的指令
// ---------------------------------------------------------------------------

/**
 * 轮次会话表：key = `${roundId}:${profileId}`。
 *
 * pi 的会话本身就带着前面的对话，所以第二步不需要把第一步的产出重新贴一遍 ——
 * 客户端只发这一步的指令，上下文由会话带着。这也是切到 pi 之后最该利用的东西。
 */
const roundSessions = new Map();

/**
 * 本轮输入（roundPrompt）只由第一步发一次。
 *
 * 会话丢失时还要靠它把问题补回去，所以第一步到达时先记在服务端；
 * 后面几步不必每步重发同一段话，前端也就不必把它塞进每个请求体。
 */
const roundPrompts = new Map();

function rememberRoundPrompt(round) {
  if (!round || typeof round.roundId !== 'string' || round.roundId === '') return;
  const text = typeof round.roundPrompt === 'string' ? round.roundPrompt.trim() : '';
  if (text === '') return;
  roundPrompts.set(round.roundId, { text, lastUsedAt: Date.now() });
}

function roundPromptFor(roundId) {
  const entry = roundPrompts.get(roundId);
  if (!entry) return '';
  entry.lastUsedAt = Date.now();
  return entry.text;
}

/** 会话的"身份"：提示词或权限变了，旧会话就不能再接着用（系统提示词是建会话时定死的）。 */
function sessionSignature(profile, instructions, permissionIds) {
  return JSON.stringify({
    profile: profile.id,
    instructions: resolveInstructions(profile, instructions),
    permissions: permissionIds,
  });
}

function safeDispose(session) {
  try {
    session.dispose();
  } catch {
    // 释放失败不该影响这一轮的结论。
  }
}

/** 释放某一轮的会话。没有就什么都不做。 */
export function disposeRoundSession(key) {
  const entry = roundSessions.get(key);
  if (!entry) return;
  roundSessions.delete(key);
  safeDispose(entry.session);
}

/**
 * 重置某个 agent 的会话：丢掉它挂着的轮次会话与本轮输入。
 *
 * pi 的会话本身记着前面的对话，所以"重置会话"要动的是这里：
 * 丢掉之后下一次运行从空白上下文开始。返回丢掉了几个会话。
 */
export function resetAgentSessions(profileId) {
  const id = typeof profileId === 'string' ? profileId.trim() : '';
  if (id === '') return 0;
  let disposed = 0;
  for (const [key, entry] of [...roundSessions]) {
    if (!key.endsWith(`:${id}`)) continue;
    roundSessions.delete(key);
    safeDispose(entry.session);
    // 本轮输入也跟着走：会话没了，那段话留着也没用。
    roundPrompts.delete(key.slice(0, key.length - id.length - 1));
    disposed += 1;
  }
  // 重置有没有真的发生，要看得见：以前这段代码在接收方身份解析之前，
  // 抛的错被 catch 吃掉，日志里只留下一句"来源认不出来"，重置从来没发生过。
  process.stderr.write(
    `[${new Date().toISOString()}] 重置会话：${id} 丢了 ${disposed} 个会话\n`,
  );
  return disposed;
}

/** 现在挂着的轮次会话有几个（测试与诊断用）。 */
export function roundSessionCount() {
  return roundSessions.size;
}

/**
 * 扫掉闲置太久的轮次会话。
 *
 * 用户跑到一半不跑了，那一轮的会话不会自己消失 —— 这里按最后使用时间兜底回收。
 */
export function sweepRoundSessions(now = Date.now()) {
  for (const [key, entry] of [...roundSessions]) {
    if (now - entry.lastUsedAt >= ROUND_SESSION_TTL_MS) disposeRoundSession(key);
  }
  // 本轮输入跟着会话一起过期：会话都没了，留着那段话也没有用。
  for (const [roundId, entry] of [...roundPrompts]) {
    if (now - entry.lastUsedAt >= ROUND_SESSION_TTL_MS) roundPrompts.delete(roundId);
  }
}

/**
 * 取这一轮这一步该用的 pi 会话。
 *
 * - 同一轮、同一身份 → 复用已有的会话（reused=true）；
 * - 没有会话（第一步、服务重启、过期回收）→ 新建；
 * - 身份变了 → 释放旧的、新建（系统提示词不能中途换）。
 *
 * 返回的 key 是释放时用的；createSession 只在测试里替换。
 */
export async function acquireRoundSession({
  round,
  profile,
  instructions,
  permissionIds,
  createSession = createPiSession,
}) {
  const args = { profile, instructions, permissionIds };
  if (!round || typeof round.roundId !== 'string' || round.roundId === '') {
    // 单次运行（不是轮次里的一步）：不登记，跑完就释放。
    return { session: await createSession(args), key: null, reused: false };
  }

  sweepRoundSessions();
  const key = `${round.roundId}:${profile.id}`;
  const signature = sessionSignature(profile, instructions, permissionIds);
  const existing = roundSessions.get(key);

  if (existing && existing.signature === signature && !existing.session.isStreaming) {
    existing.lastUsedAt = Date.now();
    return { session: existing.session, key, reused: true };
  }
  if (existing) disposeRoundSession(key);

  const session = await createSession(args);
  roundSessions.set(key, { session, signature, lastUsedAt: Date.now() });
  return { session, key, reused: false };
}

/**
 * 跑一次真实 agent，产出标准 UI message stream 分片。
 *
 * 有 handoff（上游 agent 的分析结果）时，它会被拼在问题前面一起发出去，
 * 下游 agent 就能接着上游的结论往下做。
 *
 * 多步轮次（round 给了 roundId / stepIndex / totalSteps）：
 *   - 同一轮的每一步共享一个 pi 会话，客户端只发这一步的指令；
 *   - 走完最后一步、出错、被中止 → 立刻释放会话；
 *   - 会话真的丢了（服务重启 / 过期回收）时，把本轮输入补在指令前面 ——
 *     至少让模型知道它在做什么，不指望它记得上一步。
 *
 * instructions 是字符串时用它当人设提示词（页面上可编辑的那份，空串表示不要）；
 * 没有这个字段时用该 agent 自己的默认提示词。
 */
export async function* runLiveChunks(
  prompt,
  {
    profileId,
    handoff,
    handoffFrom,
    instructions,
    permissions,
    signal,
    round,
    createSession = createPiSession,
  } = {},
) {
  const profile = resolveProfile(profileId);
  const permissionIds = resolvePermissions(profile, permissions);
  // 第一步带过来的本轮输入先记住：后面几步和"会话丢了"的兜底都要用它。
  rememberRoundPrompt(round);
  const { session, key, reused } = await acquireRoundSession({
    round,
    profile,
    instructions,
    permissionIds,
    createSession,
  });

  let message = buildHandoffPrompt(prompt, handoff, handoffFrom);
  const stepIndex = Number.isInteger(round && round.stepIndex) ? round.stepIndex : 0;
  const totalSteps = Number.isInteger(round && round.totalSteps) ? round.totalSteps : 0;
  const requestedPrompt =
    round && typeof round.roundPrompt === 'string' ? round.roundPrompt.trim() : '';
  // 请求里没带就看服务端记着的那份（第一步发过）。
  const roundPrompt =
    key === null ? requestedPrompt : requestedPrompt || roundPromptFor(round.roundId);
  if (!reused && key !== null && stepIndex > 0 && roundPrompt !== '') {
    // 会话是新造的，但这不是第一步：把本轮输入补回来（之前的步骤产出补不回来）。
    message = `【本轮输入】\n${roundPrompt}\n\n${message}`;
  }

  // 最后一步、或者调用方没说清楚一共几步（按单步处理）都要在结束时释放会话。
  const isLastStep = key !== null && (totalSteps <= 0 || stepIndex + 1 >= totalSteps);
  let status = 'done';
  let endedRound = false;
  try {
    const outcome = yield* translateSession(session, {
      prompt: message,
      signal,
      messageId: crypto.randomUUID(),
      dispose: false,
    });
    status = outcome.status;
    endedRound = outcome.roundEnded === true;
  } catch (err) {
    status = 'error';
    throw err;
  } finally {
    // 单次运行、整轮的最后一步、本步判定本轮结束（endRound）、出错、中止 ——
    // 都该在这里把会话清掉，别让一个不会再被用的会话挂到 TTL。
    if (key === null) {
      safeDispose(session);
    } else if (isLastStep || endedRound || status !== 'done') {
      disposeRoundSession(key);
    }
  }
}
