#!/usr/bin/env node
/**
 * 把 agent-vis 安装成一个 dsh profile。
 *
 * 生成的是 $DSH_HOME/profiles/agent-vis/ 下的三件东西：
 *   package.json        —— 只声明 bundle 列表（@deepseek-ai/dsh-base）
 *   cordis.yml          —— 空条目列表（profile 根）
 *   cordis.patch.yml    —— 模型路由 + 关掉宿主工具 + 挂载 agent-vis 插件
 *
 * 插件路径在安装时写成绝对 file URL：profile 里的相对路径会相对于
 * profile 目录解析，而 agent-vis 源码不在那里。
 *
 * 用法：
 *   node dsh/install.mjs [--home /path/to/dsh-home]
 * 默认 home 是 agent-vis/.data/dsh-home（已被 .gitignore 忽略）。
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { BASE_URL, CONTEXT_WINDOW, MODEL_ID, PROVIDER_NAME } from '../config.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');

/** 解析 --home，默认放在仓库自己的数据目录下。 */
function parseHome(argv) {
  const index = argv.indexOf('--home');
  if (index >= 0) {
    const value = argv[index + 1];
    if (!value) throw new Error('--home 需要一个目录参数');
    return path.resolve(value);
  }
  return path.join(ROOT, '.data', 'dsh-home');
}

function main() {
  const home = parseHome(process.argv.slice(2));
  const profileDir = path.join(home, 'profiles', 'agent-vis');
  const templateDir = path.join(HERE, 'profile');
  const pluginUrl = pathToFileURL(path.join(ROOT, 'plugin.js')).href;

  fs.mkdirSync(profileDir, { recursive: true });
  const replacements = {
    __PLUGIN_URL__: pluginUrl,
    __PROVIDER_NAME__: PROVIDER_NAME,
    __BASE_URL__: BASE_URL,
    __MODEL_ID__: MODEL_ID,
    __CONTEXT_WINDOW__: String(CONTEXT_WINDOW),
  };
  for (const name of ['package.json', 'cordis.yml', 'cordis.patch.yml']) {
    let source = fs.readFileSync(path.join(templateDir, name), 'utf8');
    for (const [token, value] of Object.entries(replacements)) source = source.replaceAll(token, value);
    fs.writeFileSync(path.join(profileDir, name), source);
  }

  process.stdout.write(
    `agent-vis profile 已写入 ${profileDir}\n` +
      `  plugin: ${pluginUrl}\n` +
      `运行：DSH_HOME=${home} node ${path.join(ROOT, 'node_modules/@deepseek-ai/dsh/lib/bin.js')} --profile agent-vis\n`,
  );
}

main();
