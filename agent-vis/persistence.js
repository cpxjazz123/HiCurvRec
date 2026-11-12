/**
 * 落盘：agent 的配置与对话记录。
 *
 * 目录结构（默认在 agent-vis/.data/ 下）：
 *
 *   agents.json             每个 agent 一份配置 + 运行时新建的 agent 身份
 *   chats/<agentId>.jsonl   每个 agent 的对话记录，一行一轮请求
 *
 * 两条硬规矩：
 *   1. 写文件先写临时文件再改名 —— 断电或进程被杀时不会留下半截 JSON；
 *   2. 读到一个写坏的文件直接报错，不悄悄当成空数据继续跑 ——
 *      "用户配的步骤没了"这种静默失败比报错难查得多。
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** 默认数据目录：agent-vis/.data/。 */
export const DEFAULT_DATA_DIR = path.join(HERE, '.data');

/** 存下来的格式版本。读到的版本不认识就报错，不猜。 */
const STATE_VERSION = 1;

/**
 * 造一个落盘句柄。
 *
 * dataDir 由调用方给：正式服务用默认目录，测试用仓库内的临时目录，
 * 这样测试跑完不会把用户自己的配置冲掉。
 */
export function createPersistence({ dataDir }) {
  if (typeof dataDir !== 'string' || dataDir === '') {
    throw new Error('createPersistence 需要一个非空的数据目录');
  }
  const agentsFile = path.join(dataDir, 'agents.json');
  const chatsDir = path.join(dataDir, 'chats');

  function ensureDirs() {
    fs.mkdirSync(chatsDir, { recursive: true });
  }

  /**
   * 读出配置表。
   *
   * 预期内的缺失：还没存过任何东西（第一次启动）—— 返回空结构。
   * 预期外的失败：文件在但内容坏了 —— 直接抛错，别兜底。
   */
  function loadAgents() {
    if (!fs.existsSync(agentsFile)) {
      return { version: STATE_VERSION, configs: {}, agents: {} };
    }
    const raw = fs.readFileSync(agentsFile, 'utf8');
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      throw new Error(
        `agents.json 不是合法的 JSON（${agentsFile}）：${(err && err.message) || String(err)}`,
      );
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error(`agents.json 的内容不是一个对象（${agentsFile}）`);
    }
    if (parsed.version !== STATE_VERSION) {
      throw new Error(
        `agents.json 的版本不认识（${agentsFile}）：${parsed.version}，认得的是 ${STATE_VERSION}`,
      );
    }
    if (!parsed.configs || typeof parsed.configs !== 'object' || Array.isArray(parsed.configs)) {
      throw new Error(`agents.json 里的 configs 不是一个对象（${agentsFile}）`);
    }
    if (!parsed.agents || typeof parsed.agents !== 'object' || Array.isArray(parsed.agents)) {
      throw new Error(`agents.json 里的 agents 不是一个对象（${agentsFile}）`);
    }
    return parsed;
  }

  /** 把配置表写回去。先写临时文件再改名，保证不会留下半截文件。 */
  function saveAgents(state) {
    ensureDirs();
    const tmp = `${agentsFile}.tmp`;
    fs.writeFileSync(tmp, `${JSON.stringify(state, null, 2)}\n`, 'utf8');
    fs.renameSync(tmp, agentsFile);
  }

  /** 追加一条对话记录（一行 JSON）。 */
  function appendChat(agentId, record) {
    ensureDirs();
    const file = chatFile(agentId);
    fs.appendFileSync(file, `${JSON.stringify(record)}\n`, 'utf8');
  }

  /**
   * 读出某个 agent 的全部对话记录，按写入顺序。
   *
   * 一行坏了就报错：把坏行悄悄丢掉，等于让用户以为那段对话没发生过。
   */
  function readChats(agentId) {
    const file = chatFile(agentId);
    if (!fs.existsSync(file)) return [];
    const raw = fs.readFileSync(file, 'utf8');
    const out = [];
    raw.split('\n').forEach((line, index) => {
      const text = line.trim();
      if (text === '') return;
      try {
        out.push(JSON.parse(text));
      } catch (err) {
        throw new Error(
          `对话记录第 ${index + 1} 行不是合法的 JSON（${file}）：${(err && err.message) || String(err)}`,
        );
      }
    });
    return out;
  }

  /**
   * 清掉某个 agent 的对话记录。
   *
   * 返回有没有真的删掉东西：文件本来就不在时返回 false ——
   * "清空"本来就是幂等的，不必当成失败。
   */
  function clearChats(agentId) {
    const file = chatFile(agentId);
    if (!fs.existsSync(file)) return false;
    fs.rmSync(file);
    return true;
  }

  /** 某个 agent 的对话记录文件。 */
  function chatFile(agentId) {
    if (typeof agentId !== 'string' || agentId === '') {
      throw new Error('对话记录必须指明是哪个 agent');
    }
    return path.join(chatsDir, `${encodeURIComponent(agentId)}.jsonl`);
  }

  return {
    dataDir,
    agentsFile,
    chatsDir,
    loadAgents,
    saveAgents,
    appendChat,
    readChats,
    clearChats,
  };
}
