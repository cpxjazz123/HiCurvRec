/**
 * idea 文件的字段契约（genrec.idea.v4）。
 *
 * idea 只回答两件事：
 *   1. 这个 idea 本身是什么（1_goal.idea）；
 *   2. 要新建哪些文件、每个文件是干什么的（4_method.new_files）。
 *
 * 文献不进文件：写 idea 时照样可以检索（检索的结果帮助判断新增性），
 * 但稿子里不存参考文献清单 —— 查重只看 ideas/_archive/，审稿方也不再核文献。
 *
 * 其余一律不写：数值与界、判据与门槛、验证步骤、开关、状态、训练计划。
 * 那些各有归属 —— 常量与它们的来由在 math.json 的 constants，退化路径与三个开关在
 * math.json 的 switches，判据是仓库契约（CLAUDE.md §25），验收是实现阶段的测试套件，
 * 路径与运行归框架（§66）。
 *
 * 之前这些都塞在 idea 里，于是模型在设计稿里排训练计划（跑几次、什么种子）、
 * 自定 success threshold、写状态；审稿的人只能逐条去挑散文，而散文是挑不干净的。
 * 现在写不出来就是写不出来：多一个键就判不合规。
 *
 * 校验只做形状，不判断机制好不好 —— 好不好是 AGENT AGAINST 的活。
 */

/** 这份契约的版本号，也是 idea 文件里 schema 字段必须写的值。 */
export const IDEA_SCHEMA_VERSION = 'genrec.idea.v5';

/** idea 文件顶层与各段允许出现的键。多一个键就是不合规。 */
const TOP_KEYS = ['schema', 'arm', 'sections'];
const SECTION_KEYS = ['1_goal', '4_method'];
const GOAL_KEYS = ['idea'];
const METHOD_KEYS = ['new_files'];

/**
 * 整份文档里都不许出现的写法：多种子计划。
 *
 * 只认「具体列出多个种子」和「N seed」这两种写法 —— 否定句（"不做多种子"）
 * 不会命中，所以不会误伤。
 */
const SEED_PATTERNS = [
  { re: /\bseed\s*\d+(\s*[/,、]\s*\d+)+/i, why: '写出了多个种子的运行计划；一次候选只跑一次（§19g）' },
  { re: /\d+\s*seeds?\b/i, why: '写出了多种子统计（如 "3 seed 均值"）；不做多种子实验（§19g）' },
];

/** new_files 的条数上限：方向不是工程量清单。 */
const MAX_NEW_FILES = 12;

function isPlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function nonEmptyString(value) {
  return typeof value === 'string' && value.trim() !== '';
}

/**
 * 检查一个对象有没有多余键。多余键一律报错 —— v4 的意义就是字段固定。
 */
function extraKeys(obj, allowed) {
  return Object.keys(obj).filter((key) => !allowed.includes(key));
}

/**
 * 校验一份 idea 文档。
 *
 * @param {unknown} doc 解析后的 JSON
 * @returns {{ok: boolean, version: string, errors: {path: string, why: string}[]}}
 */
export function validateIdeaDoc(doc) {
  const errors = [];
  const add = (path, why) => errors.push({ path, why });

  if (!isPlainObject(doc)) {
    return {
      ok: false,
      version: IDEA_SCHEMA_VERSION,
      errors: [{ path: '$', why: '顶层必须是一个 JSON 对象' }],
    };
  }

  for (const key of extraKeys(doc, TOP_KEYS)) {
    add(
      key,
      `顶层不允许这个字段（${IDEA_SCHEMA_VERSION} 顶层只有 schema / arm / sections）。` +
        `状态、日期、轮次流水归轮次登记；运行计划与判据归仓库契约。`,
    );
  }
  for (const key of TOP_KEYS) {
    if (!(key in doc)) add(key, `顶层必须有 ${key}`);
  }
  if (doc.schema !== undefined && doc.schema !== IDEA_SCHEMA_VERSION) {
    add('schema', `必须写成 "${IDEA_SCHEMA_VERSION}"`);
  }
  if (doc.arm !== undefined && !/^TIGER_[a-z0-9_]+_rq$/.test(String(doc.arm))) {
    add('arm', '必须形如 TIGER_xxx_rq（小写字母、数字、下划线），且与结果目录同名');
  }

  const sections = doc.sections;
  if (sections !== undefined) {
    if (!isPlainObject(sections)) {
      add('sections', '必须是对象');
    } else {
      for (const key of extraKeys(sections, SECTION_KEYS)) {
        add(`sections.${key}`, `这一段不属于 ${IDEA_SCHEMA_VERSION}（只有 ${SECTION_KEYS.join(' / ')}）`);
      }
      for (const key of SECTION_KEYS) {
        if (!(key in sections)) add(`sections.${key}`, `缺少这一段`);
      }
      checkGoal(sections['1_goal'], add);
      checkNewFiles(sections['4_method'], add);
    }
  }

  // 多种子计划：字段删不掉，就从值上认出来。
  const flat = JSON.stringify(doc);
  for (const { re, why } of SEED_PATTERNS) {
    const hit = re.exec(flat);
    if (hit) {
      add('$', `${why}（命中 "${hit[0]}"）`);
      break;
    }
  }

  return { ok: errors.length === 0, version: IDEA_SCHEMA_VERSION, errors };
}

function checkGoal(goal, add) {
  if (goal === undefined) return;
  if (!isPlainObject(goal)) {
    add('sections.1_goal', '必须是对象');
    return;
  }
  for (const key of extraKeys(goal, GOAL_KEYS)) {
    add(
      `sections.1_goal.${key}`,
      `${IDEA_SCHEMA_VERSION} 的 1_goal 只有 ${GOAL_KEYS.join(' / ')}；` +
        `这一项没有消费者，删掉或归位（数值与判据归 math.json 与实现阶段）。`,
    );
  }
  for (const key of GOAL_KEYS) {
    if (!(key in goal)) add(`sections.1_goal.${key}`, `必须有 ${key}`);
  }
  if (goal.idea !== undefined) {
    if (!nonEmptyString(goal.idea)) {
      add('sections.1_goal.idea', '必须是一段非空文字：这个 idea 是什么（几何对象 + 结构信号怎么进计算通路）');
    } else if (goal.idea.length > 1200) {
      add('sections.1_goal.idea', `太长了（${goal.idea.length} 字，上限 1200）：这是 idea 本身，不是正文`);
    }
  }
}

function checkNewFiles(method, add) {
  if (method === undefined) return;
  if (!isPlainObject(method)) {
    add('sections.4_method', '必须是对象');
    return;
  }
  for (const key of extraKeys(method, METHOD_KEYS)) {
    add(
      `sections.4_method.${key}`,
      `${IDEA_SCHEMA_VERSION} 的 4_method 只有 ${METHOD_KEYS.join(' / ')}；` +
        `改动清单、验证步骤与运行计划不写在这里。`,
    );
  }
  for (const key of METHOD_KEYS) {
    if (!(key in method)) add(`sections.4_method.${key}`, `必须有 ${key}`);
  }
  if (method.new_files === undefined) return;
  if (!Array.isArray(method.new_files)) {
    add('sections.4_method.new_files', '必须是数组（还没有要新建的文件就写空数组）');
    return;
  }
  if (method.new_files.length > MAX_NEW_FILES) {
    add('sections.4_method.new_files', `最多 ${MAX_NEW_FILES} 条（现在 ${method.new_files.length} 条）`);
  }
  method.new_files.forEach((item, index) => {
    if (!isPlainObject(item)) {
      add(`sections.4_method.new_files[${index}]`, '必须是对象');
      return;
    }
    for (const key of extraKeys(item, ['path', 'what'])) {
      add(`sections.4_method.new_files[${index}].${key}`, '只允许 path / what');
    }
    if (!nonEmptyString(item.path)) {
      add(`sections.4_method.new_files[${index}].path`, '必须写这个新文件放在哪（相对仓库根的路径）');
    }
    if (!nonEmptyString(item.what)) {
      add(`sections.4_method.new_files[${index}].what`, '必须写这个文件是干什么的');
    }
  });
}

/**
 * 写给模型看的骨架。
 *
 * 提示词直接用这一份，保证"要求的形状"和"校验的形状"是同一份东西 ——
 * 两处各写一遍，迟早会漂。
 */
export function ideaSkeleton() {
  return `{
  "schema": "${IDEA_SCHEMA_VERSION}",
  "arm": "TIGER_<方向>_rq",
  "sections": {
    "1_goal": {
      "idea": "这个 idea 是什么：几何对象是什么、结构信号从哪来、进计算通路的哪几处"
    },
    "4_method": {
      "new_files": [
        { "path": "<相对仓库根的路径>", "what": "这个新文件是干什么的" }
      ]
    }
  }
}`;
}

/**
 * 明确不存在的字段 —— 给模型看"这些不要写"，比只给骨架更管用。
 *
 * 每一条都注明它本来该归谁。
 */
export const IDEA_FORBIDDEN_FIELDS = [
  { field: 'date / revised', owner: 'git 与轮次登记' },
  { field: 'revise_round', owner: '轮次登记（返修流水）' },
  { field: 'result.status', owner: '轮次登记（状态）' },
  { field: '1_goal.assumption / mechanism', owner: 'v4 里这一栏叫 idea' },
  { field: '1_goal.af3_entry', owner: '结构信号怎么进通路写进 1_goal.idea 的正文' },
  { field: '1_goal.one_variable', owner: '实现阶段登记改动的文件与符号' },
  { field: '1_goal.must_hold', owner: '主线合规由审稿方逐条核（AGENT AGAINST）' },
  { field: '1_goal.novelty', owner: '查重由审稿方做；idea 只写自己是什么' },
  { field: '4_method.surfaces', owner: 'v4 里 4_method 只写要新建的文件' },
  { field: '4_method.numerics', owner: '常量与界归 math.json 的 constants，退化路径归 switches' },
  { field: '4_method.sanity_checks', owner: '验证计划与测试归实现阶段' },
  { field: '4_method.runs', owner: '框架的运行调度（§19g：只跑一次，不多种子、不做消融臂）' },
  { field: '6_evaluation（整段）', owner: '判据与门槛归仓库契约（§25）与实现阶段' },
  { field: '6_evaluation.decision_rule', owner: '仓库契约（§25：test_R@10 > 0.065）' },
  { field: '6_evaluation.reference', owner: 'ideas/baseline.json（基线事实的单一出处）' },
  { field: '5_references（整段）', owner: '文献不进文件：检索照做，差别写进 1_goal.idea 的正文' },
  { field: '5_references[].cited_by', owner: '没有消费者，删掉' },
];
