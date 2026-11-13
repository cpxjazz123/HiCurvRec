/**
 * idea 文件的字段契约（genrec.idea.v3）。
 *
 * idea 只写**假设**：这个方向赌的是什么、结构信号从哪来、进计算通路的哪几处、
 * 改了什么没改什么、与主线两条 must_hold 怎么对齐、新增性主张是什么。
 *
 * 数值、界、判据、验证步骤都不属于 idea —— 那是实现阶段的事：
 *   - 常量与它们的来由 → math.json 的 constants（name/value/why）；
 *   - 退化路径与三个开关 → math.json 的 switches；
 *   - 验收与判据 → 实现阶段的测试套件与仓库契约（§19g、§25）。
 * 之前这些都塞在 idea 里，于是模型在设计稿里排训练计划（跑几次、什么种子）、
 * 自定 success threshold、写状态；审稿的人只能逐条去挑散文，散文是挑不干净的。
 *
 * 所以这里定死字段：顶层只有三个键，每段只有那几个键，枚举值闭合。
 * 写不出来的形状（要跑 13 次训练、多个种子、自定门槛、带上 result.status、
 * 或者把数值约束与验收清单写进设计稿）不是"不合规"，是**表达不出来**。
 *
 * 校验只做形状与闭合值，不判断机制好不好 —— 好不好是 AGENT AGAINST 的活。
 */

/** 这份契约的版本号，也是 idea 文件里 schema 字段必须写的值。 */
export const IDEA_SCHEMA_VERSION = 'genrec.idea.v3';

/**
 * 候选可以动的面（CLAUDE.md §66：行为统计、双曲操作、分配规则、残差更新、
 * 新增损失）。af3_entry.enters_at 与 one_variable 只能用这几个词。
 */
export const IDEA_SURFACES = [
  'encoder_output',
  'assignment_cost',
  'residual_update',
  'codebook',
  'loss',
  'structure_stat',
  'decoder',
];

/** 文献与主线的关系：同族（只差实现）、不同族、基线方法本身。 */
export const IDEA_RELATIONS = ['same_family', 'different', 'baseline_method'];

/** 主线的两条 must_hold（CLAUDE.md 里的主线文件），两条都必须对上。 */
export const IDEA_MUST_HOLD_CLAUSES = ['must_hold#1', 'must_hold#2'];

/** idea 文件顶层与各段允许出现的键。多一个键就是不合规。 */
const TOP_KEYS = ['schema', 'arm', 'sections'];
const SECTION_KEYS = ['1_goal', '4_method', '5_references'];
const GOAL_KEYS = ['assumption', 'af3_entry', 'one_variable', 'must_hold', 'novelty'];
const METHOD_KEYS = ['surfaces'];
const WORK_KEYS = ['title', 'year', 'venue', 'doi', 'relation'];

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

function isPlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function nonEmptyString(value) {
  return typeof value === 'string' && value.trim() !== '';
}

/**
 * 检查一个对象有没有多余键。多余键一律报错 —— v2 的意义就是字段固定。
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
      checkMethod(sections['4_method'], add);
      checkReferences(sections['5_references'], add);
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
    add(`sections.1_goal.${key}`, `${IDEA_SCHEMA_VERSION} 的 1_goal 只有 ${GOAL_KEYS.join(' / ')}；这一项没有消费者，删掉或归位`);
  }
  for (const key of GOAL_KEYS) {
    if (!(key in goal)) add(`sections.1_goal.${key}`, `必须有 ${key}`);
  }
  if (goal.assumption !== undefined) {
    if (!nonEmptyString(goal.assumption)) {
      add('sections.1_goal.assumption', '必须是一段非空文字：这个方向假设了什么（几何对象 + 结构信号怎么进计算通路）');
    } else if (goal.assumption.length > 400) {
      add('sections.1_goal.assumption', `太长了（${goal.assumption.length} 字，上限 400）：这是假设，不是正文`);
    }
  }
  if (goal.af3_entry !== undefined) {
    if (!isPlainObject(goal.af3_entry)) {
      add('sections.1_goal.af3_entry', '必须是对象');
    } else {
      for (const key of extraKeys(goal.af3_entry, ['signal', 'formula', 'enters_at'])) {
        add(`sections.1_goal.af3_entry.${key}`, '只允许 signal / formula / enters_at');
      }
      if (!nonEmptyString(goal.af3_entry.signal)) {
        add('sections.1_goal.af3_entry.signal', '必须写清 AF3 给出的那个信号（是什么量、怎么算出来的）');
      }
      if (!nonEmptyString(goal.af3_entry.formula)) {
        add('sections.1_goal.af3_entry.formula', '必须写 math.json 里的公式名（指针，不是复述）');
      }
      if (goal.af3_entry.enters_at !== undefined) {
        if (!Array.isArray(goal.af3_entry.enters_at) || goal.af3_entry.enters_at.length === 0) {
          add('sections.1_goal.af3_entry.enters_at', '至少写一个面，取值只能是 ' + IDEA_SURFACES.join(' / '));
        } else {
          goal.af3_entry.enters_at.forEach((item, index) => {
            if (!IDEA_SURFACES.includes(item)) {
              add(`sections.1_goal.af3_entry.enters_at[${index}]`, `取值只能是 ${IDEA_SURFACES.join(' / ')}`);
            }
          });
        }
      }
    }
  }
  if (goal.one_variable !== undefined) {
    if (!isPlainObject(goal.one_variable)) {
      add('sections.1_goal.one_variable', '必须是对象');
    } else {
      for (const key of extraKeys(goal.one_variable, ['changed', 'unchanged'])) {
        add(`sections.1_goal.one_variable.${key}`, '只允许 changed / unchanged');
      }
      for (const key of ['changed', 'unchanged']) {
        const list = goal.one_variable[key];
        if (list === undefined) {
          add(`sections.1_goal.one_variable.${key}`, `必须有 ${key}`);
          continue;
        }
        if (!Array.isArray(list) || list.length === 0) {
          add(`sections.1_goal.one_variable.${key}`, '至少写一项');
          continue;
        }
        list.forEach((item, index) => {
          if (!IDEA_SURFACES.includes(item)) {
            add(`sections.1_goal.one_variable.${key}[${index}]`, `取值只能是 ${IDEA_SURFACES.join(' / ')}`);
          }
        });
      }
    }
  }
  if (goal.must_hold !== undefined) {
    if (!Array.isArray(goal.must_hold)) {
      add('sections.1_goal.must_hold', '必须是数组：两条 must_hold 各一项');
    } else {
      const seen = [];
      goal.must_hold.forEach((item, index) => {
        if (!isPlainObject(item)) {
          add(`sections.1_goal.must_hold[${index}]`, '必须是对象');
          return;
        }
        for (const key of extraKeys(item, ['clause', 'how'])) {
          add(`sections.1_goal.must_hold[${index}].${key}`, '只允许 clause / how');
        }
        if (!IDEA_MUST_HOLD_CLAUSES.includes(item.clause)) {
          add(`sections.1_goal.must_hold[${index}].clause`, `取值只能是 ${IDEA_MUST_HOLD_CLAUSES.join(' / ')}`);
        } else {
          seen.push(item.clause);
        }
        if (!nonEmptyString(item.how)) {
          add(`sections.1_goal.must_hold[${index}].how`, '必须写清这一条怎么满足的（写在哪、怎么算的）');
        }
      });
      for (const clause of IDEA_MUST_HOLD_CLAUSES) {
        if (!seen.includes(clause)) add('sections.1_goal.must_hold', `缺 ${clause}：两条 must_hold 都要对上`);
      }
    }
  }
  if (goal.novelty !== undefined) {
    if (!isPlainObject(goal.novelty)) {
      add('sections.1_goal.novelty', '必须是对象');
    } else {
      for (const key of extraKeys(goal.novelty, ['claim', 'inherited_from', 'closest_works'])) {
        add(`sections.1_goal.novelty.${key}`, '只允许 claim / inherited_from / closest_works');
      }
      if (!nonEmptyString(goal.novelty.claim)) {
        add('sections.1_goal.novelty.claim', '必须写清新增性到底主张哪几条');
      }
      if (goal.novelty.inherited_from !== undefined) {
        if (!Array.isArray(goal.novelty.inherited_from)) {
          add('sections.1_goal.novelty.inherited_from', '必须是数组（可以为空数组）');
        } else {
          goal.novelty.inherited_from.forEach((item, index) => {
            if (!nonEmptyString(item)) {
              add(`sections.1_goal.novelty.inherited_from[${index}]`, '必须写归档方向的名字');
            }
          });
        }
      }
      checkWorks(goal.novelty.closest_works, 'sections.1_goal.novelty.closest_works', add, true);
    }
  }
}

function checkMethod(method, add) {
  if (method === undefined) return;
  if (!isPlainObject(method)) {
    add('sections.4_method', '必须是对象');
    return;
  }
  for (const key of extraKeys(method, METHOD_KEYS)) {
    add(`sections.4_method.${key}`, `${IDEA_SCHEMA_VERSION} 的 4_method 只有 ${METHOD_KEYS.join(' / ')}；数值约束与运行计划归实现阶段与框架（§66）`);
  }
  for (const key of METHOD_KEYS) {
    if (!(key in method)) add(`sections.4_method.${key}`, `必须有 ${key}`);
  }
  if (method.surfaces !== undefined) {
    if (!Array.isArray(method.surfaces) || method.surfaces.length === 0) {
      add('sections.4_method.surfaces', '至少写一个面');
    } else {
      method.surfaces.forEach((item, index) => {
        if (!isPlainObject(item)) {
          add(`sections.4_method.surfaces[${index}]`, '必须是对象');
          return;
        }
        for (const key of extraKeys(item, ['surface', 'symbol', 'change'])) {
          add(`sections.4_method.surfaces[${index}].${key}`, '只允许 surface / symbol / change');
        }
        if (!IDEA_SURFACES.includes(item.surface)) {
          add(`sections.4_method.surfaces[${index}].surface`, `取值只能是 ${IDEA_SURFACES.join(' / ')}`);
        }
        if (!nonEmptyString(item.symbol)) {
          add(`sections.4_method.surfaces[${index}].symbol`, '必须写改在哪个符号/函数上');
        }
        if (!nonEmptyString(item.change)) {
          add(`sections.4_method.surfaces[${index}].change`, '必须写这一处改成了什么');
        }
      });
    }
  }
}

function checkWorks(list, path, add, optional) {
  if (list === undefined) {
    if (!optional) add(path, '必须是数组');
    return;
  }
  if (!Array.isArray(list)) {
    add(path, '必须是数组');
    return;
  }
  list.forEach((item, index) => {
    if (!isPlainObject(item)) {
      add(`${path}[${index}]`, '必须是对象');
      return;
    }
    for (const key of extraKeys(item, WORK_KEYS)) {
      add(`${path}[${index}].${key}`, `只允许 ${WORK_KEYS.join(' / ')}（被引数之类没有消费者，删掉）`);
    }
    for (const key of ['title', 'venue', 'doi']) {
      if (!nonEmptyString(item[key])) add(`${path}[${index}].${key}`, '必须是非空字符串');
    }
    if (!Number.isInteger(item.year)) add(`${path}[${index}].year`, '必须是整数年份');
    if (!IDEA_RELATIONS.includes(item.relation)) {
      add(`${path}[${index}].relation`, `取值只能是 ${IDEA_RELATIONS.join(' / ')}`);
    }
  });
}

function checkReferences(list, add) {
  if (list === undefined) {
    add('sections.5_references', '必须有这一段（找不到文献就写空数组）');
    return;
  }
  checkWorks(list, 'sections.5_references', add, false);
}

/**
 * 写给模型看的骨架。
 *
 * 提示词直接用这一份，保证"要求的形状"和"校验的形状"是同一份东西 ——
 * 两份各写一遍，迟早会漂。
 */
export function ideaSkeleton() {
  return `{
  "schema": "${IDEA_SCHEMA_VERSION}",
  "arm": "TIGER_<方向>_rq",
  "sections": {
    "1_goal": {
      "assumption": "假设：这个方向赌的是什么（几何对象是什么、结构信号怎么进计算通路）",
      "af3_entry": {
        "signal": "AF3 给出的那个信号（什么量、怎么算）",
        "formula": "math.json 里的公式名",
        "enters_at": ["assignment_cost"]
      },
      "one_variable": {
        "changed": ["residual_update"],
        "unchanged": ["codebook", "loss"]
      },
      "must_hold": [
        { "clause": "must_hold#1", "how": "这一条怎么满足的（写在哪、怎么算）" },
        { "clause": "must_hold#2", "how": "这一条怎么满足的（写在哪、怎么算）" }
      ],
      "novelty": {
        "claim": "新增性只主张哪几条",
        "inherited_from": ["<继承的归档方向>"],
        "closest_works": [
          { "title": "", "year": 2024, "venue": "", "doi": "", "relation": "same_family" }
        ]
      }
    },
    "4_method": {
      "surfaces": [
        { "surface": "residual_update", "symbol": "<函数/类名>", "change": "这一处改成了什么" }
      ]
    },
    "5_references": [
      { "title": "", "year": 2024, "venue": "", "doi": "", "relation": "different" }
    ]
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
  { field: '4_method.runs', owner: '框架的运行调度（§19g：只跑一次，不多种子、不做消融臂）' },
  { field: '4_method.new_files / modified_files', owner: '框架的候选树与登记（§66：路径归框架）' },
  { field: '6_evaluation（整段）', owner: '判据与门槛归仓库契约（§25）与实现阶段；idea 只写假设' },
  { field: '4_method.numerics', owner: '常量与界归 math.json 的 constants，退化路径归 switches' },
  { field: '6_evaluation.decision_rule', owner: '仓库契约（§25：test_R@10 > 0.065）' },
  { field: '6_evaluation.reference', owner: 'ideas/baseline.json（基线事实的单一出处）' },
  { field: '4_method.sanity_checks', owner: '验证计划与测试归实现阶段（idea 只写机制与判据）' },
  { field: '6_evaluation.prerequisite_checks / diagnostics', owner: '实现阶段的验证计划与评估期的记录格式' },
  { field: '1_goal.vs_archive_table', owner: 'novelty.inherited_from 与 closest_works' },
  { field: '1_goal.must_hold_alignment', owner: 'must_hold 数组（两条都要对上）' },
  { field: '5_references[].cited_by', owner: '没有消费者，删掉' },
];
