/**
 * idea 文件的字段契约（genrec.idea.v2）。
 *
 * 为什么要有这一份：模型写 idea 时是自由发挥的，写出来的东西看着完整，
 * 但里面常常混进不该由它决定的东西 —— 要跑几次训练、用什么种子、
 * 判据阈值是多少、状态是什么。这些是仓库契约和框架的职责（CLAUDE.md §19g、
 * §25、§66），一旦写进 idea，审稿的人就得去逐条挑，而散文是挑不干净的。
 *
 * 所以这里定死字段：顶层只能有三个键，每段只有那几个键，枚举值闭合，
 * 评估门槛是仓库契约里的常数。写不出来的形状（要跑 13 次训练、多个种子、
 * 自定 success threshold、带上 result.status）不是"不合规"，是**表达不出来**。
 *
 * 校验只做形状与闭合值，不判断机制好不好 —— 好不好是 AGENT AGAINST 的活。
 */

/** 这份契约的版本号，也是 idea 文件里 schema 字段必须写的值。 */
export const IDEA_SCHEMA_VERSION = 'genrec.idea.v2';

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

/** 计算精度。写不出来的就不许写。 */
export const IDEA_DTYPES = ['fp16', 'bf16', 'fp32', 'fp64'];

/** 主线的两条 must_hold（CLAUDE.md 里的主线文件），两条都必须对上。 */
export const IDEA_MUST_HOLD_CLAUSES = ['must_hold#1', 'must_hold#2'];

/** 评估指标与门槛是仓库契约（§25）：test_R@10 > 0.065。idea 只能引用，不能自定义。 */
export const IDEA_EVAL_METRIC = 'stage3.test_recall@10';
export const IDEA_EVAL_THRESHOLD = 0.065;

/** idea 文件顶层与各段允许出现的键。多一个键就是不合规。 */
const TOP_KEYS = ['schema', 'arm', 'sections'];
const SECTION_KEYS = ['1_goal', '4_method', '5_references', '6_evaluation'];
const GOAL_KEYS = ['mechanism', 'af3_entry', 'one_variable', 'must_hold', 'novelty'];
const METHOD_KEYS = ['surfaces', 'numerics', 'sanity_checks'];
const EVAL_KEYS = ['metric', 'threshold', 'comparison_arm', 'falsifier'];
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
      `顶层不允许这个字段（v2 顶层只有 schema / arm / sections）。` +
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
        add(`sections.${key}`, `这一段不属于 v2（只有 ${SECTION_KEYS.join(' / ')}）`);
      }
      for (const key of SECTION_KEYS) {
        if (!(key in sections)) add(`sections.${key}`, `缺少这一段`);
      }
      checkGoal(sections['1_goal'], add);
      checkMethod(sections['4_method'], add);
      checkReferences(sections['5_references'], add);
      checkEvaluation(sections['6_evaluation'], add);
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
    add(`sections.1_goal.${key}`, `v2 的 1_goal 只有 ${GOAL_KEYS.join(' / ')}；这一项没有消费者，删掉或归位`);
  }
  for (const key of GOAL_KEYS) {
    if (!(key in goal)) add(`sections.1_goal.${key}`, `必须有 ${key}`);
  }
  if (goal.mechanism !== undefined) {
    if (!nonEmptyString(goal.mechanism)) {
      add('sections.1_goal.mechanism', '必须是一段非空文字：几何对象 + 机制 + 几何进计算通路的哪两处');
    } else if (goal.mechanism.length > 400) {
      add('sections.1_goal.mechanism', `太长了（${goal.mechanism.length} 字，上限 400）：这是摘要，不是正文`);
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
    add(`sections.4_method.${key}`, `v2 的 4_method 只有 ${METHOD_KEYS.join(' / ')}；文件路径与运行计划归框架（§66）`);
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
  if (method.numerics !== undefined) {
    if (!isPlainObject(method.numerics)) {
      add('sections.4_method.numerics', '必须是对象');
    } else {
      for (const key of extraKeys(method.numerics, ['dtype', 'clamps', 'degenerate_to_baseline'])) {
        add(`sections.4_method.numerics.${key}`, '只允许 dtype / clamps / degenerate_to_baseline');
      }
      if (!IDEA_DTYPES.includes(method.numerics.dtype)) {
        add('sections.4_method.numerics.dtype', `取值只能是 ${IDEA_DTYPES.join(' / ')}`);
      }
      if (method.numerics.clamps !== undefined) {
        if (!Array.isArray(method.numerics.clamps)) {
          add('sections.4_method.numerics.clamps', '必须是数组（没有截断就写空数组）');
        } else {
          method.numerics.clamps.forEach((item, index) => {
            if (!isPlainObject(item)) {
              add(`sections.4_method.numerics.clamps[${index}]`, '必须是对象');
              return;
            }
            for (const key of extraKeys(item, ['name', 'value', 'why'])) {
              add(`sections.4_method.numerics.clamps[${index}].${key}`, '只允许 name / value / why');
            }
            if (!nonEmptyString(item.name)) {
              add(`sections.4_method.numerics.clamps[${index}].name`, '必须写截断的名字');
            }
            if (typeof item.value !== 'number' && typeof item.value !== 'string') {
              add(`sections.4_method.numerics.clamps[${index}].value`, '必须是数值或字符串');
            }
            if (!nonEmptyString(item.why)) {
              add(`sections.4_method.numerics.clamps[${index}].why`, '必须写为什么取这个界（要可推导，不是试出来的）');
            }
          });
        }
      }
      if (!nonEmptyString(method.numerics.degenerate_to_baseline)) {
        add('sections.4_method.numerics.degenerate_to_baseline', '必须写退化成什么（哪一项关掉后与基线逐位一致）');
      }
    }
  }
  if (method.sanity_checks !== undefined) {
    if (!Array.isArray(method.sanity_checks)) {
      add('sections.4_method.sanity_checks', '必须是数组');
    } else {
      if (method.sanity_checks.length > 8) {
        add('sections.4_method.sanity_checks', `最多 8 条（现在 ${method.sanity_checks.length} 条）：只留能机械判定的`);
      }
      method.sanity_checks.forEach((item, index) => {
        if (!nonEmptyString(item)) {
          add(`sections.4_method.sanity_checks[${index}]`, '必须是一句能机械判定的话');
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

function checkEvaluation(evalObj, add) {
  if (evalObj === undefined) return;
  if (!isPlainObject(evalObj)) {
    add('sections.6_evaluation', '必须是对象');
    return;
  }
  for (const key of extraKeys(evalObj, EVAL_KEYS)) {
    add(`sections.6_evaluation.${key}`, `v2 的 6_evaluation 只有 ${EVAL_KEYS.join(' / ')}；门槛是仓库契约，不许自定`);
  }
  for (const key of EVAL_KEYS) {
    if (!(key in evalObj)) add(`sections.6_evaluation.${key}`, `必须有 ${key}`);
  }
  if (evalObj.metric !== undefined && evalObj.metric !== IDEA_EVAL_METRIC) {
    add('sections.6_evaluation.metric', `必须是 "${IDEA_EVAL_METRIC}"（§25：最终判据是 Stage3 的 test_R@10）`);
  }
  if (evalObj.threshold !== undefined && evalObj.threshold !== IDEA_EVAL_THRESHOLD) {
    add(
      'sections.6_evaluation.threshold',
      `必须是 ${IDEA_EVAL_THRESHOLD}（§25：test_R@10 > 0.065），自定门槛无效`,
    );
  }
  if (evalObj.comparison_arm !== undefined && !nonEmptyString(evalObj.comparison_arm)) {
    add('sections.6_evaluation.comparison_arm', '必须写对照臂的名字');
  }
  if (evalObj.falsifier !== undefined && !nonEmptyString(evalObj.falsifier)) {
    add('sections.6_evaluation.falsifier', '必须写一句：什么观测会判这个机制为否');
  }
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
      "mechanism": "一句话：几何对象是什么、机制做了什么、几何进计算通路的哪两处",
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
      ],
      "numerics": {
        "dtype": "fp32",
        "clamps": [{ "name": "", "value": 0, "why": "为什么取这个界" }],
        "degenerate_to_baseline": "关掉哪一项后退化成基线（要能逐位对齐）"
      },
      "sanity_checks": ["一句能机械判定的话"]
    },
    "5_references": [
      { "title": "", "year": 2024, "venue": "", "doi": "", "relation": "different" }
    ],
    "6_evaluation": {
      "metric": "${IDEA_EVAL_METRIC}",
      "threshold": ${IDEA_EVAL_THRESHOLD},
      "comparison_arm": "<对照臂名字>",
      "falsifier": "什么观测会判否"
    }
  }
}`;
}

/**
 * v2 里明确不存在的字段 —— 给模型看"这些不要写"，比只给骨架更管用。
 *
 * 每一条都注明它本来该归谁。
 */
export const IDEA_FORBIDDEN_FIELDS = [
  { field: 'date / revised', owner: 'git 与轮次登记' },
  { field: 'revise_round', owner: '轮次登记（返修流水）' },
  { field: 'result.status', owner: '轮次登记（状态）' },
  { field: '4_method.runs', owner: '框架的运行调度（§19g：只跑一次，不多种子、不做消融臂）' },
  { field: '4_method.new_files / modified_files', owner: '框架的候选树与登记（§66：路径归框架）' },
  { field: '6_evaluation.decision_rule', owner: '仓库契约（§25：test_R@10 > 0.065）' },
  { field: '6_evaluation.reference', owner: 'ideas/baseline.json（基线事实的单一出处）' },
  { field: '6_evaluation.prerequisite_checks / diagnostics', owner: 'sanity_checks 与评估期的记录格式' },
  { field: '1_goal.vs_archive_table', owner: 'novelty.inherited_from 与 closest_works' },
  { field: '1_goal.must_hold_alignment', owner: 'must_hold 数组（两条都要对上）' },
  { field: '5_references[].cited_by', owner: '没有消费者，删掉' },
];
