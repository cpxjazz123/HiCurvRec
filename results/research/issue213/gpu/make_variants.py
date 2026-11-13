"""Generate the two accelerated builds of ORCA's count5 from the stock source.

    python3 ../gpu/make_variants.py          # run inside results/research/issue213/orca

Writes `orca_omp.cpp` (OpenMP over stages 2 and 3) and `orca_gpu.cu` (stage 1 on the device,
stages 2-3 on the CPU against flat lookups).  Neither file is committed: both are derived, and the
upstream clone is not vendored, so the patch lives here in one readable place.

The two patches are independent: everything below `openmp_patch` only adds `#pragma` lines and
per-thread scratch, everything below `gpu_patch` only replaces the stage-1 body and the two table
lookups.  Both were validated by comparing the full output with the stock single-threaded run byte
for byte, not by inspecting the numbers.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
ORCA_DIR = HERE.parent / "orca"
SOURCE = ORCA_DIR / "orca.cpp"
INCLUDE = HERE / "orca_gpu_stage1.inc"

PROGRESS_BLOCK = [
    "\t\tfrac = 100LL*x/n;",
    "\t\tif (frac!=frac_prev) {",
    "\t\t\tprintf(\"%d%%\\r\",frac);",
    "\t\t\tfrac_prev=frac;",
    "\t\t}",
]


def drop_progress(lines: list[str]) -> list[str]:
    """The progress printer reads and writes shared state; inside a parallel loop it is a race."""
    out, i = [], 0
    while i < len(lines):
        if lines[i:i + 5] == PROGRESS_BLOCK:
            i += 5
            continue
        out.append(lines[i])
        i += 1
    return out


def index_of(lines: list[str], predicate, start: int = 0) -> int:
    for i in range(start, len(lines)):
        if predicate(lines[i]):
            return i
    raise SystemExit("anchor not found in the ORCA source")


def openmp_patch(source: str) -> str:
    lines = drop_progress(source.split("\n"))

    i = index_of(lines, lambda l: l.startswith("#include <algorithm>"))
    lines.insert(i + 1, "#ifdef _OPENMP\n#include <omp.h>\n#endif")

    # stage 2: per-thread neighbour buffers and a per-thread C5 slice (the body increments the
    # counts of other nodes, so a shared array is a race)
    i = index_of(lines, lambda l: "int64 *C5 = (int64*)calloc(n,sizeof(int64));" in l)
    j = index_of(lines, lambda l: l.strip() == "for (int x=0;x<n;x++) {", i)
    lines.insert(j, """\tconst int threads = omp_get_max_threads();
\tint64 *C5_all = (int64*)calloc((size_t)threads*n, sizeof(int64));
#pragma omp parallel
\t{
\tint *neigh = (int*)malloc(n*sizeof(int)), nn;
\tint *neigh2 = (int*)malloc(n*sizeof(int)), nn2;
\tint64 *C5 = C5_all + (size_t)omp_get_thread_num()*n;   // shadows the outer array: no shared writes
#pragma omp for schedule(dynamic,64)""")
    j = index_of(lines, lambda l: l.strip() == "endTime = clock();", j)
    lines.insert(j, "\t}\n\tfor (int t=0;t<threads;t++) for (int v=0;v<n;v++) C5[v] += C5_all[(size_t)t*n+v];\n\tfree(C5_all);")

    # stage 3: the scratch arrays and the two lookup iterators are global; both must be per thread
    i = index_of(lines, lambda l: "int *common_x = (int*)calloc(n,sizeof(int));" in l)
    j = index_of(lines, lambda l: l.strip() == "for (int x=0;x<n;x++) {", i)
    lines.insert(j, """#pragma omp parallel
\t{
\tint *common_x = (int*)calloc(n,sizeof(int));
\tint *common_x_list = (int*)malloc(n*sizeof(int)); int ncx=0;
\tint *common_a = (int*)calloc(n,sizeof(int));
\tint *common_a_list = (int*)malloc(n*sizeof(int)); int nca=0;
\tunordered_map<PAIR,int,hash_PAIR>::iterator common2_it;      // the get macros read these
\tunordered_map<TRIPLE,int,hash_TRIPLE>::iterator common3_it;
#pragma omp for schedule(dynamic,32)""")
    j = index_of(lines, lambda l: l.strip() == "endTime = clock();", j)
    lines.insert(j, "\t}")
    return "\n".join(lines)


def gpu_patch(openmp_source: str) -> str:
    src = openmp_source.replace("#include <algorithm>",
                                "#include <algorithm>\n#include <vector>\n#include <cuda_runtime.h>", 1)
    src = src.replace('''#define common3_get(x) (((common3_it=common3.find(x))!=common3.end())?(common3_it->second):0)
#define common2_get(x) (((common2_it=common2.find(x))!=common2.end())?(common2_it->second):0)''',
'''static inline unsigned common2_lookup(int a, int b);
static inline unsigned common3_lookup(int a, int b, int c);
#define common3_get(x) common3_lookup((x).a,(x).b,(x).c)
#define common2_get(x) common2_lookup((x).a,(x).b)''')

    # stage 1 of count5 only: the file carries the same banner in count4 and in the edge variants
    c5 = src.index("void count5()")
    c5_end = src.index("void ecount5()", c5)
    body = src[c5:c5_end]
    start = body.index('\tprintf("stage 1 - precomputing common nodes\\n");')
    end = body.index('\t// precompute triangles that span over edges', start)
    body = (body[:start]
            + '\tprintf("stage 1 - precomputing common nodes\\n");\n'
              '\tgpu_tables_init(n, m);\n'
              '\tgpu_build_tables(n, m);\n'
            + body[end:])
    src = src[:c5] + body + src[c5_end:]

    include = INCLUDE.read_text()
    i = src.index("int64 **orbit;")
    i = src.index("\n", i) + 1
    return src[:i] + "\n" + include + "\n" + src[i:]


def main() -> int:
    source = SOURCE.read_text()
    omp = openmp_patch(source)
    (ORCA_DIR / "orca_omp.cpp").write_text(omp)
    (ORCA_DIR / "orca_gpu.cu").write_text(gpu_patch(omp))
    print("wrote orca_omp.cpp and orca_gpu.cu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
