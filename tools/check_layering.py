"""Report cross-package imports that point from a lower layer to a higher one (run from repo root)."""
import ast, collections, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1] / 'femto'
ORDER = ['femto', 'femto.depth', 'femto.dataset', 'femto.geometry', 'femto.hardware', 'femto.reports', 'femto.annotation',
         'femto.tracking', 'femto.temporal', 'femto.local', 'femto.planning', 'femto.viewer', 'femto.pipeline',
         'femto.gsplat', 'femto.capture', 'femto.app']
LAYER = {p: i for i, p in enumerate(ORDER)}

def pkg_of(dotted):
    parts = dotted.split('.')
    return '.'.join(parts[:2]) if len(parts) > 2 else 'femto'

def main():
    up = collections.defaultdict(list); total = 0
    for path in ROOT.rglob('*.py'):
        mod = path.relative_to(ROOT.parent).with_suffix('').as_posix().replace('/', '.')
        src = pkg_of(mod)
        tree = ast.parse(path.read_bytes()); top = {id(n) for n in tree.body}
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('femto.'):
                targets = [node.module]
            elif isinstance(node, ast.Import):
                targets = [a.name for a in node.names if a.name.startswith('femto.')]
            for t in targets:
                if t == 'femto.paths': continue
                dst = pkg_of(t)
                if dst != src and LAYER[dst] > LAYER[src]:
                    total += 1
                    up[(src, dst)].append(f"{mod.split('.')[-1]} -> {t.split('.')[-1]}" + ('' if id(node) in top else ' (lazy)'))
    for k in sorted(up, key=lambda k: (LAYER[k[0]], LAYER[k[1]])):
        print(f'{k[0]} -> {k[1]}')
        for e in up[k]: print('    ', e)
    print('upward edges:', total)
    return 1 if total else 0

if __name__ == '__main__':
    sys.exit(main())
