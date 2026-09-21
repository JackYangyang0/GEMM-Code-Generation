"""Expand only selected strategies' reference examples for code generation."""
import copy
import json
from functools import lru_cache
from pathlib import Path


CATALOG = Path(__file__).resolve().parents[1] / 'data/lib/strategy_examples.json'
FULL_CATALOG = CATALOG.with_name('strategy_examples_full.json')
COUPLING_CATALOG = CATALOG.with_name('strategy_coupling_contracts.json')


@lru_cache(maxsize=1)
def load_coupling_contracts():
    return json.loads(COUPLING_CATALOG.read_text(encoding='utf-8'))
EXAMPLE_RULE = (
    'Examples are conditional implementation references, not additional strategies or verified output. '
    'Check applicability against the current IR and actual parent kernel; adapt symbols, physical '
    'layouts, stage count and tail/alignment policy. Locked strategies, IR parameters and patch '
    'contracts take precedence. Never paste unresolved placeholders, change precision, remove '
    'selected optimizations or assume an example proves correctness. Return the requested patch '
    'format, not the example catalog or a full file unless explicitly authorized for repair. '
    'Validation obligations still require normal compile/numerical/safety/strategy checks.'
)


@lru_cache(maxsize=1)
def load_example_catalog():
    merged = json.loads(CATALOG.read_text(encoding='utf-8'))
    extra = json.loads(FULL_CATALOG.read_text(encoding='utf-8'))
    for key in ('examples', 'strategy_bindings'):
        if set(merged[key]) & set(extra[key]):
            raise ValueError('Duplicate strategy example definitions: '+key)
        merged[key].update(extra[key])
    return merged


def strategy_with_examples(strategy, catalog=None):
    catalog = load_example_catalog() if catalog is None else catalog
    result = copy.deepcopy(strategy)
    examples = []
    seen = set()

    def visit(node):
        if isinstance(node, dict):
            sid = node.get('strategy_id', '')
            if isinstance(sid, str) and sid:
                contracts = [copy.deepcopy(value) for prefix, value in load_coupling_contracts().items()
                             if sid.startswith(prefix)]
                if contracts:
                    node['coupling_contracts'] = contracts
                ids = node.get('implementation_example_ids',
                               catalog['strategy_bindings'].get(sid, []))
                for example_id in ids:
                    if example_id not in catalog['examples']:
                        raise ValueError(f'Unknown implementation example {example_id} for {sid}')
                    if example_id not in seen:
                        seen.add(example_id)
                        examples.append(copy.deepcopy(catalog['examples'][example_id]))
            for key, value in node.items():
                if key not in ('implementation_examples', 'implementation_example_ids'):
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    if examples:
        result['implementation_examples'] = examples
        result['implementation_example_policy'] = EXAMPLE_RULE
    return result
