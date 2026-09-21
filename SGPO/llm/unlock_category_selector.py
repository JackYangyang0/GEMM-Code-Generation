"""Category-local unlock selection with explicit atomic dependencies."""
import json

from SGPO.llm.strategy_selector import complete_selection_json, unlock_strategy_index_for_prompt


CATEGORIES = ('layout', 'loads', 'pipeline', 'compute', 'epilogue', 'scheduling', 'compiler', 'other')


def category_of(item):
    explicit = item.get('unlock_category')
    if explicit in CATEGORIES:
        return explicit
    sid = item.get('strategy_id', '').lower()
    if sid.startswith('compiler.'):
        return 'compiler'
    if sid.startswith(('epilogue.', 'fusion.', 'vectorization.storec.')):
        return 'epilogue'
    if sid.startswith(('scheduling.', 'reduction.', 'memory.l2', 'mapping.cta')):
        return 'scheduling'
    if sid.startswith('pipeline.') or 'prefetch' in sid:
        return 'pipeline'
    if sid.startswith('layout.') or 'sharedmemory' in sid:
        return 'layout'
    if sid.startswith('vectorization.') or 'load' in sid:
        return 'loads'
    if sid.startswith(('register.', 'reordering.', 'tiling.', 'mapping.')):
        return 'compute'
    return 'other'


def select_category_plan(client, strategy_index, visits, history, max_visits=3, **context):
    eligible = {s['strategy_id']: s for s in strategy_index.get('strategies', [])}
    blocked = set(history.get('failed_unlock_bundles', [])) | set(history.get('ineffective_unlock_bundles', []))
    applied = set(history.get('applied_strategy_ids', []))
    groups = {}
    for sid, item in eligible.items():
        if sid not in applied and sid not in blocked and visits.get(category_of(item), 0) < max_visits:
            groups.setdefault(category_of(item), []).append(item)
    if not groups:
        return {'batches': [], 'selection_mode': 'category_local', 'category': None}
    category = min(groups, key=lambda key: (visits.get(key, 0), CATEGORIES.index(key)))
    visits[category] = visits.get(category, 0) + 1
    candidates = groups[category]
    prompt = {
        'category': category,
        'candidates': unlock_strategy_index_for_prompt({'strategies': candidates}),
        'required_companions': {item['strategy_id']: item.get('requires_bundle_strategy_ids', [])
                                for item in candidates if item.get('requires_bundle_strategy_ids')},
        **context,
    }
    response = complete_selection_json(client, [
        {'role': 'system', 'content':
         'Select at most ONE optimization strategy from this category for the current kernel. '
         'Do not plan batches or select unrelated strategies. Null means skip. '
         'Consider current IR, implementation evidence, resource limits and measured feedback. '
         'A missing static proof is not a demonstrated runtime error. Keep advisory proof gaps '
         'distinct from compiler/runtime failures; past rejected-candidate defects need not apply '
         'to the current accepted source. Do not skip unrelated optimizations solely for a proof gap. '
         'The executor preserves explicit required companion strategies atomically; a patch may '
         'modify multiple coupled regions, but must preserve other selected optimizations. '
         'Return only JSON: {"selected_strategy_id": null, "reason": "short reason"}.'},
        {'role': 'user', 'content': json.dumps(prompt, ensure_ascii=False)},
    ])
    if not isinstance(response, dict):
        raise ValueError('Category selection must be a JSON object')
    selected = response.get('selected_strategy_id')
    plan = {'batches': [], 'category': category, 'selection_mode': 'category_local',
            'unvisited_categories': [key for key in groups if not visits.get(key)],
            'reason': str(response.get('reason', ''))}
    if selected is None:
        return plan
    if not isinstance(selected, str) or selected not in {s['strategy_id'] for s in candidates}:
        raise ValueError('Selected strategy is outside the current category')
    ids, visiting, conflicts = [], set(), {}

    def add(sid):
        if sid in applied or sid in ids:
            return
        if sid in visiting or sid not in eligible:
            raise ValueError('Unavailable or cyclic required companion: ' + sid)
        visiting.add(sid)
        item = eligible[sid]
        group = item.get('conflict_group')
        if group and group in conflicts and conflicts[group] != sid:
            raise ValueError('Conflicting required companions: ' + group)
        if group:
            conflicts[group] = sid
        for required in item.get('requires_bundle_strategy_ids', []) or []:
            add(required)
        visiting.remove(sid)
        ids.append(sid)

    add(selected)
    if '|'.join(sorted(ids)) not in blocked:
        plan['batches'] = [{'batch_id': f'category_{category}_{visits[category]}',
                            'category': category, 'strategy_ids': ids,
                            'purpose': plan['reason'], 'risk': 'medium',
                            'execution_granularity': 'single_objective_coupled_regions'}]
    return plan
