"""Keep execution failures separate from optimization evidence."""
import copy


INFRASTRUCTURE_ERRORS = frozenset({
    'KeyError', 'NameError', 'AttributeError', 'TypeError', 'PatchContractError',
    'APITimeoutError', 'APIConnectionError', 'AuthenticationError', 'RateLimitError',
})


def infrastructure_failure(summary):
    if not isinstance(summary, dict):
        return False
    if summary.get('failure_class') == 'infrastructure_error':
        return True
    if summary.get('error_type') in INFRASTRUCTURE_ERRORS:
        return True
    return any(infrastructure_failure(summary.get(k)) for k in ('last_verification', 'verification'))


def choose_unlock_steps(pending, planned, eligible_ids, history, limit):
    """FIFO carryover precedes new proposals; eligibility is rechecked each round."""
    blocked = set(history.get('failed_unlock_bundles', [])) | set(history.get('ineffective_unlock_bundles', []))
    applied = set(history.get('applied_strategy_ids', []))
    queue, rejected, seen = [], [], set()
    for step in [*pending, *planned]:
        ids = step.get('strategy_ids') or []
        key = '|'.join(sorted(set(ids)))
        if not ids or key in seen:
            continue
        seen.add(key)
        if key in blocked or any(sid in applied or sid not in eligible_ids for sid in ids):
            rejected.append({'strategy_ids': ids, 'reason': 'not eligible under current IR/history'})
            continue
        queue.append(copy.deepcopy(step))
    return queue[:limit], queue[limit:], rejected
