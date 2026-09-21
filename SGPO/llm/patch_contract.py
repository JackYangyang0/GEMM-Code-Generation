"""Normalize partial code-edit contracts without granting arbitrary IR writes."""
import copy
import re


class PatchContractError(ValueError):
    """A local strategy contract is malformed (not an LLM implementation defect)."""


def declared_ir_fields(strategy):
    fields = set((strategy.get('ir_updates') or {}).keys())
    for key in ('provides_fields', 'affected_fields', 'allowed_ir_fields'):
        fields.update(strategy.get(key) or [])
    if not fields:
        post = strategy.get('postconditions') or []
        if isinstance(post, dict):
            post = post.get('patch_ir_verification', {})

        def visit(node):
            if isinstance(node, dict):
                lhs = node.get('lhs')
                if isinstance(lhs, dict) and isinstance(lhs.get('field'), str):
                    fields.add(lhs['field'])
                for key in ('predicates', 'conditions', 'all', 'any', 'not'):
                    visit(node.get(key))
            elif isinstance(node, list):
                for item in node:
                    visit(item)
            elif isinstance(node, str):
                match = re.match(r'^\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*(?:!=|==|>=|<=|>|<)', node)
                if match:
                    fields.add(match[1])
        visit(post)
    # Postconditions can mention inputs; those are never implicit write grants.
    return sorted(f for f in fields if isinstance(f, str)
                  and not f.startswith(('hardware.', 'problem.', 'verification.', 'performance.')))


def normalize_patch_contract(strategy):
    strategy = copy.deepcopy(strategy)
    contract = strategy.get('patch_contract')
    if not contract:
        return strategy
    if not isinstance(contract, dict):
        raise PatchContractError('patch_contract must be an object')
    if 'allowed_ir_fields' not in contract:
        contract['allowed_ir_fields'] = declared_ir_fields(strategy)
        contract['ir_scope_source'] = 'strategy_declared_outputs'
    for key in ('allowed_ir_fields', 'allowed_regions'):
        values = contract.get(key)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
            raise PatchContractError(f'patch_contract.{key} must be a list of non-empty strings')
    return strategy
