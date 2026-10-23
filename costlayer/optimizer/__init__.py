"""Analysis and learning from SANITIZED traces only.

This package must not import tau2: it sees messages, tool calls, tool results, token usage and
the final reward (plus tool_kinds.json), never the task data or its gold actions.
"""
