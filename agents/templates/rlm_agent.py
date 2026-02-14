import ast
import contextlib
import io
import json
import logging
import os
from collections import defaultdict
from hashlib import blake2b
from typing import Any, Optional

import openai
from arcengine import FrameData, GameAction, GameState
from openai import OpenAI as OpenAIClient

from .llm_agents import ReasoningLLM

logger = logging.getLogger(__name__)


class RLM(ReasoningLLM):
    """Recursive language-model scaffold for ARC-AGI-3.

    - External memory (facts, transitions, per-state action stats)
    - Query tools (peek_window / python_repl / store_fact / call_subproblem)
    - Root must choose exactly ONE game action tool per turn (ARC convention)
    """

    MAX_ACTIONS = 120
    DO_OBSERVATION = True
    MODEL = "gpt-5-mini"
    MODEL_REQUIRES_TOOLS = True
    REASONING_EFFORT: Optional[str] = None

    INTERNAL_STEPS = 6
    SUB_STEPS = 4
    MAX_DEPTH = 3
    MAX_FACTS = 64
    MAX_TRANSITIONS = 64
    MAX_SUBPROBLEMS = 32
    MAX_STATES = 256
    GRID_SAMPLES = 2
    HIST_TOP_K = 8
    WINDOW_DEFAULT = 12

    _EMPTY_PARAMS: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))

        self.facts: list[dict[str, Any]] = []
        self.transitions: list[dict[str, Any]] = []
        self.subproblems: list[dict[str, Any]] = []
        self.sent_actions: list[str] = []

        self.by_state: dict[str, dict[str, dict[str, float]]] = {}
        self.state_visits: dict[str, int] = defaultdict(int)
        self.state_key: str = ""

        self.ctx: dict[str, Any] = {"globals": {}, "runs": 0}

    @property
    def name(self) -> str:
        sanitized_model_name = self.MODEL.replace("/", "-").replace(":", "-")
        return f"{super().name}.{sanitized_model_name}.rlm"

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if getattr(latest_frame, "full_reset", False) or latest_frame.state in (
            GameState.NOT_PLAYED,
            GameState.GAME_OVER,
        ):
            self._reset_memory()
            action = GameAction.RESET
            action.reasoning = {"agent": "RLM", "mode": "reset", "turn": self.action_counter}
            self._record_action(action)
            return action

        self._ingest_transition(frames, latest_frame)

        grid = self._grid(latest_frame)
        self.state_key = self._hash_grid(grid)
        if self.state_key:
            self.state_visits[self.state_key] += 1

        result = self._solve(
            latest_frame=latest_frame,
            objective="Choose the next game action.",
            focus="recent_transition",
            depth=0,
            allow_action=True,
        )

        action = result.get("action") or self._fallback_action(latest_frame)
        forced = result.get("action") is None

        if action.name == GameAction.RESET.name:
            self._reset_memory()

        action.reasoning = {
            "agent": "RLM",
            "model": self.MODEL,
            "turn": self.action_counter,
            "forced": forced,
            "state": latest_frame.state.name,
            "levels_completed": int(latest_frame.levels_completed),
            "state_key": self.state_key,
            "facts": len(self.facts),
            "transitions": len(self.transitions),
            "subproblems": len(self.subproblems),
            "trace": (result.get("trace") or [])[-8:],
        }
        self._record_action(action)
        return action

    def _solve(
        self,
        latest_frame: FrameData,
        objective: str,
        focus: str,
        depth: int,
        allow_action: bool,
        x: Any = None,
        y: Any = None,
        size: Any = None,
    ) -> dict[str, Any]:
        if depth > self.MAX_DEPTH:
            return {"status": "depth_limit", "objective": objective, "confidence": 0.0, "trace": []}

        tools = self._query_tools(include_return_insight=not allow_action)
        if allow_action:
            tools += self._action_tools(latest_frame)

        messages = [
            {"role": "system", "content": self._system_prompt(depth, allow_action)},
            {"role": "user", "content": self._user_payload(latest_frame, objective, focus, depth, allow_action, x, y, size)},
        ]

        trace: list[dict[str, Any]] = []
        steps = self.INTERNAL_STEPS if allow_action else max(1, self.SUB_STEPS - max(0, depth - 1))

        for _ in range(steps):
            msg = self._chat(messages, tools, tool_required=True)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                break

            messages.append({"role": "assistant", "tool_calls": tool_calls})

            action_call: Optional[dict[str, Any]] = None
            for tc in tool_calls:
                fn = (tc.get("function") or {})
                name = str(fn.get("name", ""))
                args = self._json_obj(fn.get("arguments"))
                tc_id = str(tc.get("id", "call_0"))

                if allow_action and self._is_game_action(name):
                    if action_call is None:
                        action_call = tc
                        messages.append({"role": "tool", "tool_call_id": tc_id, "content": json.dumps({"acknowledged": True})})
                    else:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": "Error: assistant can only call one action (tool) at a time. default to only the first chosen action.",
                            }
                        )
                    continue

                if name == "return_insight":
                    insight = str(args.get("insight", "")).strip() or "No insight."
                    evidence = str(args.get("evidence", "")).strip()
                    conf = self._safe_float(args.get("confidence"), 0.5)
                    self._remember_fact("subproblem_insight", insight, conf)
                    self._remember_subproblem(objective, depth, "insight", conf)
                    trace.append({"depth": depth, "type": "return_insight", "confidence": conf})
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": json.dumps({"stored": True})})
                    return {
                        "status": "insight",
                        "depth": depth,
                        "objective": objective,
                        "insight": insight,
                        "evidence": evidence,
                        "confidence": conf,
                        "trace": trace,
                    }

                if name == "peek_window":
                    payload = self._peek(args, latest_frame)
                    trace.append({"depth": depth, "type": "peek_window"})
                elif name == "python_repl":
                    payload = self._python(args, latest_frame)
                    trace.append({"depth": depth, "type": "python_repl"})
                elif name == "store_fact":
                    payload = self._store_fact(args)
                    trace.append({"depth": depth, "type": "store_fact"})
                elif name == "call_subproblem":
                    payload = self._solve(
                        latest_frame=latest_frame,
                        objective=str(args.get("objective", objective)),
                        focus=str(args.get("focus", focus)),
                        depth=depth + 1,
                        allow_action=False,
                        x=args.get("x"),
                        y=args.get("y"),
                        size=args.get("size"),
                    )
                    self._remember_subproblem(
                        objective=str(args.get("objective", objective)),
                        depth=depth + 1,
                        status=str(payload.get("status", "")),
                        confidence=self._safe_float(payload.get("confidence"), 0.2),
                    )
                    trace.append({"depth": depth, "type": "subproblem", "status": payload.get("status")})
                else:
                    payload = {"error": f"Unknown tool {name}"}
                    trace.append({"depth": depth, "type": "unknown_tool", "name": name})

                messages.append({"role": "tool", "tool_call_id": tc_id, "content": json.dumps(payload)})

            if action_call is not None:
                fn = (action_call.get("function") or {})
                name = str(fn.get("name", ""))
                args = self._json_obj(fn.get("arguments"))
                action = self._action_from_tool(name, args)
                trace.append({"depth": depth, "type": "action", "name": name, "args": args})
                return {"status": "action", "depth": depth, "objective": objective, "action": action, "trace": trace}

        return {"status": "no_action" if allow_action else "insight", "depth": depth, "objective": objective, "action": None, "confidence": 0.2, "trace": trace}

    def _system_prompt(self, depth: int, allow_action: bool) -> str:
        if allow_action:
            return (
                "You are the root controller of a recursive tool-using agent playing a grid puzzle game.\n"
                "IMPORTANT: Call exactly ONE available action tool per turn.\n"
                "You may use peek_window/python_repl/store_fact/call_subproblem to inspect before acting."
            )
        return (
            f"You are solving a bounded recursive subproblem (depth {depth}/{self.MAX_DEPTH}). "
            "Use tools to inspect/compute, then call return_insight."
        )

    def _user_payload(
        self,
        latest_frame: FrameData,
        objective: str,
        focus: str,
        depth: int,
        allow_action: bool,
        x: Any,
        y: Any,
        size: Any,
    ) -> str:
        payload: dict[str, Any] = {
            "task": {"objective": objective, "focus": focus, "depth": depth, "mode": "root" if allow_action else "subproblem"},
            "frame": self._frame_summary(latest_frame, include_samples=not allow_action),
            "memory": self._memory_snapshot(),
            "latest_transition": self.transitions[-1] if self.transitions else {},
            "budgets": {"internal_steps": self.INTERNAL_STEPS, "subproblem_steps": self.SUB_STEPS, "max_depth": self.MAX_DEPTH},
        }
        if x is not None and y is not None and size is not None:
            payload["task"]["window"] = {
                "x": self._safe_int(x, 0, 0, 63),
                "y": self._safe_int(y, 0, 0, 63),
                "size": self._safe_int(size, self.WINDOW_DEFAULT, 2, 32),
            }
        return json.dumps(payload, indent=2)

    def _tool(self, name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters, "strict": True}}

    def _query_tools(self, include_return_insight: bool) -> list[dict[str, Any]]:
        tools = [
            self._tool(
                "peek_window",
                "Inspect a small window from the latest planning grid.",
                {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "size": {"type": "integer"}},
                    "required": ["x", "y", "size"],
                    "additionalProperties": False,
                },
            ),
            self._tool(
                "python_repl",
                "Execute short Python over persistent ctx, frame, and memory views.",
                {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            ),
            self._tool(
                "call_subproblem",
                "Recursively solve a focused subproblem.",
                {
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string"},
                        "focus": {"type": "string", "enum": ["full_grid", "window", "recent_transition", "hypothesis_check"]},
                        "x": {"type": ["integer", "null"]},
                        "y": {"type": ["integer", "null"]},
                        "size": {"type": ["integer", "null"]},
                    },
                    "required": ["objective", "focus", "x", "y", "size"],
                    "additionalProperties": False,
                },
            ),
            self._tool(
                "store_fact",
                "Persist a durable observation in external memory.",
                {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "fact": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["category", "fact", "confidence"],
                    "additionalProperties": False,
                },
            ),
        ]
        if include_return_insight:
            tools.append(
                self._tool(
                    "return_insight",
                    "Return a concrete conclusion for this subproblem.",
                    {
                        "type": "object",
                        "properties": {"insight": {"type": "string"}, "evidence": {"type": "string"}, "confidence": {"type": "number"}},
                        "required": ["insight", "evidence", "confidence"],
                        "additionalProperties": False,
                    },
                )
            )
        return tools

    def _available_actions(self, latest_frame: FrameData) -> list[str]:
        """Return action names from the engine, falling back to ACTION1-4."""
        names: list[str] = []
        seen: set[str] = set()
        for row in list(getattr(latest_frame, "available_actions", []) or []):
            name: Optional[str] = None
            if isinstance(row, int):
                try:
                    name = GameAction.from_id(int(row)).name
                except ValueError:
                    pass
            elif isinstance(row, str):
                candidate = row.strip().upper()
                if self._is_game_action(candidate):
                    name = candidate
            elif hasattr(row, "id"):
                try:
                    raw = getattr(row, "id")
                    rid = int(raw.value) if hasattr(raw, "value") else int(raw)
                    name = GameAction.from_id(rid).name
                except Exception:
                    pass
            if name and name != GameAction.RESET.name and name not in seen:
                seen.add(name)
                names.append(name)
        return names or [GameAction.ACTION1.name, GameAction.ACTION2.name,
                         GameAction.ACTION3.name, GameAction.ACTION4.name]

    def _action_tools(self, latest_frame: FrameData) -> list[dict[str, Any]]:
        """Build action tools with tested/untested hints for the current state."""
        available = self._available_actions(latest_frame)
        tested = set()
        if self.state_key and self.state_key in self.by_state:
            tested = set(self.by_state[self.state_key].keys())

        tools: list[dict[str, Any]] = []
        for name in available:
            status = "UNTESTED in current state" if name not in tested else "already tried"
            desc = f"Emit game action {name} ({status}). One of {len(available)} available actions."
            params: dict[str, Any] = self._EMPTY_PARAMS
            if name == GameAction.ACTION6.name:
                params = {
                    "type": "object",
                    "properties": {
                        "x": {"type": "string", "description": "Int<0,63>"},
                        "y": {"type": "string", "description": "Int<0,63>"},
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                }
            tools.append(self._tool(name, desc, params))
        return tools


    def _peek(self, args: dict[str, Any], latest_frame: FrameData) -> dict[str, Any]:
        grid = self._grid(latest_frame)
        if not grid:
            return {"error": "empty_grid"}

        h, w = len(grid), len(grid[0]) if grid else 0
        size = self._safe_int(args.get("size"), self.WINDOW_DEFAULT, 2, 32)

        x = self._safe_int(args.get("x"), 0, 0, max(0, w - 1))
        y = self._safe_int(args.get("y"), 0, 0, max(0, h - 1))
        x1, y1 = min(w, x + size), min(h, y + size)

        window = [[int(grid[yy][xx]) for xx in range(x, x1)] for yy in range(y, y1)]
        return {"x": x, "y": y, "size": size, "shape": [len(window), len(window[0]) if window else 0], "window": window}

    def _store_fact(self, args: dict[str, Any]) -> dict[str, Any]:
        cat = str(args.get("category", "observation"))[:64]
        fact = str(args.get("fact", "")).strip()
        conf = self._safe_float(args.get("confidence"), 0.5)
        if not fact:
            return {"stored": False, "error": "empty_fact"}
        self._remember_fact(cat, fact, conf)
        return {"stored": True, "category": cat, "confidence": conf, "facts_total": len(self.facts)}

    def _python(self, args: dict[str, Any], latest_frame: FrameData) -> dict[str, Any]:
        code = str(args.get("code", "")).strip()
        if not code:
            return {"ok": False, "error": "empty_code"}
        if not self._safe_repl(code):
            return {"ok": False, "error": "unsafe_code"}

        ctx = self.ctx.setdefault("globals", {})
        if not isinstance(ctx, dict):
            ctx = {}
            self.ctx["globals"] = ctx

        local_env: dict[str, Any] = {
            "ctx": ctx,
            "frame": self._grid(latest_frame),
            "facts": self.facts,
            "transitions": self.transitions,
            "subproblems": self.subproblems,
            "result": None,
        }

        stdout_buf = io.StringIO()
        try:
            compiled = compile(code, "<rlm_repl>", "exec")
            with contextlib.redirect_stdout(stdout_buf):
                exec(compiled, {"__builtins__": self._safe_builtins()}, local_env)
        except Exception as exc:
            logger.debug("python_repl failed", exc_info=True)
            return {"ok": False, "error": str(exc)[:240], "stdout": stdout_buf.getvalue()[:800], "ctx_keys": self._ctx_keys()}

        if isinstance(local_env.get("ctx"), dict):
            self.ctx["globals"] = local_env["ctx"]
        self.ctx["runs"] = int(self.ctx.get("runs", 0)) + 1

        return {
            "ok": True,
            "stdout": stdout_buf.getvalue()[:800],
            "result": self._trim(local_env.get("result")),
            "ctx_keys": self._ctx_keys(),
            "runs": int(self.ctx.get("runs", 0)),
        }

    def _safe_builtins(self) -> dict[str, Any]:
        return {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }

    def _safe_repl(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        blocked = {"__import__", "compile", "eval", "exec", "globals", "input", "locals", "open", "os", "subprocess", "sys", "vars"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False
            if isinstance(node, ast.Attribute) and str(node.attr).startswith("__"):
                return False
            if isinstance(node, ast.Name) and node.id in blocked:
                return False
        return True

    def _is_game_action(self, name: str) -> bool:
        n = str(name).strip().upper()
        return any(a.name == n for a in GameAction)

    def _action_from_tool(self, name: str, args: dict[str, Any]) -> GameAction:
        action = GameAction.from_name(name)
        if action == GameAction.ACTION6:
            action.set_data(
                {
                    "x": self._safe_int(args.get("x"), 31, 0, 63),
                    "y": self._safe_int(args.get("y"), 31, 0, 63),
                }
            )
        else:
            action.set_data({})
        return action

    def _fallback_action(self, latest_frame: Optional[FrameData] = None) -> GameAction:
        candidates = [GameAction.ACTION1.name, GameAction.ACTION2.name,
                      GameAction.ACTION3.name, GameAction.ACTION4.name]
        if latest_frame is not None:
            candidates = self._available_actions(latest_frame) or candidates
        tested = self.by_state.get(self.state_key, {})
        for name in candidates:
            if name not in tested:
                return self._action_from_tool(name, {})

        def score(name: str) -> float:
            row = tested.get(name, {})
            s = max(1.0, float(row.get("samples", 1.0)))
            avg_level = float(row.get("sum_level_delta", 0.0)) / s
            avg_changed = float(row.get("sum_changed", 0.0)) / s
            return (avg_level * 1000.0) + avg_changed
        return self._action_from_tool(max(candidates, key=score), {})

    def _reset_memory(self) -> None:
        self.facts = []
        self.transitions = []
        self.subproblems = []
        self.sent_actions = []
        self.by_state = {}
        self.state_visits = defaultdict(int)
        self.state_key = ""
        self.ctx = {"globals": {}, "runs": 0}

    def _record_action(self, action: GameAction) -> None:
        self.sent_actions.append(action.name)
        self.sent_actions = self.sent_actions[-128:]

    def _remember_fact(self, category: str, fact: str, confidence: float) -> None:
        self.facts.append({"category": category[:64], "fact": fact[:400], "confidence": round(self._safe_float(confidence, 0.5), 3), "turn": int(self.action_counter)})
        self.facts = self.facts[-self.MAX_FACTS :]

    def _remember_subproblem(self, objective: str, depth: int, status: str, confidence: float) -> None:
        self.subproblems.append(
            {"turn": self.action_counter, "depth": depth, "objective": str(objective)[:160], "status": status, "confidence": round(self._safe_float(confidence, 0.5), 3)}
        )
        self.subproblems = self.subproblems[-self.MAX_SUBPROBLEMS :]

    def _ingest_transition(self, frames: list[FrameData], latest_frame: FrameData) -> None:
        if len(frames) < 2:
            return
        prev = frames[-2]
        prev_grid = self._grid(prev)
        cur_grid = self._grid(latest_frame)
        if not prev_grid or not cur_grid:
            return

        action_name = self.sent_actions[-1] if self.sent_actions else "UNKNOWN"
        level_delta = int(latest_frame.levels_completed) - int(prev.levels_completed)
        diff = self._diff(prev_grid, cur_grid)

        prev_key = self._hash_grid(prev_grid)
        cur_key = self._hash_grid(cur_grid)

        self.transitions.append(
            {"turn": max(0, self.action_counter - 1), "action": action_name, "prev_state_key": prev_key, "state_key": cur_key, "level_delta": level_delta, "diff": diff}
        )
        self.transitions = self.transitions[-self.MAX_TRANSITIONS :]

        if prev_key and action_name != "UNKNOWN":
            st = self.by_state.setdefault(prev_key, {})
            row = st.setdefault(action_name, {"samples": 0.0, "sum_changed": 0.0, "sum_level_delta": 0.0, "max_level_delta": 0.0})
            row["samples"] += 1.0
            row["sum_changed"] += float(diff.get("changed_cells", 0))
            row["sum_level_delta"] += float(level_delta)
            row["max_level_delta"] = max(row["max_level_delta"], float(level_delta))

            while len(self.by_state) > self.MAX_STATES:
                self.by_state.pop(next(iter(self.by_state)))


    def _memory_snapshot(self) -> dict[str, Any]:
        tested = self.by_state.get(self.state_key, {})
        return {
            "state_key": self.state_key,
            "state_visits": int(self.state_visits.get(self.state_key, 0)),
            "facts": self.facts[-8:],
            "recent_transitions": self.transitions[-4:],
            "recent_subproblems": self.subproblems[-4:],
            "tested_actions_for_state": tested,
            "python_repl_runs": int(self.ctx.get("runs", 0)),
            "context_globals_keys": self._ctx_keys(),
        }

    def _frame_summary(self, latest_frame: FrameData, include_samples: bool) -> dict[str, Any]:
        grids = list(getattr(latest_frame, "frame", []) or [])
        return {
            "state": latest_frame.state.name,
            "levels_completed": int(latest_frame.levels_completed),
            "win_levels": int(latest_frame.win_levels),
            "available_actions": self._available_actions(latest_frame),
            "grid_count": len(grids),
            "state_key": self._hash_grid(self._grid(latest_frame)) or None,
            "grids": [self._grid_stats(g, include_samples) for g in grids[: self.GRID_SAMPLES] if isinstance(g, list)],
        }

    def _grid_stats(self, grid: list[list[int]], include_samples: bool) -> dict[str, Any]:
        if not grid:
            return {"shape": [0, 0], "unique_values": 0}
        h, w = len(grid), len(grid[0]) if grid else 0
        hist: dict[int, int] = defaultdict(int)
        for row in grid:
            for v in row:
                hist[int(v)] += 1
        top = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)[: self.HIST_TOP_K]
        out: dict[str, Any] = {"shape": [h, w], "unique_values": len(hist), "histogram_top": {str(k): int(v) for k, v in top}}
        if include_samples:
            out["sample_rows"] = [[int(v) for v in grid[i][: min(16, w)]] for i in range(min(8, h))]
        return out

    def _grid(self, frame: FrameData) -> list[list[int]]:
        grids = list(getattr(frame, "frame", []) or [])
        if grids and isinstance(grids[0], list):
            return grids[0]
        return []

    def _hash_grid(self, grid: list[list[int]]) -> str:
        if not grid:
            return ""
        d = blake2b(digest_size=12)
        for row in grid:
            d.update(bytes(int(v) & 0xFF for v in row))
        return d.hexdigest()

    def _diff(self, a: list[list[int]], b: list[list[int]]) -> dict[str, Any]:
        h = min(len(a), len(b))
        w = min(len(a[0]), len(b[0])) if h else 0
        changed = 0
        x0 = y0 = 10**9
        x1 = y1 = -1
        for y in range(h):
            for x in range(w):
                if int(a[y][x]) != int(b[y][x]):
                    changed += 1
                    x0, y0 = min(x0, x), min(y0, y)
                    x1, y1 = max(x1, x), max(y1, y)
        bbox = {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1} if changed else None
        return {"changed_cells": changed, "bbox": bbox}

    def _ctx_keys(self) -> list[str]:
        g = self.ctx.get("globals", {})
        return sorted([str(k) for k in g.keys()])[:24] if isinstance(g, dict) else []

    def _trim(self, value: Any) -> Any:
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            return value[:400]
        if isinstance(value, dict):
            out = {}
            for i, (k, v) in enumerate(value.items()):
                if i >= 16:
                    break
                out[str(k)[:64]] = self._trim(v)
            return out
        if isinstance(value, (list, tuple)):
            return [self._trim(v) for v in list(value)[:16]]
        return str(value)[:400]

    def _chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_required: bool = False,
    ) -> dict[str, Any]:
        create_kwargs: dict[str, Any] = {"model": self.MODEL, "messages": messages}
        if tools:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = "required" if tool_required else "auto"
        if self.REASONING_EFFORT is not None:
            create_kwargs["reasoning_effort"] = self.REASONING_EFFORT

        try:
            resp = self.client.chat.completions.create(**create_kwargs)
        except openai.BadRequestError as exc:
            raise RuntimeError(f"OpenAI request failed in RLM agent: {exc}") from exc

        self.capture_reasoning_from_response(resp)
        usage = getattr(resp, "usage", None)
        total_tokens = int(getattr(usage, "total_tokens", 0)) if usage else 0
        content = resp.choices[0].message.content or ""
        self.track_tokens(total_tokens, content)
        return resp.choices[0].message.model_dump(exclude_none=True)

    def _json_obj(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str) or not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _safe_int(self, value: Any, default: int, lo: int, hi: int) -> int:
        try:
            n = int(value)
        except (TypeError, ValueError):
            n = default
        return max(lo, min(hi, n))

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            n = float(value)
        except (TypeError, ValueError):
            n = default
        return max(0.0, min(1.0, n))
