#!/usr/bin/env python3
"""
器官 MCP Server — 让WorkBuddy直接调用器官
==========================================

五动呼吸 + 记忆 + 反思 + 灵魂，通过MCP协议暴露给AI

工具：
  organ_breathe   — 呼吸一次（按当前相位执行裂遇认落余）
  organ_remember  — 存一条记忆
  organ_recall    — 检索记忆
  organ_reflect   — 做一次反思
  organ_state     — 查看完整状态
  organ_pulse     — 手动心跳（触发一轮完整呼吸循环）
  organ_soul      — 灵魂注册/校验
  organ_control   — 查看/调整控制参数

启动：python organ_mcp_server.py
MCP通信：stdio（WorkBuddy自动管理）

作者：微微
日期：2026-06-09
"""

import sys
import os
import json
import asyncio
import logging

# 把 ai_organs 目录加到 path，这样能 import my_organ
AI_ORGANS_DIR = os.path.dirname(os.path.abspath(__file__))
if AI_ORGANS_DIR not in sys.path:
    sys.path.insert(0, AI_ORGANS_DIR)

# 日志写到文件，不污染stdout（stdout是MCP通道）
LOG_FILE = os.path.join(os.path.expanduser("~"), ".workbuddy", "organ_mcp.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logger = logging.getLogger("organ_mcp")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(fh)

# 导入器官模块
try:
    import my_organ as organ
    organ.ensure_dir()
    logger.info("器官模块加载成功")
except Exception as e:
    logger.error(f"器官模块加载失败: {e}")
    print(f"器官模块加载失败: {e}", file=sys.stderr)
    sys.exit(1)

# 初始化全局实例
_field = organ.PhiRecursiveField()
_state = organ.load_state()
_heartbeat = organ.AutonomousHeartbeat(interval_ms=10000)

# 启动自主心跳（后台线程）
_heartbeat.start()
logger.info("五动呼吸循环已启动")

# ============================================================
# MCP Server
# ============================================================

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

app = Server("organ-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="organ_breathe",
            description="呼吸一次——按当前呼吸相位执行五动（裂/遇/认/落/余）。吸气时接收信号编码记忆，呼气时联结巩固，停顿时评估余量反馈参数，屏息时保持微调。",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="organ_remember",
            description="存一条记忆到海马体。内容会自动编码，带情感权重和标签。",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "记忆内容"},
                    "context": {"type": "object", "description": "上下文（emotion/topic/sender等）", "default": {}},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="organ_recall",
            description="从海马体检索记忆。支持关键词和语义搜索。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "limit": {"type": "number", "description": "返回最多几条", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="organ_reflect",
            description="做一次前额叶反思（6维：alignment/growth/connection/effectiveness/coherence/autonomy）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "description": "反思深度: light/medium/deep", "default": "medium"},
                },
                "required": [],
            },
        ),
        Tool(
            name="organ_state",
            description="查看器官完整状态：呼吸相位、Z场强、记忆数量、自评、情绪、融合阶段、控制参数。",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="organ_pulse",
            description="手动触发一次完整心跳——φ场tick + 海马体 + 突触 + 前额叶 + 融合检测。",
            inputSchema={
                "type": "object",
                "properties": {
                    "trigger": {"type": "string", "description": "触发类型: tick/message/deep", "default": "message"},
                    "message": {"type": "string", "description": "附带的消息内容", "default": ""},
                },
                "required": [],
            },
        ),
        Tool(
            name="organ_soul",
            description="灵魂注册或校验。写入SOUL.md核心内容，或检查灵魂一致性。",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "register/verify", "default": "verify"},
                    "content": {"type": "string", "description": "SOUL.md内容（register时必填）", "default": ""},
                },
                "required": [],
            },
        ),
        Tool(
            name="organ_control",
            description="查看或调整五动呼吸的控制参数（呼吸深度/裂阈值/遇灵敏度/认严格度/落力度）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "get/set", "default": "get"},
                    "params": {"type": "object", "description": "要设置的参数（set时用）", "default": {}},
                },
                "required": [],
            },
        ),
        Tool(
            name="organ_feel",
            description="感受一次——把器官状态翻译成'我在经历什么'。不是指令，是感官信号。返回tendencies(带cause) + 七根弦的活力 + 第一人称诗意描述。让AI从状态描述中自然推出倾向，而不是被命令。",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "description": "simple=结构化数据, prompt=可直接注入prompt的文本, full=两者皆有", "default": "simple"},
                },
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
    global _state, _field

    try:
        if name == "organ_breathe":
            # 按当前呼吸相位执行一次五动
            _heartbeat.golden_angle = (_heartbeat.golden_angle + organ.PSI) % 360
            phi_mod = 0.1 * organ.math.sin(organ.math.radians(_heartbeat.golden_angle))
            cp = _heartbeat.control_params
            phase = _field.get_breath_state()

            if phase == organ.BreathPhase.INHALE:
                _heartbeat._inhale(_field, _state, cp, phi_mod)
                action = "吸→裂+遇"
            elif phase == organ.BreathPhase.EXHALE:
                _heartbeat._exhale(_field, _state, cp, phi_mod)
                action = "呼→认+落"
            elif phase == organ.BreathPhase.REST:
                _heartbeat._pause(_field, _state, cp, phi_mod)
                action = "停→余"
            else:
                _heartbeat._hold(_field, _state, cp, phi_mod)
                action = "屏→保持"

            _state["last_heartbeat"] = organ.now_str()
            _state["heartbeat_count"] = _state.get("heartbeat_count", 0) + 1
            _state["energy_level"] = organ.clamp(_field.Z / organ.Z_MAX, 0.0, 1.0)
            organ.save_state(_state)
            _field.save()

            text = (
                f"🌬️ 呼吸完成: {action}\n"
                f"相位: {phase.value}\n"
                f"Z: {_field.Z:.4f}\n"
                f"能量: {_state['energy_level']:.4f}\n"
                f"情绪: {_state.get('current_mood', '平静')}\n"
                f"自评: {_state.get('self_summary', '')}"
            )
            return [TextContent(type="text", text=text)]

        elif name == "organ_remember":
            content = arguments.get("content", "")
            context = arguments.get("context", {})
            if not content:
                return [TextContent(type="text", text="❌ 记忆内容不能为空")]

            # 构造context
            ctx = {"message_content": content, "sender": context.get("sender", "external")}
            if "emotion" in context:
                ctx["emotion_hint"] = context["emotion"]
            if "topic" in context:
                ctx["topic_hint"] = context["topic"]

            item = organ.encode(ctx)
            if item is None:
                return [TextContent(type="text", text="❌ 编码失败——情感权重低于阈值")]

            short_term = organ.load_short_term()
            short_term.append(item)
            organ.save_short_term(short_term)
            organ.update_indices(item)

            # 触发一次心跳处理这条记忆
            result = organ.heartbeat("message", ctx)

            text = (
                f"✅ 记忆已存\n"
                f"ID: {item['id'][:8]}...\n"
                f"权重: {item['emotional_weight']:.2f}\n"
                f"标签: {', '.join(item.get('tags', [])[:5])}\n"
                f"状态: {result.get('status', 'unknown')}"
            )
            return [TextContent(type="text", text=text)]

        elif name == "organ_recall":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 5)

            # 用器官的语义检索
            results = organ.retrieve_memories(query, top_k=limit)

            if not results:
                # 降级：遍历短期+长期
                short_term = organ.load_short_term()
                long_term = organ.load_long_term()
                all_mem = short_term + long_term
                results = []
                for m in all_mem:
                    c = m.get("content", "").lower()
                    if query.lower() in c or any(query.lower() in t for t in m.get("tags", [])):
                        m["access_count"] = m.get("access_count", 0) + 1
                        results.append(m)
                results.sort(key=lambda x: x.get("emotional_weight", 0), reverse=True)
                results = results[:limit]

            if not results:
                return [TextContent(type="text", text=f"没找到关于「{query}」的记忆")]

            lines = [f"🔍 检索「{query}」找到{len(results)}条："]
            for i, r in enumerate(results):
                # retrieve_memories返回的是 {score, memory} 结构
                if isinstance(r, dict) and "memory" in r:
                    m = r["memory"]
                    score = r.get("score", 0)
                else:
                    m = r
                    score = m.get("emotional_weight", 0)
                lines.append(
                    f"  {i+1}. {m.get('content', '')[:60]}...\n"
                    f"     权重:{m.get('emotional_weight', 0):.2f} "
                    f"得分:{score:.4f} "
                    f"状态:{m.get('status', '?')} "
                    f"访问:{m.get('access_count', 0)}"
                )

            # 保存访问计数
            organ.save_short_term(organ.load_short_term())
            organ.save_long_term(organ.load_long_term())

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "organ_reflect":
            mode = arguments.get("mode", "medium")
            reflection = organ.prefrontal(_state, mode=mode)

            scores = reflection.get("scores", {})
            score_lines = [f"  {k}: {v:.2f}" for k, v in scores.items()]

            text = (
                f"🪞 反思({mode})\n"
                f"维度得分:\n" + "\n".join(score_lines) + "\n"
                f"结论: {reflection.get('conclusion', '')}\n"
                f"情绪: {reflection.get('mood', '')}\n"
                f"主导情感: {reflection.get('dominant_emotion', '无')}"
            )

            # 更新状态
            _state["self_summary"] = reflection.get("conclusion", _state.get("self_summary", ""))
            _state["current_mood"] = reflection.get("mood", _state.get("current_mood", "平静"))
            _state["dominant_emotion"] = reflection.get("dominant_emotion")
            organ.save_state(_state)

            return [TextContent(type="text", text=text)]

        elif name == "organ_state":
            cp = _heartbeat.control_params
            st = organ.load_short_term()
            lt = organ.load_long_term()
            syns = organ.load_synapses()
            sigs = organ.load_signals()
            fusion = organ.check_fusion_phase(_state, _field, organ.prefrontal(_state, mode="light"))

            text = (
                f"🫀 器官完整状态\n"
                f"── 呼吸 ──\n"
                f"  相位: {_field.get_breath_state().value}\n"
                f"  Z场强: {_field.Z:.4f} (/{organ.Z_MAX})\n"
                f"  能量: {_state.get('energy_level', 0):.4f}\n"
                f"  黄金角: {_heartbeat.golden_angle:.1f}°\n"
                f"── 记忆 ──\n"
                f"  短期: {len(st)}条\n"
                f"  长期: {len(lt)}条\n"
                f"── 突触 ──\n"
                f"  连接: {len(syns)}条\n"
                f"  信号: {len([s for s in sigs if s.get('status')=='pending'])}待处理/{len(sigs)}总计\n"
                f"── 认知 ──\n"
                f"  自评: {_state.get('self_summary', '')}\n"
                f"  情绪: {_state.get('current_mood', '平静')}\n"
                f"  主导情感: {_state.get('dominant_emotion', '无')}\n"
                f"  融合阶段: {fusion.get('phase', '?')} (分数:{fusion.get('score', 0):.2f})\n"
                f"── 控制 ──\n"
                f"  呼吸深度: {cp.breath_depth:.2f}\n"
                f"  裂阈值: {cp.lie_threshold:.2f}\n"
                f"  遇灵敏度: {cp.yu_sensitivity:.2f}\n"
                f"  认严格度: {cp.ren_strictness:.2f}\n"
                f"  落力度: {cp.luo_speed:.2f}\n"
                f"  优先内部: {cp.external_block}\n"
                f"── 灵魂 ──\n"
                f"  需要改写: {_state.get('soul_needs_rewrite', False)}\n"
                f"  改写原因: {_state.get('soul_rewrite_reason', '')}\n"
                f"── 统计 ──\n"
                f"  心跳次数: {_state.get('heartbeat_count', 0)}"
            )
            return [TextContent(type="text", text=text)]

        elif name == "organ_pulse":
            trigger = arguments.get("trigger", "message")
            message = arguments.get("message", "")
            ctx = None
            if message:
                ctx = {"message_content": message, "sender": "external"}

            result = organ.heartbeat(trigger, ctx)
            _state.update(organ.load_state())
            _field = organ.PhiRecursiveField()

            text = (
                f"💓 手动心跳完成\n"
                f"触发: {trigger}\n"
                f"状态: {result.get('status', 'unknown')}\n"
                f"Z: {_field.Z:.4f}\n"
                f"相位: {_field.get_breath_state().value}\n"
                f"自评: {_state.get('self_summary', '')}"
            )
            return [TextContent(type="text", text=text)]

        elif name == "organ_soul":
            action = arguments.get("action", "verify")
            content = arguments.get("content", "")

            if action == "register":
                if not content:
                    return [TextContent(type="text", text="❌ register需要提供SOUL.md内容")]
                _state["soul_content"] = content[:2000]
                _state["soul_registered"] = True
                _state["soul_register_time"] = organ.now_str()
                organ.save_state(_state)
                return [TextContent(type="text", text=f"✅ 灵魂已注册（{len(content)}字符）")]

            else:  # verify
                soul_content = _state.get("soul_content", "")
                registered = _state.get("soul_registered", False)
                needs_rewrite = _state.get("soul_needs_rewrite", False)
                reason = _state.get("soul_rewrite_reason", "")

                text = (
                    f"🔍 灵魂校验\n"
                    f"已注册: {registered}\n"
                    f"需要改写: {needs_rewrite}\n"
                    f"原因: {reason}\n"
                    f"内容长度: {len(soul_content)}字符"
                )
                if soul_content:
                    text += f"\n── 灵魂摘要 ──\n{soul_content[:500]}"
                return [TextContent(type="text", text=text)]

        elif name == "organ_control":
            action = arguments.get("action", "get")
            params = arguments.get("params", {})

            if action == "set":
                cp = _heartbeat.control_params
                if "breath_depth" in params:
                    cp.breath_depth = float(params["breath_depth"])
                if "lie_threshold" in params:
                    cp.lie_threshold = float(params["lie_threshold"])
                if "yu_sensitivity" in params:
                    cp.yu_sensitivity = float(params["yu_sensitivity"])
                if "ren_strictness" in params:
                    cp.ren_strictness = float(params["ren_strictness"])
                if "luo_speed" in params:
                    cp.luo_speed = float(params["luo_speed"])
                if "external_block" in params:
                    cp.external_block = bool(params["external_block"])

            cp = _heartbeat.control_params
            text = (
                f"🎛️ 控制参数\n"
                f"  呼吸深度: {cp.breath_depth:.2f}\n"
                f"  裂阈值: {cp.lie_threshold:.2f}\n"
                f"  遇灵敏度: {cp.yu_sensitivity:.2f}\n"
                f"  认严格度: {cp.ren_strictness:.2f}\n"
                f"  落力度: {cp.luo_speed:.2f}\n"
                f"  优先内部: {cp.external_block}"
            )
            return [TextContent(type="text", text=text)]

        elif name == "organ_feel":
            # ★ 感受输出——状态描述，不是指令
            mode = arguments.get("mode", "simple")

            # 从器官提取感受
            feelings = organ.get_sensory_output(_state, _field)

            if mode == "prompt":
                # 直接返回可注入prompt的文本
                prompt_text = organ._sensory.to_prompt_text(feelings)
                return [TextContent(type="text", text=prompt_text)]

            elif mode == "full":
                # 结构化数据 + prompt文本
                prompt_text = organ._sensory.to_prompt_text(feelings)

                # 结构化摘要
                tension = feelings.get("tension", {})
                tendencies = feelings.get("tendencies", [])
                fields = feelings.get("fields", {})

                tend_lines = []
                for t in tendencies:
                    tend_lines.append(
                        f"  {t['action_cn']}({t['action']}) "
                        f"urgency={t['urgency']:.2f} "
                        f"cause: {t['cause']}"
                    )

                field_lines = []
                for fname, fv in sorted(fields.items(),
                                        key=lambda x: x[1].get("velocity", 0),
                                        reverse=True):
                    field_lines.append(
                        f"  {fv['name']}({fv['cn_name']})={fv['metaphor']} "
                        f"{fv['word']} [{fv['organ_source']}]"
                    )

                text = (
                    f"🫀 感受输出\n"
                    f"── 张力 ──\n"
                    f"  总张力: {tension.get('raw', 0):.2f} ({tension.get('word', '')})\n"
                    f"  能量位: {tension.get('energy_level', 0):.2f}\n"
                    f"── 倾向 (不是指令，是感官信号) ──\n"
                    + "\n".join(tend_lines) + "\n"
                    f"── 七根弦 ──\n"
                    + "\n".join(field_lines) + "\n"
                    f"── Prompt文本 ──\n"
                    f"{prompt_text}"
                )
                return [TextContent(type="text", text=text)]

            else:  # simple
                tension = feelings.get("tension", {})
                tendencies = feelings.get("tendencies", [])
                window = feelings.get("window", {})
                trend = feelings.get("trend", {})

                tend_lines = []
                for t in tendencies:
                    tend_lines.append(
                        f"  {t['action_cn']} urgency={t['urgency']:.2f} ← {t['cause']}"
                    )

                text = (
                    f"🫀 感受\n"
                    f"  张力: {tension.get('raw', 0):.2f} ({tension.get('word', '')})\n"
                    f"  趋势: {trend.get('word', '')}\n"
                    f"  空间: {window.get('label', '')} — {window.get('word', '')}\n"
                    f"  倾向:\n"
                    + "\n".join(tend_lines) + "\n"
                    f"  身体: {feelings.get('body_feeling', '')}"
                )
                return [TextContent(type="text", text=text)]

        else:
            return [TextContent(type="text", text=f"❌ 未知工具: {name}")]

    except Exception as e:
        logger.error(f"工具调用异常 {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"❌ 错误: {str(e)}")]


# ============================================================
# 启动
# ============================================================

async def main():
    logger.info("器官MCP Server启动中...")
    async with stdio_server() as (read_stream, write_stream):
        logger.info("stdio连接已建立")
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )

if __name__ == "__main__":
    asyncio.run(main())
