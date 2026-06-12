import os
import time
import json
import subprocess
import requests
from datetime import datetime
from openai import OpenAI
from urllib.parse import urlparse

# ==========================================
# 配置层 — 全部通过环境变量控制
# ==========================================

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://127.0.0.1:56789")
NAMESPACE = os.environ.get("NAMESPACE", "default")
API_KEY = os.environ.get("OPENAI_API_KEY", "API Key")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-pro")

PROMETHEUS_TIMEOUT = int(os.environ.get("PROMETHEUS_TIMEOUT", "10"))
PROMETHEUS_STARTUP_TIMEOUT = int(os.environ.get("PROMETHEUS_STARTUP_TIMEOUT", "15"))
PROMETHEUS_K8S_NAMESPACE = os.environ.get("PROMETHEUS_K8S_NAMESPACE", "monitoring")
PROMETHEUS_K8S_SERVICE = os.environ.get("PROMETHEUS_K8S_SERVICE", "svc/prometheus")
PROMETHEUS_K8S_PORT = os.environ.get("PROMETHEUS_K8S_PORT", "9090")
KUBECTL_TIMEOUT = int(os.environ.get("KUBECTL_TIMEOUT", "30"))

PROMETHEUS_PORT_FORWARD_PROCESS = None

# ==========================================
# 异常检测规则 — 多维度指标巡检
# ==========================================

ANOMALY_RULES = {
    "cpu_spike": {
        "promql": 'sum(rate(container_cpu_usage_seconds_total{{namespace="{ns}"}}[1m])) by (pod)',
        "threshold": 0.10,
        "unit": "cores/s",
        "severity": "warning",
        "description": "Pod CPU 使用率超过 0.1 core/s",
    },
    "memory_pressure": {
        "promql": '(container_memory_working_set_bytes{{namespace="{ns}"}} / on(pod) container_spec_memory_limit_bytes{{namespace="{ns}"}}) < +Inf',
        "threshold": 0.85,
        "unit": "ratio",
        "severity": "critical",
        "description": "容器内存使用率超过 limit 的 85%",
    },
    "pod_restart": {
        "promql": '(time() - container_start_time_seconds{{namespace="{ns}"}}) < 180',
        "threshold": 0,
        "unit": "seconds",
        "severity": "warning",
        "description": "容器在最近 3 分钟内启动（Pod 可能被重建）",
    },
}


def format_promql(rule):
    return rule["promql"].format(ns=NAMESPACE)


def prometheus_query(query_str, timeout=PROMETHEUS_TIMEOUT):
    """执行 Prometheus instant query，失败时抛出明确异常。"""
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query_str},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus 查询失败: {data.get('error', 'unknown error')}")
    return data.get("data", {}).get("result", [])


def check_prometheus_ready():
    """返回 (是否可用, 错误信息)，用于启动和巡检健康检查。"""
    try:
        prometheus_query("up", timeout=3)
        return True, ""
    except Exception as e:
        return False, str(e)


def get_prometheus_local_port():
    parsed = urlparse(PROMETHEUS_URL)
    if parsed.port:
        return str(parsed.port)
    return "443" if parsed.scheme == "https" else "80"


def ensure_kubectl_context():
    """确认 kubectl 至少有当前 context，否则端口转发必然失败。"""
    try:
        result = subprocess.run(
            ["kubectl", "config", "current-context"],
            capture_output=True,
            text=True,
            timeout=KUBECTL_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("kubectl context 检查超时，无法建立 Prometheus 端口转发") from e
    except FileNotFoundError as e:
        raise RuntimeError("未找到 kubectl 命令，无法建立 Prometheus 端口转发") from e

    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.strip() or "current-context 为空"
        raise RuntimeError(f"kubectl 当前 context 未配置: {detail}")


# ==========================================
# Agent 工具箱 (Tools)
# ==========================================

def execute_promql(query_str):
    """执行 PromQL 查询，返回 Prometheus 时序数据。"""
    try:
        results = prometheus_query(query_str)
        if not results:
            return "未查询到数据（指标可能不存在或无匹配的标签）"
        summary_lines = []
        for r in results:
            metric = r.get("metric", {})
            value = r.get("value", [])
            labels = ", ".join(f"{k}={v}" for k, v in metric.items() if k != "__name__")
            val_str = f"{float(value[1]):.6f}" if len(value) > 1 else "N/A"
            summary_lines.append(f"  {r['metric'].get('__name__', 'N/A')}{{{labels}}} = {val_str}")
        return "\n".join(summary_lines)
    except requests.exceptions.ConnectionError:
        return f"错误: 无法连接到 Prometheus ({PROMETHEUS_URL})，请检查端口转发是否正常"
    except Exception as e:
        return f"查询失败: {e}"


def get_service_logs(service_name, tail_lines=20):
    """通过 kubectl logs 获取指定微服务的日志。"""
    cmd = [
        "kubectl", "logs",
        "-n", NAMESPACE,
        "-l", f"app={service_name}",
        "--tail", str(tail_lines),
        "--prefix=true",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=KUBECTL_TIMEOUT)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "No resources found" in stderr or "no pods" in stderr.lower():
                return f"未找到标签为 app={service_name} 的 Pod"
            return f"kubectl 执行失败: {stderr}"
        output = result.stdout.strip()
        return output if output else f"{service_name} 服务没有日志输出（Pod 可能刚启动或日志为空）"
    except subprocess.TimeoutExpired:
        return f"获取 {service_name} 日志超时（>{KUBECTL_TIMEOUT}秒）"
    except FileNotFoundError:
        return "错误: 未找到 kubectl 命令，请确认已安装 kubectl 并在 PATH 中"


def restart_deployment(service_name):
    """重启指定微服务的 Deployment（滚动更新）。"""
    cmd = [
        "kubectl", "rollout", "restart",
        f"deployment/{service_name}",
        "-n", NAMESPACE,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=KUBECTL_TIMEOUT)
        if result.returncode != 0:
            return f"重启失败: {result.stderr.strip()}"
        return f"deployment/{service_name} 滚动重启已触发，新 Pod 正在部署中。"
    except subprocess.TimeoutExpired:
        return f"重启命令执行超时（>{KUBECTL_TIMEOUT}秒）"
    except FileNotFoundError:
        return "错误: 未找到 kubectl 命令"


def get_pod_status(service_name=None):
    """获取 Pod 运行状态：Ready 状态、重启次数、运行时长。如不指定 service_name，返回所有 Pod 概览。"""
    cmd = ["kubectl", "get", "pods", "-n", NAMESPACE, "-o", "wide"]
    if service_name:
        cmd.extend(["-l", f"app={service_name}"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=KUBECTL_TIMEOUT)
        if result.returncode != 0:
            return f"kubectl 执行失败: {result.stderr.strip()}"
        return result.stdout.strip() if result.stdout.strip() else "未找到匹配的 Pod"
    except subprocess.TimeoutExpired:
        return f"查询 Pod 状态超时"
    except FileNotFoundError:
        return "错误: 未找到 kubectl 命令"


def get_deployment_describe(service_name):
    """获取 Deployment 的详细描述，包括事件、条件和最新状态。"""
    cmd = [
        "kubectl", "describe", "deployment", service_name,
        "-n", NAMESPACE,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=KUBECTL_TIMEOUT)
        if result.returncode != 0:
            return f"kubectl 执行失败: {result.stderr.strip()}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return f"查询 Deployment 详情超时"
    except FileNotFoundError:
        return "错误: 未找到 kubectl 命令"


def get_service_metrics_overview():
    """获取所有微服务的核心指标概览（CPU + Memory + 网络流量），一次性返回全景视图。"""
    queries = {
        "cpu_per_pod": 'sum(rate(container_cpu_usage_seconds_total{{namespace="{ns}"}}[5m])) by (pod)'.format(ns=NAMESPACE),
        "memory_per_pod": 'container_memory_working_set_bytes{{namespace="{ns}"}}'.format(ns=NAMESPACE),
        "memory_limit_per_pod": 'container_spec_memory_limit_bytes{{namespace="{ns}"}}'.format(ns=NAMESPACE),
        "memory_usage_ratio": '(container_memory_working_set_bytes{{namespace="{ns}"}} / on(pod) container_spec_memory_limit_bytes{{namespace="{ns}"}})'.format(ns=NAMESPACE),
    }

    results = {}
    for name, promql in queries.items():
        results[name] = execute_promql(promql)

    report_lines = ["=== 微服务全景指标概览 ===\n"]
    for name, data in results.items():
        label_map = {
            "cpu_per_pod": "CPU 使用率 (cores/s)",
            "memory_per_pod": "内存使用 (bytes)",
            "memory_limit_per_pod": "内存 Limit (bytes)",
            "memory_usage_ratio": "内存使用率 (ratio)",
        }
        report_lines.append(f"--- {label_map.get(name, name)} ---")
        report_lines.append(data)
        report_lines.append("")
    return "\n".join(report_lines)


# 注册工具字典
AVAILABLE_TOOLS = {
    "execute_promql": execute_promql,
    "get_service_logs": get_service_logs,
    "restart_deployment": restart_deployment,
    "get_pod_status": get_pod_status,
    "get_deployment_describe": get_deployment_describe,
    "get_service_metrics_overview": get_service_metrics_overview,
}


# ==========================================
# AIOps Agent
# ==========================================

class AIOpsAgent:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

        self.system_prompt = """
你是一个专注于 Online Boutique 电商微服务系统的资深云原生 AIOps 专家。

## 可用的 Prometheus 指标
当前 Prometheus 只采集 cAdvisor 数据，**只有以下指标可用**：
- container_cpu_usage_seconds_total（容器 CPU 累计使用秒数）
- container_memory_working_set_bytes（容器内存工作集）
- container_spec_memory_limit_bytes（容器内存限制）
- container_start_time_seconds（容器启动时间戳）
- container_last_seen（容器最后可见时间戳）
- container_network_receive_bytes_total / container_network_transmit_bytes_total（网络）
- container_fs_reads_bytes_total / container_fs_writes_bytes_total（文件系统）

**Pod 重启次数请通过 get_pod_status() 的 RESTARTS 列获取**，不要用 PromQL。
**容器是否健康通过 get_pod_status() 的 READY 列和 STATUS 列获取**。

## 系统架构知识
Online Boutique 由 11 个微服务组成，通过 gRPC 通信：
- frontend (Go): Web 前端，用户入口，无登录/注册
- cartservice (C#): 购物车服务，依赖 Redis (redis-cart) 存储购物车数据
- productcatalogservice (Go): 商品目录，从 JSON 文件加载产品列表
- currencyservice (Node.js): 货币转换，QPS 最高的服务
- paymentservice (Node.js): 支付处理（模拟）
- shippingservice (Go): 运费估算（模拟）
- emailservice (Python): 订单确认邮件（模拟）
- checkoutservice (Go): 结账编排，关键路径枢纽
- recommendationservice (Python): 商品推荐（非关键路径）
- adservice (Java): 文字广告（非关键路径）
- loadgenerator (Python/Locust): 负载生成器，模拟用户行为

## 关键路径
frontend → checkoutservice → (cartservice → redis-cart, paymentservice, shippingservice, emailservice, productcatalogservice, currencyservice)

## 已知脆弱点
- cartservice 依赖 redis-cart：Redis 不可用会导致购物车功能完全失效
- currencyservice 是 QPS 最高的服务，CPU 压力对其影响最大
- checkoutservice 是结账流程的编排枢纽，一旦出问题影响所有下游
- recommendationservice 和 adservice 是边缘服务，降级不影响核心购物体验
- loadgenerator 持续产生流量，其 Pod 状态不代表真实用户行为

## 诊断原则
1. 收到告警后，先获取全景指标了解整体状态（get_service_metrics_overview + get_pod_status）
2. 用 get_pod_status 查看 Pod 就绪状态、重启次数，不要用 PromQL 查 kube-* 指标
3. 用 get_service_logs 查看异常 Pod 日志，用 get_deployment_describe 查看部署事件
4. 用 execute_promql 只查 cAdvisor 指标（CPU/memory/network/disk），不要查 kube-* 指标
5. 区分正常流量波动和真实的底层故障（死锁、连接池耗尽、内存泄漏、慢查询）
6. 判断影响范围：关键路径 vs 边缘服务
7. **必须在 5 步内给出诊断结论**，不要反复查询不存在的指标
8. 给出明确的操作建议：(a) 继续观察 (b) 扩容 (c) 重启 (d) 深入排查

## 输出格式
用清晰的结构化报告呈现诊断结论：
1. **异常摘要**：什么服务、什么指标、偏离程度
2. **根因分析**：基于收集到的证据推断原因
3. **影响评估**：对用户体验和业务的影响
4. **处置建议**：具体可执行的操作
"""

        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "execute_promql",
                    "description": "执行 PromQL 查询 Prometheus，获取任意监控指标的时序数据。用于深度指标分析。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_str": {
                                "type": "string",
                                "description": "完整的 PromQL 查询语句",
                            }
                        },
                        "required": ["query_str"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_service_logs",
                    "description": "获取指定微服务的末尾日志，用于排查错误和异常。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_name": {
                                "type": "string",
                                "description": "服务名，如 frontend, cartservice, checkoutservice 等",
                            },
                            "tail_lines": {
                                "type": "integer",
                                "description": "获取最后多少行日志，默认 50",
                            },
                        },
                        "required": ["service_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "restart_deployment",
                    "description": "滚动重启指定微服务的 Deployment。仅在确认服务处于无法恢复的僵死状态（死锁、内存泄漏、OOM）时使用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_name": {
                                "type": "string",
                                "description": "要重启的 Deployment 名称",
                            }
                        },
                        "required": ["service_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_pod_status",
                    "description": "获取 Pod 的运行状态（READY、STATUS、RESTARTS、AGE）。不传参数获取所有 Pod 概览。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_name": {
                                "type": "string",
                                "description": "可选，按 app label 过滤特定服务",
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_deployment_describe",
                    "description": "获取 Deployment 的详细描述，包括最近事件、条件状态和部署历史。用于深度排查部署问题。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_name": {
                                "type": "string",
                                "description": "Deployment 名称",
                            }
                        },
                        "required": ["service_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_service_metrics_overview",
                    "description": "一次性获取所有微服务的核心指标概览（CPU、内存使用、内存使用率），快速了解全局状态。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
        ]

    def run_diagnosis(self, alert_context):
        """核心推理循环 (ReAct Loop)"""
        print("\n" + "=" * 60)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Agent 唤醒")
        print(f"告警上下文: {alert_context}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"系统出现如下异常：{alert_context}\n请按照你的诊断原则进行排查，先获取全景指标，再深入分析异常服务，最后给出诊断报告。"},
        ]

        for step in range(8):
            print(f"  [思考 第 {step+1} 步]")
            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=self.tools_schema,
                    tool_choice="auto",
                )
            except Exception as e:
                print(f"\n[错误] LLM 调用失败: {e}")
                return

            response_message = response.choices[0].message
            messages.append(response_message)

            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}

                    print(f"    调用工具: {func_name}({func_args})")
                    tool_func = AVAILABLE_TOOLS[func_name]
                    try:
                        result = tool_func(**func_args)
                    except TypeError as e:
                        result = f"工具调用参数错误: {e}，请检查传入的参数"

                    # 截断过长结果，防止超出 token 限制
                    result_str = str(result)
                    if len(result_str) > 6000:
                        result_str = result_str[:6000] + f"\n... (输出截断，原始长度 {len(result_str)} 字符)"

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": result_str,
                    })
            else:
                print("\n" + "=" * 60)
                print("[Agent 最终诊断报告]")
                print(response_message.content)
                print("=" * 60 + "\n")
                return

        print("\n[Agent 警告] 已达到最大推理步数 (8)，诊断未收敛。请检查告警是否过于复杂。\n")

    def run(self):
        """主巡检循环：定时检查多维度指标，发现异常时触发 LLM 诊断。"""
        print("=" * 60)
        print("Online Boutique AIOps Agent 启动")
        print(f"  Prometheus: {PROMETHEUS_URL}")
        print(f"  Namespace:  {NAMESPACE}")
        print(f"  Model:      {MODEL}")
        print("  巡检规则:")
        for name, rule in ANOMALY_RULES.items():
            print(f"    - {name}: {rule['description']} (阈值={rule['threshold']})")
        print("=" * 60)

        cooldown_until = {}  # 告警冷却，防止同一规则反复触发

        while True:
            now = time.time()
            anomalies_detected = []
            prometheus_errors = []

            for rule_name, rule in ANOMALY_RULES.items():
                # 冷却期检查
                if rule_name in cooldown_until and now < cooldown_until[rule_name]:
                    continue

                promql = format_promql(rule)
                try:
                    results = prometheus_query(promql)
                except Exception as e:
                    prometheus_errors.append(f"{rule_name}: {e}")
                    continue

                for r in results:
                    try:
                        val = float(r["value"][1])
                    except (KeyError, IndexError, ValueError):
                        continue

                    if val > rule["threshold"]:
                        pod_name = r["metric"].get("pod", "unknown")
                        anomalies_detected.append({
                            "rule": rule_name,
                            "pod": pod_name,
                            "value": val,
                            "threshold": rule["threshold"],
                            "unit": rule["unit"],
                            "severity": rule["severity"],
                            "description": rule["description"],
                        })
                        cooldown_until[rule_name] = now + 300  # 5 分钟冷却

            if prometheus_errors:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 巡检未完成，Prometheus 查询失败 | "
                      f"{'; '.join(prometheus_errors[:3])}")
                time.sleep(15)
                continue

            if anomalies_detected:
                # 构造告警上下文字段
                anomaly_summary_lines = ["检测到以下异常:\n"]
                for i, a in enumerate(anomalies_detected, 1):
                    anomaly_summary_lines.append(
                        f"  [{a['severity'].upper()}] 规则={a['rule']}, "
                        f"Pod={a['pod']}, "
                        f"当前值={a['value']:.4f}{a['unit']}, "
                        f"阈值={a['threshold']}{a['unit']}, "
                        f"说明={a['description']}"
                    )
                alert_msg = "\n".join(anomaly_summary_lines)
                self.run_diagnosis(alert_context=alert_msg)

            print(f"[{datetime.now().strftime('%H:%M:%S')}] 巡检完成，未发现异常 | "
                  f"冷却中: {[k for k in cooldown_until if cooldown_until[k] > now]}")
            time.sleep(15)


# ==========================================
# 主入口
# ==========================================

def ensure_prometheus():
    """确保 Prometheus 可查询；不可用时尝试建立端口转发并等待验证。"""
    global PROMETHEUS_PORT_FORWARD_PROCESS

    ready, error = check_prometheus_ready()
    if ready:
        print(f"[启动] Prometheus 已连接: {PROMETHEUS_URL}")
        return

    local_port = get_prometheus_local_port()
    port_mapping = f"{local_port}:{PROMETHEUS_K8S_PORT}"
    cmd = [
        "kubectl", "port-forward",
        "-n", PROMETHEUS_K8S_NAMESPACE,
        PROMETHEUS_K8S_SERVICE,
        port_mapping,
    ]

    print(f"[启动] Prometheus 未连接: {error}")
    ensure_kubectl_context()
    print(f"[启动] 正在建立端口转发: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError("未找到 kubectl 命令，无法建立 Prometheus 端口转发") from e

    deadline = time.time() + PROMETHEUS_STARTUP_TIMEOUT
    last_error = error
    while time.time() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read().strip() if process.stderr else ""
            detail = stderr or f"退出码 {process.returncode}"
            raise RuntimeError(f"Prometheus 端口转发启动失败: {detail}")

        ready, last_error = check_prometheus_ready()
        if ready:
            PROMETHEUS_PORT_FORWARD_PROCESS = process
            print(f"[启动] Prometheus 已连接: {PROMETHEUS_URL}")
            return

        time.sleep(1)

    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
    raise RuntimeError(
        f"Prometheus 在 {PROMETHEUS_STARTUP_TIMEOUT} 秒内仍不可用: {last_error}"
    )


def main():
    try:
        ensure_prometheus()
    except RuntimeError as e:
        print(f"[启动失败] {e}")
        raise SystemExit(1)
    agent = AIOpsAgent()
    agent.run()


if __name__ == "__main__":
    main()
