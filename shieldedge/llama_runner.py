"""
On-device Llama advisory runner for ShieldEdge.

Runs Llama 3.2 1B-Instruct through the native ExecuTorch `llama_main` runner
(XNNPACK delegate, INT4 8da4w program by default), validates the generated
explanation and combines it with the vetted remediation playbook.

If the runner/model is missing, times out, or its output fails validation, the
deterministic template explanation is used instead, so the user always gets an
answer.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time

from shieldedge.advisory_playbook import format_advisory, remediation_steps, template_explanation
from shieldedge.url_detector import OFFICIAL_DOMAINS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ET_DIR = os.path.join(BASE_DIR, "executorch")

# Preferred program first. The bf16 program is kept only as a slow baseline.
PTE_CANDIDATES = [
    os.path.join(ET_DIR, "llama3_2_1b_int4.pte"),
    os.path.join(ET_DIR, "llama3_2_1b_bf16.pte"),
    os.path.join(ET_DIR, "llama3_2_1b.pte"),
]
DEFAULT_RUNNER_BIN = os.path.join(ET_DIR, "llama_main")
DEFAULT_TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "Llama-3.2-1B-Instruct", "original", "tokenizer.model")

ASSISTANT_MARKER = "<|start_header_id|>assistant<|end_header_id|>"
MAX_NEW_TOKENS = int(os.environ.get("SHIELDEDGE_LLM_MAX_TOKENS", "96"))
MAX_STREAM_SENTENCES = 4  # keep in sync with clean_llm_text(max_sentences)
TIMEOUT_S = int(os.environ.get("SHIELDEDGE_LLM_TIMEOUT", "120"))

_URL_RE = re.compile(r"https?://|www\.|\b[a-z0-9-]+\.(com|in|xyz|top|net|org|info)\b", re.I)
_PHONE_RE = re.compile(r"\b\d{6,}\b")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
# Organisations the model must not bring up unless the evidence mentions them.
_ORG_RE = re.compile(r"\b(" + "|".join(sorted({b for b in OFFICIAL_DOMAINS if len(b) >= 3}, key=len, reverse=True))
                     + r"|state bank|rbi|cbi|police)\b", re.I)
_CONTRADICT_RE = re.compile(
    r"\b(is|looks|appears|seems)\s+(completely\s+|totally\s+|perfectly\s+)?(safe|legitimate|genuine|harmless)\b", re.I
)


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def clean_llm_text(text: str, max_sentences: int = 4) -> str:
    """Remove chat tokens / markdown, drop an unfinished last sentence, cap length."""
    text = text.split("<|eot_id|>")[0]
    text = re.sub(r"<\|[^|]*\|>", " ", text)
    text = re.sub(r"[*#`_]+", "", text)
    text = re.sub(r"^\s*(explanation|\[explanation\])\s*:?\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if sentences and not re.search(r"[.!?]$", sentences[-1]):
        sentences = sentences[:-1]  # truncated by the token budget
    return " ".join(sentences[:max_sentences]).strip()


def validate_explanation(text: str, evidence: dict) -> list:
    """Return a list of problems; an empty list means the explanation is usable."""
    problems = []
    if len(text) < 25:
        problems.append("empty or too short")
    # Anything the detector itself supplied (target + signals) may be repeated.
    supplied = " ".join([evidence.get("target") or ""] + list(evidence.get("signals", []))).lower()
    lowered = text.lower()
    if _EMAIL_RE.search(text):
        problems.append("contains an e-mail address")
    if any(num not in supplied for num in _PHONE_RE.findall(text)):
        problems.append("contains a phone/number sequence not supplied by the detector")
    for m in _URL_RE.finditer(text):
        if m.group(0).lower() not in supplied:
            problems.append("mentions a link/domain not supplied by the detector")
            break
    for m in _ORG_RE.finditer(lowered):
        org = m.group(1).lower()
        if org not in supplied and not (org == "state bank" and "sbi" in supplied):
            problems.append(f"mentions '{org}', which is not in the evidence")
            break
    if evidence.get("verdict") != "SAFE":
        for m in _CONTRADICT_RE.finditer(lowered):
            before = lowered[max(0, m.start() - 40):m.start()]
            if not re.search(r"\b(not|never|no|whether|if)\b|n't", before):  # "not certain it is safe" is fine
                problems.append("contradicts the detector verdict")
                break
    return problems


def parse_observer(output: str):
    m = re.search(r"PyTorchObserver\s+(\{.*?\})", output)
    if not m:
        return None
    try:
        raw = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return {
        "model_load_ms": raw.get("model_load_end_ms", 0) - raw.get("model_load_start_ms", 0),
        "ttft_ms": raw.get("first_token_ms", 0) - raw.get("inference_start_ms", 0),
        "generation_ms": raw.get("inference_end_ms", 0) - raw.get("inference_start_ms", 0),
        "prefill_tok_s": raw.get("prefill_token_per_sec"),
        "decode_tok_s": raw.get("decode_token_per_sec"),
        "prompt_tokens": raw.get("prompt_tokens"),
        "generated_tokens": raw.get("generated_tokens"),
    }


class LlamaAdvisoryRunner:
    def __init__(self, pte_model_path: str = None, executorch_runner_bin: str = None, tokenizer_path: str = None):
        self.pte_model_path = pte_model_path or _first_existing(PTE_CANDIDATES)
        self.executorch_runner_bin = executorch_runner_bin or DEFAULT_RUNNER_BIN
        self.tokenizer_path = tokenizer_path or DEFAULT_TOKENIZER_PATH
        self.has_executorch = bool(
            self.pte_model_path
            and os.path.exists(self.pte_model_path)
            and os.path.exists(self.executorch_runner_bin)
            and os.path.exists(self.tokenizer_path)
        )

    @property
    def model_name(self) -> str:
        return os.path.basename(self.pte_model_path) if self.pte_model_path else "none"

    # ------------------------------------------------------------------ LLM call
    def run_llm(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, stream=None) -> dict:
        """Run llama_main once. `stream` is an optional callable receiving generated text chunks."""
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
            tf.write(prompt)
            prompt_path = tf.name

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{ET_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
        cmd = [
            self.executorch_runner_bin,
            "--model_path", self.pte_model_path,
            "--tokenizer_path", self.tokenizer_path,
            "--prompt_file", prompt_path,
            "--max_new_tokens", str(max_new_tokens),
            "--temperature", "0",
        ]
        threads = os.environ.get("SHIELDEDGE_LLM_THREADS")
        if threads:
            cmd += ["--cpu_threads", threads]

        t0 = time.perf_counter()
        out_chunks = []
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
            emitted = 0
            n_markers = prompt.count(ASSISTANT_MARKER)  # the echoed prompt contains them all
            deadline = t0 + TIMEOUT_S
            while True:
                chunk = proc.stdout.read1(256)
                if not chunk:
                    break
                out_chunks.append(chunk.decode("utf-8", errors="replace"))
                if stream:
                    so_far = "".join(out_chunks)
                    if so_far.count(ASSISTANT_MARKER) >= n_markers:
                        visible = so_far.split(ASSISTANT_MARKER, n_markers)[n_markers]
                        visible = visible.split("PyTorchObserver")[0].split("<|eot_id|>")[0].lstrip()
                        # Only emit complete sentences so a token-budget cut-off is never shown.
                        ends = [m.end() for m in re.finditer(r"[.!?](\s|$)", visible)]
                        visible = visible[: ends[MAX_STREAM_SENTENCES - 1] if len(ends) >= MAX_STREAM_SENTENCES else (ends[-1] if ends else 0)]
                        if len(visible) > emitted:
                            stream(visible[emitted:])
                            emitted = len(visible)
                if time.perf_counter() > deadline:
                    proc.kill()
                    raise TimeoutError(f"llama_main exceeded {TIMEOUT_S}s")
            proc.wait(timeout=10)
            returncode = proc.returncode
        finally:
            try:
                os.remove(prompt_path)
            except OSError:
                pass

        output = "".join(out_chunks)
        generated = output.split(ASSISTANT_MARKER)[-1].split("PyTorchObserver")[0]
        return {
            "returncode": returncode,
            "raw_output": output,
            "generated": generated,
            "metrics": parse_observer(output),
            "wall_ms": (time.perf_counter() - t0) * 1000,
        }

    # ------------------------------------------------------------------ public API
    def generate_advisory(self, evidence: dict, prompt: str, stream=None, use_llm: bool = True) -> dict:
        steps = remediation_steps(evidence.get("input_type"), evidence.get("verdict"))
        caveat = evidence.get("caveat", "")
        result = {"source": "ShieldEdge template engine", "metrics": None, "validation": []}
        explanation = None

        if use_llm and self.has_executorch:
            try:
                run = self.run_llm(prompt, stream=stream)
                result["metrics"] = run["metrics"]
                result["wall_ms"] = run["wall_ms"]
                if run["returncode"] != 0:
                    result["validation"] = [f"llama_main exited with code {run['returncode']}"]
                else:
                    candidate = clean_llm_text(run["generated"])
                    problems = validate_explanation(candidate, evidence)
                    result["validation"] = problems
                    result["llm_raw"] = candidate
                    if not problems:
                        explanation = candidate
                        result["source"] = f"Llama 3.2 1B-Instruct via ExecuTorch ({self.model_name})"
            except (OSError, TimeoutError, subprocess.SubprocessError) as exc:
                result["validation"] = [f"runner error: {exc}"]
                print(f"[ShieldEdge] Llama runner failed ({exc}); using template explanation.", file=sys.stderr)

        if explanation is None:
            explanation = template_explanation(evidence)
            if use_llm and self.has_executorch:
                result["source"] = "ShieldEdge template engine (Llama output rejected: " + "; ".join(result["validation"]) + ")"

        result["explanation"] = explanation
        result["remediation"] = steps
        result["advisory"] = format_advisory(explanation, steps, caveat)
        return result

    def template_advisory(self, evidence: dict) -> dict:
        """Fast path without the LLM (used by --fast and the REST API)."""
        return self.generate_advisory(evidence, prompt="", use_llm=False)
