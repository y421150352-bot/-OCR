#!/usr/bin/env python3
"""Web UI for the final manga character clustering and speaker pipeline."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

HERE = Path(__file__).resolve().parent
DEPLOY_ROOT = HERE.parent
JOBS_ROOT = Path(os.environ.get("MANGA_WEB_JOBS", HERE / "runs" / "web_jobs")).resolve()
PIPELINE = Path(
    os.environ.get("MANGA_PIPELINE", HERE / "run_final_tail_ray_panel_pipeline.py")
).resolve()
STAGED_PIPELINE = Path(
    os.environ.get("MANGA_STAGED_PIPELINE", HERE / "run_staged_manga_pipeline.py")
).resolve()
REID_CHECKPOINT = Path(
    os.environ.get(
        "MANGA_REID_CHECKPOINT", HERE / "runs" / "samebook_supcon_cosine" / "best.pt"
    )
).resolve()
V3_CHECKPOINT = Path(
    os.environ.get(
        "MANGA_V3_CHECKPOINT",
        DEPLOY_ROOT
        / "speaker_relation_transformer"
        / "runs"
        / "geometry_text_graph_v3_prev_current_next"
        / "best.pt",
    )
).resolve()
TEXT_MODEL = Path(
    os.environ.get(
        "MANGA_TEXT_MODEL",
        DEPLOY_ROOT
        / "speaker_relation_transformer"
        / "pretrained"
        / "multilingual-e5-base",
    )
).resolve()
RTDETR_SCRIPT = Path(
    os.environ.get(
        "MANGA_RTDETR_SCRIPT", HERE.parent / "rtdetr_manga_test" / "run_inference.py"
    )
).resolve()
RTDETR_MODEL = Path(
    os.environ.get(
        "MANGA_RTDETR_MODEL", HERE.parent / "rtdetr_manga_test" / "model.onnx"
    )
).resolve()
MAGIV3_TAIL_SCRIPT = Path(
    os.environ.get("MANGA_MAGIV3_TAIL_SCRIPT", HERE / "run_magiv3_tails.py")
).resolve()
MAGIV3_MODEL = Path(
    os.environ.get("MANGA_MAGIV3_MODEL", DEPLOY_ROOT / "models" / "magiv3")
).resolve()
VLM_DEFAULT_MODEL = os.environ.get("MANGA_VLM_MODEL", "gemini-3.1-pro-preview")
ALLOWED_IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
RESULT_LOCKS: dict[str, threading.RLock] = {}
RESULT_LOCKS_GUARD = threading.Lock()

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
app.config["MAX_CONTENT_LENGTH"] = (
    int(os.environ.get("MANGA_MAX_UPLOAD_MB", "200")) * 1024 * 1024
)
JOBS_ROOT.mkdir(parents=True, exist_ok=True)


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def result_lock(job_id: str) -> threading.RLock:
    """Serialize read-modify-write review operations for one job."""
    with RESULT_LOCKS_GUARD:
        return RESULT_LOCKS.setdefault(job_id, threading.RLock())


def write_json_atomic(path: Path, payload: Any) -> None:
    """Replace a JSON file atomically so readers never observe partial data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_vlm_options() -> dict[str, Any]:
    """Read non-secret Gemini controls submitted by the web form."""
    enabled = request.form.get("vlm_enabled") == "1"
    model = request.form.get("vlm_model", VLM_DEFAULT_MODEL).strip()
    allowed_model_characters = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    if (
        not model
        or len(model) > 100
        or any(character not in allowed_model_characters for character in model)
    ):
        raise ValueError("Gemini model name is invalid")
    try:
        first_pass_threshold = float(
            request.form.get("vlm_first_pass_threshold", "0.80")
        )
        identity_threshold = float(request.form.get("vlm_identity_threshold", "0.70"))
    except ValueError as error:
        raise ValueError("Gemini confidence thresholds must be numeric") from error
    if not 0.0 <= first_pass_threshold <= 1.0 or not 0.0 <= identity_threshold <= 1.0:
        raise ValueError("Gemini confidence thresholds must be between 0 and 1")
    return {
        "enabled": enabled,
        "model": model,
        "first_pass_threshold": first_pass_threshold,
        "identity_threshold": identity_threshold,
    }


def update_job(job_id: str, **values: Any) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(values)
        snapshot = dict(JOBS[job_id])
    write_json_atomic(Path(snapshot["job_dir"]) / "job_state.json", snapshot)


def append_log(job_id: str, text: str) -> None:
    job = JOBS[job_id]
    with (Path(job["job_dir"]) / "pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
    stripped = text.strip()
    if stripped:
        update_job(
            job_id, message=stripped.splitlines()[-1][:300], updated_at=now_text()
        )


def adapter_command(
    template: str, image_dir: Path, output_dir: Path, detections: Path
) -> str:
    return template.format(
        image_dir=shlex.quote(str(image_dir)),
        output_dir=shlex.quote(str(output_dir)),
        detections=shlex.quote(str(detections)),
    )


def run_command(job_id: str, command: list[str] | str, shell: bool = False) -> None:
    shown = command if isinstance(command, str) else shlex.join(command)
    append_log(job_id, f"$ {shown}")
    process = subprocess.Popen(
        command,
        cwd=str(HERE),
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        append_log(job_id, line)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"命令执行失败，退出码 {code}: {shown}")


def run_job(job_id: str) -> None:
    job = safe_job(job_id)
    job_dir = Path(job["job_dir"])
    image_dir, output_dir = job_dir / "input", job_dir / "result"
    detections = job_dir / "detections.json"
    try:
        update_job(
            job_id,
            status="running",
            stage="检测漫画元素",
            message="正在检测人物、脸、分镜和文本",
            progress=8,
            updated_at=now_text(),
        )
        if not detections.is_file():
            template = os.environ.get("MANGA_DETECTOR_COMMAND", "").strip()
            if template:
                run_command(
                    job_id,
                    adapter_command(template, image_dir, job_dir, detections),
                    shell=True,
                )
            elif RTDETR_SCRIPT.is_file() and RTDETR_MODEL.is_file():
                detector_output = job_dir / "rtdetr_output"
                run_command(
                    job_id,
                    [
                        sys.executable,
                        "-u",
                        str(RTDETR_SCRIPT),
                        "--model",
                        str(RTDETR_MODEL),
                        "--input",
                        str(image_dir),
                        "--output",
                        str(detector_output),
                        "--threshold",
                        os.environ.get("MANGA_RTDETR_THRESHOLD", "0.5"),
                        "--input-size",
                        os.environ.get("MANGA_RTDETR_INPUT_SIZE", "1280"),
                    ],
                )
                generated = detector_output / "detections.json"
                if generated.is_file():
                    shutil.copy2(generated, detections)
            else:
                raise RuntimeError(
                    "未找到内置 RT-DETR。请将 rtdetr_manga_test 上传到 "
                    f"{RTDETR_SCRIPT.parent}，其中必须包含 run_inference.py 和 model.onnx"
                )
            if not detections.is_file():
                raise RuntimeError("RT-DETR 命令完成，但没有生成 detections.json")

        ocr_dir = job_dir / "ocr_page_bundles"
        template = os.environ.get("MANGA_OCR_COMMAND", "").strip()
        if template:
            update_job(
                job_id,
                stage="识别对白文字",
                message="正在运行 OCR",
                progress=28,
                updated_at=now_text(),
            )
            run_command(
                job_id,
                adapter_command(template, image_dir, ocr_dir, detections),
                shell=True,
            )

        magi_dir = job_dir / "magiv3_output_all"
        tails_enabled = bool(job.get("tails_enabled", False))
        template = os.environ.get("MANGA_MAGIV3_TAIL_COMMAND", "").strip()
        if not tails_enabled:
            append_log(job_id, "MagiV3 tail detection disabled for this job")
        elif MAGIV3_TAIL_SCRIPT.is_file() and MAGIV3_MODEL.is_dir():
            update_job(
                job_id,
                stage="检测气泡尾巴",
                message="正在运行 MagiV3（仅提取气泡尾巴）",
                progress=38,
                updated_at=now_text(),
            )
            run_command(
                job_id,
                [
                    sys.executable,
                    "-u",
                    str(MAGIV3_TAIL_SCRIPT),
                    "--model-dir",
                    str(MAGIV3_MODEL),
                    "--input-dir",
                    str(image_dir),
                    "--output-dir",
                    str(magi_dir),
                    "--overwrite",
                ],
            )
        elif template:
            update_job(
                job_id,
                stage="检测气泡尾巴",
                message="正在运行 MagiV3 尾巴适配器",
                progress=38,
                updated_at=now_text(),
            )
            run_command(
                job_id,
                adapter_command(template, image_dir, magi_dir, detections),
                shell=True,
            )
        else:
            raise RuntimeError(
                "已启用 MagiV3 气泡尾巴检测，但未找到模型或入口。"
                f"入口={MAGIV3_TAIL_SCRIPT}，模型={MAGIV3_MODEL}。"
                "请上传 models/magiv3，或设置 MANGA_MAGIV3_MODEL。"
            )

        if tails_enabled:
            magi_summary = magi_dir / "summary.json"
            if not magi_summary.is_file():
                raise RuntimeError("MagiV3 尾巴检测完成，但没有生成 summary.json")
            magi_stats = json.loads(magi_summary.read_text(encoding="utf-8"))
            if int(magi_stats.get("failed_count", 0)):
                raise RuntimeError(
                    f"MagiV3 有 {magi_stats['failed_count']} 页尾巴检测失败"
                )
            append_log(
                job_id,
                f"MagiV3 tails only: {magi_stats.get('total_tails', 0)} tails",
            )

        update_job(
            job_id,
            stage="角色聚类",
            message="正在提取 ReID 特征并对整本漫画聚类",
            progress=48,
            updated_at=now_text(),
        )
        command = [
            sys.executable,
            "-u",
            str(STAGED_PIPELINE),
            "--stage",
            "cluster",
            "--detections",
            str(detections),
            "--image-dir",
            str(image_dir),
            "--reid-checkpoint",
            str(REID_CHECKPOINT),
            "--output-dir",
            str(output_dir),
            "--tail-weight",
            os.environ.get("MANGA_TAIL_WEIGHT", "6.0"),
            "--top-k",
            "5",
            "--small-book-similarity",
            os.environ.get("MANGA_SMALL_BOOK_SIMILARITY", "0.72"),
        ]
        if ocr_dir.is_dir():
            command += ["--ocr-bundles-dir", str(ocr_dir)]
        if magi_dir.is_dir():
            command += ["--magi-dir", str(magi_dir)]
        run_command(job_id, command)
        result_path = output_dir / "pipeline_result.json"
        if not result_path.is_file():
            raise RuntimeError("最终管线没有生成 pipeline_result.json")
        summary = json.loads(result_path.read_text(encoding="utf-8")).get("summary", {})
        update_job(
            job_id,
            status="review",
            stage="审核角色簇",
            message=f"聚类完成：{summary.get('character_clusters', 0)} 个角色簇，请拖拽整理参考图并命名",
            summary=summary,
            progress=65,
            updated_at=now_text(),
        )
    except Exception as exc:
        append_log(job_id, traceback.format_exc())
        update_job(
            job_id,
            status="failed",
            stage="失败",
            message=str(exc),
            updated_at=now_text(),
        )


def safe_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        state = JOBS_ROOT / job_id / "job_state.json"
        if not state.is_file():
            abort(404)
        job = json.loads(state.read_text(encoding="utf-8"))
        with JOBS_LOCK:
            JOBS[job_id] = job
    return job


def load_result(job_id: str) -> tuple[dict[str, Any], Path]:
    result_path = Path(safe_job(job_id)["job_dir"]) / "result" / "pipeline_result.json"
    if not result_path.is_file():
        abort(409, "结果尚未生成")
    text = result_path.read_text(encoding="utf-8")
    try:
        result = json.loads(text)
    except json.JSONDecodeError as error:
        # Older versions wrote this file directly from several HTTP threads.
        # ``Extra data`` means the first JSON document is complete and trailing
        # bytes were left by a racing writer. Preserve the evidence, then recover
        # that complete document so the review does not need to be rerun.
        if error.msg != "Extra data":
            raise
        result, end = json.JSONDecoder().raw_decode(text)
        if not text[end:].strip():
            raise
        backup = result_path.with_name(
            f"{result_path.name}.corrupt-{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        )
        shutil.copy2(result_path, backup)
        write_json_atomic(result_path, result)
        app.logger.warning(
            "Recovered concurrent-write corruption in %s; backup=%s",
            result_path,
            backup,
        )
    return result, result_path


def write_result(result: dict[str, Any], path: Path) -> None:
    retained = [
        row
        for row in result.get("character_instances", [])
        if not row.get("excluded", False)
    ]
    library_clusters = {
        row["character_cluster_id"]
        for row in retained
        if row.get("library_member", True)
        and row.get("character_cluster_id") != "unassigned"
    }
    result.setdefault("summary", {})["character_instances"] = len(retained)
    result["summary"]["character_clusters"] = len(library_clusters)
    write_json_atomic(path, result)


def finalize_job(job_id: str) -> None:
    job = safe_job(job_id)
    job_dir = Path(job["job_dir"])
    output_dir = job_dir / "result"
    try:
        update_job(
            job_id,
            status="running",
            stage="对白人物分析",
            message="正在使用审核后的角色库运行 ReID 检索和 V3 对白分析",
            progress=75,
            updated_at=now_text(),
        )
        command = [
            sys.executable,
            "-u",
            str(STAGED_PIPELINE),
            "--stage",
            "finalize",
            "--detections",
            str(job_dir / "detections.json"),
            "--image-dir",
            str(job_dir / "input"),
            "--output-dir",
            str(output_dir),
            "--reviewed-instances",
            str(job_dir / "reviewed_instances.json"),
            "--v3-checkpoint",
            str(V3_CHECKPOINT),
            "--text-model",
            str(TEXT_MODEL),
            "--tail-weight",
            os.environ.get("MANGA_TAIL_WEIGHT", "6.0"),
            "--top-k",
            "5",
        ]
        ocr_dir = job_dir / "ocr_page_bundles"
        magi_dir = job_dir / "magiv3_output_all"
        if ocr_dir.is_dir():
            command += ["--ocr-bundles-dir", str(ocr_dir)]
        if job.get("tails_enabled") and magi_dir.is_dir():
            command += ["--magi-dir", str(magi_dir)]
        vlm = job.get("vlm", {})
        if vlm.get("enabled"):
            update_job(
                job_id,
                stage="Gemini VLM verification",
                message=f"Running {vlm['model']} speaker and identity verification",
                progress=80,
                updated_at=now_text(),
            )
            command += [
                "--vlm-top5",
                "--vlm-model",
                str(vlm["model"]),
                "--vlm-first-pass-confidence-threshold",
                str(vlm["first_pass_threshold"]),
                "--vlm-confidence-threshold",
                str(vlm["identity_threshold"]),
                "--vlm-panel-batch-size",
                "1",
                "--vlm-identity-batch-size",
                "5",
                "--vlm-timeout",
                "300",
                "--vlm-save-boards",
            ]
        run_command(job_id, command)
        result = json.loads(
            (output_dir / "pipeline_result.json").read_text(encoding="utf-8")
        )
        summary = result.get("summary", {})
        update_job(
            job_id,
            status="completed",
            stage="完成",
            message=f"完成：{summary.get('character_clusters', 0)} 个已命名角色，{summary.get('dialogues', 0)} 条对白",
            summary=summary,
            progress=100,
            updated_at=now_text(),
        )
    except Exception as exc:
        append_log(job_id, traceback.format_exc())
        update_job(
            job_id,
            status="failed",
            stage="失败",
            message=str(exc),
            updated_at=now_text(),
        )


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/api/jobs")
def create_job():
    try:
        vlm = read_vlm_options()
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    images = request.files.getlist("pages")
    valid = [
        item
        for item in images
        if Path(item.filename or "").suffix.lower() in ALLOWED_IMAGES
    ]
    if not valid:
        return jsonify({"error": "请至少选择一张漫画图片"}), 400
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    job_dir, image_dir = JOBS_ROOT / job_id, JOBS_ROOT / job_id / "input"
    image_dir.mkdir(parents=True)
    used_names, saved_names = set(), []
    for index, item in enumerate(valid, 1):
        name = secure_filename(item.filename or "") or f"page_{index:04d}.jpg"
        if name in used_names:
            name = f"{Path(name).stem}_{index}{Path(name).suffix}"
        used_names.add(name)
        item.save(image_dir / name)
        saved_names.append(name)
    detection_upload = request.files.get("detections")
    if detection_upload and detection_upload.filename:
        detection_upload.save(job_dir / "detections.json")
    job = {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "status": "queued",
        "stage": "排队",
        "message": "任务已创建",
        "progress": 2,
        "images": saved_names,
        "created_at": now_text(),
        "updated_at": now_text(),
        "tails_enabled": request.form.get("tails_enabled") == "1",
    }
    job["vlm"] = vlm
    with JOBS_LOCK:
        JOBS[job_id] = job
    update_job(job_id)
    threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = dict(safe_job(job_id))
    job.pop("job_dir", None)
    return jsonify(job)


@app.get("/api/jobs/<job_id>/result")
def job_result(job_id: str):
    return jsonify(load_result(job_id)[0])


@app.get("/api/jobs/<job_id>/log")
def job_log(job_id: str):
    path = Path(safe_job(job_id)["job_dir"]) / "pipeline.log"
    return jsonify(
        {
            "log": (
                path.read_text(encoding="utf-8", errors="replace")
                if path.is_file()
                else ""
            )
        }
    )


@app.get("/files/<job_id>/<path:relative_path>")
def job_file(job_id: str, relative_path: str):
    return send_from_directory(safe_job(job_id)["job_dir"], relative_path)


@app.post("/api/jobs/<job_id>/names")
def update_names(job_id: str):
    with result_lock(job_id):
        result, result_path = load_result(job_id)
        names = request.get_json(force=True).get("names", {})
        if not isinstance(names, dict):
            return jsonify({"error": "names 必须是对象"}), 400
        clean = {
            str(key): str(value).strip() or str(key) for key, value in names.items()
        }
        for instance in result.get("character_instances", []):
            cluster = instance.get("character_cluster_id")
            instance["character_name"] = clean.get(
                cluster, instance.get("character_name", cluster)
            )
        for page in result.get("pages", []):
            for dialogue in page.get("dialogues", []):
                cluster = dialogue.get("character_cluster_id")
                dialogue["character_name"] = clean.get(
                    cluster, dialogue.get("character_name", cluster)
                )
                for candidate in dialogue.get("top_candidates", []):
                    candidate_cluster = candidate.get("character_cluster_id")
                    candidate["character_name"] = clean.get(
                        candidate_cluster,
                        candidate.get("character_name", candidate_cluster),
                    )
        write_result(result, result_path)
        write_json_atomic(result_path.parent / "character_names.json", clean)
    return jsonify({"ok": True, "names": clean})


@app.post("/api/jobs/<job_id>/review/delete-instance")
def delete_review_instance(job_id: str):
    with result_lock(job_id):
        result, result_path = load_result(job_id)
        if result.get("stage") != "cluster_review":
            return jsonify({"error": "当前任务不在角色簇审核阶段"}), 409
        instance_id = str(request.get_json(force=True).get("instance_id", ""))
        found = False
        for row in result.get("character_instances", []):
            if row.get("instance_id") == instance_id:
                row["library_member"] = False
                row["character_cluster_id"] = "unassigned"
                row["character_name"] = "待自动检索"
                found = True
                break
        if not found:
            abort(404)
        write_result(result, result_path)
    return jsonify({"ok": True})


@app.post("/api/jobs/<job_id>/review/move-instance")
def move_review_instance(job_id: str):
    with result_lock(job_id):
        result, result_path = load_result(job_id)
        if result.get("stage") != "cluster_review":
            return jsonify({"error": "当前任务不在角色簇审核阶段"}), 409
        payload = request.get_json(force=True)
        instance_id = str(payload.get("instance_id", ""))
        target_cluster = str(payload.get("target_cluster_id", ""))
        rows = result.get("character_instances", [])
        moving = next(
            (row for row in rows if row.get("instance_id") == instance_id), None
        )
        if moving is None:
            abort(404)
        if target_cluster == "__new__":
            used_numbers = [
                int(str(row.get("character_cluster_id", "")).split("_")[-1])
                for row in rows
                if str(row.get("character_cluster_id", "")).startswith("cluster_")
                and str(row.get("character_cluster_id", "")).split("_")[-1].isdigit()
            ]
            target_cluster = f"cluster_{max(used_numbers, default=0) + 1:03d}"
            target_name = f"角色簇 {max(used_numbers, default=0) + 1:03d}"
        else:
            target_name = next(
                (
                    str(row.get("character_name", target_cluster))
                    for row in rows
                    if row.get("character_cluster_id") == target_cluster
                ),
                target_cluster,
            )
        moving["character_cluster_id"] = target_cluster
        moving["character_name"] = target_name
        moving["library_member"] = True
        moving["excluded"] = False
        write_result(result, result_path)
    return jsonify({"ok": True, "target_cluster_id": target_cluster})


@app.post("/api/jobs/<job_id>/review/merge")
def merge_review_clusters(job_id: str):
    with result_lock(job_id):
        result, result_path = load_result(job_id)
        if result.get("stage") != "cluster_review":
            return jsonify({"error": "当前任务不在角色簇审核阶段"}), 409
        payload = request.get_json(force=True)
        cluster_ids = [str(value) for value in payload.get("cluster_ids", [])]
        if len(set(cluster_ids)) < 2:
            return jsonify({"error": "请至少选择两个角色簇"}), 400
        target = cluster_ids[0]
        target_name = next(
            (
                row.get("character_name", "")
                for row in result.get("character_instances", [])
                if row.get("character_cluster_id") == target
            ),
            "",
        )
        for row in result.get("character_instances", []):
            if row.get("character_cluster_id") in cluster_ids:
                row["character_cluster_id"] = target
                row["character_name"] = target_name
        write_result(result, result_path)
    return jsonify({"ok": True, "target": target})


@app.post("/api/jobs/<job_id>/review/confirm")
def confirm_review(job_id: str):
    with result_lock(job_id):
        result, result_path = load_result(job_id)
        if result.get("stage") != "cluster_review":
            return jsonify({"error": "当前任务不在角色簇审核阶段"}), 409
        retained = [
            row
            for row in result.get("character_instances", [])
            if not row.get("excluded", False)
        ]
        if not retained:
            return jsonify({"error": "角色库不能为空"}), 400
        references = [
            row
            for row in retained
            if row.get("library_member", True)
            and row.get("character_cluster_id") != "unassigned"
        ]
        if not references:
            return jsonify({"error": "请至少保留一个角色参考图"}), 400
        unnamed = sorted(
            {
                row.get("character_name", "")
                for row in references
                if not str(row.get("character_name", "")).strip()
                or str(row.get("character_name", "")).startswith("角色簇 ")
                or str(row.get("character_name", "")).startswith("character_")
            }
        )
        if unnamed:
            return jsonify({"error": f"还有 {len(unnamed)} 个角色簇没有命名"}), 400
        reviewed_path = Path(safe_job(job_id)["job_dir"]) / "reviewed_instances.json"
        write_json_atomic(reviewed_path, retained)
        update_job(
            job_id,
            status="queued",
            stage="准备对白分析",
            message="已确认角色库",
            progress=70,
            updated_at=now_text(),
        )
    threading.Thread(target=finalize_job, args=(job_id,), daemon=True).start()
    return jsonify({"ok": True}), 202


@app.post("/api/jobs/<job_id>/dialogues/<dialogue_id>")
def update_dialogue(job_id: str, dialogue_id: str):
    with result_lock(job_id):
        result, result_path = load_result(job_id)
        payload = request.get_json(force=True)
        cluster_id, name, found = (
            str(payload.get("character_cluster_id", "")),
            str(payload.get("character_name", "")),
            False,
        )
        for page in result.get("pages", []):
            for dialogue in page.get("dialogues", []):
                if dialogue.get("dialogue_id") == dialogue_id:
                    dialogue.update(
                        character_cluster_id=cluster_id,
                        character_name=name or cluster_id,
                        speaker_source="manual_override",
                    )
                    found = True
                    break
        if not found:
            abort(404)
        write_result(result, result_path)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(
        host=os.environ.get("MANGA_WEB_HOST", "0.0.0.0"),
        port=int(os.environ.get("MANGA_WEB_PORT", "7860")),
        debug=False,
        threaded=True,
    )
