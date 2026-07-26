import os, re, hashlib, base64
from html.parser import HTMLParser

from dotenv import load_dotenv
load_dotenv(override=True)

from utils.logger_handler import logger

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredPDFLoader
from unstructured.partition.pdf import partition_pdf


# ── 表格 HTML → 纯文本 ────────────────────────────────────────────

class _TableHTMLStripper(HTMLParser):
    """将 HTML 表格转成结构化的纯文本，保留行列关系"""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell = ""
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._in_cell = False
            self._current_row.append(self._current_cell.strip())
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = []

    def handle_data(self, data):
        if self._in_cell:
            self._current_cell += data


def table_html_to_text(html: str) -> str:
    """将 Unstructured 提取的表格 HTML 转为可读的纯文本"""
    stripper = _TableHTMLStripper()
    try:
        stripper.feed(html)
    except Exception:
        return html  # 解析失败则回退到原始 HTML

    if not stripper.rows:
        return ""

    lines = []
    for row in stripper.rows:
        lines.append(" | ".join(row))
    return "\n".join(lines)


# ── 图片描述（DashScope 多模态） ────────────────────────────────────

def describe_image(
    image_path: str,
    api_key: str = "",
    model: str = "",
    prompt: str | None = None,
) -> str | None:
    """调用 DashScope 视觉模型描述图片内容，返回中文描述文本。

    默认使用 qwen3-vl-flash（性价比高），可通过 model 参数覆盖为
    qwen-vl-plus / qwen-vl-max / qwen3-vl-flash 等。
    """
    import time
    import requests

    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        logger.warning("[ImageDesc] 缺少 DASHSCOPE_API_KEY，跳过图片描述")
        return None

    if not os.path.exists(image_path):
        logger.warning(f"[ImageDesc] 图片不存在: {image_path}")
        return None

    if not model:
        model = os.getenv("VL_MODEL", "qwen3-vl-flash")

    # 读取并 base64 编码图片
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # 根据文件扩展名判断 MIME 类型
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
    mime_type = mime_map.get(ext, "image/png")

    default_prompt = (
        "请用中文详细描述这张图片中的内容，包括图表类型、关键数据、趋势、"
        "标题、坐标轴标签等。如果图片中有表格，请用文字描述表格的结构和关键数据。"
    )

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                {"type": "text", "text": prompt or default_prompt},
            ],
        }],
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                logger.info(f"[ImageDesc] 图片描述成功 ({len(content)} 字符): {os.path.basename(image_path)}")
                print(content)
                return content
            if resp.status_code == 429:
                wait = 2 * (attempt + 1)
                logger.info(f"[ImageDesc] 限流，等待 {wait}s ...")
                time.sleep(wait)
                continue
            logger.warning(f"[ImageDesc] API 返回 {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"[ImageDesc] 请求异常 (attempt {attempt+1}): {e}")
            time.sleep(1)

    return None


# ── 批量图片描述（并发 VLM 调用）────────────────────────────────────

def describe_images_batch(
    image_items: list[tuple[str, int, str]],
    api_key: str = "",
    model: str = "",
    batch_size: int = 5,
) -> dict[str, str]:
    """批量调用 DashScope VLM 描述多张图片，一次 API 请求发送多张图。

    Args:
        image_items: [(image_path, page_num, label), ...]
            label 如 "表格", "图表", 会与页码一起传给模型用于定位
        api_key: DashScope API Key
        model: VLM 模型名，默认 qwen-vl-plus（平衡速度和质量）
        batch_size: 每批最多几张图（受 token 限制，建议 3-6）

    Returns:
        {image_path: description} 字典，只包含成功描述的图片
    """
    import time, requests, concurrent.futures

    if not image_items:
        return {}

    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        logger.warning("[BatchDesc] 缺少 DASHSCOPE_API_KEY")
        return {}

    if not model:
        model = os.getenv("VL_MODEL", "qwen-vl-plus")

    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    }

    # 过滤掉不存在的图片
    valid_items = []
    for img_path, page_num, label in image_items:
        if os.path.exists(img_path) and os.path.getsize(img_path) > 6000:
            valid_items.append((img_path, page_num, label))
        else:
            logger.info(f"[BatchDesc] 跳过 {os.path.basename(img_path)} (不存在或太小)")

    if not valid_items:
        return {}

    result_map: dict[str, str] = {}

    def _process_batch(batch: list[tuple[str, int, str]]) -> None:
        content_parts = []
        batch_labels = []

        for img_path, page_num, label in batch:
            ext = os.path.splitext(img_path)[1].lower()
            mime_type = mime_map.get(ext, "image/png")
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            # 图片在前，标注在后（编号+页码+类型）
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{img_data}"},
            })
            batch_labels.append((img_path, page_num, label))
            content_parts.append({
                "type": "text",
                "text": f"[图{len(content_parts) // 2 + 1}] 第{page_num}页 {label}",
            })

        # 构造最后的描述指令（让模型按统一格式输出）
        describe_prompt = (
            "请逐一详细描述上面每张图片/图表的内容，用中文。对每张图说明："
            "图表类型、标题、关键数据、趋势、坐标轴含义、结论等。\n"
            "严格按以下格式输出，每张图以 [图N] 开头：\n"
            f"[图1]\n描述内容...\n[图2]\n描述内容...\n"
        )
        content_parts.append({"type": "text", "text": describe_prompt})

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content_parts}],
        }

        for attempt in range(3):
            try:
                resp = requests.post(
                    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                    timeout=180,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]

                    # 按 [图N] 分割回复
                    import re
                    pattern = re.compile(r"\[图(\d+)\]\s*")
                    parts = pattern.split(content)
                    # parts[0] 是第一个 [图N] 之前的文字（通常为空），然后是 N, desc, N, desc...
                    for i in range(1, len(parts), 2):
                        try:
                            idx = int(parts[i]) - 1  # 0-based
                            desc = parts[i + 1].strip() if i + 1 < len(parts) else ""
                            if 0 <= idx < len(batch_labels) and desc:
                                img_path, page_num, label = batch_labels[idx]
                                result_map[img_path] = desc
                                logger.info(f"[BatchDesc] [{label}] 描述成功 ({len(desc)} 字符)")
                        except (ValueError, IndexError):
                            continue

                    # 兜底：如果正则没匹配上，把整个回复给第一张图
                    if not any(result_map.get(p) for p, _, _ in batch_labels):
                        img_path, page_num, label = batch_labels[0]
                        result_map[img_path] = content
                        logger.info(f"[BatchDesc] [{label}] 描述成功 (fallback, {len(content)} 字符)")
                    break

                if resp.status_code == 429:
                    wait = 3 * (attempt + 1)
                    logger.info(f"[BatchDesc] 限流，等待 {wait}s ...")
                    time.sleep(wait)
                    continue

                logger.warning(f"[BatchDesc] API 返回 {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                logger.warning(f"[BatchDesc] 请求异常 (attempt {attempt+1}): {e}")
                time.sleep(1)

    # 分批执行（不能并行，否则可能触发限流更严重）
    for batch_start in range(0, len(valid_items), batch_size):
        batch = valid_items[batch_start:batch_start + batch_size]
        _process_batch(batch)

    return result_map


# ── PDF 加载（含表格 + 图片） ──────────────────────────────────────

def pdf_loader_with_table_and_image(
    filepath: str,
    extract_images: bool = False,
    describe_images: bool = False,
    describe_tables: bool = False,
    infer_tables: bool = True,
    api_key: str = "",
) -> list[Document]:
    """使用 partition_pdf 加载 PDF，提取表格 HTML + 图片，可选批量 VLM 描述。

    - extract_images=True: 把图片提取到临时目录
    - describe_images=True:  批量 VLM 描述图片内容
    - describe_tables=True:  批量 VLM 描述表格内容（需要 pdfplumber 截图）
    - infer_tables=True:    启用 unstructured 表格结构识别
    """
    import tempfile, shutil, pdfplumber

    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY", "")

    img_tmpdir = tempfile.mkdtemp(prefix="pdf_img_")
    table_tmpdir = tempfile.mkdtemp(prefix="pdf_tables_")
    docs: list[Document] = []
    stats = {"text": 0, "table": 0, "image": 0, "described": 0}

    # 一批收集所有需要 VLM 描述的项
    vlm_queue: list[tuple[str, int, str]] = []  # (image_path, page_num, label)

    try:
        partition_kwargs = {
            "filename": filepath,
            "strategy": "hi_res",
            "hi_res_model_name": "yolox_quantized",
            "languages": ["chi_sim"],
            "infer_table_structure": infer_tables,
            "extract_images_in_pdf": extract_images,
            "extract_image_block_output_dir": img_tmpdir,
            "ocr_mode": "individual_blocks",
        }

        try:
            elements = partition_pdf(**partition_kwargs)
        except Exception:
            if infer_tables:
                logger.warning(f"[PDFLoader] infer_table_structure 失败，回退: {filepath}")
                partition_kwargs["infer_table_structure"] = False
                elements = partition_pdf(**partition_kwargs)
            else:
                raise

        known_images = set()

        for el in elements:
            meta = el.metadata.to_dict() if hasattr(el.metadata, "to_dict") else {}
            category = meta.get("category", "")
            image_path = meta.get("image_path", "")
            page_num = meta.get("page_number", 0)

            if category == "Table":
                html = meta.get("text_as_html", "")
                table_text = table_html_to_text(html) if html else str(el)
                table_doc = Document(
                    page_content=f"[表格 第{page_num}页]\n{table_text}",
                    metadata={**meta, "source": os.path.basename(filepath), "element_type": "table"},
                )
                docs.append(table_doc)
                stats["table"] += 1

            elif category in ("Image", "Figure") or image_path:
                if image_path in known_images or not image_path:
                    continue
                if not os.path.exists(image_path):
                    continue
                known_images.add(image_path)
                file_size = os.path.getsize(image_path)
                if file_size < 5000:
                    continue

                text = str(el).strip()
                if text and len(text) > 5:
                    docs.append(Document(
                        page_content=text,
                        metadata={**meta, "source": os.path.basename(filepath), "element_type": "text"},
                    ))
                    stats["text"] += 1

                docs.append(Document(
                    page_content=f"[图片 第{page_num}页] {os.path.basename(image_path)}",
                    metadata={**meta, "source": os.path.basename(filepath), "element_type": "image", "image_path": image_path},
                ))
                stats["image"] += 1

                if describe_images and api_key:
                    vlm_queue.append((image_path, page_num, "图表"))

            else:
                text = str(el).strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={**meta, "source": os.path.basename(filepath), "element_type": "text"},
                    ))
                    stats["text"] += 1

        # ── 表格 VLM：用 pdfplumber 截图 ──
        if describe_tables and api_key:
            try:
                with pdfplumber.open(filepath) as pdf:
                    for page_idx, page in enumerate(pdf.pages):
                        page_num = page_idx + 1
                        found_tables = page.find_tables()
                        if not found_tables:
                            continue
                        for t_idx, tbl in enumerate(found_tables):
                            try:
                                page_img = page.to_image(resolution=150)
                                pil_img = page_img.original.convert("RGB")
                                scale = page_img.scale
                                bbox = tbl.bbox
                                crop_box = (
                                    int(bbox[0] * scale),
                                    int((page.height - bbox[3]) * scale),
                                    int(bbox[2] * scale),
                                    int((page.height - bbox[1]) * scale),
                                )
                                cropped = pil_img.crop(crop_box)
                                img_path = os.path.join(table_tmpdir, f"table_p{page_num}_t{t_idx}.png")
                                cropped.save(img_path, "PNG")
                                vlm_queue.append((img_path, page_num, "表格"))
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"[PDFLoader] pdfplumber 表格截图失败: {e}")

        # ── 批量 VLM 描述 ──
        if vlm_queue:
            logger.info(f"[PDFLoader] 批量 VLM 描述 {len(vlm_queue)} 项 ...")
            desc_map = describe_images_batch(vlm_queue, api_key=api_key)
            for img_path, _, _ in vlm_queue:
                desc = desc_map.get(img_path)
                if desc:
                    # 从路径中反查 page_num
                    for _p, _pn, _label in vlm_queue:
                        if _p == img_path:
                            break
                    else:
                        _pn = 0
                    docs.append(Document(
                        page_content=f"[VLM描述 第{_pn}页]\n{desc}",
                        metadata={
                            "source": os.path.basename(filepath),
                            "page_number": _pn,
                            "element_type": "image_description",
                            "image_path": img_path,
                        },
                    ))
                    stats["described"] += 1

    finally:
        shutil.rmtree(img_tmpdir, ignore_errors=True)
        shutil.rmtree(table_tmpdir, ignore_errors=True)
        for _dir in (os.path.join(os.getcwd(), "figures"),):
            if os.path.isdir(_dir):
                shutil.rmtree(_dir, ignore_errors=True)

    logger.info(
        f"[PDFLoader] {os.path.basename(filepath)}: "
        f"文本={stats['text']}, 表格={stats['table']}, "
        f"图片={stats['image']}, VLM描述={stats['described']}"
    )
    return docs


# ── 混合 PDF 加载（离线表格 + VLM 描述） ───────────────────────────

def pdf_loader_hybrid(
    filepath: str,
    describe_tables: bool = False,
    describe_images: bool = False,
    api_key: str = "",
    temp_dir: str = "",
) -> list[Document]:
    """混合模式加载 PDF：unstructured（文字）+ pdfplumber（表格）+ 批量 VLM。

    - 文字：使用 unstructured hi_res (HF_HUB_OFFLINE=1)
    - 表格：使用 pdfplumber 离线提取 → 结构化 Markdown + 可选 VLM 描述
    - 图片：使用 unstructured 提取嵌入图片 → 可选 VLM 描述

    VLM 描述统一走批量接口 describe_images_batch，一次 API 请求发送多张图。
    """
    import tempfile, shutil, pdfplumber

    docs: list[Document] = []
    stats = {"text": 0, "table": 0, "image": 0, "described": 0}

    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY", "")

    # 收集所有需要 VLM 描述的项
    vlm_queue: list[tuple[str, int, str]] = []

    # ── 1. 用 unstructured 提取文本 + 嵌入图片 ──
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    img_tmpdir = tempfile.mkdtemp(prefix="pdf_images_")

    try:
        all_elements = partition_pdf(
            filename=filepath,
            strategy="hi_res",
            hi_res_model_name="yolox_quantized",
            languages=["chi_sim"],
            infer_table_structure=False,
            extract_images_in_pdf=True,
            extract_image_block_output_dir=img_tmpdir,
            ocr_mode="individual_blocks",
        )

        known_images = set()

        for el in all_elements:
            meta = el.metadata.to_dict() if hasattr(el.metadata, "to_dict") else {}
            image_path = meta.get("image_path", "")
            page_num = meta.get("page_number", 0)

            if image_path:
                text = str(el).strip()
                if text and len(text) > 5:
                    docs.append(Document(
                        page_content=text,
                        metadata={**meta, "source": os.path.basename(filepath), "element_type": "text"},
                    ))
                    stats["text"] += 1

                if image_path in known_images or not os.path.exists(image_path):
                    continue
                known_images.add(image_path)

                stats["image"] += 1

                docs.append(Document(
                    page_content=f"[图片 第{page_num}页] {os.path.basename(image_path)}",
                    metadata={
                        "source": os.path.basename(filepath),
                        "page_number": page_num,
                        "element_type": "image",
                        "image_path": image_path,
                    },
                ))

                file_size = os.path.getsize(image_path)
                if describe_images and api_key and file_size > 6000:
                    vlm_queue.append((image_path, page_num, "图表"))
            else:
                text = str(el).strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={**meta, "source": os.path.basename(filepath), "element_type": "text"},
                    ))
                    stats["text"] += 1

    except Exception as e:
        logger.warning(f"[HybridLoader] unstructured 解析失败: {e}")

    # ── 2. 用 pdfplumber 提取表格 ──
    table_tmpdir = temp_dir or tempfile.mkdtemp(prefix="pdf_tables_")

    try:
        with pdfplumber.open(filepath) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                tables = page.extract_tables()

                if not tables:
                    continue

                for t_idx, table_data in enumerate(tables):
                    if not table_data or len(table_data) < 2:
                        continue

                    clean_rows = []
                    for row in table_data:
                        clean_row = [str(c).strip() if c else "" for c in row]
                        if any(clean_row):
                            clean_rows.append(clean_row)

                    if not clean_rows:
                        continue

                    header = clean_rows[0]
                    md_lines = [" | ".join(header)]
                    md_lines.append(" | ".join(["---"] * len(header)))
                    for row in clean_rows[1:]:
                        padded = row + [""] * (len(header) - len(row))
                        md_lines.append(" | ".join(padded[:len(header)]))

                    md_table = "\n".join(md_lines)

                    # 截图表格区域，加入 VLM 队列
                    table_img_path = ""
                    if describe_tables and api_key:
                        try:
                            found_tables = page.find_tables()
                            if found_tables and t_idx < len(found_tables):
                                table_bbox = found_tables[t_idx].bbox
                                page_img = page.to_image(resolution=150)
                                pil_img = page_img.original.convert("RGB")
                                scale = page_img.scale
                                crop_box = (
                                    int(table_bbox[0] * scale),
                                    int((page.height - table_bbox[3]) * scale),
                                    int(table_bbox[2] * scale),
                                    int((page.height - table_bbox[1]) * scale),
                                )
                                cropped = pil_img.crop(crop_box)
                                table_img_path = os.path.join(table_tmpdir, f"table_p{page_num}_t{t_idx}.png")
                                cropped.save(table_img_path, "PNG")
                                vlm_queue.append((table_img_path, page_num, "表格"))
                        except Exception:
                            pass

                    docs.append(Document(
                        page_content=f"[表格 第{page_num}页]\n{md_table}",
                        metadata={
                            "source": os.path.basename(filepath),
                            "page_number": page_num,
                            "element_type": "table",
                            "table_index": t_idx,
                            "table_image_path": table_img_path or "",
                        },
                    ))
                    stats["table"] += 1

    except Exception as e:
        logger.error(f"[HybridLoader] pdfplumber 表格提取失败: {e}", exc_info=True)

    # ── 3. 批量 VLM 描述 ──
    if vlm_queue:
        logger.info(f"[HybridLoader] 批量 VLM 描述 {len(vlm_queue)} 项 ...")
        desc_map = describe_images_batch(vlm_queue, api_key=api_key)
        for img_path, page_num, _label in vlm_queue:
            desc = desc_map.get(img_path)
            if desc:
                docs.append(Document(
                    page_content=f"[VLM描述 第{page_num}页]\n{desc}",
                    metadata={
                        "source": os.path.basename(filepath),
                        "page_number": page_num,
                        "element_type": "image_description",
                        "image_path": img_path,
                    },
                ))
                stats["described"] += 1

    # 清理所有临时文件
    shutil.rmtree(img_tmpdir, ignore_errors=True)
    if not temp_dir:
        shutil.rmtree(table_tmpdir, ignore_errors=True)
    cwd_figures = os.path.join(os.getcwd(), "figures")
    if os.path.isdir(cwd_figures):
        shutil.rmtree(cwd_figures, ignore_errors=True)

    logger.info(
        f"[HybridLoader] {os.path.basename(filepath)}: "
        f"文本={stats['text']}, 表格={stats['table']}, "
        f"图片={stats['image']}, VLM描述={stats['described']}"
    )
    return docs


def get_file_md5_hex(filepath:str):         # 获取文件的md5的十六进制字符串
    if not os.path.exists(filepath):
        logger.error(f"[MD5计算]文件{filepath}不存在")
        return
    if not os.path.isfile(filepath):
        logger.error(f"[MD5计算]路径{filepath}不是文件")
        return

    md5_obj = hashlib.md5()
    chunk_size=4096     # 4KB分片，避免文件过大爆内存
    try:
            with open(filepath, "rb")as f:  # 必须二进制读取
                while chunk:=f.read(chunk_size):
                    md5_obj.update(chunk)

            md5_hex=md5_obj.hexdigest()
            return md5_hex
    except Exception as  e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None

def listdir_with_allowed_type(path:str,allowed_types:tuple[str]):   # 返回文件夹内的文件列表（允许的文件后缀）
    files= []
    if not os.path.isdir(path):
        logger.error(f"[lisdir_with_allowed_type]{path}不是文件夹")
        return allowed_types

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path,f))
    return tuple(files)

def pdf_loader(filepath:str,passwd=None)->list[Document]:

    return PyPDFLoader(filepath,passwd).load()


def pdf_loader3(filepath:str,passwd=None)->list[Document]:
    # 1. 配置加载器
    return UnstructuredPDFLoader(
    file_path=filepath,
    mode="elements",           # 关键：按语义元素拆分
    strategy="fast",  # 替换 hi_res
    languages=["chi_sim"]
).load()


def pdf_loader2(filepath: str, passwd=None) -> list[Document]:
    return UnstructuredPDFLoader(
        file_path=filepath,
        mode="elements",  # 保持元素独立，获取元数据
        strategy="hi_res",
        languages=["chi_sim"],
        hi_res_model_name="yolox_quantized",
        infer_table_structure=False,  # ★ 关键：生成表格HTML
        extract_image_block_types=["Image", "Figure"],  # ★ 替代 extract_images
    ).load()

def pdf_loader4(filepath: str, passwd=None) -> list[Document]:
    elements = partition_pdf(
        filename=filepath,
        strategy="hi_res",
        chunking_strategy="by_page",  # 在这里使用是有效的
        hi_res_model_name="yolox_quantized",
        languages=["chi_sim"],
        extract_images_in_pdf=False,   # 如果不需要图片文字，设为 False 可提速
        ocr_mode="never",              # 纯文本型PDF无需OCR
    )
    # 转成 Document 列表
    return [
        Document(page_content=el.text, metadata=el.metadata.to_dict())
        for el in elements if el.text
    ]




def txt_loader(filepath:str)->list[Document]:

    return TextLoader(filepath,encoding="utf-8").load()
