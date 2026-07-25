import os,hashlib
from utils.logger_handler import logger

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredPDFLoader
from unstructured.partition.pdf import partition_pdf


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
        mode="elements",               # 保持元素拆分（表格、图片会作为独立元素）
        strategy="hi_res",             # 必须启用，以识别表格和图片区域
        languages=["chi_sim"],
        hi_res_model_name="yolox_quantized",  # ① 使用更快的量化模型（比默认模型快 2~3 倍）
        extract_images=True,           # 提取图片（以便后续OCR或向量化）
        # ③ 开启并行加速（适用于多页PDF）
        split_pdf_page=True,           # 将PDF拆分成多页并行处理
        split_pdf_concurrency_level=5, # 并行线程数（默认为5）
        # image_output_dir=None         # 如果不需要保存到磁盘，可不设置
    ).load()

def pdf_loader4(filepath: str, passwd=None) -> list[Document]:
    elements = partition_pdf(
        filename=filepath,
        strategy="hi_res",
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
