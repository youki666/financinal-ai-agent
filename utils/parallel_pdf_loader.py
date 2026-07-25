import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
import traceback
from utils.logger_handler import logger

# ========== 单文件处理函数（可独立测试） ==========
def process_single_pdf(
    filepath: str,
    strategy: str = "hi_res",
    model_name: str = "yolox_quantized",
    languages: list = ["chi_sim"],
    extract_images: bool = False,
    ocr_mode: str = "never"
) -> list[Document]:
    """
    处理单个PDF文件，返回Document列表。
    所有参数均可自定义。
    """
    try:
        elements = partition_pdf(
            filename=filepath,
            strategy=strategy,
            hi_res_model_name=model_name,
            languages=languages,
            extract_images_in_pdf=extract_images,
            ocr_mode=ocr_mode,
        )
        docs = []
        for el in elements:
            if hasattr(el, "text") and el.text:
                metadata = el.metadata.to_dict() if hasattr(el.metadata, "to_dict") else {}
                docs.append(Document(page_content=el.text, metadata=metadata))
                # logger.info(docs)
                # logger.info("======docs==========")
        return docs
    except Exception as e:
        print(f"❌ 处理失败 {filepath}: {e}")
        traceback.print_exc()
        return []   # 返回空列表，不中断整体流程

# ========== 批量并行处理 ==========
def process_multiple_pdfs(
    file_list: list,
    max_workers: int = 4,
    desc: str = "处理PDF文件",
    **kwargs
) -> list[Document]:
    """
    并行处理多个PDF文件。
    :param file_list: PDF文件路径列表
    :param max_workers: 并行进程数（建议 ≤ CPU核心数）
    :param desc: 进度条描述
    :param kwargs: 传递给 process_single_pdf 的其他参数
    :return: 所有Document对象的列表
    """
    all_docs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_pdf, f, **kwargs): f for f in file_list
        }
        # 使用tqdm显示进度
        for future in tqdm(as_completed(future_to_file), total=len(file_list), desc=desc):
            filepath = future_to_file[future]
            try:
                docs = future.result()
                all_docs.extend(docs)
            except Exception as e:
                print(f"⚠️ {filepath} 发生意外错误: {e}")
    return all_docs

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 1. 设置PDF文件夹路径
    pdf_dir = "./pdfs"
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
        print(f"📁 请将PDF文件放入 {pdf_dir} 目录后重新运行")
    else:
        pdf_files = [
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if f.lower().endswith('.pdf')
        ]
        if not pdf_files:
            print("⚠️ 没有找到PDF文件")
        else:
            print(f"📄 找到 {len(pdf_files)} 个PDF文件，开始并行处理...")
            # 2. 调用并行处理（可根据需要调整参数）
            all_documents = process_multiple_pdfs(
                file_list=pdf_files,
                max_workers=4,            # 根据CPU核心数调整
                strategy="hi_res",
                model_name="yolox_quantized",
                languages=["chi_sim"],
                extract_images=False,
                ocr_mode="never"
            )
            print(f"✅ 共提取 {len(all_documents)} 个Document元素")
            # 可选：保存结果（pickle/json）
            # import pickle
            # with open("extracted_docs.pkl", "wb") as f:
            #     pickle.dump(all_documents, f)