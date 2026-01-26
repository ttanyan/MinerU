# Copyright (c) Opendatalab. All rights reserved.

import base64
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import click
import gradio as gr
from gradio_pdf import PDF
from loguru import logger

log_level = os.getenv("MINERU_LOG_LEVEL", "INFO").upper()
logger.remove()  # 移除默认handler
logger.add(sys.stderr, level=log_level)  # 添加新handler

from mineru.cli.common import prepare_env, read_fn, aio_do_parse, pdf_suffixes, image_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.hash_utils import str_sha256


# --- 新增：标准的树状思维导图生成函数 ---
def md_to_markmap_html(md_content):
    """
    将 Markdown 转换为标准的树状思维导图 (Markmap)
    """
    if not md_content:
        return ""

    # 转义 Markdown 中的反引号和符号，防止破坏 JS 字符串
    safe_md = md_content.replace('`', '\\`').replace('$', '\\$')

    # 完整的 HTML + Markmap 渲染引擎
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            html, body, #mindmap {{ width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden; background-color: white; }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-view"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-toolbar"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-lib"></script>
    </head>
    <body>
        <svg id="mindmap"></svg>
        <script>
            // 等待页面完全加载后再初始化
            document.addEventListener('DOMContentLoaded', function() {{
                try {{
                    const {{ Markmap, loadCSS, loadJS, Transformer }} = window.markmap;
                    const transformer = new Transformer();
                    const {{ root, features }} = transformer.transform(`{safe_md}`);
                    const {{ styles, scripts }} = transformer.getAssets();

                    if (styles) loadCSS(styles);
                    if (scripts) loadJS(scripts);

                    // 增加延迟确保资源加载完成
                    setTimeout(() => {{
                        const mm = Markmap.create('#mindmap', {{
                            autoFit: true,
                            fitRatio: 0.9,
                            initialExpandLevel: -1
                        }}, root);

                        // 添加错误处理
                        if (mm) {{
                            console.log("Markmap created successfully");
                        }} else {{
                            console.error("Failed to create Markmap");
                        }}
                    }}, 500); // 延迟500毫秒以确保资源加载
                }} catch (error) {{
                    console.error('Error initializing markmap:', error);
                }}
            }});
        </script>
    </body>
    </html>
    """

    # 使用 iframe 封装，彻底解决渲染失效问题
    iframe_srcdoc = full_html.replace('"', '&quot;')
    iframe_code = f"""
    <iframe srcdoc="{iframe_srcdoc}" style="width: 100%; height: 800px; border: 1px solid #ddd; border-radius: 8px;" sandbox="allow-scripts"></iframe>
    """
    return iframe_code


# ────────────── 新增：根据上一级标题自动补全下一级标题 ──────────────
def auto_promote_paragraphs_to_subheading(text):
    lines = text.splitlines()
    result = []
    in_section = False
    empty_count = 0

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('# '):
            result.append(line)
            in_section = True
            empty_count = 0
            continue

        if stripped.startswith('#'):
            result.append(line)
            in_section = False
            empty_count = 0
            continue

        if not stripped:
            result.append(line)
            empty_count += 1
            if empty_count >= 2:
                in_section = False
            continue

        # 跳过图片、列表、代码等特殊行
        if (
                stripped.startswith('![') or
                stripped.startswith('>') or
                stripped.startswith('```') or
                re.match(r'^[-*+] ', stripped) or
                re.match(r'^\d+\. ', stripped)
        ):
            result.append(line)
            empty_count = 0
            continue

        empty_count = 0
        if in_section:
            result.append('## ' + stripped)
        else:
            result.append(line)

    return '\n'.join(result)


async def parse_pdf(doc_path, output_dir, end_page_id, is_ocr, formula_enable, table_enable, language, backend, url):
    os.makedirs(output_dir, exist_ok=True)
    try:
        file_name = f'{safe_stem(Path(doc_path).stem)}_{time.strftime("%y%m%d_%H%M%S")}'
        pdf_data = read_fn(doc_path)
        if backend.startswith("vlm"):
            parse_method = "vlm"
        else:
            parse_method = 'ocr' if is_ocr else 'auto'

        if backend.startswith("hybrid"):
            env_name = f"hybrid_{parse_method}"
        else:
            env_name = parse_method

        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, env_name)
        await aio_do_parse(
            output_dir=output_dir,
            pdf_file_names=[file_name],
            pdf_bytes_list=[pdf_data],
            p_lang_list=[language],
            parse_method=parse_method,
            end_page_id=end_page_id,
            formula_enable=formula_enable,
            table_enable=table_enable,
            backend=backend,
            server_url=url,
        )
        return local_md_dir, file_name
    except Exception as e:
        logger.exception(e)
        return None


def compress_directory_to_zip(directory_path, output_zip_path):
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, directory_path)
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def replace_image_with_base64(markdown_text, image_dir_path):
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'

    def replace(match):
        relative_path = match.group(1)
        if relative_path.endswith('.jpg'):
            full_path = os.path.join(image_dir_path, relative_path)
            base64_image = image_to_base64(full_path)
            return f'![{relative_path}](data:image/jpeg;base64,{base64_image})'
        return match.group(0)

    return re.sub(pattern, replace, markdown_text)


async def to_markdown(file_path, end_pages=10, is_ocr=False, formula_enable=True, table_enable=True, language="ch",
                      backend="pipeline", url=None):
    # 如果language包含()，则提取括号前的内容作为实际语言
    if '(' in language and ')' in language:
        language = language.split('(')[0].strip()
    file_path = to_pdf(file_path)
    # 获取识别的md文件以及压缩包文件路径
    local_md_dir, file_name = await parse_pdf(file_path, './output', end_pages - 1, is_ocr, formula_enable,
                                              table_enable, language, backend, url)
    archive_zip_path = os.path.join('./output', str_sha256(local_md_dir) + '.zip')
    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    if zip_archive_success == 0:
        logger.info('Compression successful')
    else:
        logger.error('Compression failed')
    md_path = os.path.join(local_md_dir, file_name + '.md')
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()

    # ────────────── 自动补全：根据 # 标题补全后续段落为 ## ──────────────
    txt_content = auto_promote_paragraphs_to_subheading(txt_content)
    # ────────────────────────────────────────────────────────────────

    md_content = replace_image_with_base64(txt_content, local_md_dir)

    # 生成思维导图HTML - 使用新的实现
    mind_map_html = md_to_markmap_html(txt_content)

    # 返回转换后的PDF路径
    new_pdf_path = os.path.join(local_md_dir, file_name + '_layout.pdf')

    return md_content, txt_content, archive_zip_path, new_pdf_path, mind_map_html


latex_delimiters_type_all = [
    {'left': '$$', 'right': '$$', 'display': True},
    {'left': '$', 'right': '$', 'display': False},
    {'left': '\\(', 'right': '\\)', 'display': False},
    {'left': '\\[', 'right': '\\]', 'display': True},
]

latex_delimiters_type_a = [
    {'left': '$$', 'right': '$$', 'display': True},
    {'left': '$', 'right': '$', 'display': False},
]

latex_delimiters_type_b = [
    {'left': '\\(', 'right': '\\)', 'display': False},
    {'left': '\\[', 'right': '\\]', 'display': True},
]

other_lang = ['ch (Chinese, English, Chinese Traditional)', 'en (English)', 'korean', 'japan']
all_lang = [*other_lang]


def safe_stem(file_path):
    stem = Path(file_path).stem
    return re.sub(r'[^\w.]', '_', stem)


def to_pdf(file_path):
    if file_path is None: return None
    pdf_bytes = read_fn(file_path)
    unique_filename = f'{safe_stem(file_path)}.pdf'
    tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)
    with open(tmp_file_path, 'wb') as tmp_pdf_file:
        tmp_pdf_file.write(pdf_bytes)
    return tmp_file_path


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option(
    '--enable-example',
    'example_enable',
    type=bool,
    help="Enable example files for input."
         "The example files to be input need to be placed in the `example` folder within the directory where the command is currently executed.",
    default=True,
)
@click.option(
    '--enable-http-client',
    'http_client_enable',
    type=bool,
    help="Enable http-client backend to link openai-compatible servers.",
    default=False,
)
@click.option(
    '--enable-api',
    'api_enable',
    type=bool,
    help="Enable gradio API for serving the application.",
    default=True,
)
@click.option(
    '--max-convert-pages',
    'max_convert_pages',
    type=int,
    help="Set the maximum number of pages to convert from PDF to Markdown.",
    default=1000,
)
@click.option(
    '--server-name',
    'server_name',
    type=str,
    help="Set the server name for the Gradio app.",
    default=None,
)
@click.option(
    '--server-port',
    'server_port',
    type=int,
    help="Set the server port for the Gradio app.",
    default=None,
)
@click.option(
    '--latex-delimiters-type',
    'latex_delimiters_type',
    type=click.Choice(['a', 'b', 'all']),
    help="Set the type of LaTeX delimiters to use in Markdown rendering:"
         "'a' for type '$', 'b' for type '()[]', 'all' for both types.",
    default='all',
)
def main(ctx,
         example_enable,
         http_client_enable,
         api_enable, max_convert_pages,
         server_name, server_port, latex_delimiters_type, **kwargs
         ):
    # 检测系统语言环境，默认为中文
    import locale
    import os

    def detect_language():
        # 检查环境变量
        lang = os.getenv('LANG', '')
        if 'zh' in lang.lower() or 'chinese' in lang.lower():
            return 'zh'

        # 检查系统默认locale
        try:
            default_locale = locale.getdefaultlocale()[0]
            if default_locale and 'zh' in default_locale.lower():
                return 'zh'
        except:
            pass

        # 默认返回中文
        return 'zh'

    detected_lang = detect_language()

    # 创建 i18n 实例，支持中英文，默认为中文
    i18n = gr.I18n(
        en={
            "upload_file": "Please upload a PDF or image",
            "max_pages": "Max convert pages",
            "backend": "Backend",
            "server_url": "Server URL",
            "server_url_info": "OpenAI-compatible server URL for http-client backend.",
            "recognition_options": "Recognition Options",
            "table_enable": "Enable table recognition",
            "table_info": "If disabled, tables will be shown as images.",
            "formula_label_vlm": "Enable display formula recognition",
            "formula_label_pipeline": "Enable formula recognition",
            "formula_label_hybrid": "Enable inline formula recognition",
            "formula_info_vlm": "If disabled, display formulas will be shown as images.",
            "formula_info_pipeline": "If disabled, display formulas will be shown as images, and inline formulas will not be detected or parsed.",
            "formula_info_hybrid": "If disabled, inline formulas will not be detected or parsed.",
            "ocr_language": "OCR Language",
            "ocr_language_info": "Select the OCR language for image-based PDFs and images.",
            "force_ocr": "Force enable OCR",
            "force_ocr_info": "Enable only if the result is extremely poor. Requires correct OCR language.",
            "convert": "Convert",
            "clear": "Clear",
            "pdf_preview": "PDF preview",
            "examples": "Examples:",
            "convert_result": "Convert result",
            "md_rendering": "Markdown rendering",
            "md_text": "Markdown text",
            "mind_map": "Mind Map",  # 新增
            "backend_info_vlm": "High-precision parsing via VLM, supports Chinese and English documents only.",
            "backend_info_pipeline": "Traditional Multi-model pipeline parsing, supports multiple languages, hallucination-free.",
            "backend_info_hybrid": "High-precision hybrid parsing, supports multiple languages.",
            "backend_info_default": "Select the backend engine for document parsing.",
        },
        zh={
            "upload_file": "请上传 PDF 或图片",
            "max_pages": "最大转换页数",
            "backend": "解析后端",
            "server_url": "服务器地址",
            "server_url_info": "http-client 后端的 OpenAI 兼容服务器地址。",
            "recognition_options": "识别选项",
            "table_enable": "启用表格识别",
            "table_info": "禁用后，表格将显示为图片。",
            "formula_label_vlm": "启用行间公式识别",
            "formula_label_pipeline": "启用公式识别",
            "formula_label_hybrid": "启用行内公式识别",
            "formula_info_vlm": "禁用后，行间公式将显示为图片。",
            "formula_info_pipeline": "禁用后，行间公式将显示为图片，行内公式将不会被检测或解析。",
            "formula_info_hybrid": "禁用后，行内公式将不会被检测或解析。",
            "ocr_language": "OCR 语言",
            "ocr_language_info": "为扫描版 PDF 和图片选择 OCR 语言。",
            "force_ocr": "强制启用 OCR",
            "force_ocr_info": "仅在识别效果极差时启用，需选择正确的 OCR 语言。",
            "convert": "转换",
            "clear": "清除",
            "pdf_preview": "PDF 预览",
            "examples": "示例：",
            "convert_result": "转换结果",
            "md_rendering": "Markdown 渲染",
            "md_text": "Markdown 文本",
            "mind_map": "思维导图",  # 新增
            "backend_info_vlm": "多模态大模型高精度解析，仅支持中英文文档。",
            "backend_info_pipeline": "传统多模型管道解析，支持多语言，无幻觉。",
            "backend_info_hybrid": "高精度混合解析，支持多语言。",
            "backend_info_default": "选择文档解析的后端引擎。",
        },
    )

    # 根据后端类型获取公式识别标签（闭包函数以支持 i18n）
    def get_formula_label(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("formula_label_vlm")
        elif backend_choice == "pipeline":
            return i18n("formula_label_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("formula_label_hybrid")
        else:
            return i18n("formula_label_pipeline")

    def get_formula_info(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("formula_info_vlm")
        elif backend_choice == "pipeline":
            return i18n("formula_info_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("formula_info_hybrid")
        else:
            return ""

    def get_backend_info(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("backend_info_vlm")
        elif backend_choice == "pipeline":
            return i18n("backend_info_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("backend_info_hybrid")
        else:
            return i18n("backend_info_default")

    # 更新界面函数
    def update_interface(backend_choice):
        formula_label_update = gr.update(label=get_formula_label(backend_choice), info=get_formula_info(backend_choice))
        backend_info_update = gr.update(info=get_backend_info(backend_choice))
        if "http-client" in backend_choice:
            client_options_update = gr.update(visible=True)
        else:
            client_options_update = gr.update(visible=False)
        if "vlm" in backend_choice:
            ocr_options_update = gr.update(visible=False)
        else:
            ocr_options_update = gr.update(visible=True)

        return client_options_update, ocr_options_update, formula_label_update, backend_info_update

    kwargs.update(arg_parse(ctx))

    if latex_delimiters_type == 'a':
        latex_delimiters = latex_delimiters_type_a
    elif latex_delimiters_type == 'b':
        latex_delimiters = latex_delimiters_type_b
    elif latex_delimiters_type == 'all':
        latex_delimiters = latex_delimiters_type_all
    else:
        raise ValueError(f"Invalid latex delimiters type: {latex_delimiters_type}.")

    vlm_engine = get_vlm_engine("auto", is_async=True)
    if vlm_engine in ["transformers", "mlx-engine"]:
        http_client_enable = True
    else:
        try:
            logger.info(f"Start init {vlm_engine}...")
            from mineru.backend.vlm.vlm_analyze import ModelSingleton
            model_singleton = ModelSingleton()
            predictor = model_singleton.get_model(
                vlm_engine,
                None,
                None,
                **kwargs
            )
            logger.info(f"{vlm_engine} init successfully.")
        except Exception as e:
            logger.exception(e)

    suffixes = [f".{suffix}" for suffix in pdf_suffixes + image_suffixes]
    with gr.Blocks(title="多模态思维导图助手",
                   fill_height=True) as demo:
        # gr.HTML(header)
        gr.HTML("<h1 style='text-align: center;'>多模态思维导图助手</h1>")
        with gr.Row():
            with gr.Column(variant='panel', scale=5):
                with gr.Row():
                    input_file = gr.File(label=i18n("upload_file"), file_types=suffixes)
                with gr.Row():
                    max_pages = gr.Slider(1, max_convert_pages, max_convert_pages, step=1, label=i18n("max_pages"))
                with gr.Row():
                    drop_list = ["pipeline", "vlm-auto-engine", "hybrid-auto-engine"]
                    preferred_option = "hybrid-auto-engine"
                    if http_client_enable:
                        drop_list.extend(["vlm-http-client", "hybrid-http-client"])
                    backend = gr.Dropdown(drop_list, label=i18n("backend"), value=preferred_option,
                                          info=get_backend_info(preferred_option))
                with gr.Row(visible=False) as client_options:
                    url = gr.Textbox(label=i18n("server_url"), value='http://localhost:30000',
                                     placeholder='http://localhost:30000', info=i18n("server_url_info"))
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown(i18n("recognition_options"))
                        table_enable = gr.Checkbox(label=i18n("table_enable"), value=True, info=i18n("table_info"))
                        formula_enable = gr.Checkbox(label=get_formula_label(preferred_option), value=True,
                                                     info=get_formula_info(preferred_option))
                    with gr.Column(visible=False) as ocr_options:
                        language = gr.Dropdown(all_lang, label=i18n("ocr_language"),
                                               value='ch (Chinese, English, Chinese Traditional)',
                                               info=i18n("ocr_language_info"))
                        is_ocr = gr.Checkbox(label=i18n("force_ocr"), value=False, info=i18n("force_ocr_info"))
                with gr.Row():
                    change_bu = gr.Button(i18n("convert"))
                    clear_bu = gr.ClearButton(value=i18n("clear"))
                pdf_show = PDF(label=i18n("pdf_preview"), interactive=False, visible=True, height=800)
                if example_enable:
                    example_root = os.path.join(os.getcwd(), 'examples')
                    if os.path.exists(example_root):
                        gr.Examples(
                            label=i18n("examples"),
                            examples=[os.path.join(example_root, _) for _ in os.listdir(example_root) if
                                      _.endswith(tuple(suffixes))],
                            inputs=input_file
                        )

            with gr.Column(variant='panel', scale=5):
                output_file = gr.File(label=i18n("convert_result"), interactive=False)
                with gr.Tabs():
                    with gr.Tab(i18n("md_rendering")):
                        md = gr.Markdown(
                            label=i18n("md_rendering"),
                            height=1200,
                            show_copy_button=True,
                            latex_delimiters=latex_delimiters,
                            line_breaks=True
                        )
                    with gr.Tab(i18n("md_text")):
                        md_text = gr.TextArea(
                            lines=45,
                            show_copy_button=True,
                            label=i18n("md_text")
                        )
                    with gr.Tab(i18n("mind_map")):  # 新增的思维导图tab
                        mind_map = gr.HTML(label=i18n("mind_map"))  # 使用HTML组件来渲染markmap

        # 添加事件处理
        backend.change(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options, formula_enable, backend],
            api_name=False
        )
        # 添加demo.load事件，在页面加载时触发一次界面更新
        demo.load(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options, formula_enable, backend],
            api_name=False
        )
        clear_bu.add([input_file, md, pdf_show, md_text, output_file, is_ocr, mind_map])

        input_file.change(
            fn=to_pdf,
            inputs=input_file,
            outputs=pdf_show,
            api_name="to_pdf" if api_enable else False
        )
        change_bu.click(
            fn=to_markdown,
            inputs=[input_file, max_pages, is_ocr, formula_enable, table_enable, language, backend, url],
            outputs=[md, md_text, output_file, pdf_show, mind_map],  # 添加mind_map输出
            api_name="to_markdown" if api_enable else False
        )

    footer_links = ["gradio", "settings"]
    if api_enable:
        footer_links.append("api")
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        show_api=api_enable,
        i18n=i18n,
        # title = "多模态思维导图助手",
        # favicon_path = "logo.png"
    )


if __name__ == '__main__':
    main()