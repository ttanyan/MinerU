/**
 * Word 转 PDF 工具
 * 使用 mammoth 将 Word 文档转换为 HTML，然后使用 html2pdf.js 转换为 PDF
 */

declare global {
  interface Window {
    mammoth: {
      convertToHtml: (arrayBuffer: ArrayBuffer) => Promise<{ value: string }>;
      convertToHtmlOptions: (options: object) => object;
    };
    html2pdf: {
      (): {
        set: (options: object) => ReturnType<typeof html2pdf>;
        from: (element: HTMLElement | string) => ReturnType<typeof html2pdf>;
        save: () => Promise<string>;
        outputPdf: (type: string) => Promise<Blob>;
      };
    };
  }
}

/**
 * 动态加载 mammoth 库
 */
async function loadMammoth(): Promise<void> {
  if (typeof window.mammoth === 'undefined') {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/mammoth@1.6.0/mammoth.browser.min.js';
      script.onload = () => resolve();
      script.onerror = () => reject(new Error('Failed to load mammoth library'));
      document.head.appendChild(script);
    });
  }
}

/**
 * 动态加载 html2pdf 库
 */
async function loadHtml2Pdf(): Promise<void> {
  if (typeof window.html2pdf === 'undefined') {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/html2pdf.js@0.10.1/dist/html2pdf.bundle.min.js';
      script.onload = () => resolve();
      script.onerror = () => reject(new Error('Failed to load html2pdf library'));
      document.head.appendChild(script);
    });
  }
}

/**
 * 将 Word 文档转换为 PDF
 * @param file Word 文件
 * @param onProgress 进度回调
 * @returns PDF 文件
 */
export async function convertWordToPdf(
  file: File,
  onProgress?: (progress: number) => void
): Promise<File> {
  console.log('convertWordToPdf called with file:', file.name, file.size)
  onProgress?.(10);

  // 加载必要的库
  console.log('Loading mammoth library...')
  await loadMammoth();
  console.log('Mammoth library loaded')
  onProgress?.(20);

  console.log('Loading html2pdf library...')
  await loadHtml2Pdf();
  console.log('html2pdf library loaded')
  onProgress?.(30);

  // 读取 Word 文件
  console.log('Reading Word file...')
  const arrayBuffer = await file.arrayBuffer();
  console.log('Word file read successfully')
  onProgress?.(50);

  // 转换为 HTML
  console.log('Converting Word to HTML...')
  const result = await window.mammoth.convertToHtml({ arrayBuffer });
  console.log('Word to HTML conversion successful, HTML length:', result.value.length)
  const html = result.value;
  onProgress?.(70);

  // 创建临时容器
  console.log('Creating temporary container...')
  const container = document.createElement('div');
  container.innerHTML = `
    <div style="font-family: 'Microsoft YaHei', Arial, sans-serif; padding: 20px;">
      ${html}
    </div>
  `;
  container.style.width = '595px'; // A4 width in pixels at 72 DPI
  document.body.appendChild(container);
  console.log('Temporary container created')

  onProgress?.(80);

  // 生成 PDF
  console.log('Generating PDF...')
  const pdfOptions = {
    margin: 10,
    filename: file.name.replace(/\.(docx?|doc)$/i, '.pdf'),
    image: { type: 'jpeg', quality: 0.98 },
    html2canvas: { scale: 2, useCORS: true },
    jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
  };

  const pdfBlob = await window.html2pdf()
    .set(pdfOptions)
    .from(container)
    .outputPdf('blob');
  console.log('PDF generated successfully, blob size:', pdfBlob.size)

  onProgress?.(95);

  // 清理
  document.body.removeChild(container);
  console.log('Temporary container removed')

  // 转换为 File 对象
  const pdfFile = new File([pdfBlob], pdfOptions.filename, {
    type: 'application/pdf'
  });
  console.log('PDF file created:', pdfFile.name, pdfFile.size)

  onProgress?.(100);

  return pdfFile;
}

/**
 * 检查文件是否为 Word 文档
 */
export function isWordFile(file: File): boolean {
  const fileName = file.name.toLowerCase();
  return fileName.endsWith('.docx') || fileName.endsWith('.doc');
}