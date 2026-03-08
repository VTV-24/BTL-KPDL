"""
src/evaluation/report.py
========================
Chức năng xuất báo cáo PDF từ notebook.
"""

import os
import subprocess
from pathlib import Path


def export_notebook_to_pdf(notebook_path: str, output_path: str) -> bool:
    """
    Xuất notebook Jupyter thành file PDF.

    Args:
        notebook_path: Đường dẫn đến file notebook (.ipynb)
        output_path: Đường dẫn output cho file PDF

    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        # Đảm bảo đường dẫn tuyệt đối
        notebook_path = Path(notebook_path).resolve()
        output_path = Path(output_path).resolve()

        # Tạo thư mục output nếu chưa có
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Lệnh nbconvert
        cmd = [
            "jupyter", "nbconvert",
            "--to", "pdf",
            str(notebook_path),
            "--output", output_path.name,
            "--output-dir", str(output_path.parent)
        ]

        # Chạy lệnh
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Đã xuất PDF thành công: {output_path}")
            return True
        else:
            print(f"❌ Lỗi xuất PDF: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Lỗi không mong muốn: {e}")
        return False


def generate_final_report(notebook_dir: str = "notebooks",
                         reports_dir: str = "outputs/reports") -> str:
    """
    Tạo báo cáo cuối cùng từ notebook 07.

    Args:
        notebook_dir: Thư mục chứa notebooks
        reports_dir: Thư mục output cho reports

    Returns:
        str: Đường dẫn đến file PDF đã tạo
    """
    notebook_path = os.path.join(notebook_dir, "07_evaluation_report.ipynb")
    pdf_path = os.path.join(reports_dir, "final_report.pdf")

    success = export_notebook_to_pdf(notebook_path, pdf_path)

    if success:
        return pdf_path
    else:
        raise RuntimeError("Không thể tạo báo cáo PDF")


if __name__ == "__main__":
    # Chạy từ root directory
    generate_final_report()