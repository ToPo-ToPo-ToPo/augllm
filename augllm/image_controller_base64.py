
import io
import os
import base64
from PIL import Image
import fitz  # PyMuPDF
#----------------------------------------------------------------------
# ユーザー入力（パスのリストや単一パス）を受け取り、
# LLMに渡せるBase64文字列のリストを返すメイン関数
#----------------------------------------------------------------------
def process_images_to_base64(user_images, max_size=2048):
    
    encoded_images = []

    # 単一のパス文字列ならリストに変換
    if isinstance(user_images, str):
        user_images = [user_images]
        
    if not user_images:
        return []
    
    for entry in user_images:
        # 画像ファイルの場合
        if is_image_file(entry):
            try:
                b64 = encode_image_to_base64(entry, max_size=max_size)
                encoded_images.append(b64)
            except Exception as e:
                print(f"Warning: Failed to process image {entry}: {e}")
        
        # PDFファイルの場合（全ページを画像化して追加）
        elif entry.lower().endswith(".pdf") and os.path.exists(entry):
            try:
                pdf_b64_list = pdf_to_base64_images(entry, max_size=max_size)
                encoded_images.extend(pdf_b64_list)
            except Exception as e:
                print(f"Warning: Failed to process PDF {entry}: {e}")
        
        else:
            # ファイルが見つからない、または対応していない形式
            print(f"Warning: Skipped invalid file path: {entry}")

    return encoded_images

#----------------------------------------------------------------------
# 画像パスを受け取り、リサイズしてBase64文字列を返す
#----------------------------------------------------------------------
def encode_image_to_base64(image_path, max_size=2048):
    try:
        # 画像を開く
        with Image.open(image_path) as img:
            # RGBAなどをRGBに変換
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # リサイズ（アスペクト比維持）
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            
            # メモリ上のバッファに保存
            buffered = io.BytesIO()

            # フォーマット
            # img.save(buffered, format="JPEG", quality=85)  # JPEG: 軽量だがノイズあり
            img.save(buffered, format="PNG")  # 基本は PNG (画質最優先・文書向け)
            # img.save(buffered, format="WEBP", lossless=True)  # WebP (PNG並みの画質で、サイズは軽量。最近のモデルなら対応していることが多い)
            
            # Base64エンコード
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str
            
    except Exception as e:
        raise RuntimeError(f"Error encoding image {image_path}: {e}")

#----------------------------------------------------------------------
# PDFを読み込み、各ページを画像化してBase64リストで返す（ファイル保存なし）
#----------------------------------------------------------------------
def pdf_to_base64_images(pdf_path, max_size=2048):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    b64_list = []
    
    print(f"Converting PDF '{os.path.basename(pdf_path)}' for high-precision reading...")
    
    for page in doc:
        # 【重要】DPIを上げる
        # デフォルト(72dpi)だと文字がボヤけます。200dpi程度あれば細かい数字もクッキリします。
        # max_sizeで後でリサイズされるとしても、ソースの品質が良いことが重要です。
        pix = page.get_pixmap(dpi=200) 
        
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # リサイズ処理
        # 技術資料はアスペクト比が命なので、thumbnail（比率維持）は必須
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        
        buffered = io.BytesIO()
        
        # 【重要】PNG形式を指定 (数値を守るため)
        img.save(buffered, format="PNG")
        
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        b64_list.append(img_str)
        
    doc.close()
    return b64_list

#----------------------------------------------------------------------
# 拡張子判定
#----------------------------------------------------------------------
def is_image_file(file_path):
    if not file_path or not isinstance(file_path, str):
        return False
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
    _, ext = os.path.splitext(file_path.lower())
    return ext in image_extensions and os.path.exists(file_path)