
import io
from PIL import Image
import os
import uuid
from tqdm import tqdm
import fitz # PyMuPDFのものを使用
import base64
import hashlib
#--------------------------------------------------------------------------------------
# キャッシュを設定
#--------------------------------------------------------------------------------------
def get_image_cache_path(image_path, cache_dir):
    with open(image_path, 'rb') as f:
        data = f.read()
        h = hashlib.sha256(data).hexdigest()
    filename = f"{h}.png"
    return os.path.join(cache_dir, filename)

#--------------------------------------------------------------------------------------
# 画像をリサイズして キャッシュで画像を保存し、LLMにパスを返す関数
#--------------------------------------------------------------------------------------
def shrink_and_cache_image(image_path, cache_dir, max_size=512):
    os.makedirs(cache_dir, exist_ok=True)
    cached_path = get_image_cache_path(image_path, cache_dir)

    if os.path.exists(cached_path):
        return cached_path  # キャッシュ済み

    # 画像処理して保存
    image = Image.open(image_path)
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    image.thumbnail((max_size, max_size), Image.LANCZOS)
    image.save(cached_path, format='PNG')

    return cached_path

#--------------------------------------------------------------------------------------
# 画像をリサイズして Base64 エンコードされた文字列で返す関数（Ollamaに安全に渡せる形式）
#--------------------------------------------------------------------------------------
def shrink_image(image_path, max_size=512):
    try:
        image = Image.open(image_path)
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        image.thumbnail((max_size, max_size), Image.LANCZOS)

        # メモリ上のバッファにPNGで保存
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)

        # base64文字列に変換して返す（Ollamaの仕様に対応）
        base64_str = base64.b64encode(buf.read()).decode('utf-8')
        return base64_str
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        raise

#--------------------------------------------------------------------------------------
# パスで画像を取得する場合
#--------------------------------------------------------------------------------------
def shrink_image_to_path(image_path, cache_dir, max_size=512):
    try:
        #
        image = Image.open(image_path)

        # RGBA や P モードを RGB に変換
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')

        # 最大サイズにリサイズ
        image.thumbnail((max_size, max_size), Image.LANCZOS)

        # ディレクトリが存在しない場合は作成
        os.makedirs(cache_dir, exist_ok=True)

        # 一時ファイル名をユニークに生成
        filename = f"ollama_img_{uuid.uuid4().hex}.png"
        save_path = os.path.join(cache_dir, filename)

        # 画像保存
        image.save(save_path, format='PNG')

        #
        return save_path

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        raise

#--------------------------------------------------------------------------------------
# 一時保存した画像を全て削除
#--------------------------------------------------------------------------------------
def delete_temp_images(temp_image_paths):
    for path in temp_image_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Warning: Failed to delete temporary image {path}: {e}")

#--------------------------------------------------------------------------------------
# 入力が画像ファイルかどうかを判定する関数 
#--------------------------------------------------------------------------------------
def is_image_file(file_path):
    
    if not file_path or not isinstance(file_path, str):
        return False
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'] # webpも追加
    _, ext = os.path.splitext(file_path.lower())
    return ext in image_extensions and os.path.exists(file_path)

#--------------------------------------------------------------------------------------
# PDFを画像に変換して保存
#--------------------------------------------------------------------------------------
def pdf_to_images(pdf_path, output_dir):
    
    # PDFの存在確認
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDFファイルが見つかりません: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    image_paths = []
    print(f"Converting PDF '{os.path.basename(pdf_path)}' to images...")
    for page_num in tqdm(range(len(doc)), desc="Extracting pages"):
        page = doc.load_page(page_num)
        # 解像度を少し上げてみる (必要に応じて調整)
        pix = page.get_pixmap(dpi=150) 
        img_path = os.path.join(output_dir, f"{page_num}.png")
        pix.save(img_path)
        image_paths.append(img_path)
    doc.close()
    print(f"Saved {len(image_paths)} images to '{output_dir}'")

    # エラーチェック
    if not image_paths:
        print("No images were extracted from the PDF.")
        exit()
    
    #
    return image_paths

