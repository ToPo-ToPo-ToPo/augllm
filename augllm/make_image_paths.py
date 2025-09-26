
#  画像処理関連の関数
from .image_controller import shrink_and_cache_image, is_image_file 
#----------------------------------------------------------------------
# 画像のパスのリストを更新
#----------------------------------------------------------------------
'''def update_image_paths(image_paths, user_images):

    # 単一画像の場合はリストに変換
    if isinstance(user_images, str):
        user_images = [user_images]
        
    # リストとして画像を追加
    for image in user_images: 
        if is_image_file(image):
            try:
                path = shrink_and_cache_image(image_path=image, cache_dir=cache_dir)
                image_paths.append(path)
            except Exception as e:
                print(f"Warning: Failed to shrink image: {e}")
    
    #
    return image_paths'''

#----------------------------------------------------------------------
# 画像データを入力し、画像のパスのリストを返す
#----------------------------------------------------------------------
def make_image_paths(user_images, cache_dir):

    #
    image_paths = []

    # 単一画像の場合はリストに変換
    if isinstance(user_images, str):
        user_images = [user_images]
        
    # リストとして画像を追加
    for image in user_images: 
        if is_image_file(image):
            try:
                path = shrink_and_cache_image(image_path=image, cache_dir=cache_dir)
                image_paths.append(path)
            except Exception as e:
                print(f"Warning: Failed to shrink image: {e}")
    
    #
    return image_paths
