
# Author: Shun Ogawa (a.k.a. "ToPo")
# Copyright (c) 2025 Shun Ogawa (a.k.a. "ToPo")
# License: Apache License Version 2.0

from augllm import AugmentedLLM, LLMInterface, PromptBuilder
#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # 使用するLLMを指定
    model_path = "gemma3:4b"
    
    # モデルのインスタンスを生成
    llm = AugmentedLLM(
        llm=LLMInterface(
            model_name=model_path,
            options = {
                "temperature": 0.5,
                "top_k": 20.0,
                "top_p": 0.95,
            }
        ),
        prompt_builder=PromptBuilder(
            system_prompt_text="あなたは、AIアシスタントです。",
            system_prompt_images=""
        ),
        cache_dir="cache/",
        tools=None,
    )
    
    # 入力文章
    user_test = "次の画像の内容について説明してください。"
    
    # 入力画像
    # データが保存されている箇所
    DATA_PATH = "database/"
    user_images = [
        DATA_PATH + "bc.jpg",
    ]
    
    # LLMへ問い合わせ
    response = llm.respond(
        user_text=user_test, 
        user_images=user_images, 
        stream=True
    )
    for chunk in response:
        print(chunk, end="", flush=True)
    
