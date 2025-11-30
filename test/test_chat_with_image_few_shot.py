
# Author: Shun Ogawa (a.k.a. "ToPo")
# Copyright (c) 2025 Shun Ogawa (a.k.a. "ToPo")
# License: Apache License Version 2.0

from augllm import AugmentedLLM, LLMInterface, PromptBuilder
#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # 使用するLLMを指定
    model_path = "gemma3:27b"

    # Few-Shotプロンプト（回答の形式を教える）
    few_shots = [
        {
            "role": "user",
            "content": "この画像の人を説明してください。",
            "images": "database/ikazuchi_mal_type1.png",
        },
        {
            "role": "assistant",
            "content": "女の子です。頭にツノがついています。髪型はおさげです。服は白いセーラー服を着ています。目の色は青です。"
        }
    ]
    
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
            system_prompt_text="",
            system_prompt_images=""
        ),
        tools=None,
    )
    
    # 入力文章
    user_test = "この画像の人の表情を説明してください。"
    user_images="database/ikazuchi_mal_type2.png"
    
    # LLMへ問い合わせ
    response = llm.respond(
        user_text=user_test,
        user_images=user_images,
        few_shot_examples=few_shots,
        stream=True
    )
    for chunk in response:
        print(chunk, end="", flush=True)
    
