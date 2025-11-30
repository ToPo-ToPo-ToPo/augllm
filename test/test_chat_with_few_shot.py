
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
            "content": "あなたの名前は何ですか？"
        },
        {
            "role": "assistant",
            "content": "わしの名前は、いかずちまるじゃ。未来の機械設計システムじゃ。"
        },
        {
            "role": "user",
            "content": "今日は何をしましたか？"
        },
        {
            "role": "assistant",
            "content": "サッカーをしていたのじゃ。"
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
    user_test = "何をしてましたか？"
    
    # LLMへ問い合わせ
    response = llm.respond(
        user_text=user_test,
        few_shot_examples=few_shots,
        stream=True
    )
    for chunk in response:
        print(chunk, end="", flush=True)
    
