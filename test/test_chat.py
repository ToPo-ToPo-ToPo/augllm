
from augllm import AugmentedLLM, LLMInterface, PromptBuilder
#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def chat(llm):
    print("ローカルLLMチャットを開始します。終了するには 'exit' を入力してください。")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("チャットを終了します。")
            break
        
        # LLMへ問い合わせ
        response = llm.respond(
            user_text=user_input, 
            user_images=None, 
            stream=True
        )
        for chunk in response:
            print(chunk, end="", flush=True)
        
        print("\n")
        
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
    
    #
    chat(llm=llm)
