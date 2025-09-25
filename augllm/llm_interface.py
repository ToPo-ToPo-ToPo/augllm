

import ollama
import subprocess
#=======================================================================================================
# LLMを使うためのインターフェース
#=======================================================================================================
class LLMInterface:

    #---------------------------------------------------------------
    # パラメータの初期化
    #---------------------------------------------------------------
    def __init__(self, model_name, options=None):
        self.model = model_name
        self.options = options or {}

    #---------------------------------------------------------------
    # 回答を返す
    #---------------------------------------------------------------
    def chat(self, messages, stream=False):
        return ollama.chat(
            model=self.model,
            messages=messages,
            stream=stream,
            options=self.options
        )
    #--------------------------------------------------------------------------------------
    # Ollamaでロードしたモデルを止める
    #--------------------------------------------------------------------------------------
    def free_model(self):
        
        try:
            result = subprocess.run(
                ["ollama", "stop", self.model],
                check=True,
                capture_output=True, # stdout/stderrをキャプチャ
                text=True
            )
        
        except subprocess.CalledProcessError as e:
            # stop コマンドはモデルがロードされていない場合エラーを返すことがある
            print(f"モデル '{self.model}' の停止中にエラーが発生しました (モデルがロードされていない可能性があります): {e.stderr}")
