
#================================================================================
# エージェントへ入力するプロンプトを作成するクラス
#================================================================================
class PromptBuilder:

    #--------------------------------------------------------------------------
    # 初期化
    #--------------------------------------------------------------------------
    def __init__(self, system_prompt_text=None, system_prompt_images=None):
        
        # 文章システムプロンプトの設定
        self.system_prompt_text = system_prompt_text or ""
        
        # 画像システムプロンプトの設定
        self.system_prompt_images = system_prompt_images or []

    #--------------------------------------------------------------------------
    # システムプロンプトを作成する
    #--------------------------------------------------------------------------
    def build_system_prompt(self, rag_context_text=None):
        
        # 初期化 
        parts = []

        # 文章のシステムプロンプトがある場合
        if self.system_prompt_text:
            parts.append(self.system_prompt_text)

        # 文章データのRAG検索結果がある場合
        if rag_context_text:
            parts.append(rag_context_text)

        return "\n\n".join(parts)
    
    #--------------------------------------------------------------------------
    # システムプロンプトを初期化
    #--------------------------------------------------------------------------
    def reset(self):
        self.system_prompt_text = None
        self.system_prompt_images = []

    #--------------------------------------------------------------------------
    # システムプロンプトを更新
    #--------------------------------------------------------------------------
    def set_system_prompt(self, prompt_text):
        self.system_prompt_text = prompt_text

