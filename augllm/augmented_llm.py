

from typing import Optional
from .make_image_paths import make_image_paths
from .function_calling import register_tools, generate_system_prompt
from .tool_handler import select_tool
#===================================================================================
# Coreの部分
#===================================================================================
class AugmentedLLM:

    #---------------------------------------------------------------
    # パラメータの初期化
    #---------------------------------------------------------------
    def __init__(self, llm, prompt_builder, cache_dir, tools=None, tool_selector_name="gemma3:4b"):

        # LLMの設定
        self.llm = llm
        self.prompt_builder = prompt_builder
        
        # ツールの定義
        self.tool_dict = register_tools(tools) if tools is not None else None
        self.tool_selector_name = tool_selector_name
        
        # 内部データの保存先
        self.cache_dir = cache_dir
        
        # 結果保存変数
        self.report_text = None
        self.report_images = None
        self.report_data = None
    
    #---------------------------------------------------------------
    # インスタンスを生成する関数
    #---------------------------------------------------------------
    @classmethod
    def from_config(cls, agent_info, llm, prompt_builder, tools=None):
        return cls(
            name=agent_info['name'],
            llm=llm,
            prompt_builder=prompt_builder,
            tools=tools,
            max_chat_history=agent_info['max_chat_history']
        )
    
    #---------------------------------------------------------------
    # 
    #---------------------------------------------------------------
    def get_name(self) -> str:
        return self.name
    
    #---------------------------------------------------------------
    # ロードしたモデルを止める
    #---------------------------------------------------------------
    def free_model(self):
        self.llm.free_model()
    
    #---------------------------------------------------------------
    # ツール実行結果を整形して、LLMに渡すための最終的なメッセージを作成
    #---------------------------------------------------------------
    def build_messages_with_tool_result(self, system_prompt, tool_execution_results, memory, local_history):
        
        # 初期化
        tool_results_summary_for_llm: Optional[str] = None
        
        # ツールの結果が何かある場合 YAML関連の応答 (コールあり、空コール、または処理中エラー)
        if tool_execution_results is not None:
            
            # 実際にツールコールがあったか、処理中エラーがあった
            if tool_execution_results:

                # ツールの実行結果を整理 
                formatted_results_list = []
                for res in tool_execution_results:

                    # データ取得
                    func_name = res["name"]
                    result_data = res["result"] # これはstatus, data, message を含む辞書

                    # dataフィールドはmessageに含めるか、必要に応じて別途整形してLLMに渡す
                    status = result_data.get("status", "unknown")
                    message = result_data.get("message", "結果メッセージなし")
                    images = result_data.get("images", None)  # ツールを実行した結果、画像が戻る場合がある
                    
                    # ツールの実行結果に関する追加プロンプト
                    if status == "success":
                        formatted_results_list.append(f"- {message}")
                    # ツールの実行結果を無視する場合（検索したけど該当しなかったなど）
                    elif status == "ignore":
                        pass
                    # Error or unknown                       
                    else:
                        print(f"- '{func_name}' の処理中に問題が発生しました。: {message}") 
                
                # ツールの実行結果がある場合はサマリーを作成
                if formatted_results_list:
                    tool_results_summary_for_llm = "以下の内部情報に基づき回答してください。：\n" + "\n".join(formatted_results_list)
        
        # システムプロンプトの指示に従い、ツール結果を 'user' ロールで履歴に追加
        if tool_results_summary_for_llm:
            # 文章の追加
            local_history[0]["content"] += tool_results_summary_for_llm
            # 画像の追加
            if images:
                #
                image_paths = []
                # 画像のパスの取得
                query_image_paths = make_image_paths(user_images=images, cache_dir=self.cache_dir)
                image_paths.extend(query_image_paths)
                # メッセージに追加
                local_history[0]["images"] = image_paths 
        
        # 最終応答のためのメッセージリストを作成
        # historyには、ユーザープロンプト、LLMのツール試行応答、ツール結果の要約(あれば) が含まれる
        messages = [system_prompt] + memory + local_history
        
        #
        return messages
    
    #---------------------------------------------------------------
    # LLMに渡すための最終的なメッセージを作成
    #---------------------------------------------------------------
    def build_messages(self, system_prompt, memory, local_history):
        return [system_prompt] + memory + local_history
    
    #---------------------------------------------------------------
    # LLMに渡すための最終的なメッセージを作成
    #---------------------------------------------------------------
    def build_user_message(self, text, images=None):
        
        # テキストの入力
        user_message = {
            "role": "user", 
            "content": text
        }

        # 入力画像がある場合 通常の画像データ->画像のパスに変換
        if images:
            #
            image_paths = []
            # 画像のパスの取得
            query_image_paths = make_image_paths(user_images=images, cache_dir=self.cache_dir)
            image_paths.extend(query_image_paths)
            # メッセージに追加
            user_message["images"] = image_paths 
        
        return user_message
    
    #---------------------------------------------------------------
    # 回答を生成
    #---------------------------------------------------------------
    def respond(self, user_text, user_images=None, memory=[], stream=False):
        
        # ユーザーの入力を整形
        query_message = self.build_user_message(
            text=user_text, 
            images=user_images
        )
        
        # 初期化
        messages = None
        local_history = None
        
        # 何かツールがある場合
        if self.tool_dict:

            # ルール呼び出し機能を持ったシステムプロンプトを作成
            tool_select_prompt = generate_system_prompt(functions=self.tool_dict)
            tool_select_system_prompt = {
                "role": "system",
                "content": f"{tool_select_prompt}"
            }
            
            # LLMでツールを呼び出し、結果を取得
            tool_execution_results = select_tool(
                llm=self.llm,
                tool_dict=self.tool_dict,
                system_prompt=tool_select_system_prompt,
                user_message=query_message
            )

            # 最終的なメッセージ作成のためlocal_historyを初期化
            local_history = [query_message]
            
            # 最終的なメッセージに対するシステムプロンプト
            system_prompt={
                "role": "system",
                "content": f"{self.prompt_builder.system_prompt_text}"
            }
            
            # ツールの実行結果の整理と最終的なメッセージを作成
            # local_historyを更新
            messages = self.build_messages_with_tool_result(
                system_prompt=system_prompt,
                tool_execution_results=tool_execution_results,
                memory=memory,
                local_history=local_history
            )
        # ツールが全く定義されていない場合 -> 通常の回答生成
        else:
            # local_historyを初期化
            local_history = [query_message]

            # システムプロンプトを作成
            system_prompt = {
                "role": "system",
                "content": f"{self.prompt_builder.system_prompt_text}"
            }
            
            # 最終的なメッセージを作成
            messages = self.build_messages(
                system_prompt=system_prompt,
                memory=memory,
                local_history=local_history
            )
            
        # 最終的な回答を生成
        full_reply = ""
        # ストリーミング出力
        if stream:
            found = False  # 何か返されたか確認
            for chunk in self._stream_llm_response(messages=messages):
                found = True
                full_reply += chunk
                yield chunk
            if not found:
                yield "[ストリーミング応答が得られませんでした]"            
        # 通常の生成
        else:
            #
            try:
                reply = self.llm.chat(messages=messages)["message"]["content"]
            except Exception as e:
                yield f"[エラー: {str(e)}]"
                return
            #
            full_reply = reply 
            yield full_reply
        
        # 報告用のデータ作成
        self.report_text = f"{full_reply}"
        self.report_images = None
        self.report_images = local_history[0].get("images", None)
        self.report_data = None
    
    #---------------------------------------------------------------
    # 
    #---------------------------------------------------------------
    def _stream_llm_response(self, messages):
        try:
            # 回答を生成
            generator = self.llm.chat(messages=messages, stream=True)
            if generator is None:
                yield "[エラー: モデルがストリーミング応答を返しませんでした]"
                return
            #
            any_chunk = False
            for chunk in generator:
                any_chunk = True
                # 通常生成
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
                
                # エラー情報をストリームで返す    
                elif "error" in chunk:
                    print(f"Error from llm response: {chunk['error']}")
                    yield f"\n[エラー: {chunk['error']}]"

            if not any_chunk:
                yield "[エラー: モデルからの応答が空でした]" 
        
        except Exception as e:
            yield f"[エラー発生: {str(e)}]"  