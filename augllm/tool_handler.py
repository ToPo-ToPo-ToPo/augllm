
from typing import List, Tuple, Optional
from .function_calling import cast_arguments, extract_tool_calls_from_yaml
from .llm_interface import LLMInterface
#---------------------------------------------------------------
# ツール呼び出し用LLMのロード
#---------------------------------------------------------------
def load_tool_selector(tool_selector_name: str):
    llm = LLMInterface(
        model_name = tool_selector_name,
        options = {
            "temperature": 0.0,
            "top_k": 1.0,
            "top_p": 1.0,
        }
    )
    return llm

#---------------------------------------------------------------
# ツール呼び出し用LLMの解放
#---------------------------------------------------------------
def free_tool_selector(llm):
    llm.free_model()

#---------------------------------------------------------------
# ユーザー入力からツールを選択・実行し、その結果と履歴を返す
#---------------------------------------------------------------
def select_tool(llm, tool_dict, system_prompt: dict, user_message: dict) -> Tuple[Optional[List[dict]]]:

    # 入力メッセージの作成    
    messages = [system_prompt] + [user_message]
    llm_response_content: Optional[str] = None
    
    # ツール実行結果の初期化
    tool_execution_results = []
    called_tool_names = []
    
    # 必ず実行するツールの処理
    context = {
        "query_text": user_message['content'],
        # 他に必要な共通情報があればここに追加
    }

    # エージェントが所有しているツールをすべて確認し使うものをまとめる
    for tool_name in tool_dict:
        # ツールの取得
        tool = tool_dict.get(tool_name)
        # mustフラグかつすでにツールが実行されていない場合
        if tool.must == True and tool.name() not in called_tool_names:
            raw_args = tool.prepare_args(context)
            args = cast_arguments(tool._run, raw_args)
            print(f"Calling tool '{tool_name}' with args: {args}")
            result_dict = tool.run(**args)
            tool_execution_results.append({
                "name": tool_name, 
                "result": result_dict
            })
            # 実行されたツール名を保持
            called_tool_names.append(tool_name)

    # LLMがクエリから関連するツールを選択
    try:
        # ツール選択LLMをロード
        tool_selector_name = 'gemma3:4b'
        tool_selector = load_tool_selector(tool_selector_name=tool_selector_name)
        
        # LLMに問い合わせ
        result = tool_selector.chat(messages=messages)
        llm_response_content = result["message"]["content"].strip()

        # ツール選択LLMの解放
        free_tool_selector(llm=tool_selector)

        # YAMLからツール呼び出しを抽出
        tool_calls = extract_tool_calls_from_yaml(llm_response_content)
        
        # ツールが呼び出されていた場合
        if tool_calls:
            # 使用すると判断されたすべてのツールを実行
            for call in tool_calls:
                
                # ツールの取得
                tool_name = call["function"]["name"]
                tool = tool_dict.get(tool_name)
                
                # mustフラグのついたツールはすでに実行されているはずなので処理しない
                if tool.must == True:
                    continue

                # 実行結果を追加
                tool_execution_results.append(
                    run_tool_call(
                        tool_name=tool_name,
                        raw_args=call["function"]["arguments"],
                        tool = tool
                    )
                )
                # 実行されたツール名を保持
                called_tool_names.append(tool_name)
        #
        return tool_execution_results

    # ツール呼び出しが不要と判断されたケース
    except ValueError:
        return None if not tool_execution_results else tool_execution_results

    # ツール実行しのエラー処理
    except Exception as e:
        print(f"\n❌ Unexpected Error in select_tool: {e}")
        return [{
            "name": "system_processing_error",
            "result": {
                "status": "error",
                "data": None,
                "message": f"ツール実行中にエラー: {e}. LLM応答: {llm_response_content or 'N/A'}"
            }
        }]

#---------------------------------------------------------------
# ツール呼び出しの定型処理
#---------------------------------------------------------------
def run_tool_call(tool_name, raw_args, tool) -> dict:

    if not tool:
        print(f"[ERROR]: Function '{tool_name}' is not defined.")
        return {
            "name": tool_name,
            "result": {
                "status": "error",
                "data": None,
                "message": f"ツール '{tool_name}' は定義されていません。"
            }
        }

    try:
        args = cast_arguments(tool._run, raw_args)
        print(f"Calling tool '{tool_name}' with args: {args}")
        result_dict = tool.run(**args)
        return {"name": tool_name, "result": result_dict}

    except Exception as e:
        print(f"[ERROR]: Tool '{tool_name}' execution failed: {e}")
        return {
            "name": tool_name,
            "result": {
                "status": "error",
                "data": None,
                "message": f"ツール '{tool_name}' の実行中にエラー: {e}"
            }
        }