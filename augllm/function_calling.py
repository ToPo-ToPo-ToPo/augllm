
# Author: Shun Ogawa (a.k.a. "ToPo")
# Copyright (c) 2025 Shun Ogawa (a.k.a. "ToPo")
# License: Apache License Version 2.0

import inspect
from typing import Callable, Any, Dict, List, Optional
import yaml
import re
from .tool import Tool
#------------------------------------------------------------------------------------
# 関数 (ツール) を名前にマッピングして登録します。
#------------------------------------------------------------------------------------
def register_tools(tools: List[Tool]) -> Dict[str, Tool]:
    return {tool.name(): tool for tool in tools}

#------------------------------------------------------------------------------------
# 呼び出し可能なターゲット (関数やメソッド) を取得します。
# 関数、クラス、インスタンスを受け取って、signatureを取得可能な対象を返す
#------------------------------------------------------------------------------------
def get_callable_target(fn_or_instance: Any) -> Callable:
    #
    if inspect.isfunction(fn_or_instance):
        return fn_or_instance
    # ツールインスタンスの場合、_runメソッドを優先
    elif hasattr(fn_or_instance, "_run") and callable(getattr(fn_or_instance, "_run")):
        return getattr(fn_or_instance, "_run")
    elif hasattr(fn_or_instance, "run") and callable(getattr(fn_or_instance, "run")):
        return getattr(fn_or_instance, "run")
    elif inspect.isclass(fn_or_instance):
        if hasattr(fn_or_instance, "run") and callable(getattr(fn_or_instance, "run")): # クラスメソッド run
            return getattr(fn_or_instance, "run")
        # クラスの __call__ (通常は __init__) はツールの実行シグネチャとは異なるため、ここでは対象外とする
        raise TypeError(f"クラス {fn_or_instance} から実行可能なターゲットメソッド (run または _run) を特定できませんでした。")
    elif callable(fn_or_instance): # その他の callable オブジェクト
        return fn_or_instance
    else:
        raise TypeError(f"{fn_or_instance} は関数、クラス、または適切な callable ではありません。")

#------------------------------------------------------------------------------------
# LLMにツール呼び出しのルールと関数一覧を伝えるシステムプロンプトを生成します。（YAML版）
#------------------------------------------------------------------------------------
def generate_system_prompt(functions: Dict[str, Callable]) -> str:
    #
    descriptions = []
    for name, tool_instance in functions.items():
        try:
            # tool_instanceの_runメソッドのシグネチャを取得する
            target_callable = get_callable_target(tool_instance)
            sig = inspect.signature(target_callable)
            
            params_list = []
            for k, v in sig.parameters.items():
                if k == 'self': # 'self' パラメータは除外
                    continue
                param_type = v.annotation.__name__ if v.annotation != inspect.Parameter.empty else 'Any'
                if v.default != inspect.Parameter.empty:
                    params_list.append(f"{k}: {param_type} (optional, default={v.default})")
                else:
                    params_list.append(f"{k}: {param_type}")
            params = ", ".join(params_list)
            
            # ツールのdescriptionを使用
            docstring = tool_instance.description().strip()
            descriptions.append(f"- `{name}({params})`: {docstring}")
        except Exception as e:
            descriptions.append(f"- `{name}()`: <定義情報の取得に失敗: {e}>")

    # Function calling用のプロンプトを作成
    prompt = f"""ユーザーの指示に応じて、以下のツールを呼び出します。
### ツール呼び出しのルール（厳守）：

1. ユーザーの指示に基づいてツールが必要な場合、以下の**厳密な YAML 形式**のみで返答してください：

```yaml
tool_calls:
  - tool:
      name: <ツール名>
      arguments:
        <引数名1>: <値1>
        <引数名2>: <値2>
        ...
```
- `tool_calls` はリストです。
- 各要素は `tool` オブジェクトを含みます。
- `tool` オブジェクトは `name` と `arguments` を含みます。
- `arguments` は引数名をキー、値をバリューとするマップ（辞書）です。
- YAMLコードブロック (```yaml ... ```) で囲んでください。

ツール一覧：
{chr(10).join(descriptions)}

2. 関数を使う必要がないときは通常の自然言語で返答してください。

この2つのルールを厳守して、ユーザーの要求に最適な形で応答してください。
"""
    return prompt

#------------------------------------------------------------------------------------
# 関数のシグネチャに基づいて、生の引数辞書を適切な型にキャストします。
#------------------------------------------------------------------------------------
def cast_arguments(func_callable: Callable, raw_args: dict) -> dict:
    sig = inspect.signature(func_callable)
    casted_args = {}

    valid_param_names = {name for name in sig.parameters if name != 'self'}

    for name, value in raw_args.items():
        if name not in valid_param_names:
            print(f"Warning: Argument '{name}' is not a valid parameter for {func_callable.__name__}. It will be ignored.")
            continue

        param = sig.parameters[name]
        target_type = param.annotation

        if target_type == inspect.Parameter.empty or target_type == Any:
            casted_args[name] = value
            continue

        try:
            if target_type == str:
                casted_args[name] = str(value)
            elif target_type == int:
                casted_args[name] = int(value)
            elif target_type == float:
                casted_args[name] = float(value)
            elif target_type == bool:
                if isinstance(value, str):
                    casted_args[name] = value.lower() in ['true', 'yes', 'on', '1']
                else:
                    casted_args[name] = bool(value)
            else: # リスト、辞書などはYAMLパーサーが適切に変換していることを期待
                casted_args[name] = value
        except (ValueError, TypeError) as e:
            print(f"Warning: Failed to cast argument '{name}' (value: '{value}') to type {target_type.__name__}. Using original value. Error: {e}")
            casted_args[name] = value

    # デフォルト値を持つ引数が省略された場合の処理
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if name not in casted_args and param.default is not inspect.Parameter.empty:
            casted_args[name] = param.default
        elif name not in casted_args and param.default is inspect.Parameter.empty : #必須引数が指定されてない
            print(f"Warning: Required argument '{name}' for {func_callable.__name__} was not provided by LLM.")
            # ここでエラーをraiseするか、Noneを設定するかは設計による
            # casted_args[name] = None # またはエラー

    return casted_args

#------------------------------------------------------------------------------------
# LLMの応答からYAML形式のtool_callsセクションを抽出・解析します。
#------------------------------------------------------------------------------------
def extract_tool_calls_from_yaml(content: str) -> List[Dict[str, Any]]:
    content = content.strip()

    # YAMLコードブロックの検出
    match = re.search(r"```yaml\s*(.*?)\s*```", content, re.DOTALL)

    yaml_str: Optional[str]
    if match:
        yaml_str = match.group(1).strip()
    elif content.startswith("tool_calls:"): # YAMLブロックなしで直接YAMLが始まる場合も許容
        yaml_str = content
    else:
        yaml_str = None

    if not yaml_str:
        raise ValueError("YAMLコードブロックまたはtool_callsで始まるYAML文字列が見つかりませんでした。")

    try:
        parsed = yaml.safe_load(yaml_str)

        if not isinstance(parsed, dict) or "tool_calls" not in parsed:
            raise ValueError("YAMLのルートが辞書型でなく、'tool_calls'キーが含まれていません。")

        calls = parsed["tool_calls"] # 修正: キー名からコロンを削除
        if not isinstance(calls, list):
            # LLMが単一のツールコールをリストでなく直接記述した場合を許容する（例：tool_calls: {tool: ...}）
            if isinstance(calls, dict) and "tool" in calls:
                calls = [calls] # リストに変換
            else:
                raise ValueError("'tool_calls'の値はリストである必要があります。")

        extracted_tool_calls = []
        for item in calls:
            if not isinstance(item, dict) or "tool" not in item:
                raise ValueError("各tool_callエントリは辞書型で、'tool'キーを含む必要があります。")

            tool_data = item["tool"] # 修正: キー名からコロンを削除
            if not isinstance(tool_data, dict):
                raise ValueError("'tool'の値は辞書型である必要があります。")

            name = tool_data.get("name") # 修正: .getを使用し、キー名からコロンを削除
            arguments = tool_data.get("arguments", {}) # 修正: .getを使用し、引数がない場合は空の辞書をデフォルトに

            if not isinstance(name, str) or not name: # 名前が文字列で、空でないことを確認
                raise ValueError(f"無効なツール名です: '{name}'。ツール名は文字列である必要があります。")
            if not isinstance(arguments, dict):
                raise ValueError(f"ツールの引数 ('arguments') は辞書型である必要がありますが、'{type(arguments)}'でした (ツール: {name})。")

            # OpenAI互換の形式に少し寄せる
            extracted_tool_calls.append({
                "function": {"name": name, "arguments": arguments} 
            })
        return extracted_tool_calls

    except yaml.YAMLError as e:
        raise ValueError(f"YAML解析エラー: {e}\n解析対象のYAML文字列:\n---\n{yaml_str}\n---")
    except Exception as e: # その他のパース関連エラー
        raise ValueError(f"tool_callsの解析中に予期せぬエラーが発生しました: {e}\n解析対象のYAML文字列:\n---\n{yaml_str}\n---")
