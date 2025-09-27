
# Author: Shun Ogawa (a.k.a. "ToPo")
# Copyright (c) 2025 Shun Ogawa (a.k.a. "ToPo")
# License: Apache License Version 2.0

from abc import ABC, abstractmethod
#=============================================================================================
# ツールクラス(抽象クラス)
#=============================================================================================
class Tool(ABC):

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    def prepare_args(self, context: dict) -> dict:
        """
        context: 実行時に利用できるすべての情報（query_text, user_idなど）
        return: raw_args として渡す辞書
        """
        pass

    #--------------------------------------------------------------
    # ツールの実行インターフェース
    #--------------------------------------------------------------
    def run(self, **kwargs):
        return self._run(**kwargs)
    
    #--------------------------------------------------------------
    # ツールの実装部分
    #--------------------------------------------------------------
    @abstractmethod
    def _run(self, **kwargs):
        pass


