import logging
from agents.base_agent import LLMAgent, Msg
from dl.predictor import DLPredictor
from dl.backtester import Backtester

logger = logging.getLogger(__name__)


class DLPredictorAgent(LLMAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 在 Agent 初始化时预加载模型权重，避免每次 reply() 重复读取磁盘
        print(f"  [{self.__class__.__name__}] 预加载深度学习模型权重...")
        self._predictor = DLPredictor(commodity_key=self.commodity["key"])
        self._backtester = Backtester(model_type="lstm", commodity_key=self.commodity["key"])

    def reply(self, x=None):
        if x is None:
            return super().reply(x)
        print(f"\n[{self.name}] 正在调用多维特征深度学习引擎进行推演 ...")
        try:
            # 使用预加载的模型进行预测
            dl_result = self._predictor.predict_next_day()

            # 附带回测绩效，帮助首席策略官评估模型可信度
            try:
                backtest_summary = self._backtester.get_summary_for_agent(test_days=250)
                dl_result += f"\n\n{backtest_summary}"
            except Exception as e:
                logger.warning(f"回测摘要获取失败（不影响主预测）: {e}")

            msg = Msg(name=x.name, role=x.role, content=dl_result)
            return super().reply(msg)
        except Exception as e:
            logger.error(f"深度学习引擎异常: {e}", exc_info=True)
            msg = Msg(name=x.name, role=x.role, content=f"深度学习引擎异常：{e}")
            return super().reply(msg)
