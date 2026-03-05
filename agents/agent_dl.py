import logging
from agents.base_agent import LLMAgent, Msg
from dl.predictor import DLPredictor
from dl.backtester import Backtester

logger = logging.getLogger(__name__)


class DLPredictorAgent(LLMAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # pre-load model weights at agent init time to avoid re-reading disk on every reply()
        print(f"  [{self.__class__.__name__}] Pre-loading deep learning model weights...")
        self._predictor = DLPredictor(commodity_key=self.commodity["key"])
        self._backtester = Backtester(model_type="lstm", commodity_key=self.commodity["key"])

    def reply(self, x=None):
        if x is None:
            return super().reply(x)
        print(f"\n[{self.name}] Running multi-feature deep learning engine for inference ...")
        try:
            # use the pre-loaded model to generate predictions
            dl_result = self._predictor.predict_next_day()

            # include backtest performance to help the chief strategist assess model credibility
            try:
                backtest_summary = self._backtester.get_summary_for_agent(test_days=250)
                dl_result += f"\n\n{backtest_summary}"
            except Exception as e:
                logger.warning(f"Failed to retrieve backtest summary (does not affect main prediction): {e}")

            msg = Msg(name=x.name, role=x.role, content=dl_result)
            return super().reply(msg)
        except Exception as e:
            logger.error(f"Deep learning engine error: {e}", exc_info=True)
            msg = Msg(name=x.name, role=x.role, content=f"Deep learning engine error: {e}")
            return super().reply(msg)
