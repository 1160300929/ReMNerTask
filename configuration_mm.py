from transformers import BertConfig,LxmertConfig
#
#visual configs for mm model
#
class MMBertConfig(BertConfig):
    def __init__(self,
                 visual_dim=2048):
        self.visual_embed = visual_dim



class MMLxmertConfig(LxmertConfig):
    def __init__(self,
                 visual_dim=2048):
        self.visual_embed = visual_dim



