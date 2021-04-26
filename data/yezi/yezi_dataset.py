import pandas as pd
import datasets
import os

FEATURES = datasets.Features(
    {
        "text": datasets.Value("string"),
    }
)

class YeziConfig(datasets.BuilderConfig):

    data_file : str

class Yezi(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS=YeziConfig

    def _info(self):
        return datasets.DatasetInfo(features=FEATURES)

    def _split_generators(self, dl_manager):
        
        if os.path.exists(self.config.data_file):
            raise ValueError(f'The data dile do not exists, got {self.config.data_file}')

        return [datasets.SplitGenerator(name=datasets.split.TRAIN, gen_kwargs={'data_file':self.config.data_file})]

    def _generate_examples(self, data_file):
        
        df=pd.read_excel(data_file, usecols=['B'])
        