import numpy as np
from dataclasses import dataclass

from test.datamodel.BaseDataModel import BaseDataModel


@dataclass
class MitdbDataModel(BaseDataModel):
    def getPidList(self):
        return np.unique(self.np_data['pid'])

    def getAfibData(self):
        return self.json_data['afib']

    def getAfibData(self, pid: str):
        return self.json_data['afib'][pid]

    def getAflData(self):
        return self.json_data['afl']

    def getAflData(self, pid: str):
        return self.json_data['afl'][pid]

    def getPidSignal(self, pid: str):
        return self.raw_data[pid]['ecg']

    def getPidFs(self, pid: str):
        return self.raw_data[pid]['fs']
