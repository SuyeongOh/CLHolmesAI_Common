from dataclasses import dataclass


@dataclass
class AccuracyDataModel:
    t_count: int = 0
    p_count: int = 0
    tp_count: int = 0

    def getSense(self):
        if self.t_count != 0:
            return self.tp_count / self.t_count
        else:
            return -1

    def getPpv(self):
        if self.p_count != 0:
            return self.tp_count / self.p_count
        else :
            return -1