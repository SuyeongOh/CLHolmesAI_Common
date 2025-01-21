from dataclasses import dataclass

@dataclass
class BaseDataModel:
    np_data: dict
    json_data: dict
    raw_data: dict