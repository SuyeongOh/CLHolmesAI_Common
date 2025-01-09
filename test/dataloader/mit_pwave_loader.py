from test.dataloader.mit_arrhythmia_loader import MitArrhythmiaLoader


class MitPwaveLoader(MitArrhythmiaLoader):
    dataset_path = 'mit-bih-arrhythmia-database-p-wave-annotations-1.0.0'