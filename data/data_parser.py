from data.mit_bih_arrhythmia import MIT_BIH_ARRHYTMIA


class DataParser:
    mit_parser = MIT_BIH_ARRHYTMIA()

    def run_eval(self):
        self.mit_parser.run("arrhythmia")
        self.mit_parser.run("stress")


DataParser().run_eval()