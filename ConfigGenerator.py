import os
import json

class ConfigGenerator:
    @classmethod
    def generate_config_files(cls, file_dir: str = "configs/"):
        for sr_ratio in ConfigGenerator.sr_ratio_range():
            for num_real_shots in ConfigGenerator.num_real_shots_range():
                for ci_ratio in ConfigGenerator.ci_ratio_range():
                    ConfigGenerator.save_config(sr_ratio, num_real_shots, ci_ratio, file_dir)

    def save_config(sr_ratio, num_real_shots, ci_ratio, file_dir):
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        name = f"sr{sr_ratio}_nrs{num_real_shots}_ci{ci_ratio}"

        config = {
            "name": name,
            "sr_ratio": sr_ratio,
            "num_real_shots": num_real_shots,
            "ci_ratio": ci_ratio
        }

        with open(os.path.join(file_dir, name), "w") as f:
            json.dump(config, f)
    
    @staticmethod
    def get_num_files():
        return len(ConfigGenerator.sr_ratio_range()) * len(ConfigGenerator.num_real_shots_range()) * len(ConfigGenerator.ci_ratio_range())

    @staticmethod
    def get_results_shape():
        return (len(ConfigGenerator.sr_ratio_range()), len(ConfigGenerator.num_real_shots_range()), len(ConfigGenerator.ci_ratio_range()))

    @staticmethod
    def sr_ratio_range():
        return [x/10 for x in range(1, 10, 2)]

    @staticmethod
    def num_real_shots_range():
        return [2**i for i in range(1, 10, 2)]

    @staticmethod
    def ci_ratio_range():
        return [x/10 for x in range(1, 10, 2)]