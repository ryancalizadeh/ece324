import os
import json

class ConfigGenerator:
    @classmethod
    def generate_config_files(cls, file_dir: str = "configs/"):
        for i, ci_ratio in enumerate(ConfigGenerator.ci_ratio_range()):
            for j, num_real_shots in enumerate(ConfigGenerator.num_real_shots_range()):
                for k, sr_ratio in enumerate(ConfigGenerator.sr_ratio_range()):
                    ConfigGenerator.save_config(ci_ratio, num_real_shots, sr_ratio, file_dir, i, j, k)

    @staticmethod
    def save_config(ci_ratio, num_real_shots, sr_ratio, file_dir, ci_ratio_index, num_real_shots_index, sr_ratio_index):
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        name = f"ci{ci_ratio}_nrs{num_real_shots}_sr{sr_ratio}"

        config = {
            "name": name,
            "ci_ratio": ci_ratio,
            "num_real_shots": num_real_shots,
            "sr_ratio": sr_ratio,
            "ci_ratio_index": ci_ratio_index,
            "num_real_shots_index": num_real_shots_index,
            "sr_ratio_index": sr_ratio_index
        }

        with open(os.path.join(file_dir, name), "w") as f:
            json.dump(config, f)
    
    @staticmethod
    def get_num_files():
        return len(ConfigGenerator.ci_ratio_range()) * len(ConfigGenerator.num_real_shots_range()) * len(ConfigGenerator.sr_ratio_range())

    @staticmethod
    def get_results_shape():
        return (len(ConfigGenerator.ci_ratio_range()), len(ConfigGenerator.num_real_shots_range()), len(ConfigGenerator.sr_ratio_range()))

    @staticmethod
    def ci_ratio_range():
        return [x/10 for x in range(2, 11, 2)]

    @staticmethod
    def num_real_shots_range():
        return [2**i for i in range(2, 11, 2)]

    @staticmethod
    def sr_ratio_range():
        return [x/10 for x in range(2, 11, 2)]
