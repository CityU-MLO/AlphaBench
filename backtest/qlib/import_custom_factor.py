import os


def generate_custom_alpha_handler_code(
    fields_list,
    names_list,
    qlib_dir="~/miniconda3/envs/qlib/lib/python3.9/site-packages/qlib",
):
    """
    Generate and save a CustomAlphaHandler.py file to the qlib contrib/data directory.
    
    Args:
        fields_list (list): List of feature fields
        names_list (list): List of feature names
        qlib_dir (str): Path to qlib installation directory (default is miniconda env path)
    """
    # Expand user path (~)
    qlib_dir = os.path.expanduser(qlib_dir)

    # Prepare the output path
    output_dir = os.path.join(qlib_dir, "contrib", "data")
    output_path = os.path.join(output_dir, "CustomAlphaHandler.py")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the string for fields and names
    fields_str = repr(fields_list)
    names_str = repr(names_list)

    # Template with all necessary imports and complete class definition
    template = f"""from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.processor import Processor
from ...utils import get_callable_kwargs
from ...data.dataset import processor as processor_module
from inspect import getfullargspec

_DEFAULT_LEARN_PROCESSORS = [
    {{"class": "DropnaLabel"}},
    {{"class": "CSZScoreNorm", "kwargs": {{"fields_group": "label"}}}},
]
_DEFAULT_INFER_PROCESSORS = [
    {{"class": "ProcessInf", "kwargs": {{}}}},
    {{"class": "ZScoreNorm", "kwargs": {{}}}},
    {{"class": "Fillna", "kwargs": {{}}}},
]

def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure fit_start_time and fit_end_time are not None."
                pkwargs.update(
                    {{
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }}
                )
            proc_config = {{"class": klass.__name__, "kwargs": pkwargs}}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


class CustomAlphaHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {{
            "class": "QlibDataLoader",
            "kwargs": {{
                "config": {{
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                }},
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            }},
        }}

        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )

    def get_feature_config(self):
        fields = {fields_str}
        names = {names_str}
        return fields, names

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
"""

    # Save to file
    with open(output_path, "w") as f:
        f.write(template)

    print(f"CustomAlphaHandler.py successfully saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    fields = ["$close", "$open", "$high", "$low", "$volume"]
    names = ["close", "open", "high", "low", "volume"]

    generate_custom_alpha_handler_code(fields, names)
