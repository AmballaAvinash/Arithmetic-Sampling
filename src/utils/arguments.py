import argparse


class Arguments(argparse.ArgumentParser):
    def __init__(self, groups=None):
        super().__init__(conflict_handler='resolve')
        # Common flags
        self.add_argument(
            "--out_dir", type=str, default="outputs"
        )
        self.add_argument(
            "--clean_out_dir", type=str, default="outputs"
        )
        self.add_argument(
            "--debug", action="store_true",
        )
        self.add_argument(
            "--verbose", action="store_true",
        )
        self.add_argument(
            "--seed", type=int, default=17
        )
        self.add_argument(
            "--run_id", type=int, default=None
        )

        if type(groups) is not list:
            groups = [groups]

        # Group-wise flags (group means a specific file/module in your code)
        for group in groups:
            if group == "llm":
                self.add_argument(
                    "--model", type=str
                )
                self.add_argument(
                    "--is_chat", action='store_true'
                )
                self.add_argument(
                    "--load_in_8bit", action="store_false"
                )
            if group == "self_consistency":
                self.add_argument(
                    "--eval_inf_fn_key", type=str
                )
                self.add_argument(
                    "--eval_split", type=str
                )
                self.add_argument( 
                    "--dataset_name", type=str
                )
                self.add_argument(
                    "--dataset_subname", type=str
                )
                self.add_argument(
                    "--eval_n_samples", type=int
                )
                self.add_argument(
                    "--eval_retrieval_strategy", type=str
                )
                self.add_argument(
                    "--eval_output_sampling_strategy", type=str
                )
                self.add_argument(
                    "--eval_output_temperature", type=str
                )
                self.add_argument(
                    "--eval_output_top_k", type=str
                )
                self.add_argument(
                    "--eval_output_top_p", type=str
                )
            
            pass
            # Examples:
            # if group == "args_group_1":
            #     self.add_argument(
            #         "--arg1", type=str
            #     )
            #     self.add_argument(
            #         "--arg2", type=str, required=True
            #     )
            #     self.add_argument(
            #         "--arg3", action="store_true"
            #     )
            #
            # if group == "args_group_2":
            #     self.add_argument(
            #         "--arg4", type=int, default=None
            #     )
            #     self.add_argument(
            #         "--arg5", type=int, default=4
            #     )
            #     self.add_argument(
            #         "--arg6", nargs='+', default=None,
            #         help="Accepts one or more strings as a space-separated list: 'option1' 'option2' 'option3'"
            #     )
            #     self.add_argument(
            #         "--arg7", action=argparse.BooleanOptionalAction, default=True,
            #     )
            #     self.add_argument(
            #         "--arg8", action=argparse.BooleanOptionalAction, default=False,
            #     )
            #     self.add_argument(
            #         "--arg9", default="option1", choices=['option1', 'option2', 'option3']
            #     )
