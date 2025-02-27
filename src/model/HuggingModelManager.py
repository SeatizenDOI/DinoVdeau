
import os
from pathlib import Path
from argparse import Namespace
from datetime import date, datetime
from huggingface_hub import HfApi, HfFolder
from transformers import AutoConfig, AutoModelForImageClassification

from .model_setup import create_head
from ..utils.utils import get_config_env
from ..utils.enums import LabelType, get_training_type_from_args, ClassificationType

class HuggingModelManager():
    def __init__(self, args: Namespace, label_type: LabelType) -> None:
        self.args = args
        self.training_type = get_training_type_from_args(args)
        self.config_env = get_config_env(args.config_path)
        self.label_type = label_type

        self.model_name = ""
        self.model_name_with_username = None
        self.output_dir = Path("")
        self.resume_from_checkpoint, self.latest_checkpoint = None, None

        self.model = None
    

    def setup_model_dir(self) -> None:
        """ Return session_name and output_dir computed by args. """

        now = datetime.now()
        elapsed_second_in_day = now.hour * 3600 + now.minute * 60 + now.second
        today = f'{date.today().strftime("%Y_%m_%d")}_{elapsed_second_in_day}'
        model_without_enterprise = self.args.model_name.split("/")[1]
        model_size = model_without_enterprise[model_without_enterprise.find("-")+1:]
        freeze = 'unfreeze' if self.args.no_freeze else 'freeze'
        test_data = "prova_" if self.args.test_data else ""
        training_type = "_monolabel" if self.training_type == ClassificationType.MONOLABEL else ""
        label_type = "_probs" if self.label_type == LabelType.PROBS else ""

        self.model_name = f"{self.args.new_model_name}-{model_size}-{today}-{test_data}bs{self.args.batch_size}_{freeze}{training_type}{label_type}"
        self.output_dir = Path(self.config_env["MODEL_PATH"], self.model_name)

        
        if self.args.resume:
            self.model_name = self.config_env["MODEL_NAME"]
            self.output_dir = Path(self.config_env["MODEL_PATH"], self.model_name)
            checkpoints = [f for f in self.output_dir.iterdir() if f.name.startswith('checkpoint')]
            if len(checkpoints) != 0:
                self.latest_checkpoint = max(checkpoints, key=os.path.getctime)
                self.resume_from_checkpoint = self.latest_checkpoint
            
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Hugging face have a model name len limit fix to 96 so
        if len(self.model_name) > 96:
            self.model_name = self.model_name[0:96] 


    def setup_model(self, label_names: list, id2label: dict, label2id: dict):

        model_config = AutoConfig.from_pretrained(
            self.args.model_name,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id,
            problem_type=self.training_type.value, # Multi or monolabel
            image_size=self.args.image_size
        )
        # Get hidden_size number from model
        hidden_size = 1024 # Default value if not found.
        if hasattr(model_config, "hidden_size"):
            hidden_size = getattr(model_config, "hidden_size")
        if hasattr(model_config, "hidden_sizes"):
            hidden_size = getattr(model_config, "hidden_sizes")[-1]

        if not self.args.disable_web:
            self.model = AutoModelForImageClassification.from_pretrained(self.args.model_name, config=model_config, ignore_mismatched_sizes=True)
        else:
            model_name = self.config_env["LOCAL_MODEL_PATH"] if self.config_env["LOCAL_MODEL_PATH"] != '' else self.args.model_name

            self.model = AutoModelForImageClassification.from_pretrained(model_name, config=model_config, ignore_mismatched_sizes=True)

        if not(self.args.no_custom_head):
            self.model.classifier = create_head(hidden_size * 2, model_config.num_labels)

        if not self.args.no_freeze:
            model_name = self.args.model_name.split("/")[1].split("-")[0] # Extract dinov2 from facebook/dinov2-large
            for name, param in self.model.named_parameters():
                if name.startswith(model_name):
                    param.requires_grad = False
    
    
    def send_data_to_hugging_face(self) -> None:
        """ Send files to hugging face."""
        
        # Send data to huggingface.
        token = HfFolder.get_token()
        hf_api = HfApi(token=token)
        try:
            username = hf_api.whoami()["name"]
        except:
            print("User not found with hugging face token provide.")
            return
        
        repo_id = f"{username}/{self.model_name}"
        try:
            repo_url = hf_api.create_repo(token=token, repo_id=repo_id, private=False, exist_ok=True)
            print(f"Repository URL: {repo_url}")
        except Exception as e:
            raise NameError(f"Error creating repository: {e}")

        all_files = [f for f in self.output_dir.iterdir() if f.is_file() and f.name != "model.safetensors"]

        for filepath in all_files:
            hf_api.upload_file(
                token=token,
                path_or_fileobj=filepath,
                path_in_repo=filepath.name,
                repo_id=self.model_name_with_username,
                commit_message=f"Upload {filepath.name}"
            )

        print(f"All files successfully uploaded to the Hub: {repo_url}")


    def get_hf_username(self) -> str:
        hf_api = HfApi(token=self.config_env["HUGGINGFACE_TOKEN"])
        try:
            username = hf_api.whoami()["name"]
        except:
            raise NameError("User not found with hugging face token provide.")

        return username
    
    def get_model_name_with_username(self) -> str:
        if self.model_name_with_username == None:
            self.model_name_with_username = f"{self.get_hf_username()}/{self.model_name}"
        
        return self.model_name_with_username