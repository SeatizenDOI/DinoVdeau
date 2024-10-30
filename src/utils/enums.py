from enum import Enum
from argparse import Namespace

class ClassificationType(Enum):
    MULTILABEL = "multi_label_classification"
    MONOLABEL = "single_label_classification"


class LabelType(Enum):
    BIN = "binary"
    PROBS = "probabilities"
    

def get_training_type_from_args(args: Namespace) -> ClassificationType:
    """ Return ClassificationType object from args. """
    if args.training_type == "multilabel": return ClassificationType.MULTILABEL
    elif args.training_type == "monolabel": return ClassificationType.MONOLABEL
    else:
        raise NameError(f"Argument provide for training type not found: {args.training_type}.") 
