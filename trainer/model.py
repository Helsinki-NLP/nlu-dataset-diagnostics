import torch.optim as optim
import torch.nn as nn
from transformers import RobertaForSequenceClassification


# Specify the Transformer model
class BertModel(nn.Module):
    def __init__(self):
        """Defines the transformer model to be used.
        """
        super(BertModel, self).__init__()

        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

    def forward(self, x, attention_mask, labels):
        return self.model(x, attention_mask=attention_mask, labels=labels)


def create(args, device):
    """
    Create the model

    Args:
      args: experiment parameters.
      device: device.
    """
    model = BertModel().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    return model, optimizer
