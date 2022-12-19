import torch
import transformers
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import metrics
import pandas as pd


# multi class or multi lable: to change the loss function, 
#                               num classes, dataset lables and metrics   


class datasets:
    def __init__(self, texts, targets = None, max_len = 64):
        self.texts = texts
        self.targets = targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case = False
        ) 
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens = True,
            max_length = self.max_len,
            padding = "max_length",
        )
        resp = {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float)
            # "targets": torch.tensor(self.targets[idx], dtype=torch.long)
            # "targets": torch.tensor(self.targets[idx]#vector#, dtype=torch.float)
        }
        return resp

class textmodel(nn.Module):
    def __init__(self, num_classes, num_train_steps):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(
            "bert-base-uncased", return_dict = False
        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, num_classes)
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"
    
    def fetch_optimizer(self):
        opt = AdamW(self.parameters(), lr = 1e-4)
        return opt
    
    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps = 0, num_training_steps = self.num_train_steps
        )
        return sch
    
    def loss(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets) 
        # return nn.CrossEntropyLoss()(outputs, targets)
    
    def moniter_metrics(self, outputs, targets):
        outputs = nn.Sigmoid(outputs).detach().cpu().numpy() >= 0.5
        # outputs = torch.argmax(outputs, axis = 1).cpu().detach().numpy()
        targets = targets.detach().cpu().numpy()
        return {
            "accuracy": metrics.accuracy_score(targets, outputs)
        }        
    
    def forward(self, ids, mask, token_type_ids, targets = None):
        _, x = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids)
        x = self.bert_drop(x)
        x = self.out(x)
        if targets is not None:
            loss = self.loss(x, targets=targets)
            met = self.moniter_metrics(x, targets=targets)
            return x, loss, met
        return x, None, {}

def train_model(fold, train_batch_size = 16, epochs = 10):
    df = pd.read_csv("")
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_test = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = datasets(df_train.review.values, df_train.sentiment.values)
    test_dataset = datasets(df_test.review.values, df_test.sentiment.values)

    num_train_steps = int(len(df_train) / train_batch_size * epochs)
    model = textmodel(num_classes=1, num_train_steps= num_train_steps)

    # es = tez.callbacks.EarlyStopping(monitor="valid_loss", patients= 3)
    model.fit(
        train_dataset,
        valid_dataset = test_dataset,
        device = "cuda",
        epochs = 10,
        train_bs = train_batch_size,
        # callback = [es]
    )


    torch.save(model, "model.pkl")
    model = torch.load("model.pkl", device = "cuda")
    torch.save(model.state_dict(), 'model_parameters.pkl')
    model.load_state_dict(torch.load("model_parameters.pkl"))

    preds = model.predict(test_dataset, device = "cuda")


if __name__ == '__main__':
    train_model() 
    # new note     