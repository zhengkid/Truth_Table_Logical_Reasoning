import os
from together import Together


if __name__ == '__main__':

    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    # print(os.environ.get("WANDB_API_KEY"))

    train_size = 200
    with_exp = True
    round = 3
    n_epoch = 1 if round == 1 or round == 2 else 5
    n_checkpoints = 1 if round == 1 or round == 2 else 5

    assert train_size in [200, 400], "Train size must be 200 or 400"

    if train_size == 200:
        # if not with_exp:
        #     file_name = 'file-6280b07b-8d14-44a6-bdf8-5536f4f55578'
        # else:
        if round == 1:
            file_name = 'file-f146b3d7-8169-403a-8da0-07777df1a46c'
        elif round == 2:
            file_name = 'file-1de5663e-9430-468e-aae2-83d12dcfada7'
        elif round ==3:
            file_name = 'file-8ed661fd-88b4-4927-aad4-05e02be5af72'

    # Trigger fine-tuning job
    resp = client.fine_tuning.create(
        suffix = f"Star-r3",
        model= "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        training_file=file_name,
        n_checkpoints=n_checkpoints,
        n_epochs=n_epoch,
        batch_size=16,
        learning_rate=2e-5,
        # wandb_api_key=os.environ.get("WANDB_API_KEY"),
    )

    print(resp)
    print(resp.id)