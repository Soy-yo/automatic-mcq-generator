# Automatic MCQ Generator
Implementation of Master's Degree project about Automatic Multiple Choice
Question Generation (Camilo José Cela University, Madrid, Spain).

To be able to run all scripts and the app, first it is necessary to install all
requirements:
```shell
pip install -r requirements.txt
```

## Run training
Create a json file that will store training configurations with the following
format (default values shown here).

```json
{
  "model_name": "t5-small",
  "save_to": ".",
  "preprocess_config": {
    "batch_size": 512,
    "use_validation": true,
    "num_proc": null,
    "ag": {
      "max_input_length": null,
      "max_target_length": null
    },
    "qg": {
      "max_input_length": null,
      "max_target_length": null
    },
    "dg": {
      "max_input_length": null,
      "max_target_length": null
    }
  },
  "training_args": {
    "key": "value"
  }
}
```
Then, run mcq/train.py with this config file:
```shell
python mcq/train.py --config path/to/config.json
```

### Valid configurations
#### model_name - str
Name of the model used for training.

#### use_validation - bool
Whether to run validation during training or not.

#### preprocess_config - object
Preprocess configurations

- batch_size - int. Batch size used for preprocessing to speed it up.
- num_proc - int. Number of processors to be used for preprocessing.
- ag/qg/dg - object. Specific configurations for each task (answer generation, 
  question generation and distractor generation).
  - max_input_length - int. Maximum number of tokens used for input.
  - max_target_length - int. Maximum number of tokens used for target.

#### training_args - object
See [transformers.TrainingArguments](https://huggingface.co/docs/transformers/v4.22.1/en/main_classes/trainer#transformers.TrainingArguments).

## Run evaluation
Once a model has been trained, one can run the evaluation script
mcq/evaluation.py.
```shell
python mcq/train.py [-a] [-q] [-d] [-t tokenizer] path/to/model
```
Arguments `-a`, `-q` and `-d` enable AG, QG and DG evaluation, respectively,
`-t` specifies the tokenizer, if needed (defaults to `t5-small`) and the only
mandatory argument is the path to the trained model.

## Run app
Create a .flaskenv file inside the `app` folder that sets at least the `MODEL`
env variable. Then, run the app with:
```shell
flask --app .\app\mcq_app.py run
```