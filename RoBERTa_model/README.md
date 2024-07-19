# RoBERTa model part:
This part is the RoBERTa model. Among them, RoBERTa_model.ipynb contains the 
preprocessing and model adjustment of the RoBERTa model. Since the overall labeled 
data is relatively large, we divided it into sections according to the section name: 
Australia news, UK news, World news and Opinion.
### Update: Functionality of `roberta_new.py` and `load-roberta.py`

**`roberta_new.py`**:
- This script is used for fine-tuning and generating the RoBERTa model. It includes the necessary code to train and save the fine-tuned model.

**`load-roberta.py`**:
- This script is designed to load the fine-tuned RoBERTa model and use it to generate tokenized articles. It facilitates the loading of the model and applying it to your text data.

Please refer to these scripts to understand their specific functionalities and how to use them in your workflow. If you have any questions or need further assistance, feel free to contact me.

Because our adjusted model is about 400MB, we use Google Drive to store it.
Here is the fine_tuned RoBERTa model: 
https://drive.google.com/file/d/1oOg8mb0OtHiwEHzivOhoHcCWjvxXhHWh/view?usp=sharing
The fine-tuned model was generated using `roberta_new.py` from this folder.
If you want to run it, please use `load-roberta.py` to load it.

Our labelled Guardian data is also very large and is stored in the Google Drive link below. 
We have also prepared versions classified according to different sections, 
namely Australia news, UK news, World news and Opinion.
Here is Roberta’s version of the labeled Guardian data: 
https://drive.google.com/file/d/1bkZC6LQyO04AaRAgRzuMwWvkoxpQJP7U/view?usp=sharing
