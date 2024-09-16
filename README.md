<div align="center">

# FineTuning Large Language Models
![image](https://github.com/user-attachments/assets/b37062ba-6425-4263-a8f0-ffa0be4a697c)

  
</div>
<br>
### What is Fine-tuning?

Fine-tuning is a machine learning technique where a pre-trained model is further trained (or fine-tuned) on a new dataset, usually smaller and domain-specific, to adapt it to a particular task. In this process, the pre-trained model retains the knowledge it has learned during its initial training and applies that to the new task, often with fewer resources and training time compared to training a model from scratch.

Fine-tuning is popular in NLP, computer vision, and other AI fields, especially when using large-scale models like **BERT**, **GPT**, **T5**, or **ResNet**, which are pre-trained on general datasets.

### Key Steps in Fine-tuning
1. **Load Pre-trained Model**: Start with a model pre-trained on a large, diverse dataset.
2. **Adapt Architecture**: Adjust the model's layers or output to match the specific task (e.g., for classification or generation).
3. **Train on New Dataset**: Train the model on a new, smaller dataset specific to your task, often using a smaller learning rate to avoid overfitting or disrupting the pre-trained weights.

---

### Challenges in Fine-tuning

1. **Overfitting**: When fine-tuning on a small dataset, there’s a risk of the model overfitting and losing its generalization capabilities.
   - **Solution**: Use techniques like data augmentation, early stopping, and regularization. You can also freeze some pre-trained layers and only fine-tune the last few layers to prevent overfitting.

2. **Catastrophic Forgetting**: The model may "forget" the general knowledge it learned during pre-training when fine-tuned on a small, task-specific dataset.
   - **Solution**: Use a lower learning rate or freeze parts of the model (e.g., lower layers) to preserve the pre-trained knowledge.

3. **Limited Training Data**: Fine-tuning often involves working with smaller datasets, which may not be sufficient to adapt the model effectively.
   - **Solution**: Use data augmentation, transfer learning (by leveraging pre-trained models), and regularization techniques. Additionally, combining multiple small datasets can help.

4. **Domain Mismatch**: If there is a large difference between the domain of the pre-trained model and the target task (e.g., fine-tuning a model trained on English for use in a different language), performance might degrade.
   - **Solution**: Gradual unfreezing, where you gradually unfreeze the model’s layers and fine-tune deeper layers slowly to adapt to the new domain, can help.

5. **Hyperparameter Tuning**: Finding the right hyperparameters (e.g., learning rate, batch size, weight decay) can be challenging during fine-tuning.
   - **Solution**: Use grid search, random search, or more sophisticated approaches like Bayesian optimization to find the best hyperparameters. Start with lower learning rates since pre-trained models are sensitive to large updates.

6. **Computational Resources**: Fine-tuning large models, especially transformer-based models, can require significant computational resources, especially in terms of memory and processing power.
   - **Solution**: Use techniques like **[`Low-Rank Adaptation (LoRA)`](https://github.com/zeyadusf/topics-in-nlp-llm/tree/main/PEFT%20(Parameter-Efficient%20Fine-Tuning)/LoRA)** or other methods of **[`Parameter-Efficient Fine-Tuning (PEFT)`](https://github.com/zeyadusf/topics-in-nlp-llm/tree/main/PEFT%20(Parameter-Efficient%20Fine-Tuning))**, which reduces memory usage, or opt for 4-bit or 8-bit quantization to reduce model size.

7. **Evaluation and Validation**: Properly evaluating a fine-tuned model on new data can be difficult if the dataset is unbalanced or there are no standard metrics for the task.
   - **Solution**: Use cross-validation, domain-specific evaluation metrics (e.g., BLEU, ROUGE for text, F1 for classification), and create robust validation sets.

8. **Bias in Pre-trained Models**: The pre-trained models might carry biases from the data they were initially trained on, which can impact performance on new tasks.
   - **Solution**: Bias mitigation techniques, like re-sampling the training data or fine-tuning on more representative data, can help reduce the impact of unwanted biases.

---

<div align="center"> 
<br>
# Projects 
</div>

<table style="width:100%">
  <tr>
    <th>#</th>
    <th>Project Name</th>
    <th>Model Name</th>
    <th>Task</th>
    <th>GitHub</th>
    <th>Kaggle</th>
    <th>Hugging Face</th>
    <th>Space</th>
    <th>Notes</th>
  </tr>
  
  <tr>
    <td>1</td>
    <td>DAIGT</td>
    <td><b>DeBERTa</b></td>
    <td><b>Classification</b></td>
    <td><a href="https://github.com/zeyadusf/DAIGT-Catch-the-AI">DAIGT | Catch the AI</a></td>
    <td><a href="https://www.kaggle.com/code/zeyadusf/daigt-deberta">DAIGT | DeBERTa</a></td>
    <td><a href="https://huggingface.co/zeyadusf/deberta-DAIGT-MODELS">deberta-DAIGT-MODELS</a></td>
    <td><a href="https://huggingface.co/spaces/zeyadusf/Detection-of-AI-Generated-Text">Detection-of-AI-Generated-Text</a></td>
    <td>
      <i>Part of my Graduation Project</i><br>
      <a href="https://www.catchtheai.tech/">Catch The AI</a>
    </td>
  </tr>

  <tr>
    <td>2</td>
    <td>DAIGT</td>
    <td><b>RoBERTa</b></td>
    <td><b>Classification</b></td>
    <td><a href="https://github.com/zeyadusf/DAIGT-Catch-the-AI">DAIGT | Catch the AI</a></td>
    <td><a href="https://www.kaggle.com/code/zeyadusf/daigt-roberta">DAIGT | RoBERTa</a></td>
    <td><a href="https://huggingface.co/zeyadusf/roberta-DAIGT-kaggle">roberta-DAIGT-kaggle</a></td>
    <td><a href="https://huggingface.co/spaces/zeyadusf/Detection-of-AI-Generated-Text">Detection-of-AI-Generated-Text</a></td>
    <td>
      <i>Part of my Graduation Project</i><br>
      <a href="https://www.catchtheai.tech/">Catch The AI</a>
    </td>
  </tr>

  <tr>
    <td>3</td>
    <td>DAIGT</td> 
    <td><b>BERT</b></td>
    <td><b>Classification</b></td>
    <td><a href="https://github.com/zeyadusf/DAIGT-Catch-the-AI">DAIGT | Catch the AI</a></td>
    <td><a href="https://www.kaggle.com/code/zeyadusf/daigt-bert">DAIGT | BERT</a></td>
    <td><a href="https://huggingface.co/zeyadusf/bert-DAIGT-MODELS">bert-DAIGT-MODELS</a></td>
    <td><a href="https://huggingface.co/spaces/zeyadusf/Detection-of-AI-Generated-Text">Detection-of-AI-Generated-Text</a></td>
    <td>
      <i>Part of my Graduation Project</i><br>
      <a href="https://www.catchtheai.tech/">Catch The AI</a>
    </td>
  </tr>

  <tr>
    <td>4</td>
    <td>DAIGT</td> 
    <td><b>DistilBERT</b></td>
    <td><b>Classification</b></td>
    <td><a href="https://github.com/zeyadusf/DAIGT-Catch-the-AI">DAIGT | Catch the AI</a></td>
    <td><a href="https://www.kaggle.com/code/zeyadusf/daigt-distilbert">DAIGT | DistilBERT</a></td>
    <td><a href="https://huggingface.co/zeyadusf/distilbert-DAIGT-MODELS">distilbert-DAIGT-MODELS</a></td>
    <td><a href="https://huggingface.co/spaces/zeyadusf/Detection-of-AI-Generated-Text">Detection-of-AI-Generated-Text</a></td>
    <td>
      <i>Part of my Graduation Project</i><br>
      <a href="https://www.catchtheai.tech/">Catch The AI</a>
    </td>
  </tr>

  <tr>
    <td>5</td>
    <td>Summarization-by-Finetuning-FlanT5-LoRA</td> 
    <td><b>FlanT5</b></td>
    <td><b>Summarization</b></td>
    <td><a href="https://github.com/zeyadusf/Summarization-by-Finetuning-FlanT5-LoRA">Summarization-by-Finetuning-FlanT5-LoRA</a></td>
    <td><a href="https://www.kaggle.com/code/zeyadusf/summarization-by-finetuning-flant5-lora">Summarization by Finetuning FlanT5-LoRA</a></td>
    <td><a href="https://huggingface.co/zeyadusf/FlanT5Summarization-samsum">FlanT5Summarization-samsum </a></td>
    <td><a href="https://huggingface.co/spaces/zeyadusf/Summarizationflant5">Summarization by Flan-T5-Large with PEFT</a></td>
    <td>
      <i>use PEFT and LoRA</i><br>
    </td>
  </tr>
<tr>
    <td>6</td>
    <td>Finetune Llama2</td> 
    <td><b>Llama2</b></td>
    <td><b>Text Generation</b></td>
    <td><a href="https://github.com/zeyadusf/FineTune-Llama2">FineTune-Llama2</a></td>
    <td><a href="https://www.kaggle.com/code/zeyadusf/finetune-llama2">FineTune-Llama2</a></td>
    <td><a href="https://huggingface.co/zeyadusf/llama2-miniguanaco">llama2-miniguanaco </a></td>
    <td><a href=#">---</a></td>
    <td>
      <i>...</i><br>
    </td>
  </tr>

  <tr>
    <td>7</td>
    <td>...</td> 
    <td><b>...</b></td>
    <td><b>...</b></td>
    <td><a href="#">...</a></td>
    <td><a href="#">...</a></td>
    <td><a href="#">... </a></td>
    <td><a href=#">...</a></td>
    <td>
      <i>...</i><br>
    </td>
  </tr>

</table>
</div>


<hr>


## 📞 Contact :

<!--social media-->
<div align="center">
<a href="https://www.kaggle.com/zeyadusf" target="blank">
  <img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/kaggle.svg" alt="zeyadusf" height="30" width="40" />
</a>


<a href="https://huggingface.co/zeyadusf" target="blank">
  <img align="center" src="https://github.com/zeyadusf/zeyadusf/assets/83798621/5c3db142-cda7-4c55-bcce-cc09d5b3aa50" alt="zeyadusf" height="40" width="40" />
</a> 

 <a href="https://github.com/zeyadusf" target="blank">
   <img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/github.svg" alt="zeyadusf" height="30" width="40" />
 </a>
  
<a href="https://www.linkedin.com/in/zeyadusf/" target="blank">
  <img align="center" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linkedin/linkedin-original.svg" alt="Zeyad Usf" height="30" width="40" />
  </a>
  
  
  <a href="https://www.facebook.com/ziayd.yosif" target="blank">
    <img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/facebook.svg" alt="Zeyad Usf" height="30" width="40" />
  </a>
  
<a href="https://www.instagram.com/zeyadusf" target="blank">
  <img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="zeyadusf" height="30" width="40" />
</a> 
</div>



