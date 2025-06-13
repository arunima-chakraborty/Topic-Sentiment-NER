# 🔍 Multi Headed Model for Topic, Sentiment and NER extraction 

A robust model designed on distilbert that can do BIOES topic extraction, sentiment analysis and NER detection. 

I have divided this project into 2 parts. 

1. [PTQ: 16 bit model training with ONNX conversion](https://github.com/arunima-chakraborty/Topic-Sentiment-NER/tree/main/16-bit):
   - 
   - [model_class.py](https://github.com/arunima-chakraborty/Topic-Sentiment-NER/blob/main/16-bit/model_class.py): The model is defined here. I have used 3 heads alongwith different weights assigned to each of the heads to define the priorities of the heads
   - [data_class.py](data_class.py): The data is formatted here. I've used multilingual data and the topics tagged in BIOES tagging and the NER is tagged as O, PER, ORG n MISC.
   - [training-model.ipynb](https://github.com/arunima-chakraborty/Topic-Sentiment-NER/blob/main/16-bit/training-model.ipynb): The model is trained in 16 bit precision and used to predict a sample text.
   - [onnx-conversion.py](https://github.com/arunima-chakraborty/Topic-Sentiment-NER/blob/main/16-bit/onnx-conversion.py): Model is quantized to 8 bit precision (Post Training  to reduce the size of the model and improve the latency in production
   - [postprocessing-onnx.ipynb](https://github.com/arunima-chakraborty/Topic-Sentiment-NER/blob/main/16-bit/postprocessing-onnx.ipynb): Added the postprocessing steps to dividing the text into chunks of <50 words and predict from the onnx model. Postprocessing includes merge the subwords and getting the entire meaningful phrase and the final sentiment from the whole text (max sentiment from all the chunks)
     
2. [QAT: Quantized Model on LoRA Config](https://github.com/arunima-chakraborty/Topic-Sentiment-NER/tree/main/LORA%20Config)
   - 
   - [lora-training.ipynb](https://github.com/arunima-chakraborty/Topic-Sentiment-NER/blob/main/LORA%20Config/lora-training.ipynb): Model class and data class are defined same as PTQ version, and have added LORA config parameters to train the model in 8 bit format itself. 




