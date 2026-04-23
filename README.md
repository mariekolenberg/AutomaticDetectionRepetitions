This repository contains the NLP models described in the paper 'Automatic Detection of Direct and Self Repetitions in Naturalistic Speech Recordings of French- and Dutch-Speaking Autistic Children'. 
The models aim at detecting repetitions in the speech of a target child, more specifically direct repetitions from utterances of other speakers pronounced less than 10 seconds before the target utterance, 
or self-repetitions (from utterances spoken by the target child inside the same sound file). For more information about the goals and the working of the models, see the paper. 
Using the `preprocess_textgrid` function in 'Extract_repetition_candidates', the user can **extract all pairs of utterances** from a Textgrid file that are candidates for a source-repetition pair.
```From Extract_repetition_candidates import preprocess_textgrid
textgrid_file= 'My_example.Textgrid'
non_speech_tiers= ['Ignore_this_tier','Ignore_this_one_too']

test= textgrid_preprocessing(textgrid_file= textgrid_file, child_tier='Transcription', non_speech_tiers= non_speech_tiers)
rep_df= test.get_repetition_candidates(repetition_type='direct')
```
The resulting dataframe looks like this:

<img width="1202" height="94" alt="Example_utterances" src="https://github.com/user-attachments/assets/01ee604f-7131-4970-9af5-6e7dcc6607f9" />

Then, the user can **predict** whether the candidate utterance pairs are repetitive or non-repetitive.
For this purpose, we have developed two models in 'Predict'.
The **first model** computes linguistic cosine similarity scores to distinguish repetitive from non-repetitive utterance pairs. 
The prediction is based on comparing the similarity score with a threshold determined in gold standard annotated data. 
Our thresholds are specific for Dutch and French, and using the default thresholds will thus only work for data in these languages. 
The user can specify own thresholds for use of the model on other languages. For a reflexion on the cross-linguistic applicability of the model, see the paper.
```From Predict import CosSim_predict
source_texts= ['ik ga naar de zee','mama','ik eet een appel']
repetition_texts= ['ik ga naar de winkel','mama', 'jij eet een banaan']
cossim_model= CosSim_predict(source_texts=source_texts, repetition_texts=repetition_texts,language='nl')
preds_cossim= cossim_model.predict_repetitions(repetition_type='self-repetition',vector_type='semantic')
```
The **second model type** are BERT-models that we finetuned on our gold standard annotation data in Dutch and French.
The models therefore only work on data in these languages. The finetuned BERT models can be found at https://huggingface.co/m0183394/AutomaticRepetitionDetection.

```From Predict import BERT_predict
source_texts= ['ik ga naar de zee','mama','ik eet een appel']
repetition_texts= ['ik ga naar de winkel','mama', 'jij eet een banaan']
preds_BERT= BERT_predict(language= 'nl', repetition_type= 'self-repetition', source_texts= source_texts, repetition_texts=repetition_texts,
               max_len=50, return_all_probs=False, device='cpu')
```

